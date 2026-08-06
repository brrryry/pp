import os
import sys
import json
import logging
import zipfile
import io
import shutil
import hashlib
import requests
import numpy as np
import pandas as pd
import threading
import time
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse
from osrparse import Replay
from osrparse.utils import Key
import rosu_pp_py as rosu
import joblib

# Add parent directory to path to find config.py and src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from server.core.parser import parse_osu_file
from server.core.features import extract_map_features
from server.core.portfolio import compute_mechanical_portfolio, compute_map_portfolio_skills

# Set up logging
from logger_setup import setup_logger
logger = setup_logger("main")

app = FastAPI(title="Osu! PP & Skill Profiler API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPLAYS_DIR = "data/replays"
MAPS_DIR = config.maps_path
SUMMARY_CSV = "data/replays/replays_summary.csv"

# Index local maps on startup
local_maps_cache = {}

MAP_SKILLS_CACHE_FILE = "data/map_skills_cache.json"
map_skills_cache = {}
map_skills_cache_lock = threading.Lock()

# Hybrid Caching: Detect Redis URL, fall back to SQLite
use_redis = False
redis_client = None

redis_url = os.getenv("REDIS_URL") or getattr(config, 'redis_url', None)
if redis_url:
    try:
        import redis
        redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        use_redis = True
        logger.info(f"Connected to Redis cache at {redis_url}")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis ({e}). Falling back to SQLite cache.")

SQLITE_PATH = "data/osu_profiler.db"

def init_db():
    # Setup SQLite tables (always create database file to persist hits, even if Redis is active for skills caching)
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS maps_hash_cache (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_maps_hash ON maps_hash_cache(hash)")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS map_skills_cache (
                beatmap_hash TEXT PRIMARY KEY,
                skills_json TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS replay_hits_cache (
                replay_file TEXT PRIMARY KEY,
                hits_json TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS replay_mechanical_skills (
                replay_file TEXT PRIMARY KEY,
                skills_json TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to initialize SQLite database: {e}")

# Initialize Database Cache tables
import sqlite3
init_db()

from server.DatabaseManager import DatabaseManager
from server.pipelines.ReplayAnalysisPipeline import ReplayAnalysisPipeline
from server.pipelines.BeatmapIngestionPipeline import BeatmapIngestionPipeline

db = DatabaseManager(SQLITE_PATH)
db.init_db()

def save_replay_hits(replay_file, hits_json):
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO replay_hits_cache (replay_file, hits_json)
            VALUES (?, ?)
        """, (replay_file, hits_json))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to save replay hits to SQLite: {e}")

def get_replay_hits(replay_file):
    if not os.path.exists(SQLITE_PATH):
        return None
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT hits_json FROM replay_hits_cache WHERE replay_file = ?", (replay_file,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        logger.error(f"Failed to get replay hits from SQLite: {e}")
        return None

def save_mechanical_skills(replay_file, skills_dict):
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO replay_mechanical_skills (replay_file, skills_json)
            VALUES (?, ?)
        """, (replay_file, json.dumps(skills_dict)))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to save mechanical skills to SQLite: {e}")

def get_mechanical_skills(replay_file):
    if not os.path.exists(SQLITE_PATH):
        return None
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT skills_json FROM replay_mechanical_skills WHERE replay_file = ?", (replay_file,))
        row = cursor.fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    except Exception as e:
        logger.error(f"Failed to get mechanical skills from SQLite: {e}")
        return None

def load_map_skills_cache():
    global map_skills_cache
    map_skills_cache = {}
    
    if use_redis:
        try:
            cached = redis_client.hgetall("map_skills_cache")
            if cached:
                with map_skills_cache_lock:
                    for k, v in cached.items():
                        map_skills_cache[k] = json.loads(v)
                logger.info(f"Loaded {len(cached)} cached map skill profiles from Redis.")
        except Exception as e:
            logger.error(f"Failed to load map skills from Redis: {e}")
    else:
        if os.path.exists(SQLITE_PATH):
            try:
                conn = sqlite3.connect(SQLITE_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT beatmap_hash, skills_json FROM map_skills_cache")
                rows = cursor.fetchall()
                conn.close()
                
                with map_skills_cache_lock:
                    for h, json_str in rows:
                        map_skills_cache[h] = json.loads(json_str)
                logger.info(f"Loaded {len(rows)} cached map skill profiles from SQLite.")
            except Exception as e:
                logger.error(f"Failed to load map skills cache from SQLite: {e}")

def save_map_skills_cache(beatmap_hash, skills_only):
    # Update memory representation
    with map_skills_cache_lock:
        map_skills_cache[beatmap_hash] = skills_only
        
    # Persist entry
    json_str = json.dumps(skills_only)
    if use_redis:
        try:
            redis_client.hset("map_skills_cache", beatmap_hash, json_str)
        except Exception as e:
            logger.error(f"Failed to save map skills to Redis: {e}")
    else:
        try:
            conn = sqlite3.connect(SQLITE_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO map_skills_cache (beatmap_hash, skills_json)
                VALUES (?, ?)
            """, (beatmap_hash, json_str))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save map skills to SQLite: {e}")

# Load the cache on startup
load_map_skills_cache()

def get_cached_map_skills(osu_path, beatmap_hash):
    global map_skills_cache
    with map_skills_cache_lock:
        if beatmap_hash in map_skills_cache:
            return map_skills_cache[beatmap_hash]
        
    profile = compute_map_skills_profile(osu_path)
    if profile:
        skills_only = {
            "SnapAim": profile.get("SnapAim", 0),
            "FlowAim": profile.get("FlowAim", 0),
            "Speed": profile.get("Speed", 0),
            "Streaming": profile.get("Streaming", 0),
            "Stamina": profile.get("Stamina", 0),
            "Tech": profile.get("Tech", 0),
            "FingerControl": profile.get("FingerControl", 0),
            "Precision": profile.get("Precision", 0),
            "Reading": profile.get("Reading", 0),
            "VisualDensity": profile.get("VisualDensity", 0),
            "AimControl": profile.get("AimControl", 0)
        }
        save_map_skills_cache(beatmap_hash, skills_only)
        return skills_only
    return None

def compute_md5(file_path):
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception:
        return None

def scan_local_maps():
    global local_maps_cache
    local_maps_cache = {}
    
    # Load cache from SQLite or Redis
    cache_data = {}
    if use_redis:
        try:
            cached = redis_client.hgetall("maps_metadata_cache")
            if cached:
                for k, v in cached.items():
                    cache_data[k] = json.loads(v)
        except Exception:
            pass
    else:
        if os.path.exists(SQLITE_PATH):
            try:
                conn = sqlite3.connect(SQLITE_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT path, hash, mtime, size FROM maps_hash_cache")
                rows = cursor.fetchall()
                conn.close()
                for path, h, mtime, size in rows:
                    cache_data[path] = {
                        'hash': h,
                        'mtime': mtime,
                        'size': size
                    }
            except Exception:
                pass
            
    # Gather all file paths to index
    file_paths = []
    
    # 1. Server maps cache folder (completely isolated from the user's hard drive)
    if os.path.exists(MAPS_DIR):
        for f in os.listdir(MAPS_DIR):
            if f.endswith(".osu"):
                file_paths.append(os.path.abspath(os.path.join(MAPS_DIR, f)))
                        
    new_cache_data = {}
    cache_updated = False
    
    for path in file_paths:
        try:
            stat = os.stat(path)
            mtime = stat.st_mtime
            size = stat.st_size
            
            # Use path as unique key in cache
            if path in cache_data and cache_data[path].get('mtime') == mtime and cache_data[path].get('size') == size:
                h = cache_data[path]['hash']
                new_cache_data[path] = cache_data[path]
                local_maps_cache[h] = path
            else:
                h = compute_md5(path)
                if h:
                    new_cache_data[path] = {
                        'hash': h,
                        'mtime': mtime,
                        'size': size
                    }
                    local_maps_cache[h] = path
                    cache_updated = True
        except Exception:
            pass
            
    # Save cache if updated or if files were removed
    if cache_updated or len(new_cache_data) != len(cache_data):
        if use_redis:
            try:
                deleted_keys = set(cache_data.keys()) - set(new_cache_data.keys())
                if deleted_keys:
                    redis_client.hdel("maps_metadata_cache", *deleted_keys)
                    for k in deleted_keys:
                        old_hash = cache_data[k]['hash']
                        redis_client.hdel("maps_hash_cache", old_hash)
                        
                for path, meta in new_cache_data.items():
                    if path not in cache_data or cache_data[path] != meta:
                        redis_client.hset("maps_metadata_cache", path, json.dumps(meta))
                        redis_client.hset("maps_hash_cache", meta['hash'], path)
            except Exception as e:
                logger.error(f"Failed to save maps metadata cache to Redis: {e}")
        else:
            try:
                conn = sqlite3.connect(SQLITE_PATH)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM maps_hash_cache")
                for path, meta in new_cache_data.items():
                    cursor.execute("""
                        INSERT INTO maps_hash_cache (path, hash, mtime, size)
                        VALUES (?, ?, ?, ?)
                    """, (path, meta['hash'], meta['mtime'], meta['size']))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to save maps metadata cache to SQLite: {e}")
            
    logger.info(f"Loaded {len(local_maps_cache)} local maps into cache.")
 
def register_beatmap(path, beatmap_hash):
    local_maps_cache[beatmap_hash] = path
    try:
        stat = os.stat(path)
        mtime = stat.st_mtime
        size = stat.st_size
    except Exception:
        mtime = 0.0
        size = 0
        
    if use_redis:
        try:
            meta = {"hash": beatmap_hash, "mtime": mtime, "size": size}
            redis_client.hset("maps_metadata_cache", path, json.dumps(meta))
            redis_client.hset("maps_hash_cache", beatmap_hash, path)
        except Exception as e:
            logger.error(f"Failed to register beatmap in Redis: {e}")
    else:
        try:
            conn = sqlite3.connect(SQLITE_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO maps_hash_cache (path, hash, mtime, size)
                VALUES (?, ?, ?, ?)
            """, (path, beatmap_hash, mtime, size))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to register beatmap in SQLite: {e}")

def ingest_single_map(path, beatmap_hash):
    """
    Registers a map in the path cache and parses/ingests it into the relational database.
    """
    register_beatmap(path, beatmap_hash)
    try:
        parsed = parse_osu_file(path)
        if parsed:
            feats = extract_map_features(parsed)
            if feats:
                feats['file_hash'] = beatmap_hash
                if 'mapset_id' not in feats or not feats['mapset_id']:
                    feats['mapset_id'] = parsed.get('metadata', {}).get('BeatmapSetID', 0)
                return db.add_map(feats)
    except Exception as e:
        logger.error(f"Failed to ingest single map {beatmap_hash} into relational DB: {e}")
    return None

# Initialize map indexing
scan_local_maps()

def download_and_extract_mapset(beatmapset_id, expected_checksum):
    logger.info(f"Downloading beatmapset {beatmapset_id}...")
    
    # Mirror endpoints list
    mirrors = [
        ("Nerinyan", f"https://api.nerinyan.moe/d/{beatmapset_id}"),
        ("OsuDirect", f"https://osu.direct/api/d/{beatmapset_id}"),
        ("Chimu", f"https://api.chimu.moe/v1/download/{beatmapset_id}"),
        ("Sayobot", f"https://txy1.sayobot.cn/beatmaps/download/full/{beatmapset_id}")
    ]
    
    response = None
    success = False
    
    for mirror_name, url in mirrors:
        logger.info(f"Attempting download from {mirror_name}...")
        max_retries = 2
        backoff = 2.0
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code in (429, 503):
                    logger.warning(f"{mirror_name} busy ({response.status_code}). Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                response.raise_for_status()
                # Verify that the response content is actually a zip file (PK header)
                if response.content.startswith(b"PK\x03\x04"):
                    success = True
                    break
                else:
                    logger.warning(f"Response from {mirror_name} is not a valid ZIP file archive.")
                    break
            except Exception as e:
                logger.warning(f"Error downloading from {mirror_name}: {e}")
                time.sleep(1.0)
        
        if success:
            logger.info(f"Successfully downloaded beatmapset {beatmapset_id} from {mirror_name}!")
            break
            
    if not success or response is None:
        logger.error(f"Failed to download mapset {beatmapset_id} from all mirrors.")
        return None
        
    try:
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        extracted_path = None
        existing_count = len([f for f in os.listdir(MAPS_DIR) if f.endswith(".osu")])
        idx = existing_count
        
        for file in zip_file.namelist():
            if file.endswith(".osu"):
                zip_file.extract(file, MAPS_DIR)
                old_path = os.path.join(MAPS_DIR, file)
                new_name = f"{beatmapset_id}_{idx}.osu"
                new_path = os.path.join(MAPS_DIR, new_name)
                
                if os.path.exists(new_path):
                    os.remove(new_path)
                os.rename(old_path, new_path)
                
                h = compute_md5(new_path)
                if h:
                    register_beatmap(new_path, h)
                    if h == expected_checksum:
                        extracted_path = new_path
                idx += 1
        return extracted_path
    except Exception as e:
        logger.error(f"Download error: {e}")
        return None

osu_client = None

def get_osu_client():
    global osu_client
    if osu_client is None:
        try:
            from osu import Client
            osu_client = Client.from_credentials(
                config.osu_api_client_id,
                config.osu_api_client_secret,
                config.osu_api_redirect_uri
            )
        except Exception as e:
            logger.error(f"Failed to initialize osu! API client: {e}")
    return osu_client

def get_beatmap_path(beatmap_hash, download_if_missing=True):
    if beatmap_hash in local_maps_cache:
        return local_maps_cache[beatmap_hash]
        
    # Check if the map file already exists on disk
    disk_path = os.path.join(MAPS_DIR, f"{beatmap_hash}.osu")
    if os.path.exists(disk_path):
        local_maps_cache[beatmap_hash] = disk_path
        return disk_path
    
    if not download_if_missing:
        return None
        
    # Query API lookup
    try:
        client = get_osu_client()
        if client:
            bm = client.lookup_beatmap(checksum=beatmap_hash)
            if bm:
                return download_and_extract_mapset(bm.beatmapset_id, beatmap_hash)
    except Exception as e:
        logger.error(f"API lookup failed: {e}")
    return None

# Note: compute_mechanical_portfolio is now imported from server.core.portfolio

def process_replay_data(replay_path, osu_map_path):
    replay = Replay.from_path(replay_path)
    parsed_map = parse_osu_file(osu_map_path)
    
    # Extract clicks
    current_time = 0
    clicks = []
    prev_keys = 0
    for event in replay.replay_data:
        current_time += event.time_delta
        keys_val = event.keys.value
        for mask in (1, 2, 4, 8):
            if (keys_val & mask) and not (prev_keys & mask):
                clicks.append((current_time, event.x, event.y, mask))
        prev_keys = keys_val
        
    # Get difficulty settings and mods to calculate exact hit windows and radius
    import osrparse
    
    od = float(parsed_map['difficulty'].get('OverallDifficulty', 8.0))
    cs = float(parsed_map['difficulty'].get('CircleSize', 4.0))
    
    speed_multiplier = 1.0
    if osrparse.Mod.HardRock in replay.mods:
        od = min(10.0, od * 1.4)
        cs = min(10.0, cs * 1.3)
    elif osrparse.Mod.Easy in replay.mods:
        od = od * 0.5
        cs = cs * 0.5
        
    if osrparse.Mod.DoubleTime in replay.mods or osrparse.Mod.Nightcore in replay.mods:
        speed_multiplier = 1.5
    elif osrparse.Mod.HalfTime in replay.mods:
        speed_multiplier = 0.75
        
    # Calculate target hit windows in real-time milliseconds
    window_300 = 80.0 - 6.0 * od
    window_100 = 140.0 - 8.0 * od
    window_50 = 200.0 - 10.0 * od
    radius = 54.4 - 4.48 * cs
    
    hit_objects = parsed_map.get('hit_objects', [])
    clicks_left = clicks.copy()
    hit_results = []
    window = 150.0  # Search window in game-time (approx. 100ms real-time on DT)
    
    for obj in hit_objects:
        if obj['type'] == 'spinner':
            continue
        target_time = obj['time']
        target_x = obj['x']
        target_y = obj['y']
        
        best_click = None
        best_idx = -1
        min_diff = float('inf')
        
        for i, click in enumerate(clicks_left):
            click_time = click[0]
            diff = abs(click_time - target_time)
            if diff < window and diff < min_diff:
                min_diff = diff
                best_click = click
                best_idx = i
                
        if best_click is not None:
            click_time, click_x, click_y, click_mask = best_click
            timing_offset = click_time - target_time
            real_offset = timing_offset / speed_multiplier  # Convert game-time delta to real-time
            aim_distance = ((click_x - target_x)**2 + (click_y - target_y)**2)**0.5
            dx = click_x - target_x
            dy = click_y - target_y
            
            # Judgment scoring: must hit circle spatially and satisfy temporal hit windows
            is_hit = False
            score = 0
            if aim_distance <= radius and abs(real_offset) <= window_50:
                is_hit = True
                if abs(real_offset) <= window_300:
                    score = 300
                elif abs(real_offset) <= window_100:
                    score = 100
                else:
                    score = 50
            
            hit_results.append({
                'target_time': target_time,
                'target_x': target_x,
                'target_y': target_y,
                'type': obj['type'],
                'hit': is_hit,
                'score': score,
                'timing_offset': timing_offset,
                'aim_distance': aim_distance,
                'dx': dx,
                'dy': dy
            })
            clicks_left.pop(best_idx)
        else:
            hit_results.append({
                'target_time': target_time,
                'target_x': target_x,
                'target_y': target_y,
                'type': obj['type'],
                'hit': False,
                'score': 0,
                'timing_offset': None,
                'aim_distance': None,
                'dx': None,
                'dy': None
            })
            
    df_hits = pd.DataFrame(hit_results)
    if df_hits.empty:
        return None
        
    hits_only = df_hits[df_hits['hit'] == True]
    total_notes = len(df_hits)
    hits_count = len(hits_only)
    misses_count = total_notes - hits_count
    
    # Calculate exact binned accuracy percentage matching osu! formula
    count_300 = len(df_hits[df_hits['score'] == 300])
    count_100 = len(df_hits[df_hits['score'] == 100])
    count_50 = len(df_hits[df_hits['score'] == 50])
    
    accuracy = 0.0
    if total_notes > 0:
        accuracy = (300 * count_300 + 100 * count_100 + 50 * count_50) / (300 * total_notes) * 100.0
    
    avg_offset = hits_only['timing_offset'].mean() if hits_count > 0 else 0.0
    abs_offset = hits_only['timing_offset'].abs().mean() if hits_count > 0 else 0.0
    std_offset = hits_only['timing_offset'].std() if hits_count > 1 else 0.0
    unstable_rate = std_offset * 10.0 if not pd.isna(std_offset) else 0.0
    avg_aim_error = hits_only['aim_distance'].mean() if hits_count > 0 else 0.0
    std_aim_error = hits_only['aim_distance'].std() if hits_count > 1 else 0.0
    
    basename = os.path.basename(replay_path).replace('.osr', '')
    
    # Compute map features directly from the parsed map (already in memory)
    map_features = {}
    try:
        feats = extract_map_features(parsed_map)
        if feats:
            map_features = feats
    except Exception:
        pass
            
    # Default parameters if feature extraction failed
    density = map_features.get('density_notes_per_sec', total_notes / (parsed_map['hit_objects'][-1]['time'] / 1000.0) if len(parsed_map['hit_objects']) > 0 else 1.0)
    duration = (parsed_map['hit_objects'][-1]['time'] - parsed_map['hit_objects'][0]['time']) / 1000.0 if len(parsed_map['hit_objects']) > 0 else 100.0
    slider_ratio = map_features.get('sliders_ratio', 0.2)
    angle_sharp = map_features.get('angle_sharp_ratio', 0.3)
    velocity = map_features.get('velocity_mean', 1.0)
    
    analysis = {
        'replay_file': os.path.basename(replay_path),
        'beatmap_hash': replay.beatmap_hash,
        'player': replay.username,
        'map_title': parsed_map['metadata'].get('Title', 'Unknown'),
        'map_artist': parsed_map['metadata'].get('Artist', 'Unknown'),
        'difficulty_name': parsed_map['metadata'].get('Version', 'Unknown'),
        'mods': str(replay.mods),
        'total_notes': total_notes,
        'hits': hits_count,
        'misses': misses_count,
        'accuracy_percent': accuracy,
        'avg_offset_ms': avg_offset if not pd.isna(avg_offset) else 0.0,
        'abs_offset_ms': abs_offset if not pd.isna(abs_offset) else 0.0,
        'unstable_rate': unstable_rate,
        'avg_aim_error_px': avg_aim_error if not pd.isna(avg_aim_error) else 0.0,
        'std_aim_error_px': std_aim_error if not pd.isna(std_aim_error) else 0.0,
        # Save structural details for profile math
        'map_density': density,
        'map_duration': duration,
        'map_slider_ratio': slider_ratio,
        'map_angle_sharp_ratio': angle_sharp,
        'map_velocity': velocity
    }
    
    # Save hit arrays to SQLite database cache
    hits_json = df_hits.to_json(orient='records')
    save_replay_hits(os.path.basename(replay_path), hits_json)
    
    # Compute and save per-note mechanical skill portfolio
    try:
        mech_skills = compute_mechanical_portfolio(hit_results, parsed_map.get('difficulty', {}))
        if mech_skills:
            save_mechanical_skills(os.path.basename(replay_path), mech_skills)
            analysis['mechanical_skills'] = mech_skills
    except Exception as e:
        logger.error(f"Failed to compute mechanical portfolio: {e}")
        
    return analysis

def download_missing_maps_bg(hashes):
    logger.info(f"Background thread starting to download {len(hashes)} missing maps...")
    downloaded = 0
    
    for h in hashes[:15]:
        try:
            if db.get_map_portfolio(h):
                continue
                
            disk_path = os.path.join(MAPS_DIR, f"{h}.osu")
            if not os.path.exists(disk_path):
                osu_path = get_beatmap_path(h)
            else:
                osu_path = disk_path
                
            if osu_path and os.path.exists(osu_path):
                db_map_id = db.get_map_id_by_hash(h)
                if not db_map_id:
                    ingest_single_map(osu_path, h)
                downloaded += 1
                time.sleep(1.2)
        except Exception as e:
            logger.error(f"Failed to download/ingest map {h} in background: {e}")
            
    logger.info(f"Background map download complete. Successfully ingested {downloaded} maps.")

@app.get("/api/users")
def get_users():
    try:
        players = db.get_unique_players()
        if not players and os.path.exists(SUMMARY_CSV):
            df = pd.read_csv(SUMMARY_CSV)
            players = df['player'].unique().tolist()
        return players
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/{username}")
def get_user_profile(username: str):
    try:
        # 1. Fetch local plays from database
        local_plays = db.get_player_replays(username)
        
        # Initialize default player stats
        total_plays = 0
        avg_accuracy = 90.0
        avg_ur = 0.0
        avg_aim_error = 0.0
        history = []
        mechanical_skills_list = []
        user_plays = pd.DataFrame()
        
        if local_plays:
            total_plays = len(local_plays)
            avg_accuracy = sum(p['accuracy_percent'] for p in local_plays) / total_plays
            
            # UR and Aim error averages (filtering out None/0.0 values)
            urs = [p['unstable_rate'] for p in local_plays if p['unstable_rate'] is not None and p['unstable_rate'] > 0]
            avg_ur = sum(urs) / len(urs) if urs else 0.0
            
            aim_errors = [p['avg_aim_error_px'] for p in local_plays if p['avg_aim_error_px'] is not None]
            avg_aim_error = sum(aim_errors) / len(aim_errors) if aim_errors else 0.0
            
            history = local_plays
            
            # Gather mechanical skills
            for p in local_plays:
                m_json = p.get('mechanical_json')
                if m_json:
                    if isinstance(m_json, str):
                        mech = json.loads(m_json)
                    else:
                        mech = m_json
                    if mech:
                        mechanical_skills_list.append(mech)
        else:
            # Fallback to CSV for reverse compatibility
            if os.path.exists(SUMMARY_CSV):
                df = pd.read_csv(SUMMARY_CSV)
                user_plays = df[df['player'].str.lower() == username.lower()].copy()
            
            if not user_plays.empty:
                user_plays['accuracy_percent'] = pd.to_numeric(user_plays['accuracy_percent'], errors='coerce')
                user_plays['unstable_rate'] = pd.to_numeric(user_plays['unstable_rate'], errors='coerce')
                user_plays['avg_aim_error_px'] = pd.to_numeric(user_plays['avg_aim_error_px'], errors='coerce')
                
                user_plays = user_plays.fillna({
                    'accuracy_percent': 90.0,
                    'unstable_rate': 200.0,
                    'avg_aim_error_px': 15.0
                })
                
                total_plays = len(user_plays)
                avg_accuracy = user_plays['accuracy_percent'].mean()
                avg_ur = user_plays['unstable_rate'].mean()
                avg_aim_error = user_plays['avg_aim_error_px'].mean()
                history = user_plays.to_dict(orient='records')
                
                for _, row in user_plays.iterrows():
                    replay_file = row.get('replay_file', '')
                    if replay_file:
                        mech = db.get_mechanical_skills(replay_file)
                        if mech:
                            mechanical_skills_list.append(mech)
                            
        # 2. Fetch official plays from osu! API (with caching)
        top_plays = []
        recent_plays = []
        
        cached_scores = db.get_cached_api_scores(username)
        if cached_scores:
            top_plays, recent_plays = cached_scores
            logger.info(f"API Cache HIT for player: {username}")
        else:
            logger.info(f"API Cache MISS for player: {username}. Fetching from official API...")
            client = get_osu_client()
            if client:
                try:
                    user = client.get_user(username, key='username')
                    if user:
                        try:
                            scores_1 = client.get_user_scores(user.id, type="best", limit=100, offset=0)
                        except Exception as e:
                            logger.error(f"Error fetching top plays offset 0: {e}")
                            scores_1 = []
                            
                        try:
                            scores_2 = client.get_user_scores(user.id, type="best", limit=100, offset=100)
                        except Exception as e:
                            logger.error(f"Error fetching top plays offset 100: {e}")
                            scores_2 = []
                            
                        api_top_scores = scores_1 + scores_2
                        
                        for s in api_top_scores:
                            try:
                                title = s.beatmapset.title if s.beatmapset else (s.beatmap.beatmapset.title if s.beatmap and s.beatmap.beatmapset else "Unknown")
                                artist = s.beatmapset.artist if s.beatmapset else (s.beatmap.beatmapset.artist if s.beatmap and s.beatmap.beatmapset else "Unknown")
                                version = s.beatmap.version if s.beatmap else "Unknown"
                                
                                great = s.statistics.great or 0
                                ok_hits = s.statistics.ok or 0
                                meh = s.statistics.meh or 0
                                miss = s.statistics.miss or 0
                                
                                top_plays.append({
                                    'replay_file': None,
                                    'player': username,
                                    'map_title': title,
                                    'map_artist': artist,
                                    'difficulty_name': version,
                                    'mods': "+".join([str(m) for m in s.mods]) if s.mods else "NoMod",
                                    'total_notes': (s.beatmap.count_circles + s.beatmap.count_sliders + s.beatmap.count_spinners) if s.beatmap else (great + ok_hits + meh + miss),
                                    'hits': great + ok_hits + meh,
                                    'misses': miss,
                                    'accuracy_percent': (s.accuracy * 100.0) if s.accuracy else 0.0,
                                    'avg_offset_ms': None,
                                    'abs_offset_ms': None,
                                    'unstable_rate': None,
                                    'avg_aim_error_px': None,
                                    'std_aim_error_px': None,
                                    'pp': float(s.pp) if s.pp is not None else None,
                                    'weight': round(float(s.weight.percentage) if s.weight else 100.0, 1),
                                    'created_at': str(s.created_at) if hasattr(s, 'created_at') else None,
                                    'is_api_score': True,
                                    'score_type': 'best',
                                    'beatmap_hash': s.beatmap.checksum if s.beatmap else None
                                })
                            except Exception as e:
                                logger.error(f"Error mapping top score: {e}")
                                
                        try:
                            api_recent_scores = client.get_user_scores(user.id, type="recent", limit=50, include_fails=True)
                        except Exception as e:
                            logger.error(f"Error fetching recent plays: {e}")
                            api_recent_scores = []
                            
                        for s in api_recent_scores:
                            try:
                                title = s.beatmapset.title if s.beatmapset else (s.beatmap.beatmapset.title if s.beatmap and s.beatmap.beatmapset else "Unknown")
                                artist = s.beatmapset.artist if s.beatmapset else (s.beatmap.beatmapset.artist if s.beatmap and s.beatmap.beatmapset else "Unknown")
                                version = s.beatmap.version if s.beatmap else "Unknown"
                                
                                great = s.statistics.great or 0
                                ok_hits = s.statistics.ok or 0
                                meh = s.statistics.meh or 0
                                miss = s.statistics.miss or 0
                                
                                recent_plays.append({
                                    'replay_file': None,
                                    'player': username,
                                    'map_title': title,
                                    'map_artist': artist,
                                    'difficulty_name': version,
                                    'mods': "+".join([str(m) for m in s.mods]) if s.mods else "NoMod",
                                    'total_notes': (s.beatmap.count_circles + s.beatmap.count_sliders + s.beatmap.count_spinners) if s.beatmap else (great + ok_hits + meh + miss),
                                    'hits': great + ok_hits + meh,
                                    'misses': miss,
                                    'accuracy_percent': (s.accuracy * 100.0) if s.accuracy else 0.0,
                                    'avg_offset_ms': None,
                                    'abs_offset_ms': None,
                                    'unstable_rate': None,
                                    'avg_aim_error_px': None,
                                    'std_aim_error_px': None,
                                    'pp': float(s.pp) if s.pp is not None else None,
                                    'created_at': str(s.created_at) if hasattr(s, 'created_at') else None,
                                    'is_api_score': True,
                                    'score_type': 'recent'
                                })
                            except Exception as e:
                                logger.error(f"Error mapping recent score: {e}")
                                
                        db.save_api_scores(username, top_plays, recent_plays)
                except Exception as e:
                    logger.error(f"Error querying user profile from osu! API: {e}")

        if total_plays == 0 and top_plays:
            total_plays = len(top_plays)
            avg_accuracy = sum(p['accuracy_percent'] for p in top_plays) / len(top_plays)
            avg_ur = 0.0
            avg_aim_error = 0.0

        KEY_MAP = {
            "SnapAim": "Snap Aim",
            "Speed": "Speed",
            "Streaming": "Streaming",
            "FingerControl": "Finger Control",
            "Reading": "Reading",
            "VisualDensity": "Visual Density",
            "Tech": "Tech",
            "AimControl": "Aim Control",
            "FlowAim": "Flow Aim",
            "Precision": "Precision",
            "Stamina": "Stamina"
        }

        # Calculate skill levels based on the demands of maps in their Top 200 list
        api_skills = {}
        weighted_skills_sum = {k: 0.0 for k in KEY_MAP.values()}
        weights_sum = 0.0
        missing_hashes = []

        for play in top_plays:
            h = play.get('beatmap_hash')
            if not h:
                continue

            # 1. Try DB first
            skills = db.get_map_portfolio(h)
            
            # 2. Try in-memory legacy cache second
            if not skills:
                with map_skills_cache_lock:
                    skills = map_skills_cache.get(h)
            
            # 3. Try local disk path third
            if not skills:
                osu_path = local_maps_cache.get(h)
                if not osu_path:
                    disk_path = os.path.join(MAPS_DIR, f"{h}.osu")
                    if os.path.exists(disk_path):
                        osu_path = disk_path
                        local_maps_cache[h] = disk_path
                        
                if osu_path and os.path.exists(osu_path):
                    skills = get_cached_map_skills(osu_path, h)

            # If found, aggregate skills
            if skills:
                skills_mapped = {}
                for k, v in skills.items():
                    user_key = KEY_MAP.get(k)
                    if user_key:
                        skills_mapped[user_key] = v
                
                acc = play['accuracy_percent'] / 100.0
                misses = play['misses']
                perf_multiplier = (acc ** 2) * (0.95 ** misses)
                weight = play.get('weight', 100.0) / 100.0

                for user_k in KEY_MAP.values():
                    if user_k in skills_mapped:
                        weighted_skills_sum[user_k] += skills_mapped[user_k] * perf_multiplier * weight
                weights_sum += weight
            else:
                missing_hashes.append(h)

        # Trigger background downloads asynchronously if missing
        if missing_hashes:
            threading.Thread(
                target=download_missing_maps_bg,
                args=(missing_hashes,),
                daemon=True
            ).start()

        if weights_sum > 0:
            api_skills = {k: weighted_skills_sum[k] / weights_sum for k in KEY_MAP.values()}

        # Build mechanical portfolio from stored per-replay skills
        mechanical_portfolio = {}
        if mechanical_skills_list:
            axes = mechanical_skills_list[0].keys()
            for axis in axes:
                vals = [m[axis] for m in mechanical_skills_list if axis in m]
                display_axis = KEY_MAP.get(axis, axis)
                mechanical_portfolio[display_axis] = round(float(np.mean(vals)), 1) if vals else 50.0

        potential_portfolio = {}
        if api_skills:
            potential_portfolio = {k: round(v, 1) for k, v in api_skills.items()}

        if local_plays or not user_plays.empty or top_plays:
            # Map history items to remove mechanical_json and prevent bandwidth load
            history_cleaned = []
            for play in history:
                p_copy = play.copy()
                if 'mechanical_json' in p_copy:
                    del p_copy['mechanical_json']
                history_cleaned.append(p_copy)

            return {
                'username': username,
                'summary': {
                    'total_plays': total_plays,
                    'avg_accuracy': avg_accuracy,
                    'avg_ur': avg_ur,
                    'avg_aim_error': avg_aim_error
                },
                'skills': {
                    'potential': potential_portfolio,
                    'mechanical': mechanical_portfolio
                },
                'plays': history_cleaned,
                'top_plays': top_plays,
                'recent_plays': recent_plays
            }
        else:
            raise HTTPException(status_code=404, detail=f"User {username} has no local replays or API profile.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def weighted_stats(values, weights):
    """
    Computes weighted mean, weighted standard deviation, and weighted percentiles (10th, 50th, 90th).
    """
    values = np.array(values)
    weights = np.array(weights)
    if len(values) == 0:
        return {
            'mean': 95.0,
            'std': 2.0,
            'p10': 92.0,
            'p50': 95.0,
            'p90': 98.0
        }
    weights = weights / np.sum(weights)
    
    # Weighted mean
    mean = np.sum(values * weights)
    
    # Weighted variance/std
    variance = np.sum(weights * (values - mean) ** 2)
    std = np.sqrt(variance)
    
    # Weighted percentile helper
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumulative_weights = np.cumsum(sorted_weights)
    
    def get_percentile(p):
        idx = np.searchsorted(cumulative_weights, p / 100.0)
        idx = min(idx, len(sorted_values) - 1)
        return float(sorted_values[idx])
        
    return {
        'mean': float(mean),
        'std': float(std) if std > 0.0 else 1.0,
        'p10': get_percentile(10),
        'p50': get_percentile(50),
        'p90': get_percentile(90)
    }

@app.get("/api/predict")
def predict_accuracy(username: str, beatmap_hash: str, mods: Optional[str] = None):
    """
    Predicts the accuracy a player will achieve on a specific beatmap, using our trained ML model.
    """
    if not load_accuracy_predictor_model():
        raise HTTPException(status_code=503, detail="Accuracy predictor model is not loaded or not trained yet.")
        
    try:
        KEY_MAP = {
            "SnapAim": "Snap Aim",
            "FlowAim": "Flow Aim",
            "Speed": "Speed",
            "Streaming": "Streaming",
            "Stamina": "Stamina",
            "Tech": "Tech",
            "FingerControl": "Finger Control",
            "Precision": "Precision",
            "Reading": "Reading",
            "VisualDensity": "Visual Density",
            "AimControl": "Aim Control"
        }
        
        profile = {
            'avg_accuracy': 95.0,
            'peak_accuracy': 98.0,
            'volatility_accuracy': 2.0,
            
            'avg_ur': 100.0,
            'peak_ur': 80.0,
            'volatility_ur': 15.0,
            
            'avg_aim_error': 15.0,
            'peak_aim_error': 10.0,
            'volatility_aim_error': 3.0,
            
            'aim_cs_slope': 1.5,
            'tap_density_slope': 5.0
        }
        
        for skill in KEY_MAP.keys():
            profile[f'user_potential_{skill}'] = 50.0
            profile[f'user_mechanical_{skill}'] = 50.0

        # Fetch local plays
        local_plays = db.get_player_replays(username)
        mechanical_skills_list = []
        
        if local_plays:
            accuracies = []
            urs = []
            aim_errors = []
            weights = []
            
            # Connect to DB to check map stats for slopes
            conn = sqlite3.connect(SQLITE_PATH)
            cursor = conn.cursor()
            
            cs_pairs = []
            density_pairs = []
            
            for i, play in enumerate(local_plays):
                # Recency weight: 30-play half-life style decay
                w = 0.95 ** (len(local_plays) - 1 - i)
                
                if 'mechanical_json' in play and play['mechanical_json']:
                    try:
                        mech = json.loads(play['mechanical_json'])
                        mechanical_skills_list.append((mech, w))
                    except Exception:
                        pass
                        
                if play.get('accuracy') is not None:
                    accuracies.append(play['accuracy'])
                    weights.append(w)
                if play.get('unstable_rate') is not None:
                    urs.append(play['unstable_rate'])
                if play.get('avg_aim_error_px') is not None:
                    aim_errors.append(play['avg_aim_error_px'])
                    
                # Collect data for slopes
                map_id = play.get('map_id')
                if map_id is not None:
                    cursor.execute("SELECT circle_size, density_notes_per_sec FROM map_stats WHERE map_id = ?", (map_id,))
                    res = cursor.fetchone()
                    if res:
                        cs, density = res
                        if play.get('avg_aim_error_px') is not None:
                            cs_pairs.append((cs, play['avg_aim_error_px']))
                        if play.get('unstable_rate') is not None:
                            density_pairs.append((density, play['unstable_rate']))
                            
            conn.close()
            
            # Fit CS-Aim slope
            if len(cs_pairs) >= 3:
                try:
                    xs = np.array([p[0] for p in cs_pairs])
                    ys = np.array([p[1] for p in cs_pairs])
                    A = np.vstack([xs, np.ones(len(xs))]).T
                    m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
                    profile['aim_cs_slope'] = float(m)
                except Exception:
                    pass
                    
            # Fit Tapping Density-UR slope
            if len(density_pairs) >= 3:
                try:
                    xs = np.array([p[0] for p in density_pairs])
                    ys = np.array([p[1] for p in density_pairs])
                    A = np.vstack([xs, np.ones(len(xs))]).T
                    m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
                    profile['tap_density_slope'] = float(m)
                except Exception:
                    pass

            # Compute weighted stats
            if accuracies:
                w_acc = weights[:len(accuracies)]
                stats_acc = weighted_stats(accuracies, w_acc)
                profile['avg_accuracy'] = stats_acc['mean']
                profile['peak_accuracy'] = stats_acc['p90']
                profile['volatility_accuracy'] = stats_acc['std']
                
            if urs:
                w_ur = weights[:len(urs)]
                stats_ur = weighted_stats(urs, w_ur)
                profile['avg_ur'] = stats_ur['mean']
                profile['peak_ur'] = stats_ur['p10']
                profile['volatility_ur'] = stats_ur['std']
                
            if aim_errors:
                w_aim = weights[:len(aim_errors)]
                stats_aim = weighted_stats(aim_errors, w_aim)
                profile['avg_aim_error'] = stats_aim['mean']
                profile['peak_aim_error'] = stats_aim['p10']
                profile['volatility_aim_error'] = stats_aim['std']

            if mechanical_skills_list:
                for skill in KEY_MAP.keys():
                    vals = [m[0][skill] for m in mechanical_skills_list if skill in m[0]]
                    w_vals = [m[1] for m in mechanical_skills_list if skill in m[0]]
                    if vals:
                        profile[f'user_mechanical_{skill}'] = weighted_stats(vals, w_vals)['mean']

        # Fetch cached API plays
        cached_scores = db.get_cached_api_scores(username)
        top_plays = []
        if cached_scores:
            top_plays = cached_scores[0]

        if top_plays:
            accuracies = [p['accuracy_percent'] for p in top_plays if 'accuracy_percent' in p]
            weights = [p.get('weight', 100.0) / 100.0 for p in top_plays if 'accuracy_percent' in p]
            if accuracies:
                stats_acc = weighted_stats(accuracies, weights)
                profile['avg_accuracy'] = stats_acc['mean']
                profile['peak_accuracy'] = stats_acc['p90']
                profile['volatility_accuracy'] = stats_acc['std']

            weighted_skills_sum = {k: 0.0 for k in KEY_MAP.keys()}
            weights_sum = 0.0
            for play in top_plays:
                h = play.get('beatmap_hash')
                if not h:
                    continue
                skills = db.get_map_portfolio(h)
                if skills:
                    acc = play.get('accuracy_percent', 95.0) / 100.0
                    misses = play.get('misses', 0)
                    perf_multiplier = (acc ** 2) * (0.95 ** misses)
                    weight = (play.get('weight', 100.0) / 100.0) * perf_multiplier
                    
                    for k in KEY_MAP.keys():
                        if k in skills:
                            weighted_skills_sum[k] += skills[k] * weight
                    weights_sum += weight

            if weights_sum > 0:
                for k in KEY_MAP.keys():
                    profile[f'user_potential_{k}'] = weighted_skills_sum[k] / weights_sum

        # 3. Get map stats and portfolio
        map_id = db.get_map_id_by_hash(beatmap_hash)
        if not map_id:
            # Try to lookup and download/ingest map set
            osu_path = get_beatmap_path(beatmap_hash)
            if osu_path and os.path.exists(osu_path):
                map_id = ingest_single_map(osu_path, beatmap_hash)
                
        if not map_id:
            raise HTTPException(status_code=404, detail="Beatmap stats or details not found in database.")

        # Query stats
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM map_stats WHERE map_id = ?", (map_id,))
        stat_res = cursor.fetchone()
        cursor.execute("SELECT * FROM map_portfolios WHERE map_id = ?", (map_id,))
        port_res = cursor.fetchone()
        
        cursor.execute("PRAGMA table_info(map_stats)")
        stat_cols = [c[1] for c in cursor.fetchall()]
        stat_dict = dict(zip(stat_cols, stat_res))
        
        cursor.execute("PRAGMA table_info(map_portfolios)")
        port_cols = [c[1] for c in cursor.fetchall()]
        port_dict = dict(zip(port_cols, port_res))
        conn.close()
        
        # Build features dictionary
        features = dict(profile)
        
        features['map_cs'] = stat_dict.get('circle_size', 4.0)
        features['map_od'] = stat_dict.get('overall_difficulty', 8.0)
        features['map_ar'] = stat_dict.get('approach_rate', 9.0)
        features['map_hp'] = stat_dict.get('hp_drain', 5.0)
        features['map_star_rating'] = stat_dict.get('star_rating', 5.0)
        features['map_duration'] = stat_dict.get('duration_seconds', 120.0)
        features['map_density'] = stat_dict.get('density_notes_per_sec', 3.0)
        features['map_total_objects'] = stat_dict.get('total_objects', 500)
        features['map_sliders_ratio'] = stat_dict.get('sliders_ratio', 0.2)
        
        for k in KEY_MAP.keys():
            features[f'map_demand_{k}'] = port_dict.get(k, 50.0)
            
        # Parse mods
        mods_str = str(mods or 'NoMod').upper()
        features['mod_DT'] = 1 if ('DT' in mods_str or 'NC' in mods_str) else 0
        features['mod_HR'] = 1 if 'HR' in mods_str else 0
        features['mod_EZ'] = 1 if 'EZ' in mods_str else 0
        features['mod_HD'] = 1 if 'HD' in mods_str else 0
        features['mod_FL'] = 1 if 'FL' in mods_str else 0
        features['mod_HT'] = 1 if 'HT' in mods_str else 0
        
        # 4. Perform Inference
        feature_cols = accuracy_predictor_package['feature_cols']
        model = accuracy_predictor_package['model']
        
        # Construct exact input vector matching training column order
        input_vector = []
        for col in feature_cols:
            val = features.get(col, 0.0)
            if np.isnan(val):
                val = 0.0
            input_vector.append(val)
            
        input_df = pd.DataFrame([input_vector], columns=feature_cols)
        
        # Predict
        predicted_accuracy = float(model.predict(input_df)[0])
        predicted_accuracy = max(50.0, min(100.0, predicted_accuracy))
        
        return {
            "status": "success",
            "username": username,
            "beatmap_hash": beatmap_hash,
            "mods": mods_str,
            "predicted_accuracy": round(predicted_accuracy, 2)
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/analyze")
def upload_replay(file: UploadFile = File(...), osu_file: Optional[UploadFile] = File(None)):
    os.makedirs(REPLAYS_DIR, exist_ok=True)
    temp_path = os.path.join(REPLAYS_DIR, file.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        replay = Replay.from_path(temp_path)
        h = replay.beatmap_hash
        osu_path = None
        
        if osu_file:
            target_osu_path = os.path.join(MAPS_DIR, f"{h}.osu")
            os.makedirs(MAPS_DIR, exist_ok=True)
            with open(target_osu_path, "wb") as f_osu:
                shutil.copyfileobj(osu_file.file, f_osu)
            register_beatmap(target_osu_path, h)
            osu_path = target_osu_path
            logger.info(f"Accepted uploaded .osu file from client for beatmap hash: {h}")
            
        if not osu_path:
            osu_path = get_beatmap_path(h)
            
        if not osu_path or not os.path.exists(osu_path):
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=404, detail=f"Corresponding beatmap hash {h} not found on local system or Osu! API lookup.")
            
        # Ingest the map set if missing from database
        map_id = db.get_map_id_by_hash(h)
        if not map_id:
            map_id = ingest_single_map(osu_path, h)
            if not map_id:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise HTTPException(status_code=500, detail="Failed to ingest beatmap into database.")
                
        # Analyze replay using pipeline
        analyzer = ReplayAnalysisPipeline(db, replays_dir=REPLAYS_DIR, maps_dir=MAPS_DIR)
        pipeline_result = analyzer.analyze_replay(temp_path)
        if not pipeline_result:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=400, detail="Replay data contains no compatible hit objects or analysis failed.")
            
        # Create backward-compatible analysis dict
        parsed_map = parse_osu_file(osu_path)
        metadata = parsed_map.get('metadata', {})
        mods_str = "+".join([str(m).replace("Mod.", "") for m in replay.mods]) if replay.mods else "NoMod"
        
        analysis = {
            'replay_file': os.path.basename(temp_path),
            'player': replay.username,
            'map_title': metadata.get('Title', 'Unknown'),
            'map_artist': metadata.get('Artist', 'Unknown'),
            'difficulty_name': metadata.get('Version', 'Unknown'),
            'mods': mods_str,
            'accuracy_percent': pipeline_result['accuracy'],
            'unstable_rate': pipeline_result['unstable_rate'],
            'avg_aim_error_px': pipeline_result['avg_aim_error']
        }
        
        # Append analysis to global CSV
        new_row = pd.DataFrame([analysis])
        if os.path.exists(SUMMARY_CSV):
            df_old = pd.read_csv(SUMMARY_CSV)
            df_old = df_old[df_old['replay_file'] != analysis['replay_file']]
            df_new = pd.concat([df_old, new_row], ignore_index=True)
        else:
            df_new = new_row
            
        df_new.to_csv(SUMMARY_CSV, index=False)
        return {"status": "success", "analysis": analysis}
        
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))

ml_params = None

def load_ml_model():
    global ml_params
    if ml_params is not None:
        return True
    try:
        model_dir = "data/model_results"
        params_path = os.path.join(model_dir, "model_parameters.json")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                ml_params = json.load(f)
            logger.info("Successfully loaded ML Ridge model parameters from JSON for Star Rating predictions.")
            return True
    except Exception as e:
        logger.error(f"Failed to load ML model JSON: {e}")
    return False

accuracy_predictor_package = None

def load_accuracy_predictor_model():
    global accuracy_predictor_package
    if accuracy_predictor_package is not None:
        return True
    try:
        model_dir = "data/model_results"
        model_path = os.path.join(model_dir, "accuracy_predictor.joblib")
        if os.path.exists(model_path):
            accuracy_predictor_package = joblib.load(model_path)
            logger.info("Successfully loaded ML Accuracy Predictor model package.")
            return True
        else:
            logger.warning(f"Accuracy predictor model file not found at {model_path}. Run train_predictor.py first.")
    except Exception as e:
        logger.error(f"Failed to load Accuracy Predictor model: {e}")
    return False

def get_official_star_rating(osu_path):
    """
    Calculates the exact official osu! star rating locally using the official rust library rosu-pp-py.
    """
    try:
        if os.path.exists(osu_path):
            map_data = rosu.Beatmap(path=osu_path)
            diff_attrs = rosu.Difficulty().calculate(map_data)
            return round(float(diff_attrs.stars), 2)
    except Exception as e:
        logger.error(f"Failed to compute official SR locally: {e}")
    return None

def compute_map_skills_profile(osu_path):
    try:
        parsed = parse_osu_file(osu_path)
        feats = extract_map_features(parsed)
        if not feats:
            return None
            
        difficulty = parsed.get('difficulty', {})
        cs = float(difficulty.get('CircleSize', 4.0))
        od = float(difficulty.get('OverallDifficulty', 8.0))
        ar = float(difficulty.get('ApproachRate', 9.0))
        density = feats.get("density_notes_per_sec", 4.0)
        duration = feats.get("duration_seconds", 120.0)
        
        # Compute map portfolio skills from the core portfolio library
        portfolio = compute_map_portfolio_skills(feats)
        snap_aim = portfolio['SnapAim']
        flow_aim = portfolio['FlowAim']
        speed = portfolio['Speed']
        streaming = portfolio['Streaming']
        stamina = portfolio['Stamina']
        tech = portfolio['Tech']
        finger_control = portfolio['FingerControl']
        precision = portfolio['Precision']
        reading = portfolio['Reading']
        visual_density = portfolio['VisualDensity']
        aim_control = portfolio['AimControl']
        
        title = feats.get("title", "Unknown")
        artist = feats.get("artist", "Unknown")
        version = feats.get("version", "Unknown")
        
        # ML Star Rating prediction
        ai_sr = None
        official_sr = get_official_star_rating(osu_path)
        sr_diff = None
        map_class = "Balanced"
        
        if load_ml_model():
            try:
                feat_vector = []
                for col in ml_params["feature_names"]:
                    val = feats.get(col, 0.0)
                    if np.isnan(val):
                        val = 0.0
                    feat_vector.append(val)
                
                # Perform scaling math manually
                feat_vector = np.array(feat_vector)
                mean = np.array(ml_params["scaler_mean"])
                scale = np.array(ml_params["scaler_scale"])
                scaled_vector = (feat_vector - mean) / scale
                
                # Perform Ridge Regression prediction using dot product
                coefs = np.array(ml_params["ridge_coef"])
                intercept = ml_params["ridge_intercept"]
                prediction = np.dot(scaled_vector, coefs) + intercept
                
                ai_sr = round(float(prediction), 2)
                
                if not official_sr:
                    official_sr = round(float((snap_aim * 0.01) + (flow_aim * 0.01) + (speed * 0.015) + (precision * 0.01) + 1.2), 2)
                
                sr_diff = round(ai_sr - official_sr, 2)
                
                if sr_diff > 0.35:
                    map_class = "Tech / Underweighted"
                elif sr_diff < -0.35:
                    map_class = "PP Farm / Overweighted"
                else:
                    map_class = "Balanced"
            except Exception as e:
                logger.error(f"ML prediction run failed: {e}")
                
        skills_dict = {
            "SnapAim": round(snap_aim, 1),
            "FlowAim": round(flow_aim, 1),
            "Speed": round(speed, 1),
            "Streaming": round(streaming, 1),
            "Stamina": round(stamina, 1),
            "Tech": round(tech, 1),
            "FingerControl": round(finger_control, 1),
            "Precision": round(precision, 1),
            "Reading": round(reading, 1),
            "VisualDensity": round(visual_density, 1),
            "AimControl": round(aim_control, 1)
        }
        
        max_val = max(skills_dict.values())
        if max_val > 0:
            normalized_dict = {k: round((v / max_val) * 100.0, 1) for k, v in skills_dict.items()}
        else:
            normalized_dict = {k: 5.0 for k in skills_dict.keys()}
            
        return {
            "SnapAim": skills_dict["SnapAim"],
            "FlowAim": skills_dict["FlowAim"],
            "Speed": skills_dict["Speed"],
            "Streaming": skills_dict["Streaming"],
            "Stamina": skills_dict["Stamina"],
            "Tech": skills_dict["Tech"],
            "FingerControl": skills_dict["FingerControl"],
            "Precision": skills_dict["Precision"],
            "Reading": skills_dict["Reading"],
            "VisualDensity": skills_dict["VisualDensity"],
            "AimControl": skills_dict["AimControl"],
            "Normalized": normalized_dict,
            "Title": title,
            "Artist": artist,
            "Difficulty": version,
            "Creator": feats.get("creator"),
            "CS": cs,
            "OD": od,
            "AR": ar,
            "TotalObjects": feats.get("total_objects", 0),
            "Density": round(density, 2),
            "Duration": round(duration, 1),
            "AISr": ai_sr,
            "OfficialSr": official_sr,
            "SrDiff": sr_diff,
            "Class": map_class
        }
    except Exception as e:
        logger.error(f"Error computing map profile: {e}")
        return None

@app.get("/api/map/skills/{replay_basename}")
def get_map_skills_endpoint(replay_basename: str):
    """Calculates and returns the skillset requirements for the map played in this replay."""
    if not os.path.exists(SUMMARY_CSV):
        raise HTTPException(status_code=404, detail="Replay summary CSV not found.")
        
    try:
        df = pd.read_csv(SUMMARY_CSV)
        # Find exact replay matching the base name prefix
        match = df[df['replay_file'].str.startswith(replay_basename)].copy()
        if match.empty:
            raise HTTPException(status_code=404, detail="Replay score not found.")
            
        rfile = match.iloc[0]['replay_file']
        rpath = os.path.join(REPLAYS_DIR, rfile)
        
        if not os.path.exists(rpath):
            raise HTTPException(status_code=404, detail="Replay file missing from storage.")
            
        r = Replay.from_path(rpath)
        h = r.beatmap_hash
        osu_path = get_beatmap_path(h)
        if not osu_path or not os.path.exists(osu_path):
            raise HTTPException(status_code=404, detail=f"Corresponding beatmap file missing for hash {h}")
            
        profile = compute_map_skills_profile(osu_path)
        if not profile:
            raise HTTPException(status_code=500, detail="Failed to calculate map skillset profile.")
            
        return profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mapset")
def list_mapsets(page: int = 1, per_page: int = 50, search: str = None):
    """
    Lists all locally available mapsets with pagination.
    Query params:
      - page: page number (default 1)
      - per_page: results per page (default 50, max 200)
      - search: optional search string to filter by title/artist/difficulty name
    """
    per_page = min(max(1, per_page), 200)
    page = max(1, page)
    
    os.makedirs(MAPS_DIR, exist_ok=True)
    osu_files = [f for f in os.listdir(MAPS_DIR) if f.endswith(".osu")]
    
    # Group files by mapset ID (filename pattern: {mapset_id}_{index}.osu)
    mapset_groups = {}
    for f in osu_files:
        parts = f.rsplit("_", 1)
        if len(parts) == 2:
            mapset_id = parts[0]
            if mapset_id not in mapset_groups:
                mapset_groups[mapset_id] = []
            mapset_groups[mapset_id].append(f)
    
    # Load star ratings cache for metadata enrichment
    sr_cache = {}
    sr_cache_path = "data/star_ratings_cache.json"
    if os.path.exists(sr_cache_path):
        try:
            with open(sr_cache_path, "r") as f:
                sr_cache = json.load(f)
        except Exception:
            pass
    
    # Build summary entries for each mapset
    mapsets = []
    for mid, files in sorted(mapset_groups.items(), key=lambda x: x[0]):
        entry = {
            "mapset_id": mid,
            "difficulty_count": len(files),
            "difficulties": [],
            "title": None,
            "artist": None,
            "creator": None,
        }
        
        # Try to read metadata from the first .osu file in the set
        try:
            first_path = os.path.join(MAPS_DIR, files[0])
            parsed = parse_osu_file(first_path)
            meta = parsed.get("metadata", {})
            entry["title"] = meta.get("Title", None)
            entry["artist"] = meta.get("Artist", None)
            entry["creator"] = meta.get("Creator", None)
        except Exception:
            pass
        
        # Enrich with star ratings from cache
        if mid in sr_cache:
            for diff_name, sr in sr_cache[mid].items():
                entry["difficulties"].append({"name": diff_name, "star_rating": sr})
        else:
            # Fall back to listing filenames
            for f_name in files:
                entry["difficulties"].append({"name": f_name, "star_rating": None})
        
        mapsets.append(entry)
    
    # Apply search filter
    if search:
        q = search.lower()
        filtered = []
        for m in mapsets:
            title = (m["title"] or "").lower()
            artist = (m["artist"] or "").lower()
            creator = (m["creator"] or "").lower()
            diff_names = " ".join(d["name"].lower() for d in m["difficulties"])
            if q in title or q in artist or q in creator or q in diff_names or q in m["mapset_id"]:
                filtered.append(m)
        mapsets = filtered
    
    total = len(mapsets)
    total_pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    end = start + per_page
    
    return {
        "total_mapsets": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "mapsets": mapsets[start:end]
    }

@app.get("/mapset/{mapset_id}")
def get_mapset_details(mapset_id: int, version: str = None):
    """
    Returns detailed skillset stats and ML star predictions for each map in the mapset.
    Supports ?version=DifficultyName query parameter to filter by a specific difficulty.
    """
    os.makedirs(MAPS_DIR, exist_ok=True)
    files = [f for f in os.listdir(MAPS_DIR) if f.startswith(f"{mapset_id}_") and f.endswith(".osu")]
    
    # Failover: if not found locally, try downloading the mapset on the fly!
    if not files:
        logger.info(f"Mapset {mapset_id} not found locally. Attempting to download on the fly...")
        download_and_extract_mapset(mapset_id, None)
        scan_local_maps()
        files = [f for f in os.listdir(MAPS_DIR) if f.startswith(f"{mapset_id}_") and f.endswith(".osu")]
        
    if not files:
        raise HTTPException(status_code=404, detail=f"Mapset {mapset_id} not found locally and download failed.")
        
    results = []
    for f in files:
        path = os.path.join(MAPS_DIR, f)
        profile = compute_map_skills_profile(path)
        if profile:
            profile["Filename"] = f
            results.append(profile)
            
    if not results:
        raise HTTPException(status_code=500, detail="Failed to parse any maps in the mapset.")
        
    if version:
        filtered = [p for p in results if p["Difficulty"].lower() == version.lower()]
        if not filtered:
            filtered = [p for p in results if version.lower() in p["Difficulty"].lower()]
        if not filtered:
            raise HTTPException(status_code=404, detail=f"Difficulty '{version}' not found in mapset. Available: {[p['Difficulty'] for p in results]}")
        return filtered[0]
        
    return results

@app.get("/api/hits/{replay_basename}")
def get_hits_details(replay_basename: str):
    """Returns the hit coordinate records for scatter/timing distributions."""
    replay_file = f"{replay_basename}.osr"
    hits_json = get_replay_hits(replay_file)
    if not hits_json:
        raise HTTPException(status_code=404, detail="Hits detail data not found in database.")
    try:
        return json.loads(hits_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/map/{beatmap_hash}")
def get_map_details(beatmap_hash: str):
    """Returns details, 11-axis skillset profile, and scores leaderboard for a specific beatmap hash."""
    osu_path = get_beatmap_path(beatmap_hash)
    if not osu_path or not os.path.exists(osu_path):
        raise HTTPException(status_code=404, detail="Beatmap file not found on server cache or API mirrors.")
        
    try:
        parsed_map = parse_osu_file(osu_path)
        metadata = parsed_map.get('metadata', {})
        difficulty = parsed_map.get('difficulty', {})
        timing_points = parsed_map.get('timing_points', [])
        
        # Calculate skills profile (try DB first)
        skills = db.get_map_portfolio(beatmap_hash)
        if not skills:
            skills = get_cached_map_skills(osu_path, beatmap_hash)
        
        bpm = 120.0
        if timing_points:
            uninherited_tps = [tp for tp in timing_points if tp.get('uninherited', 1) == 1]
            if uninherited_tps and uninherited_tps[0]['beat_length'] > 0:
                bpm = round(60000.0 / uninherited_tps[0]['beat_length'], 1)
        
        # Query scores leaderboard (try DB first)
        leaderboard = db.get_map_leaderboard(beatmap_hash)
        if not leaderboard and os.path.exists(SUMMARY_CSV):
            df = pd.read_csv(SUMMARY_CSV)
            match = df[df['beatmap_hash'] == beatmap_hash]
            if not match.empty:
                match = match.sort_values(by=['accuracy_percent', 'unstable_rate'], ascending=[False, True])
                for _, row in match.iterrows():
                    leaderboard.append({
                        'replay_file': row['replay_file'],
                        'player': row['player'],
                        'mods': row['mods'],
                        'accuracy': row['accuracy_percent'],
                        'unstable_rate': row['unstable_rate'],
                        'avg_aim_error': row['avg_aim_error_px'],
                        'misses': row['misses']
                    })
                    
        return {
            'hash': beatmap_hash,
            'title': metadata.get('Title', 'Unknown'),
            'artist': metadata.get('Artist', 'Unknown'),
            'difficulty_name': metadata.get('Version', 'Unknown'),
            'creator': metadata.get('Creator', 'Unknown'),
            'bpm': bpm,
            'cs': float(difficulty.get('CircleSize', 4.0)),
            'od': float(difficulty.get('OverallDifficulty', 8.0)),
            'ar': float(difficulty.get('ApproachRate', 9.0)),
            'hp': float(difficulty.get('HPDrainRate', 5.0)),
            'skills': skills,
            'leaderboard': leaderboard
        }
    except Exception as e:
        logger.error(f"Error serving /api/map/{beatmap_hash}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve static web folder
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
os.makedirs(static_dir, exist_ok=True)

# SPA fallback router to support clean URLs with react-router-dom client routing
@app.exception_handler(404)
async def spa_fallback_handler(request, exc):
    # If the request is not for /api/ and not for /static/ files directly, return index.html
    if not request.url.path.startswith("/api/") and not request.url.path.startswith("/static/"):
        index_path = os.path.join(static_dir, "index.html")
        # Check in dist folder first if Vite build exists
        dist_index = os.path.join(static_dir, "dist", "index.html")
        if os.path.exists(dist_index):
            return FileResponse(dist_index)
        elif os.path.exists(index_path):
            return FileResponse(index_path)
    raise exc

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
def generate_missing_hits_data():
    logger.info("Startup check: Verifying interactive hits details JSON data...")
    if not os.path.exists(SUMMARY_CSV):
        return
    try:
        df = pd.read_csv(SUMMARY_CSV)
        for _, row in df.iterrows():
            rfile = row['replay_file']
            rpath = os.path.join(REPLAYS_DIR, rfile)
            if os.path.exists(rpath) and not get_replay_hits(rfile):
                try:
                    r = Replay.from_path(rpath)
                    h = r.beatmap_hash
                    osu_path = get_beatmap_path(h)
                    if osu_path and os.path.exists(osu_path):
                        logger.info(f"Regenerating hits details JSON for existing play: {rfile}")
                        process_replay_data(rpath, osu_path)
                except Exception as ex:
                    logger.error(f"Error compiling play data for {rfile}: {ex}")
    except Exception as e:
        logger.error(f"Startup check failed: {e}")
        
    try:
        load_accuracy_predictor_model()
    except Exception as e:
        logger.error(f"Failed to initialize accuracy predictor on startup: {e}")

@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    import uvicorn
    # Load settings from config
    host = getattr(config, 'server_host', '127.0.0.1')
    port = getattr(config, 'server_port', 8000)
    uvicorn.run(app, host=host, port=port)
