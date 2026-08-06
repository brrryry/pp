import os
import sys
import hashlib
import json
import logging
import argparse
import zipfile
import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from osrparse import Replay
from osrparse.utils import Key

# Add parent directory to path to find config.py and src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from core.parser import parse_osu_file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPLAYS_DIR = "data/replays"
MAPS_DIR = config.maps_path
SUMMARY_CSV = "data/replays/replays_summary.csv"

def compute_md5(file_path):
    """Computes the MD5 checksum of a file."""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file {file_path}: {e}")
        return None

def scan_local_maps():
    """Scans local maps directory and returns a dictionary of md5_hash -> file_path."""
    logger.info("Scanning local maps folder to index MD5 checksums...")
    hash_map = {}
    if not os.path.exists(MAPS_DIR):
        return hash_map
        
    for f in os.listdir(MAPS_DIR):
        if f.endswith(".osu"):
            path = os.path.join(MAPS_DIR, f)
            h = compute_md5(path)
            if h:
                hash_map[h] = path
    logger.info(f"Indexed {len(hash_map)} unique local .osu files.")
    return hash_map

def download_and_extract_mapset(beatmapset_id, expected_checksum, local_maps_cache):
    """Downloads a beatmapset from the mirror and extracts its .osu files."""
    logger.info(f"Attempting to download beatmapset {beatmapset_id} from mirror...")
    url = f"https://api.nerinyan.moe/d/{beatmapset_id}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        extracted_files = []
        
        # Count existing .osu files to assign a unique index
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
                
                # Update cache
                h = compute_md5(new_path)
                if h:
                    local_maps_cache[h] = new_path
                    if h == expected_checksum:
                        extracted_files.append(new_path)
                idx += 1
                
        if extracted_files:
            logger.info(f"Successfully downloaded and found matching beatmap difficulty locally!")
            return extracted_files[0]
            
    except Exception as e:
        logger.error(f"Failed to download/extract mapset {beatmapset_id}: {e}")
        
    return None

def get_or_download_beatmap(beatmap_hash, local_maps_cache):
    """Finds the beatmap path locally or downloads it from the API using lookup."""
    # 1. Check local cache
    if beatmap_hash in local_maps_cache:
        logger.info("Found matching beatmap in local cache.")
        return local_maps_cache[beatmap_hash]
        
    # 2. Query Osu! API for beatmapset ID
    logger.info(f"Beatmap hash {beatmap_hash} not found locally. Querying Osu! API...")
    try:
        from osu import Client
        client = Client.from_credentials(
            config.osu_api_client_id,
            config.osu_api_client_secret,
            config.osu_api_redirect_uri
        )
        bm = client.lookup_beatmap(checksum=beatmap_hash)
        if bm:
            logger.info(f"Osu! API matched beatmap to set {bm.beatmapset_id} ({bm.version}).")
            # 3. Download the beatmapset
            return download_and_extract_mapset(bm.beatmapset_id, beatmap_hash, local_maps_cache)
        else:
            logger.warning("Beatmap lookup returned no matches on the Osu! API.")
    except Exception as e:
        logger.error(f"Error looking up beatmap on Osu! API: {e}")
        
    return None

def analyze_replay(replay_path, osu_map_path):
    """Parses and matches replay click frames to hit objects, returns analysis dict."""
    logger.info(f"Parsing replay: {os.path.basename(replay_path)}")
    replay = Replay.from_path(replay_path)
    
    logger.info(f"Parsing beatmap: {os.path.basename(osu_map_path)}")
    parsed_map = parse_osu_file(osu_map_path)
    
    # 1. Parse absolute timestamps of replay events
    current_time = 0
    clicks = []
    prev_keys = 0
    
    for event in replay.replay_data:
        current_time += event.time_delta
        keys_val = event.keys.value
        
        # Check for key-down transition (M1=1, M2=2, K1=4, K2=8)
        for mask in (1, 2, 4, 8):
            if (keys_val & mask) and not (prev_keys & mask):
                clicks.append((current_time, event.x, event.y, mask))
        prev_keys = keys_val
        
    logger.info(f"Extracted {len(clicks)} clicks from replay.")
    
    # 2. Match hit objects
    # Osu! hit objects list
    hit_objects = parsed_map.get('hit_objects', [])
    
    clicks_left = clicks.copy()
    hit_results = []
    
    # 150ms hit window (standard 50s hit window is ~135ms to ~150ms depending on OD)
    window = 150.0 
    
    for obj in hit_objects:
        if obj['type'] == 'spinner':
            continue  # Spinners do not have aiming targets or specific click keys
            
        target_time = obj['time']
        target_x = obj['x']
        target_y = obj['y']
        
        # Find closest click in the window
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
            timing_offset = click_time - target_time # negative: early, positive: late
            aim_distance = ((click_x - target_x)**2 + (click_y - target_y)**2)**0.5
            dx = click_x - target_x
            dy = click_y - target_y
            
            hit_results.append({
                'target_time': target_time,
                'target_x': target_x,
                'target_y': target_y,
                'type': obj['type'],
                'hit': True,
                'timing_offset': timing_offset,
                'aim_distance': aim_distance,
                'dx': dx,
                'dy': dy,
                'click_time': click_time
            })
            clicks_left.pop(best_idx)
        else:
            # Miss
            hit_results.append({
                'target_time': target_time,
                'target_x': target_x,
                'target_y': target_y,
                'type': obj['type'],
                'hit': False,
                'timing_offset': None,
                'aim_distance': None,
                'dx': None,
                'dy': None,
                'click_time': None
            })
            
    df_hits = pd.DataFrame(hit_results)
    
    if df_hits.empty:
        logger.warning("No hit objects matched.")
        return None, None
        
    hits_only = df_hits[df_hits['hit'] == True]
    
    # Calculate performance metrics
    total_notes = len(df_hits)
    hits_count = len(hits_only)
    misses_count = total_notes - hits_count
    hit_ratio = hits_count / total_notes if total_notes > 0 else 0
    
    if not hits_only.empty:
        avg_offset = hits_only['timing_offset'].mean()
        abs_offset = hits_only['timing_offset'].abs().mean()
        std_offset = hits_only['timing_offset'].std()
        # Unstable Rate (UR) is Standard Deviation * 10
        unstable_rate = std_offset * 10.0 if not pd.isna(std_offset) else 0.0
        
        avg_aim_error = hits_only['aim_distance'].mean()
        std_aim_error = hits_only['aim_distance'].std()
    else:
        avg_offset, abs_offset, std_offset, unstable_rate = 0.0, 0.0, 0.0, 0.0
        avg_aim_error, std_aim_error = 0.0, 0.0
        
    analysis = {
        'replay_file': os.path.basename(replay_path),
        'replay_id': replay.replay_id,
        'replay_hash': replay.replay_hash,
        'player': replay.username,
        'map_title': parsed_map['metadata'].get('Title', 'Unknown'),
        'map_artist': parsed_map['metadata'].get('Artist', 'Unknown'),
        'difficulty_name': parsed_map['metadata'].get('Version', 'Unknown'),
        'mods': str(replay.mods),
        'total_notes': total_notes,
        'hits': hits_count,
        'misses': misses_count,
        'accuracy_percent': hit_ratio * 100.0,
        'avg_offset_ms': avg_offset,
        'abs_offset_ms': abs_offset,
        'unstable_rate': unstable_rate,
        'avg_aim_error_px': avg_aim_error,
        'std_aim_error_px': std_aim_error
    }
    
    return analysis, df_hits

def plot_replay_results(analysis, df_hits, output_dir):
    """Generates and saves performance diagnostic plots."""
    os.makedirs(output_dir, exist_ok=True)
    basename = analysis['replay_file'].replace('.osr', '')
    
    hits_only = df_hits[df_hits['hit'] == True].copy()
    if hits_only.empty:
        return
        
    # Set visual theme
    sns.set_theme(style="darkgrid")
    
    # Plot 1: Timing Error Distribution (Histogram)
    plt.figure(figsize=(8, 5))
    sns.histplot(data=hits_only, x="timing_offset", kde=True, bins=30, color="teal")
    plt.axvline(x=0, color='red', linestyle='--', label='Perfect Timing')
    plt.axvline(x=analysis['avg_offset_ms'], color='yellow', linestyle='-', label=f"Mean Offset ({analysis['avg_offset_ms']:.1f}ms)")
    plt.title(f"Timing Offset Distribution (UR: {analysis['unstable_rate']:.1f})")
    plt.xlabel("Offset (ms) - Early (<0) vs Late (>0)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{basename}_timing_distribution.png"), dpi=150)
    plt.close()
    
    # Plot 2: Aim Landing coordinates relative to note center (Scatter)
    plt.figure(figsize=(6, 6))
    # Draw Circle boundary (roughly typical CS4 radius is ~36px)
    circle = plt.Circle((0, 0), 36, color='grey', fill=False, linestyle=':', label='Note Boundary (CS4)')
    plt.gca().add_patch(circle)
    sns.scatterplot(data=hits_only, x="dx", y="dy", alpha=0.6, color="purple", label='Hits')
    plt.axhline(0, color='black', alpha=0.3)
    plt.axvline(0, color='black', alpha=0.3)
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    plt.title("Aim Landing Error (Distance from Note Center)")
    plt.xlabel("X Error (pixels)")
    plt.ylabel("Y Error (pixels)")
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{basename}_aim_scatter.png"), dpi=150)
    plt.close()
    
    # Plot 3: Fatigue / Offset drift over song time
    plt.figure(figsize=(10, 5))
    # Convert time to seconds
    hits_only['time_sec'] = hits_only['target_time'] / 1000.0
    sns.lineplot(data=hits_only, x="time_sec", y="timing_offset", color="royalblue", alpha=0.4, label='Offset Trend')
    sns.scatterplot(data=hits_only, x="time_sec", y="timing_offset", hue="aim_distance", palette="flare", alpha=0.7, size="aim_distance", sizes=(10, 100))
    plt.axhline(0, color='red', alpha=0.5, linestyle='--')
    plt.title("Timing and Aim Drift Over Time (Fatigue Tracking)")
    plt.xlabel("Song Time (seconds)")
    plt.ylabel("Timing Offset (ms)")
    plt.legend(title="Aim Error (px)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{basename}_fatigue_drift.png"), dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze osu! replay files (.osr).")
    parser.add_argument("--replay", type=str, help="Specific replay file to analyze. If omitted, analyzes all in replays folder.")
    args = parser.parse_args()
    
    os.makedirs(REPLAYS_DIR, exist_ok=True)
    os.makedirs(MAPS_DIR, exist_ok=True)
    
    # Index local maps by MD5
    maps_cache = scan_local_maps()
    
    # Determine replays to process
    if args.replay:
        replay_files = [args.replay]
    else:
        replay_files = [os.path.join(REPLAYS_DIR, f) for f in os.listdir(REPLAYS_DIR) if f.endswith(".osr") and f != "test.osr"]
        
    if not replay_files:
        logger.warning(f"No replay files found in '{REPLAYS_DIR}'. Place some .osr files in there first!")
        return

    logger.info(f"Found {len(replay_files)} replays to process.")
    
    summaries = []
    
    for rpath in replay_files:
        if not os.path.exists(rpath):
            logger.error(f"Replay file not found: {rpath}")
            continue
            
        try:
            r = Replay.from_path(rpath)
            h = r.beatmap_hash
            logger.info("-" * 50)
            logger.info(f"Processing: {os.path.basename(rpath)}")
            
            # Lookup / Download corresponding map
            osu_path = get_or_download_beatmap(h, maps_cache)
            
            if not osu_path or not os.path.exists(osu_path):
                logger.error(f"Could not find or download beatmap matching hash: {h}. Skipping.")
                continue
                
            # Perform Analysis
            analysis, df_hits = analyze_replay(rpath, osu_path)
            
            if analysis:
                summaries.append(analysis)
                
                # Generate plots
                plot_dir = os.path.join(REPLAYS_DIR, f"plots_{os.path.basename(rpath).replace('.osr', '')}")
                plot_replay_results(analysis, df_hits, plot_dir)
                logger.info(f"Saved diagnostic plots to: {plot_dir}")
                
                # Print quick screen report
                print(f"\n--- Replay Dashboard: {analysis['map_title']} [{analysis['difficulty_name']}] ---")
                print(f"Player: {analysis['player']} | Mods: {analysis['mods']}")
                print(f"Hit Accuracy: {analysis['accuracy_percent']:.2f}% ({analysis['hits']}/{analysis['total_notes']})")
                print(f"Unstable Rate (UR): {analysis['unstable_rate']:.2f}")
                print(f"Timing Bias: {analysis['avg_offset_ms']:.2f} ms (Mean absolute: {analysis['abs_offset_ms']:.2f} ms)")
                print(f"Average Aim Error: {analysis['avg_aim_error_px']:.2f} px")
                print(f"--------------------------------------------------------------------------------\n")
                
        except Exception as e:
            logger.error(f"Error processing replay {rpath}: {e}")
            
    # Export summaries comparison CSV
    if summaries:
        df_summary = pd.DataFrame(summaries)
        df_summary.to_csv(SUMMARY_CSV, index=False)
        logger.info(f"\nSuccessfully compiled results for {len(summaries)} replays and saved to '{SUMMARY_CSV}'.")
        print("\nAll Replays Comparative Summary:")
        print(df_summary[['player', 'map_title', 'difficulty_name', 'accuracy_percent', 'unstable_rate', 'avg_aim_error_px']])
    else:
        logger.warning("No replays were successfully analyzed.")

if __name__ == "__main__":
    main()
