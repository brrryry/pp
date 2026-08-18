import hashlib
import logging
import time
import uuid
import os
from dataclasses import dataclass
from typing import Optional
import torch

import requests
# pyrefly: ignore [missing-import]
import redis
# pyrefly: ignore [missing-import]
from rq import Queue, Retry
# pyrefly: ignore [missing-import]
from osrparse import Replay
import config
from datetime import datetime
from datetime import timezone

# Import your classes – they will be used inside the worker functions
from osu import Client
from core.DatabaseManager import DatabaseManager
from core.BeatmapIngestor import BeatmapIngestor
from core.RecommendationEngine import RecommendationEngine
from core.Embedder import MapVAE

logging.basicConfig(level=logging.INFO,
                    format='[ReplayIngestor] %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cpu")

def _request_osu_access_token(client_id: int, client_secret: str) -> Optional[str]:
    """Request an OAuth access token from the osu! API."""
    response = requests.post(
        "https://osu.ppy.sh/oauth/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
            "scope": "public"
        }
    )

    if response.status_code != 200:
        logger.error(f"Failed to get token: {response.status_code} - {response.text}")
        return None

    return response.json().get("access_token")

def datetime_to_windows_ticks(dt: datetime) -> int:
    # Ensure dt is timezone-aware and set to UTC to avoid timezone shifts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
        
    # Unix epoch in Windows ticks is 621355968000000000
    unix_time = dt.timestamp()
    return int(unix_time * 10_000_000) + 621355968000000000

# ---------- Worker‑side global state ----------
_worker_state = None

def get_worker_state():
    """Lazily initialise dependencies inside the RQ worker process."""
    global _worker_state
    if _worker_state is None:
        db = DatabaseManager()
        r_url = os.environ.get("REDIS_URL") or getattr(config, "REDIS_URL", getattr(config, "redis_url", "redis://localhost:6379/0"))
        try:
            redis_client = redis.from_url(r_url, socket_connect_timeout=1)
            redis_client.ping()
        except Exception:
            redis_client = redis.from_url("redis://localhost:6379/0", socket_connect_timeout=1)
        osu_client = Client.from_credentials(
            config.OSU_API_CLIENT_ID,
            config.OSU_API_CLIENT_SECRET,
            config.OSU_API_REDIRECT_URI
        )
        mirrors = config.MIRRORS
        embedder = MapVAE(
            input_size = config.INPUT_SIZE,
            hidden_size = config.HIDDEN_SIZE,
            embedding_size = config.EMBEDDING_SIZE,
            dropout = config.DROPOUT
        )
        embedder.load_state_dict(torch.load(config.EMBEDDER_FILE_PATH, map_location=device))
        embedder.eval()
        # This is a placeholder; you can also pass model path via environment.
        beatmap_ingestor = BeatmapIngestor(
            db_manager=db,
            embed_model=embedder,
            osu_client=osu_client,
            mirrors=mirrors
        )
        recommendation_engine = RecommendationEngine(
            db_manager=db,
            model_folder="models"
        )
        _worker_state = {
            'db': db,
            'redis': redis_client,
            'osu_client': osu_client,
            'beatmap_ingestor': beatmap_ingestor,
            'recommendation_engine': recommendation_engine
        }
    return _worker_state

# ---------- Module‐level job functions ----------
def calculate_mastery(accuracy: float, misses: int, total: int, max_combo: int) -> float:
    if total == 0:
        return 0.0
    acc_factor = accuracy ** 3
    miss_factor = max(0.0, 1.0 - (misses / total))
    combo_factor = max_combo / total
    return acc_factor * miss_factor * combo_factor

def calculate_mods_int(mods_list) -> int:
    """Safely convert score.mods list from osu SDK to bitwise int bitmask."""
    if not mods_list:
        return 0
    mods_int = 0
    mod_map = {
        'NF': 1, 'NOFAIL': 1,
        'EZ': 2, 'EASY': 2,
        'TD': 4, 'TOUCHDEVICE': 4,
        'HD': 8, 'HIDDEN': 8,
        'HR': 16, 'HARDROCK': 16,
        'SD': 32, 'SUDDENDEATH': 32,
        'DT': 64, 'DOUBLETIME': 64,
        'RX': 128, 'RELAX': 128,
        'HT': 256, 'HALFTIME': 256,
        'NC': 576, 'NIGHTCORE': 576,
        'FL': 1024, 'FLASHLIGHT': 1024,
        'SO': 4096, 'SPUNOUT': 4096,
        'PF': 16384, 'PERFECT': 16384
    }
    for m in mods_list:
        mod_obj = getattr(m, 'mod', m)
        val = getattr(mod_obj, 'value', mod_obj)
        name = getattr(mod_obj, 'name', str(mod_obj))
        
        v_str = str(val).upper().strip() if val is not None else ''
        n_str = str(name).upper().strip() if name is not None else ''
        if v_str == 'CL' or n_str == 'CLASSIC':
            continue
        
        if v_str in mod_map:
            mods_int |= mod_map[v_str]
        elif n_str in mod_map:
            mods_int |= mod_map[n_str]
        elif isinstance(val, int):
            mods_int |= val
    return mods_int

def wait_for_beatmap(redis_client, mapset_id: int, timeout: int = 120) -> bool:
    """Block until beatmap ingestion for `mapset_id` is complete."""
    status_key = f"job:beatmap_{mapset_id}"

    # Fast path 1: already done in Redis?
    current = redis_client.hget(status_key, "status")
    if current and current.decode() in ["ready", "already_exists"]:
        return True

    # Fast path 2: already exists in DB?
    try:
        state = get_worker_state()
        db = state['db']
        if db.find_mapset_by_id(mapset_id):
            redis_client.hset(status_key, "status", "ready")
            return True
    except Exception:
        pass

    # Subscribe and wait
    pubsub = redis_client.pubsub()
    pubsub.subscribe("beatmap_events")
    deadline = time.time() + timeout
    poll_interval = 2  # seconds between polling fallbacks

    try:
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break

            # Listen for pub/sub message (non-blocking with short timeout)
            message = pubsub.get_message(ignore_subscribe_messages=True,
                                         timeout=min(poll_interval, remaining))
            if message and message["type"] == "message":
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode()
                if data == f"ready:{mapset_id}":
                    logger.info(f"Received pub/sub ready signal for mapset {mapset_id}")
                    return True

            # Polling fallback: check the status key directly
            current = redis_client.hget(status_key, "status")
            if current and current.decode() == "ready":
                logger.info(f"Polling detected ready status for mapset {mapset_id}")
                return True

        logger.warning(f"Timed out waiting for beatmap {mapset_id} after {timeout}s")
        return False
    finally:
        pubsub.unsubscribe("beatmap_events")
        pubsub.close()

def _resolve_mapset_id(db, map_hash: str) -> Optional[int]:
    """Look up the mapset_id for a beatmap hash via the osu! API.
    Uses db._osu_api_get() to enforce the global 1 req/s rate limit.
    """
    try:
        token = db._get_osu_token()  # Uses cached token via Redis
        if not token:
            logger.error("Could not obtain osu! API access token")
            return None
        headers = {"Authorization": f"Bearer {token}"}
        response = db._osu_api_get(
            f"https://osu.ppy.sh/api/v2/beatmaps/lookup?checksum={map_hash}",
            headers=headers
        )
        logger.info(f"Osu API lookup response status code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                if 'beatmapset_id' in data and data['beatmapset_id']:
                    return int(data['beatmapset_id'])
                elif 'beatmapset' in data and isinstance(data['beatmapset'], dict) and 'id' in data['beatmapset']:
                    return int(data['beatmapset']['id'])
    except Exception as e:
        logger.error(f"Failed to resolve mapset_id for hash {map_hash}: {str(e)}")
    return None

def process_replay_file_job(file_path: str, job_id: str):
    """RQ worker function for a single replay file."""
    state = get_worker_state()
    db = state['db']
    redis_client = state['redis']
    osu_client = state['osu_client']
    beatmap_ingestor = state['beatmap_ingestor']
    recommendation_engine = state['recommendation_engine']

    try:
        # Parse replay
        replay = Replay.from_path(file_path)
        replay_hash = replay.replay_hash

        if db.find_replay_by_hash(replay_hash) or db.find_replay_by_username_and_map_hash(replay.username, replay.beatmap_hash):
            redis_client.hset(f"job:{job_id}", mapping={
                "status": "skipped",
                "reason": "duplicate"
            })
            logger.info(f"Replay {replay_hash} already exists")
            return

        username = replay.username
        map_hash = replay.beatmap_hash
        mods = replay.mods.value if hasattr(replay.mods, 'value') else int(replay.mods)
        misses = replay.count_miss
        total = replay.count_300 + replay.count_100 + replay.count_50 + misses
        accuracy = (300 * replay.count_300 + 100 * replay.count_100 + 50 * replay.count_50) / (300 * total) if total > 0 else 0.0
        max_combo = replay.max_combo
        mastery = calculate_mastery(accuracy, misses, total, max_combo)

        # Check if beatmap already exists in DB
        mapset_id = db.get_mapset_id_by_hash(map_hash)

        if mapset_id is None:
            # Beatmap not in DB — resolve mapset_id via osu! API
            mapset_id = _resolve_mapset_id(db, map_hash)
            if mapset_id is None:
                logger.error(f"Could not resolve mapset_id for map hash {map_hash}")
                redis_client.hset(f"job:{job_id}", mapping={
                    "status": "failed",
                    "error": f"Could not resolve mapset_id for hash {map_hash}"
                })
                return

            # Kick off beatmap ingestion (enqueues to beatmap_ingestion queue)
            logger.info(f"Beatmap not found locally — ingesting mapset {mapset_id}")
            beatmap_ingestor.ingest_mapset(mapset_id)

            # Block until the beatmap worker finishes
            beatmap_ready = wait_for_beatmap(redis_client, mapset_id, timeout=120)
        else:
            beatmap_ready = True

        # Insert replay
        db.add_replay(
            username=username,
            replay_hash=replay_hash,
            map_hash=map_hash,
            mods=mods,
            accuracy=accuracy,
            misses=misses,
            max_combo=max_combo,
            mastery_score=mastery
        )

        if beatmap_ready:
            # Immediately update recommendation engine for this user
            osu_id = db.get_osu_id_from_username(username)
            if osu_id:
                logger.info(f"Replay {replay_hash} is ready, refreshing recommendation engine for user {osu_id}")
                recommendation_engine.refresh_user(osu_id)
            redis_client.hset(f"job:{job_id}", mapping={
                "status": "ready",
                "replay_hash": replay_hash
            })
        else:
            # Beatmap ingestion timed out — mark as pending for later retry
            redis_client.hset(f"job:{job_id}", mapping={
                "status": "pending_beatmap",
                "replay_hash": replay_hash,
                "mapset_id": str(mapset_id)
            })
            logger.warning(f"Replay {replay_hash} saved but beatmap {mapset_id} not ready yet")

        logger.info(f"Processed replay {replay_hash} for {username}")
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        redis_client.hset(f"job:{job_id}", mapping={
            "status": "failed",
            "error": str(e)
        })
        raise

def fetch_top_replays_job(user_id: int, limit: int = 100, recent_limit: int = 50, job_id: str = ""):
    """RQ worker function to fetch top 100 best plays and recent 50 plays from osu! API."""
    state = get_worker_state()
    db = state['db']
    redis_client = state['redis']
    osu_client = state['osu_client']
    beatmap_ingestor = state['beatmap_ingestor']

    try:
        # Resolve username string to numeric osu! user ID if necessary
        numeric_user_id = user_id
        try:
            numeric_user_id = int(user_id)
        except (ValueError, TypeError):
            u_obj = osu_client.get_user(user_id)
            if u_obj:
                numeric_user_id = u_obj.id
                db.add_user(str(u_obj.id), u_obj.username)
            else:
                raise ValueError(f"osu! user '{user_id}' not found.")

        # 1. Fetch top 100 best scores
        best_scores = []
        try:
            best_scores = osu_client.get_user_scores(user=numeric_user_id, mode="osu", type="best", limit=limit) or []
        except Exception as err:
            logger.error(f"Error fetching best scores for user {numeric_user_id}: {err}")

        # 2. Fetch recent scores (last 50 plays)
        recent_scores = []
        try:
            recent_scores = osu_client.get_user_scores(user=numeric_user_id, mode="osu", type="recent", limit=recent_limit) or []
        except Exception as err:
            logger.error(f"Error fetching recent scores for user {numeric_user_id}: {err}")

        all_scores = list(best_scores) + list(recent_scores)
        seen = set()
        processed = 0

        mapsets_to_ingest = set()
        for score in all_scores:
            if not score or not getattr(score, 'beatmap', None):
                continue
            map_hash = getattr(score.beatmap, 'checksum', None)
            score_id = getattr(score, 'id', None) or f"{map_hash}_{getattr(score, 'created_at', processed)}"
            if score_id in seen:
                continue
            seen.add(score_id)

            username = score.user.username if getattr(score, 'user', None) else str(user_id)
            accuracy = getattr(score, 'accuracy', 0.0) or 0.0
            mods_int = calculate_mods_int(score.mods)
            mapset_id = getattr(score.beatmap, 'beatmapset_id', None)
            if mapset_id:
                mapsets_to_ingest.add(mapset_id)

            misses = getattr(score.statistics, 'miss', 0) if getattr(score, 'statistics', None) else 0
            if misses is None:
                misses = 0
            max_combo = score.max_combo if getattr(score, 'max_combo', None) is not None else 0

            circles = getattr(score.beatmap, 'count_circles', 0) or 0
            sliders = getattr(score.beatmap, 'count_sliders', 0) or 0
            spinners = getattr(score.beatmap, 'count_spinners', 0) or 0
            total = circles + sliders + spinners
            if total == 0:
                total = max_combo or 1

            mastery = calculate_mastery(accuracy, misses, total, max_combo)

            bm = score.beatmap
            beatmap_id = getattr(bm, 'id', None)
            bm_set = getattr(bm, 'beatmapset', None)
            title = getattr(bm_set, 'title', None) or getattr(bm, 'title', 'Unknown Title')
            artist = getattr(bm_set, 'artist', None) or getattr(bm, 'artist', 'Unknown Artist')
            creator = getattr(bm_set, 'creator', None) or getattr(bm, 'creator', 'Unknown Mapper')
            version = getattr(bm, 'version', None) or 'Normal'
            sr = getattr(bm, 'difficulty_rating', None) or getattr(bm, 'sr', 0.0)

            if map_hash:
                db.add_map(
                    map_id=beatmap_id,
                    mapset_id=mapset_id,
                    map_hash=map_hash,
                    title=title,
                    artist=artist,
                    creator=creator,
                    version=version,
                    sr=sr
                )

            db.add_replay(
                username=username,
                replay_hash=None,
                map_hash=map_hash,
                mods=mods_int,
                accuracy=accuracy,
                misses=misses,
                max_combo=max_combo,
                mastery_score=mastery
            )
            processed += 1

        # Enqueue missing beatmaps for background processing without blocking top-play ingestion
        for ms_id in mapsets_to_ingest:
            beatmap_ingestor.ingest_mapset(ms_id)

        recommendation_engine = state.get('recommendation_engine')
        if recommendation_engine:
            try:
                recommendation_engine.refresh_user(str(numeric_user_id))
                logger.info(f"Refreshed ALS recommendation vector for user {numeric_user_id}")
            except Exception as ref_err:
                logger.error(f"Failed to refresh recommendation vector for user {numeric_user_id}: {ref_err}")

        redis_client.hset(f"job:{job_id}", mapping={
            "status": "ready",
            "processed": str(processed)
        })
        logger.info(f"Fetched {processed} total replays (best & recent) for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to fetch replays for user {user_id}: {e}")
        redis_client.hset(f"job:{job_id}", mapping={
            "status": "failed",
            "error": str(e)
        })
        raise

# ---------- The ReplayIngestor class (main process) ----------
class ReplayIngestor:
    def __init__(self, db_manager: DatabaseManager, beatmap_ingestor: BeatmapIngestor,
                 recommendation_engine: RecommendationEngine,
                 redis_url: str = config.redis_url):
        # These are used only for enqueuing jobs – they are NOT passed to workers.
        self.db = db_manager
        self.beatmap_ingestor = beatmap_ingestor
        try:
            self.redis = redis.from_url(redis_url, socket_connect_timeout=1)
            self.redis.ping()
        except Exception:
            self.redis = redis.from_url("redis://localhost:6379/0", socket_connect_timeout=1)
        self.queue = Queue('replay_ingestion', connection=self.redis, default_timeout=1800)

    def ingest_replay(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Replay file not found: {file_path}")
        job_id = f"replay_{uuid.uuid4().hex[:8]}"
        # Enqueue the module‑level function with only primitive args
        job = self.queue.enqueue(
            'core.ReplayIngestor.process_replay_file_job',
            file_path,
            job_id,
            job_id=job_id,
            job_timeout=1800,
            retry=Retry(max=2, interval=[10, 60]),
            description=f"Ingest replay {os.path.basename(file_path)}"
        )
        self.redis.hset(f"job:{job_id}", mapping={
            "status": "queued",
            "file": os.path.basename(file_path),
            "created_at": time.time()
        })
        self.redis.expire(f"job:{job_id}", 86400)
        logger.info(f"Enqueued replay job {job_id}")
        return job_id

    def get_active_top_replay_job(self, user_id) -> Optional[str]:
        """Check if a top replay ingestion job for user_id is currently active or queued."""
        clean_uid = str(user_id).strip().lower()
        active_key = f"active_job:topreplay:{clean_uid}"
        try:
            existing_job_id = self.redis.get(active_key)
            if existing_job_id:
                job_id_str = existing_job_id.decode() if isinstance(existing_job_id, bytes) else str(existing_job_id)
                status_data = self.get_job_status(job_id_str)
                status = status_data.get("status")
                if status in ["queued", "started", "processing", "downloading"]:
                    return job_id_str
                else:
                    self.redis.delete(active_key)
        except Exception:
            pass
        return None

    def ingest_top_replays(self, user_id: int, limit: int = 100, recent_limit: int = 50, force: bool = False) -> str:
        clean_uid = str(user_id).strip().lower()
        if not force:
            active_job = self.get_active_top_replay_job(clean_uid)
            if active_job:
                logger.info(f"Re-using existing active job {active_job} for user {user_id}")
                return active_job
        else:
            self.redis.delete(f"active_job:topreplay:{clean_uid}")

        job_id = f"topreplay_{clean_uid}_{uuid.uuid4().hex[:8]}"
        job = self.queue.enqueue(
            fetch_top_replays_job,
            user_id,
            limit,
            recent_limit,
            job_id,
            job_id=job_id,
            job_timeout=1800,
            retry=Retry(max=2, interval=[10, 60]),
            description=f"Fetch top {limit} & recent {recent_limit} replays for user {user_id}"
        )
        self.redis.hset(f"job:{job_id}", mapping={
            "status": "queued",
            "user_id": user_id,
            "limit": limit,
            "recent_limit": recent_limit,
            "created_at": time.time()
        })
        self.redis.expire(f"job:{job_id}", 86400)
        self.redis.set(f"active_job:topreplay:{clean_uid}", job_id, ex=1800)
        return job_id

    def get_job_status(self, job_id: str) -> dict:
        data = self.redis.hgetall(f"job:{job_id}")
        if not data:
            return {"status": "not_found"}
        return {k.decode(): v.decode() for k, v in data.items()}