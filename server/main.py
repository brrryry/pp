import os
import sys
import json
import hashlib
import threading
import numpy as np
import pandas as pd
from osrparse import Replay
from osrparse.utils import Key
import joblib
# pyrefly: ignore [missing-import]
import redis
import sqlite3
# pyrefly: ignore [missing-import]
import uvicorn
from osu import Client
import torch

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse

from core.DatabaseManager import DatabaseManager
from core.BeatmapIngestor import BeatmapIngestor
from core.Embedder import MapVAE
from core.ReplayIngestor import ReplayIngestor
from core.RecommendationEngine import RecommendationEngine
from core.Projector import MapProjector

# Add parent directory to path to find config.py and src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
# Set up logging
from logger_setup import setup_logger
logger = setup_logger("main")

# INIT OSU CLIENT
client = Client.from_credentials(
    config.OSU_API_CLIENT_ID,
    config.OSU_API_CLIENT_SECRET,
    config.OSU_API_REDIRECT_URI
)

# INIT DATABASE
db = DatabaseManager()

# INIT EMBEDDER MODEL
device = torch.device("cpu")
embed_model_weights = torch.load(config.EMBEDDER_FILE_PATH, map_location=device)
embed_model = MapVAE(
    input_size = config.INPUT_SIZE,
    hidden_size = config.HIDDEN_SIZE,
    embedding_size = config.EMBEDDING_SIZE,
    dropout = config.DROPOUT
)
embed_model.load_state_dict(embed_model_weights)
embed_model.eval()

# INIT BEATMAP INGESTOR
beatmap_ingestor = BeatmapIngestor(
    db_manager=db,
    embed_model=embed_model,
    osu_client=client,
    mirrors = config.MIRRORS,
)

# INIT RECOMMENDATION ENGINE
recommender = RecommendationEngine(
    db_manager = db,
    model_folder = config.RECOMMENDATION_ENGINE_FOLDER
)
if recommender.user_factors.shape[0] == 0:
    logger.info("Fitting global ALS recommendation model from database replays...")
    recommender.fit_global_als()

# INIT REPLAY INGESTOR
replay_ingestor = ReplayIngestor(
    db_manager = db,
    beatmap_ingestor = beatmap_ingestor,
    recommendation_engine = recommender,
)

# INIT MAP PROJECTOR
_map_projector = MapProjector()


def _startup_backfill():
    """
    Background thread: clear stale Redis recommendation caches and
    backfill any maps in the DB that still have a null map_id.
    Uses a Redis distributed lock so only ONE of the gunicorn workers
    performs the backfill — the others exit immediately.
    """
    redis_client = getattr(db, 'redis_client', None)

    # Distributed lock: only one worker should run the backfill.
    # The lock expires after 10 minutes to handle crash recovery.
    LOCK_KEY = "lock:startup_backfill"
    LOCK_TTL_MS = 600_000  # 10 minutes in milliseconds
    acquired = False
    if redis_client:
        try:
            # SET NX PX: set only if not exists, with TTL in milliseconds
            acquired = bool(redis_client.set(LOCK_KEY, "1", nx=True, px=LOCK_TTL_MS))
        except Exception as e:
            logger.warning(f"Could not acquire backfill lock: {e}")
            acquired = True  # Proceed without lock if Redis is unavailable
    else:
        acquired = True  # No Redis — always proceed (single-process mode)

    if not acquired:
        logger.info("Startup backfill skipped: another worker is already running it.")
        return

    try:
        # 1. Clear stale recommendation caches (targeted — do NOT clear cache:map:*)
        if redis_client:
            try:
                cleared_total = 0
                for pattern in ["cache:endpoint:recs:*", "cache:recs:*"]:
                    stale_keys = list(redis_client.scan_iter(pattern))
                    if stale_keys:
                        redis_client.delete(*stale_keys)
                        cleared_total += len(stale_keys)
                if cleared_total:
                    logger.info(f"Cleared {cleared_total} stale recommendation cache keys.")
            except Exception as e:
                logger.warning(f"Could not clear stale Redis caches: {e}")

        # 2. Backfill maps with missing map_id — rate-limited globally via _osu_api_get (1 req/s)
        rows_attempted = 0
        batch_size = 50
        while True:
            n = db.backfill_missing_map_ids(limit=batch_size)
            rows_attempted += n
            if n < batch_size:
                # Fewer rows than batch_size means the table is exhausted
                break
        if rows_attempted:
            logger.info(f"Startup backfill complete: attempted {rows_attempted} maps with missing map_id.")
        else:
            logger.info("Startup backfill: no maps with missing map_id found in database.")

        # 3. Backfill missing 2D UMAP coordinates for maps with embeddings
        try:
            updated_coords = db.backfill_missing_coordinates()
            if updated_coords:
                logger.info(f"Startup backfill complete: updated 2D UMAP coordinates for {updated_coords} maps.")
        except Exception as e_cb:
            logger.warning(f"Coordinate backfill encountered an issue: {e_cb}")
    finally:
        # Always release the lock when done so the next restart can re-run
        if redis_client and acquired:
            try:
                redis_client.delete(LOCK_KEY)
            except Exception:
                pass


_backfill_thread = threading.Thread(target=_startup_backfill, daemon=True, name="startup-backfill")
_backfill_thread.start()

# INIT APP
app = FastAPI(title="Osu! PP & Skill Profiler API")


# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_replay")
def upload_replay(file: UploadFile = File(...)):
    if not file.filename.endswith(".osr"):
        raise HTTPException(status_code=400, detail="File must be an .osr file")
    # use replay ingestor using file path
    with open(file.filename, "wb") as f:
        f.write(file.file.read())
    job_id = replay_ingestor.ingest_replay(file.filename)
    return {"job_id": job_id}
    
@app.delete("/delete_replay")
def delete_replay(identifier: str):
    if db.remove_replay(identifier):
        return {"message": "Replay deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Replay not found")

@app.delete("/delete_map")
def delete_map(identifier: str):
    # Try deleting by mapset_id (if integer) or by map_hash (if string hash)
    if identifier.isdigit():
        if db.delete_mapset_by_id(int(identifier)):
            return {"message": f"Mapset {identifier} deleted successfully"}
    elif db.remove_map_by_hash(identifier):
        return {"message": f"Map {identifier} deleted successfully"}
    raise HTTPException(status_code=404, detail="Map not found")

@app.get("/jobs")
def get_job_status(job_id: str):
    if not job_id or job_id.lower() in ("null", "undefined", "none"):
        raise HTTPException(status_code=400, detail="Invalid job_id")
    job = replay_ingestor.get_job_status(job_id)
    return job


def resolve_user_info(osu_id: str):
    """Resolves username or numeric ID to numeric_id string and user dict."""
    user = db.get_user(osu_id)
    if user and user.get("osu_id"):
        return str(user["osu_id"]), user
    try:
        osu_user = client.get_user(osu_id)
        if osu_user:
            numeric_id = str(osu_user.id)
            db.add_user(numeric_id, osu_user.username)
            u_obj = {"osu_id": numeric_id, "username": osu_user.username}
            return numeric_id, u_obj
    except Exception as e:
        logger.error(f"Error resolving user '{osu_id}': {e}")
    return str(osu_id), user

@app.get("/users")
def get_user(osu_id: str):
    numeric_id, user = resolve_user_info(osu_id)
    active_job = replay_ingestor.get_active_top_replay_job(numeric_id)
    if active_job:
        return {"user": user, "job_id": active_job}

    if user is None:
        raise HTTPException(status_code=404, detail=f"User '{osu_id}' not found")
    return {"user": user}

@app.get("/user/replays")
def get_user_replays(osu_id: str):
    numeric_id, user = resolve_user_info(osu_id)
    active_job = replay_ingestor.get_active_top_replay_job(numeric_id)
    if active_job:
        return {"job_id": active_job}

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    replays = db.get_user_replays(numeric_id)
    tries = 3
    while (replays is None or len(replays) < 25) and tries > 0:
        job_id = replay_ingestor.ingest_top_replays(numeric_id, limit=100, recent_limit=50)
        tries -= 1
        return {"job_id": job_id}
    return {"replays": replays}

@app.post("/user/recalibrate")
def recalibrate_user(osu_id: str = Query(..., description="osu! User ID or Username")):
    """
    Forces a fresh fetch of top 100 best plays and 50 recent plays from osu! API v2.
    """
    numeric_id, user = resolve_user_info(osu_id)
    target_id = numeric_id if numeric_id else osu_id
    job_id = replay_ingestor.ingest_top_replays(target_id, limit=100, recent_limit=50, force=True)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "queued"})

@app.get("/user/recommended_maps")
def get_recommendations(
    osu_id: str = Query(..., description="osu! User ID or Username"),
    k: int = Query(10, description="Number of map recommendations to return"),
    exclude_played: bool = Query(True, description="Whether to exclude maps already played by the user")
):
    """
    Get personalized beatmap recommendations for a user.
    """
    clean_id = str(osu_id).strip().lower()
    numeric_id = db.get_osu_id_from_username(clean_id)
    user = db.get_user(numeric_id if numeric_id else osu_id)
    target_uid = numeric_id if numeric_id else osu_id
    str_uid = str(target_uid).strip().lower()
    redis_key = f"cache:endpoint:recs:{str_uid}:{k}:{exclude_played}"

    replays = db.get_user_replays(target_uid)
    if (replays is None or len(replays) < 25):
        numeric_id, user = resolve_user_info(target_uid)
        target_uid = numeric_id if numeric_id else target_uid
        active_job = replay_ingestor.get_active_top_replay_job(target_uid)
        if not active_job:
            try:
                active_job = replay_ingestor.ingest_top_replays(target_uid, limit=100, recent_limit=50)
            except Exception as e_ingest:
                logger.warning(f"Could not trigger top replay ingestion for user '{target_uid}': {e_ingest}")
        if active_job:
            return {"recommended_maps": [], "job_id": active_job, "status": "ingesting"}

    if db and getattr(db, 'redis_client', None):
        try:
            cached = db.redis_client.get(redis_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

    if user is None and (replays is None or len(replays) == 0):
        raise HTTPException(status_code=404, detail="User not found")

    try:
        recommended_maps = recommender.get_user_recommendations(target_uid, k=k, exclude_played=exclude_played)
        if not recommended_maps and replays:
            recommender.refresh_user(target_uid)
            recommended_maps = recommender.get_user_recommendations(target_uid, k=k, exclude_played=exclude_played)
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        recommended_maps = []

    target_hashes = [r['map_hash'] for r in recommended_maps]
    influential_plays = recommender.get_influential_plays_batch(target_uid, target_hashes)

    final_recommended_maps = []
    # for each recommended map, turn its embed into 2D coordinates for "fog of war"
    for recommended_map in recommended_maps:
        h = recommended_map['map_hash']
        map_ = db.get_map_by_hash(h) or {}
        cx = map_.get('coord_x', 0.0)
        cy = map_.get('coord_y', 0.0)
        if (cx == 0.0 or cy == 0.0):
            emb_vec = recommender.get_embed_by_hash(h)
            if emb_vec is not None:
                cx, cy = _map_projector.transform(emb_vec)
                if cx != 0.0 or cy != 0.0:
                    db.update_map_coordinates(h, cx, cy)
        recommended_map['coord_x'] = cx
        recommended_map['coord_y'] = cy
        recommended_map['map_id'] = map_.get('beatmap_id') or map_.get('map_id')
        recommended_map['creator'] = map_.get('creator') or 'Unknown Creator'
        recommended_map['title'] = map_.get('title') or 'Unknown Title'
        recommended_map['artist'] = map_.get('artist') or 'Unknown Artist'
        recommended_map['difficulty'] = map_.get('difficulty') or map_.get('version') or 'Normal'
        sr_val = float(map_.get('sr') or map_.get('star_rating') or 0.0)
        recommended_map['star_rating'] = sr_val
        recommended_map['sr'] = sr_val
        plays_list = influential_plays.get(h, [])
        recommended_map['influential_plays'] = plays_list
        recommended_map['influential_play'] = plays_list[0] if len(plays_list) > 0 else None
        final_recommended_maps.append(recommended_map)
    
    result = {"recommended_maps": final_recommended_maps}
    if db and getattr(db, 'redis_client', None):
        try:
            db.redis_client.set(redis_key, json.dumps(result), ex=900)
        except Exception:
            pass

    return result

candidate_paths = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "static")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static")),
    os.path.abspath("static"),
    os.path.abspath("../static"),
]

static_dir = None
for p in candidate_paths:
    if os.path.exists(p) and os.path.isfile(os.path.join(p, "index.html")):
        static_dir = p
        break

if static_dir:
    logger.info(f"Mounting static files from: {static_dir}")
    assets_dir = os.path.join(static_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
else:
    logger.warning("Static directory containing index.html was not found!")

@app.get("/")
def read_root():
    if static_dir and os.path.exists(os.path.join(static_dir, "index.html")):
        return FileResponse(os.path.join(static_dir, "index.html"))
    raise HTTPException(status_code=404, detail="Frontend static index.html not found.")

@app.get("/{full_path:path}")
def catch_all_spa(full_path: str):
    if static_dir:
        # 1. Check if full_path is a static asset file on disk (e.g. assets, favicon)
        target_file = os.path.join(static_dir, full_path)
        if os.path.isfile(target_file):
            return FileResponse(target_file)

        # 2. Exclude OpenAPI / Swagger docs from SPA fallback
        if full_path.startswith(("docs", "openapi.json", "redoc", "api/")):
            raise HTTPException(status_code=404, detail="API route not found")

        # 3. Serve SPA index.html for all frontend routes (e.g. /users/username, /users/)
        index_file = os.path.join(static_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)

    raise HTTPException(status_code=404, detail="Not Found")

if __name__ == "__main__":
    # Load settings from config
    host = getattr(config, 'SERVER_HOST', getattr(config, 'server_host', '127.0.0.1'))
    port = getattr(config, 'SERVER_PORT', getattr(config, 'server_port', 8000))
    uvicorn.run(app, host=host, port=port)
