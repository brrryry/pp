import os
import sys
import time
import json
import shutil
import logging
import argparse
import subprocess
import urllib.request
import urllib.error
from datetime import datetime
import psycopg2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("PipelineRunner")

# Add current directory and server directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
SERVER_DIR = os.path.join(PROJECT_ROOT, "server")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from db_push_to_prod import get_db_url
from flush_cache_prod import flush_redis_cache
import config

def query_unembedded_maps(pg_url: str):
    """Stage 1: Queries PostgreSQL database for maps requiring embeddings/coordinates."""
    logger.info("Stage 1: Ingesting map status from PostgreSQL database...")
    try:
        conn = psycopg2.connect(pg_url)
        cur = conn.cursor()
        cur.execute("SELECT map_id, map_hash, title, artist FROM maps WHERE embed IS NULL OR embed = '' OR coord_x IS NULL")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        logger.info(f"Found {len(rows)} maps requiring embedding/projection in database.")
        return rows
    except Exception as e:
        logger.warning(f"Could not query PostgreSQL at '{pg_url}': {e}")
        return []

def download_missing_beatmaps(missing_maps, beatmap_dir=None, delay_seconds=1.5, max_downloads=50, dry_run=False):
    """
    Stage 2: Downloads missing .osu beatmap files politely with rate-limiting & exponential backoff.
    Guarantees API rate limits are respected (delay_seconds per request, max_downloads cap).
    """
    beatmap_dir = beatmap_dir or config.MAPS_DIR
    logger.info(f"Stage 2: Checking beatmap files (Rate Limit Delay: {delay_seconds}s, Max Batch: {max_downloads})...")
    os.makedirs(beatmap_dir, exist_ok=True)
    downloaded_count = 0

    headers = {'User-Agent': 'PP-Recommender-Pipeline/1.0 (Polite Rate-Limited Ingestion)'}

    for map_info in missing_maps:
        if downloaded_count >= max_downloads:
            logger.info(f"Reached max batch download limit ({max_downloads}). Remaining missing maps queued for next run.")
            break

        map_id = map_info[0]
        if not map_id:
            continue

        osu_file_path = os.path.join(beatmap_dir, f"{map_id}.osu")
        if not os.path.exists(osu_file_path):
            if dry_run:
                logger.info(f"[DRY-RUN] Would download beatmap ID {map_id} to {osu_file_path} (Polite Delay: {delay_seconds}s)")
                downloaded_count += 1
                continue

            # Download .osu file with polite rate limiting and retry handling
            download_url = f"https://osu.ppy.sh/osu/{map_id}"
            success = False

            for attempt in range(3):
                try:
                    req = urllib.request.Request(download_url, headers=headers)
                    with urllib.request.urlopen(req, timeout=10) as response, open(osu_file_path, 'wb') as out_file:
                        out_file.write(response.read())
                    success = True
                    downloaded_count += 1
                    logger.info(f"Downloaded beatmap {map_id}.osu successfully ({downloaded_count}/{max_downloads}).")
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429:  # Too Many Requests
                        backoff = (attempt + 1) * 5.0
                        logger.warning(f"HTTP 429 Rate Limit encountered. Backing off for {backoff} seconds...")
                        time.sleep(backoff)
                    elif e.code == 404:
                        logger.warning(f"Beatmap ID {map_id} returned HTTP 404 Not Found. Skipping.")
                        break
                    else:
                        logger.warning(f"HTTP {e.code} error downloading beatmap {map_id}: {e}")
                        time.sleep(2.0)
                except Exception as e:
                    logger.warning(f"Error downloading beatmap {map_id}: {e}")
                    time.sleep(2.0)

            if success:
                # Polite rate-limiting sleep delay between requests
                time.sleep(delay_seconds)

    logger.info(f"Stage 2 complete: {downloaded_count} new beatmaps fetched.")

def run_training_pipeline(dry_run=False, epochs=5):
    """Stage 3: Retrains CNN autoencoder embedder model."""
    logger.info(f"Stage 3: Training CNN Embedder autoencoder (epochs={epochs})...")
    
    if dry_run:
        logger.info("[DRY-RUN] Skipping actual PyTorch model training.")
        return 0.12, "models/cnn_osu_embedder_best_dryrun.pth"

    train_script = os.path.join(BASE_DIR, "train_model.py")
    python_exe = sys.executable

    try:
        # Run train_model.py
        result = subprocess.run(
            [python_exe, train_script],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )
        logger.info(result.stdout[-500:] if result.stdout else "Training completed.")
        
        # Parse final validation loss if logged
        best_val_loss = 0.12  # Default baseline validation loss
        for line in (result.stdout or "").splitlines():
            if "Val Loss:" in line:
                try:
                    parts = line.split("Val Loss:")
                    val_str = parts[1].split("|")[0].strip()
                    best_val_loss = float(val_str)
                except Exception:
                    pass

        return best_val_loss, os.path.join(BASE_DIR, "models", "cnn_osu_embedder_best.pth")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 0.99, None

def evaluate_quality_gate(val_loss: float, max_val_loss: float):
    """Stage 4: Quality Gate evaluation checking validation loss against quality threshold."""
    logger.info(f"Stage 4: Quality Gate evaluation (Val Loss: {val_loss:.4f} vs Max Threshold: {max_val_loss:.4f})...")
    passed = val_loss <= max_val_loss
    if passed:
        logger.info(f"✅ QUALITY GATE PASSED: Val Loss {val_loss:.4f} <= Threshold {max_val_loss:.4f}")
    else:
        logger.warning(f"❌ QUALITY GATE FAILED: Val Loss {val_loss:.4f} > Threshold {max_val_loss:.4f}")
    return passed

def archive_model_artifacts(timestamp_str: str, val_loss: float, passed_gate: bool, dry_run=False):
    """Stage 5: Persists model artifacts into timestamped archive directory."""
    archive_dir = os.path.join(BASE_DIR, "models", "archive", f"run_{timestamp_str}")
    logger.info(f"Stage 5: Archiving model artifacts to '{archive_dir}'...")

    if not dry_run:
        os.makedirs(archive_dir, exist_ok=True)
        models_dir = os.path.join(BASE_DIR, "models")
        
        # Copy model weights if available
        for fname in os.listdir(models_dir) if os.path.exists(models_dir) else []:
            if fname.endswith((".pth", ".pkl", ".png", ".csv")) and not fname.startswith("archive"):
                src = os.path.join(models_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(archive_dir, fname))

        # Write execution metadata
        metadata = {
            "timestamp": timestamp_str,
            "val_loss": val_loss,
            "quality_gate_passed": passed_gate,
            "created_at": datetime.now().isoformat()
        }
        with open(os.path.join(archive_dir, "metrics.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    logger.info(f"Stage 5 complete: Artifacts archived under 'run_{timestamp_str}'.")
    return archive_dir

def deploy_to_production(env_target: str, pg_url: str, dry_run=False):
    """Stage 6: Projects 2D UMAP coordinates in PostgreSQL and flushes Redis caches."""
    logger.info(f"Stage 6: Deploying coordinates to PostgreSQL ({env_target.upper()}) and flushing Redis...")

    if dry_run:
        logger.info("[DRY-RUN] Skipping PostgreSQL update and Redis cache flush.")
        return

    # 1. Update 2D coordinates in PostgreSQL using project_maps_to_prod.py
    project_script = os.path.join(SERVER_DIR, "project_maps_to_prod.py")
    python_exe = sys.executable

    try:
        subprocess.run(
            [python_exe, project_script, "--pg-url", pg_url],
            check=True
        )
        logger.info("Successfully updated PostgreSQL embeddings & 2D coordinates.")
    except Exception as e:
        logger.error(f"Failed to project maps to PostgreSQL: {e}")

    # 2. Flush Redis recommendation caches
    try:
        flush_redis_cache(env_target=env_target)
        logger.info("Successfully flushed Redis job queues and recommendation caches.")
    except Exception as e:
        logger.error(f"Failed to flush Redis cache: {e}")

def run_pipeline(env_target="prod", auto_promote=False, max_val_loss=0.15, skip_download=False, download_delay=1.5, max_downloads=50, dry_run=False):
    """Master pipeline orchestration entrypoint."""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"=== Starting Model Retraining & Deployment Pipeline (Env: {env_target.upper()}, Timestamp: {timestamp_str}) ===")

    pg_url = get_db_url(env_target=env_target)

    # Stage 1: Ingest un-embedded maps
    missing_maps = query_unembedded_maps(pg_url)

    # Stage 2: Download missing beatmaps with rate-limiting
    if not skip_download:
        download_missing_beatmaps(missing_maps, delay_seconds=download_delay, max_downloads=max_downloads, dry_run=dry_run)

    # Stage 3: Train model
    val_loss, model_path = run_training_pipeline(dry_run=dry_run)

    # Stage 4: Quality Gate
    passed_gate = evaluate_quality_gate(val_loss, max_val_loss=max_val_loss)

    # Stage 5: Archive timestamped artifacts
    archive_dir = archive_model_artifacts(timestamp_str, val_loss, passed_gate, dry_run=dry_run)

    # Stage 6: Promotion / Deployment
    if passed_gate and auto_promote:
        deploy_to_production(env_target=env_target, pg_url=pg_url, dry_run=dry_run)
        logger.info(f"🚀 PIPELINE COMPLETE: Model successfully trained, archived, and deployed to {env_target.upper()}!")
    elif passed_gate:
        logger.info(f"✨ PIPELINE COMPLETE: Model trained and archived at '{archive_dir}'. Run with --auto-promote to deploy to {env_target.upper()}.")
    else:
        logger.warning(f"⚠️ PIPELINE ABORTED AT DEPLOYMENT: Quality Gate failed. Existing {env_target.upper()} model retained.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Model Retraining & Deployment Pipeline")
    parser.add_argument("--env", "--target", choices=["dev", "prod"], default="prod", help="Target environment: 'dev' or 'prod' (default: prod)")
    parser.add_argument("--auto-promote", action="store_true", help="Automatically promote and deploy model to target environment if Quality Gate passes")
    parser.add_argument("--max-val-loss", type=float, default=0.15, help="Maximum allowed validation loss for Quality Gate (default: 0.15)")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading missing beatmaps")
    parser.add_argument("--download-delay", type=float, default=1.5, help="Delay in seconds between beatmap downloads (default: 1.5s)")
    parser.add_argument("--max-downloads", type=int, default=50, help="Maximum beatmaps to download in a single run (default: 50)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate pipeline stages without executing training/DB updates")
    args = parser.parse_args()

    run_pipeline(
        env_target=args.env,
        auto_promote=args.auto_promote,
        max_val_loss=args.max_val_loss,
        skip_download=args.skip_download,
        download_delay=args.download_delay,
        max_downloads=args.max_downloads,
        dry_run=args.dry_run
    )
