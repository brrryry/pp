import os
import sys
import glob
import logging
import hashlib
import shutil
import sqlite3
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from osrparse import Replay

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config

db_file = config.DB_FILE

# Configure logging
logging.basicConfig(level=logging.INFO, format="[Replay Utility] %(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("import_replays")

def calculate_mastery(acc, misses, total_obj, combo, max_combo):
    """Normalizes game performance into a clean 0.0 - 1.0 confidence score."""
    if total_obj == 0 or max_combo == 0:
        return 0.0
    acc_factor = acc ** 3
    miss_factor = max(0.0, 1.0 - (misses / total_obj))
    combo_factor = combo / max_combo
    return acc_factor * miss_factor * combo_factor  

def compute_md5(file_path):
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    except Exception:
        return None

def process_replay_file(file_path, beatmap_index=None):
    """
    Parses a single local .osr replay file and extracts clean metrics.
    Integrates a map database lookup via the replay's MD5 beatmap hash.
    """
    # Note: beatmap_index is passed but not actively used for metrics calculations 
    # in this fallback snippet. It remains available for your database expansion.
    replay = Replay.from_path(file_path)
    c300 = replay.count_300
    c100 = replay.count_100
    c50 = replay.count_50
    misses = replay.count_miss
    max_combo = replay.max_combo

    # 1. FIXED ACCURACY FORMULA (Enforced correct mathematical precedence)
    total_hits = c300 + c100 + c50 + misses
    if total_hits == 0:
        accuracy = 0.0
    else:
        actual_score_points = (300 * c300) + (100 * c100) + (50 * c50)
        maximum_score_points = 300 * total_hits
        accuracy = actual_score_points / maximum_score_points

    # Fallback placeholders if database lookup isn't passed in your local environment yet
    total_obj = total_hits
    max_map_combo = max_combo

    # 3. COMPUTE FINAL CONFIDENCE SCORE
    mastery_score = calculate_mastery(accuracy, misses, total_obj, max_combo, max_map_combo)

    # Return structured dictionary including the clean mastery score ready for database seeding
    return {
        'username': replay.username,
        'replay_hash': replay.replay_hash,
        'map_hash': replay.beatmap_hash,
        'mods': replay.mods.value,  # Bitmask integer tracking HD, HR, DT configurations
        'accuracy': accuracy,
        'misses': misses,
        'max_combo': max_combo,
        'mastery_score': mastery_score
    }

def worker_task(file_path, seed_only, replays_dir):
    """
    Worker task wrapping file management operations and replay processing 
    so it can be safely executed inside a separate process.
    """
    filename = os.path.basename(file_path)

    # check if replay already exists in db and has same md5 hash
    # if it does, skip
    replay_hash = Replay.from_path(file_path).replay_hash
    if replay_hash:
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT replay_hash FROM replays WHERE replay_hash = ?", (replay_hash,))
            if cursor.fetchone():
                conn.close()
                return False, None, "Replay already exists in database"
            conn.close()
        except Exception as e:
            logger.error(f"Failed to check database: {e}")

    try:
        # If not in seed-only mode, copy file to replays_dir
        if not seed_only:
            target_replay_path = os.path.join(replays_dir, filename)
            if os.path.abspath(file_path) != os.path.abspath(target_replay_path):
                os.makedirs(replays_dir, exist_ok=True)
                shutil.copy2(file_path, target_replay_path)
            current_file_path = target_replay_path
        else:
            current_file_path = file_path

        # Process replay file and extract mastery metrics
        # (beatmap_index omitted here to avoid massive IPC overhead passing the dict to workers)
        replay_data = process_replay_file(current_file_path)
        return True, replay_data, None
    except Exception as e:
        return False, None, f"Failed to process {filename}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Import and seed osu! replays into processed database.")
    parser.add_argument(
        "--dir", type=str, default=None, help="Directory containing the raw .osr replay files to import."
    )
    parser.add_argument(
        "--beatmaps_dir", type=str, default=None, help="Optional directory containing local .osu files to match replay hashes."
    )
    parser.add_argument(
        "--limit", type=int, default=5000, help="Maximum number of replays to import/process (use -1 to process all)."
    )
    parser.add_argument(
        "--replays_dir", type=str, default=config.REPLAYS_DIR, help="Directory where imported replays are stored/read from."
    )
    parser.add_argument(
        "--db_file", type=str, default=config.DB_FILE, help="SQLite database file path to seed."
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of parallel workers (defaults to CPU count)."
    )
    args = parser.parse_args()

    # If --dir is not provided, we run in seed-only mode using --replays_dir
    seed_only = args.dir is None

    if seed_only:
        if not os.path.exists(args.replays_dir):
            logger.error(f"Replays directory not found at {args.replays_dir}. Cannot run seeding.")
            return
        files = glob.glob(os.path.join(args.replays_dir, "*.osr"))
        logger.info(f"Seed-only mode active. Found {len(files)} replay files in directory: {args.replays_dir}")
    else:
        if not os.path.exists(args.dir):
            logger.error(f"Input replays directory not found at {args.dir}")
            return
        files = glob.glob(os.path.join(args.dir, "*.osr"))
        logger.info(f"Import mode active. Found {len(files)} raw replay files in directory: {args.dir}")

    if len(files) == 0:
        logger.warning("No replay files to process.")
        return

    # Enforce import/processing limit if args.limit > 0 and len(files) > args.limit:
    if args.limit > 0 and len(files) > args.limit:
        logger.info(f"Limiting processing to first {args.limit} replays.")
        files = files[:args.limit]

    # Index local beatmaps folder if provided
    beatmap_index = {}
    if args.beatmaps_dir and os.path.exists(args.beatmaps_dir):
        logger.info(f"Scanning and indexing beatmaps folder: {args.beatmaps_dir}...")
        for root, _, f_list in os.walk(args.beatmaps_dir):
            for f in f_list:
                if f.endswith('.osu'):
                    p = os.path.join(root, f)
                    h_val = compute_md5(p)
                    if h_val:
                        beatmap_index[h_val] = p
        logger.info(f"Successfully indexed {len(beatmap_index)} local beatmap files.")

    # Ensure DB parent directory exists
    db_dir = os.path.dirname(args.db_file)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(args.db_file)
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS replays ("
        "username TEXT, "
        "replay_hash TEXT UNIQUE, "
        "map_hash TEXT, "
        "mods INTEGER, "
        "accuracy REAL, "
        "misses INTEGER, "
        "max_combo INTEGER, "
        "mastery_score REAL"
        ")"
    )

    processed = 0
    
    # Process files using ProcessPoolExecutor
    logger.info(f"Starting parallel processing with {args.workers or 'all available'} CPU workers...")
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit tasks to the process pool
        future_to_file = {
            executor.submit(worker_task, file_path, seed_only, args.replays_dir): file_path 
            for file_path in files
        }
        
        # Gather results as they finish and stream them safely into SQLite
        for future in tqdm(as_completed(future_to_file), total=len(files), desc="Processing replays"):
            success, replay_data, error_msg = future.result()
            
            if success and replay_data:
                try:
                    cursor.execute(
                        "INSERT INTO replays VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            replay_data['username'],
                            replay_data['replay_hash'],
                            replay_data['map_hash'],
                            replay_data['mods'],
                            replay_data['accuracy'],
                            replay_data['misses'],
                            replay_data['max_combo'],
                            replay_data['mastery_score']
                        )
                    )
                    processed += 1
                except sqlite3.IntegrityError:
                    # Catch duplicate hash inserts if UNIQUE constraint triggers
                    pass
                except Exception as db_err:
                    logger.error(f"Database insertion failed: {db_err}")
            else:
                if error_msg:
                    logger.error(error_msg)

    conn.commit()
    conn.close()
    logger.info(f"Successfully processed and seeded {processed}/{len(files)} replays into {args.db_file}")

if __name__ == "__main__":
    main()