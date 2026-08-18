import os
import sys
import glob
import logging
import hashlib
import sqlite3
import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import rosu_pp_py as rosu
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from ml_parser import osu_to_ml_sequence, parse_osu_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="[Beatmap Utility] %(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("seed_maps")

# Global variables initialized in worker processes
_model = None
_device = None

def init_worker(model_path):
    global _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model architecture locally within worker
    class LSTMEmbedder(nn.Module):
        def __init__(self, input_size=9, hidden_size=128, output_size=32, num_layers=2):
            super(LSTMEmbedder, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    # Instantiate model and load checkpoint weights
    try:
        _model = LSTMEmbedder()
        _model.load_state_dict(torch.load(model_path, map_location=_device))
        _model.to(_device)
        _model.eval()
    except Exception as e:
        logger.error(f"Worker failed to load model from {model_path}: {e}")
        raise e

def worker_task(file_path):
    global _model, _device
    try:
        # 1. Compute MD5 hash of file content
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        map_hash = hasher.hexdigest()

        # 2. Parse metadata fields
        res = parse_osu_file(file_path)
        metadata = res.get('metadata', {})

        map_id_str = metadata.get('BeatmapID', '0')
        mapset_id_str = metadata.get('BeatmapSetID', '0')

        try:
            map_id = int(map_id_str)
        except ValueError:
            map_id = 0
        try:
            mapset_id = int(mapset_id_str)
        except ValueError:
            mapset_id = 0

        title = metadata.get('Title', '')
        artist = metadata.get('Artist', '')
        creator = metadata.get('Creator', '')
        version = metadata.get('Version', '')

        # 3. Generate ML sequence (shape [2000, 9])
        sequence = osu_to_ml_sequence(file_path, max_seq_len=2000)

        # 4. Generate Embedding via PyTorch model
        with torch.no_grad():
            sequence_tensor = sequence.clone().detach().float().unsqueeze(0).to(_device)
            embedding = _model(sequence_tensor)
            embedding_np = embedding.squeeze(0).cpu().numpy()

        # 5. Get star rating using rosu_pp_py
        beatmap = rosu.Beatmap(path=file_path)
        diff = rosu.Difficulty().calculate(beatmap)
        sr = diff.stars

        # Serialize embedding as a JSON list (standard for storing INT[]/FLOAT[] arrays in SQLite)
        embed_json = json.dumps(embedding_np.tolist())

        return True, {
            'map_id': map_id,
            'mapset_id': mapset_id,
            'map_hash': map_hash,
            'title': title,
            'artist': artist,
            'creator': creator,
            'version': version,
            'embed': embed_json,
            'sr': sr
        }, None
    except Exception as e:
        return False, None, f"Failed to process {os.path.basename(file_path)}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Process beatmaps and save embeddings to database.")
    parser.add_argument(
        "--maps_dir",
        type=str,
        default=config.MAPS_DIR,
        help="Directory containing local .osu beatmap files."
    )
    parser.add_argument(
        "--db_file",
        type=str,
        default=config.DB_FILE,
        help="SQLite database file path to seed."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/lstm_osu_embedder_best_1.pth",
        help="Path to the trained LSTM encoder model weights."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Maximum number of beatmaps to process (default: -1 for all)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (defaults to CPU count)."
    )

    args = parser.parse_args()

    # Search for model file in multiple locations
    model_path = args.model_path
    if not os.path.exists(model_path):
        # Check sibling path options
        alternatives = [
            "lstm_osu_embedder_best_1.pth",
            "models/lstm_osu_embedder_best_1.pth",
            os.path.join(os.path.dirname(__file__), "models", "lstm_osu_embedder_best_1.pth"),
            os.path.join(os.path.dirname(__file__), "lstm_osu_embedder_best_1.pth")
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                model_path = alt
                break

    if not os.path.exists(model_path):
        logger.error(f"Encoder model weights not found at {args.model_path} or alternatives.")
        return

    # Collect .osu files
    if not os.path.exists(args.maps_dir):
        logger.error(f"Beatmaps directory not found at {args.maps_dir}")
        return

    files = glob.glob(os.path.join(args.maps_dir, "**/*.osu"), recursive=True)
    logger.info(f"Found {len(files)} beatmap files in: {args.maps_dir}")

    if len(files) == 0:
        logger.warning("No beatmaps to process.")
        return

    # Enforce limit if specified
    if args.limit > 0 and len(files) > args.limit:
        logger.info(f"Limiting processing to first {args.limit} beatmaps.")
        files = files[:args.limit]

    # Ensure DB folder exists
    db_dir = os.path.dirname(args.db_file)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # Initialize SQLite schema
    conn = sqlite3.connect(args.db_file)
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS maps ("
        "map_id INTEGER, "
        "mapset_id INTEGER, "
        "map_hash TEXT UNIQUE PRIMARY KEY, "
        "title TEXT, "
        "artist TEXT, "
        "creator TEXT, "
        "version TEXT, "
        "embed TEXT, "
        "sr FLOAT"
        ")"
    )
    conn.commit()

    processed = 0

    # Start multiprocessing spawn context
    ctx = multiprocessing.get_context("spawn")
    logger.info(f"Starting parallel beatmap processing with {args.workers or 'all available'} CPU workers...")

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=init_worker, initargs=(model_path,)) as executor:
        # Submit task for each file
        future_to_file = {
            executor.submit(worker_task, file_path): file_path for file_path in files
        }

        # Gather results and save to SQLite
        for future in tqdm(as_completed(future_to_file), total=len(files), desc="Generating embeddings"):
            success, map_data, error_msg = future.result()

            if success and map_data:
                try:
                    cursor.execute(
                        "INSERT OR REPLACE INTO maps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            map_data['map_id'],
                            map_data['mapset_id'],
                            map_data['map_hash'],
                            map_data['title'],
                            map_data['artist'],
                            map_data['creator'],
                            map_data['version'],
                            map_data['embed'],
                            map_data['sr']
                        )
                    )
                    processed += 1
                except Exception as db_err:
                    logger.error(f"Database write failed for map {map_data['map_hash']}: {db_err}")
            else:
                if error_msg:
                    logger.error(error_msg)

    conn.commit()
    conn.close()

    logger.info(f"Successfully processed and seeded {processed}/{len(files)} maps into {args.db_file}")

if __name__ == "__main__":
    main()