import os
import sys
import sqlite3
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import rosu_pp_py as rosu

# Set up logging to console and file
from logger_setup import setup_logger
logger = setup_logger("dataset")

def print(*args, **kwargs):
    import builtins
    builtins.print(*args, **kwargs)
    msg = " ".join(str(arg) for arg in args)
    logger.info(msg)

# Add parent directory to path to find config and src package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.parser import parse_osu_file
from core.features import extract_map_features
import config

DATASET_DB = "data/osu_profiler.db"
TABLE_NAME = "beatmaps"

def process_single_map(args):
    """
    Worker function to process a single map file. Runs in a separate process.
    Calculates difficulty/star rating locally using rosu-pp-py.
    """
    filename, maps_dir, star_threshold = args
    file_path = os.path.join(maps_dir, filename)
    parts = filename.split('_')
    if not parts:
        return None
    beatmapset_id_str = parts[0]
    
    try:
        # 1. Calculate difficulty/star rating locally using rosu-pp-py
        map_data = rosu.Beatmap(path=file_path)
        diff_attrs = rosu.Difficulty().calculate(map_data)
        star_rating = diff_attrs.stars
        
        if star_rating is None or star_rating < star_threshold:
            os.remove(file_path)
            return None
            
        # 2. Parse the map for custom features
        parsed = parse_osu_file(file_path)
        
        # 3. Extract custom features
        features = extract_map_features(parsed)
        if features is None:
            return None
            
        # 4. Attach target rating and metadata
        features['star_rating'] = star_rating
        features['filename'] = filename
        features['beatmapset_id'] = beatmapset_id_str
        return features
    except Exception:
        return None

def infer_sqlite_type(val):
    """Helper to map Python data types to SQLite data types."""
    if isinstance(val, int):
        return "INTEGER"
    elif isinstance(val, float):
        return "REAL"
    return "TEXT"

def build_dataset(maps_dir=None, limit=None, star_threshold=5.0, max_workers=None):
    if maps_dir is None:
        maps_dir = config.maps_path

    print(f"Scanning maps in '{maps_dir}'...")
    if not os.path.exists(maps_dir):
        print(f"Error: Maps directory '{maps_dir}' does not exist.")
        return

    osu_files = [f for f in os.listdir(maps_dir) if f.endswith(".osu")]
    if limit:
        osu_files = osu_files[:limit]

    print(f"Processing {len(osu_files)} map files using ProcessPoolExecutor...")

    tasks = [(f, maps_dir, star_threshold) for f in osu_files]
    compiled_features = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_map, tasks),
            total=len(tasks),
            desc="Processing beatmaps"
        ))
        
        compiled_features = [r for r in results if r is not None]

    if not compiled_features:
        print("No beatmaps were successfully processed with matching star ratings.")
        return

    # Ensure the output data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Drop existing beatmaps table for a clean rebuild, but preserve other tables (cache, hits, etc.)
    conn_pre = sqlite3.connect(DATASET_DB)
    conn_pre.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    conn_pre.commit()
    conn_pre.close()

    # Establish connection to SQLite
    conn = sqlite3.connect(DATASET_DB)
    cursor = conn.cursor()

    # 1. Dynamically build the table schema based on the keys of the first item
    sample_item = compiled_features[0]
    columns_definition = []
    columns_order = list(sample_item.keys())

    for col in columns_order:
        sqlite_type = infer_sqlite_type(sample_item[col])
        columns_definition.append(f'"{col}" {sqlite_type}')

    create_table_sql = f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} ({', '.join(columns_definition)});"
    cursor.execute(create_table_sql)

     # 2. Prepare the bulk insert statement
    placeholders = ", ".join(["?"] * len(columns_order))
    
    # Escape column names safely outside the f-string to support Python 3.11 and older
    escaped_columns = ", ".join(f'"{c}"' for c in columns_order)
    insert_sql = f"INSERT INTO {TABLE_NAME} ({escaped_columns}) VALUES ({placeholders});"

    # 3. Format the data into clean tuples matching the columns order
    data_tuples = []
    for item in compiled_features:
        # Defaults to None (NULL in SQL) if a feature is somehow missing a key
        data_tuples.append(tuple(item.get(col, None) for col in columns_order))

    # 4. Write data to SQLite using ultra-fast bulk execution
    cursor.executemany(insert_sql, data_tuples)
    
    # Create indexes for columns you will search through frequently (highly speeds up query performance)
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_beatmapset_id ON {TABLE_NAME} (beatmapset_id);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_star_rating ON {TABLE_NAME} (star_rating);")

    conn.commit()
    
    # Fetch final count directly from SQL for the confirmation log
    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
    total_rows = cursor.fetchone()[0]
    conn.close()

    print(f"\nSuccessfully compiled features for {total_rows} maps!")
    print(f"Saved dataset directly to SQLite database at {DATASET_DB}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Osu! map feature dataset builder")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--workers", type=int, default=None, help="Number of processes to use")
    parser.add_argument("--star_threshold", type=float, default=5.0, help="Minimum star rating to include")
    
    args = parser.parse_args()
    build_dataset(limit=args.limit, max_workers=args.workers, star_threshold=args.star_threshold)
