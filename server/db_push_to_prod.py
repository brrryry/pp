import os
import sys
import sqlite3
import logging
import argparse
from urllib.parse import urlparse

try:
    import psycopg2
    from psycopg2 import extras
except ImportError:
    print("Error: psycopg2 is required to run this script. Run 'pip install psycopg2-binary'")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

try:
    from core.DatabaseManager import DatabaseManager
except Exception:
    DatabaseManager = None

try:
    import config
    DEFAULT_SQLITE_PATH = config.DB_FILE
except ImportError:
    DEFAULT_SQLITE_PATH = "data/pp.db"

logging.basicConfig(level=logging.INFO, format="[DB Push] %(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DBPush")

def sanitize_cell(val):
    if isinstance(val, str):
        return val.replace('\x00', '')
    elif isinstance(val, bytes):
        try:
            return val.decode('utf-8', errors='ignore').replace('\x00', '')
        except Exception:
            return None
    return val

def sanitize_rows(rows):
    return [tuple(sanitize_cell(cell) for cell in row) for row in rows]

def get_db_url(env_target: str = "prod", host: str = None, port: int = None) -> str:
    """
    Constructs the PostgreSQL connection URL for 'dev' or 'prod' environment using credentials from server/.env.
    """
    env_paths = [
        os.path.join(SCRIPT_DIR, ".env"),
        os.path.join(PARENT_DIR, "server", ".env"),
        os.path.join(PARENT_DIR, ".env"),
        os.path.join(os.getcwd(), "server", ".env"),
        os.path.join(os.getcwd(), ".env")
    ]
    
    env_db_url = None
    env_password = None
    env_user = None
    env_dbname = None

    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DATABASE_URL="):
                        env_db_url = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("POSTGRES_PASSWORD="):
                        env_password = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("POSTGRES_USER="):
                        env_user = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("POSTGRES_DB="):
                        env_dbname = line.split("=", 1)[1].strip().strip('"').strip("'")

    user = env_user or "ppuser"
    password = env_password or "pppassword"
    dbname = env_dbname or "ppdb"

    if env_db_url:
        try:
            parsed = urlparse(env_db_url)
            if parsed.username:
                user = parsed.username
            if parsed.password:
                password = parsed.password
            if parsed.path and parsed.path != '/':
                dbname = parsed.path.lstrip('/')
        except Exception as e:
            logger.warning(f"Failed to parse DATABASE_URL from .env: {e}")

    if env_target.lower() == "dev":
        target_host = host or "localhost"
        target_port = port or 5433
    else:
        target_host = host or "bryan-nas.gkhomenetwork.lan"
        target_port = port or 5433

    return f"postgresql://{user}:{password}@{target_host}:{target_port}/{dbname}"

def ensure_schema_exists(pg_conn):
    """Initializes schema using CREATE TABLE IF NOT EXISTS."""
    cur = pg_conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS maps (
            id SERIAL PRIMARY KEY,
            map_id INTEGER,
            mapset_id INTEGER,
            map_hash TEXT UNIQUE NOT NULL,
            title TEXT,
            artist TEXT,
            creator TEXT,
            version TEXT,
            embed TEXT,
            sr DOUBLE PRECISION,
            coord_x DOUBLE PRECISION DEFAULT 0.0,
            coord_y DOUBLE PRECISION DEFAULT 0.0,
            projection_version INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS replays (
            user_and_map_hash TEXT PRIMARY KEY,
            username TEXT,
            replay_hash TEXT UNIQUE,
            map_hash TEXT,
            mods INTEGER,
            accuracy DOUBLE PRECISION,
            misses INTEGER,
            max_combo INTEGER,
            mastery_score DOUBLE PRECISION
        );

        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            osu_id BIGINT UNIQUE
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            osu_id BIGINT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            CONSTRAINT fk_sessions_user FOREIGN KEY(osu_id) REFERENCES users(osu_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_maps_map_hash ON maps (map_hash);
        CREATE INDEX IF NOT EXISTS idx_maps_mapset_id ON maps (mapset_id);
        CREATE INDEX IF NOT EXISTS idx_replays_username ON replays (username);
        CREATE INDEX IF NOT EXISTS idx_replays_map_hash ON replays (map_hash);
        CREATE INDEX IF NOT EXISTS idx_users_osu_id ON users (osu_id);
    """)
    pg_conn.commit()
    cur.close()

def push_to_db(sqlite_path: str, pg_url: str):
    logger.info(f"Starting push from SQLite '{sqlite_path}' to PostgreSQL '{pg_url}'...")

    if not os.path.exists(sqlite_path):
        candidates = [
            os.path.join(SCRIPT_DIR, sqlite_path),
            os.path.join(PARENT_DIR, sqlite_path),
            os.path.join(SCRIPT_DIR, "data", "pp.db"),
            os.path.join(PARENT_DIR, "data", "pp.db"),
            os.path.join(os.getcwd(), sqlite_path),
            os.path.join(os.getcwd(), "data", "pp.db"),
        ]
        found = False
        for cand in candidates:
            if os.path.exists(cand):
                sqlite_path = cand
                found = True
                break
        if not found:
            logger.error(f"SQLite file not found at: {sqlite_path}")
            sys.exit(1)

    logger.info(f"Using SQLite source file: {sqlite_path}")

    # 1. Connect to PostgreSQL and SQLite
    try:
        pg_conn = psycopg2.connect(pg_url)
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL at '{pg_url}': {e}")
        sys.exit(1)

    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()

    # Ensure PostgreSQL schema exists
    logger.info("Ensuring PostgreSQL schema exists...")
    if DatabaseManager:
        try:
            db_mgr = DatabaseManager(pg_url)
            db_mgr.initialize_schema()
        except Exception:
            ensure_schema_exists(pg_conn)
    else:
        ensure_schema_exists(pg_conn)

    try:
        # --- Push USERS ---
        logger.info("Pushing table 'users'...")
        sqlite_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if sqlite_cur.fetchone():
            sqlite_cur.execute("SELECT username, osu_id FROM users")
            user_rows = sqlite_cur.fetchall()
            logger.info(f"Found {len(user_rows)} users in SQLite.")

            if user_rows:
                sanitized_users = sanitize_rows(user_rows)
                user_insert_sql = """
                    INSERT INTO users (username, osu_id)
                    VALUES %s
                    ON CONFLICT (username) DO UPDATE SET osu_id = COALESCE(EXCLUDED.osu_id, users.osu_id);
                """
                extras.execute_values(pg_cur, user_insert_sql, sanitized_users, page_size=1000)
                pg_conn.commit()

        # --- Push MAPS ---
        logger.info("Pushing table 'maps'...")
        sqlite_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='maps'")
        if sqlite_cur.fetchone():
            sqlite_cur.execute("PRAGMA table_info(maps)")
            cols = [c[1] for c in sqlite_cur.fetchall()]
            
            has_coord_x = 'coord_x' in cols
            has_proj_ver = 'projection_version' in cols

            if has_coord_x and has_proj_ver:
                sqlite_cur.execute("SELECT map_id, mapset_id, map_hash, title, artist, creator, version, embed, sr, coord_x, coord_y, projection_version FROM maps")
            elif has_coord_x:
                sqlite_cur.execute("SELECT map_id, mapset_id, map_hash, title, artist, creator, version, embed, sr, coord_x, coord_y, 0 FROM maps")
            else:
                sqlite_cur.execute("SELECT map_id, mapset_id, map_hash, title, artist, creator, version, embed, sr, 0.0, 0.0, 0 FROM maps")
                
            map_rows = sqlite_cur.fetchall()
            logger.info(f"Found {len(map_rows)} maps in SQLite.")

            if map_rows:
                sanitized_maps = sanitize_rows(map_rows)
                map_insert_sql = """
                    INSERT INTO maps (map_id, mapset_id, map_hash, title, artist, creator, version, embed, sr, coord_x, coord_y, projection_version)
                    VALUES %s
                    ON CONFLICT (map_hash) DO UPDATE SET
                        map_id = COALESCE(EXCLUDED.map_id, maps.map_id),
                        mapset_id = EXCLUDED.mapset_id,
                        title = EXCLUDED.title,
                        artist = EXCLUDED.artist,
                        creator = EXCLUDED.creator,
                        version = EXCLUDED.version,
                        embed = COALESCE(EXCLUDED.embed, maps.embed),
                        sr = EXCLUDED.sr,
                        coord_x = CASE WHEN EXCLUDED.coord_x != 0.0 THEN EXCLUDED.coord_x ELSE maps.coord_x END,
                        coord_y = CASE WHEN EXCLUDED.coord_y != 0.0 THEN EXCLUDED.coord_y ELSE maps.coord_y END,
                        projection_version = CASE WHEN EXCLUDED.projection_version != 0 THEN EXCLUDED.projection_version ELSE maps.projection_version END;
                """
                extras.execute_values(pg_cur, map_insert_sql, sanitized_maps, page_size=2000)
                pg_conn.commit()

        # --- Push REPLAYS ---
        logger.info("Pushing table 'replays'...")
        sqlite_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='replays'")
        if sqlite_cur.fetchone():
            sqlite_cur.execute("""
                SELECT user_and_map_hash, username, replay_hash, map_hash, mods, accuracy, misses, max_combo, mastery_score
                FROM replays
            """)
            replay_rows = sqlite_cur.fetchall()
            logger.info(f"Found {len(replay_rows)} replays in SQLite.")

            if replay_rows:
                sanitized_replays = []
                for r in replay_rows:
                    user_and_map_hash, username, replay_hash, map_hash, mods, accuracy, misses, max_combo, mastery_score = r
                    
                    user_and_map_hash = sanitize_cell(user_and_map_hash)
                    username = sanitize_cell(username)
                    map_hash = sanitize_cell(map_hash)
                    replay_hash = sanitize_cell(replay_hash)
                    
                    if replay_hash is not None and not str(replay_hash).strip():
                        replay_hash = None

                    sanitized_replays.append((user_and_map_hash, username, replay_hash, map_hash, mods, accuracy, misses, max_combo, mastery_score))

                replay_insert_sql = """
                    INSERT INTO replays (user_and_map_hash, username, replay_hash, map_hash, mods, accuracy, misses, max_combo, mastery_score)
                    VALUES %s
                    ON CONFLICT (user_and_map_hash) DO UPDATE SET
                        username = EXCLUDED.username,
                        replay_hash = COALESCE(EXCLUDED.replay_hash, replays.replay_hash),
                        map_hash = EXCLUDED.map_hash,
                        mods = EXCLUDED.mods,
                        accuracy = EXCLUDED.accuracy,
                        misses = EXCLUDED.misses,
                        max_combo = EXCLUDED.max_combo,
                        mastery_score = EXCLUDED.mastery_score;
                """
                extras.execute_values(pg_cur, replay_insert_sql, sanitized_replays, page_size=5000)
                pg_conn.commit()

        # --- Push SESSIONS ---
        sqlite_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
        if sqlite_cur.fetchone():
            sqlite_cur.execute("SELECT session_id, osu_id, start_time, end_time FROM sessions")
            session_rows = sqlite_cur.fetchall()
            if session_rows:
                logger.info(f"Pushing {len(session_rows)} sessions...")
                sanitized_sessions = sanitize_rows(session_rows)
                session_insert_sql = """
                    INSERT INTO sessions (session_id, osu_id, start_time, end_time)
                    VALUES %s
                    ON CONFLICT (session_id) DO NOTHING;
                """
                extras.execute_values(pg_cur, session_insert_sql, sanitized_sessions, page_size=1000)
                pg_conn.commit()

        # --- Verification ---
        logger.info("\n========== DB PUSH SUMMARY ==========")
        for table in ['users', 'maps', 'replays', 'sessions']:
            sqlite_cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if sqlite_cur.fetchone():
                sqlite_cur.execute(f"SELECT COUNT(*) FROM {table}")
                sq_count = sqlite_cur.fetchone()[0]
                
                pg_cur.execute(f"SELECT COUNT(*) FROM {table}")
                pg_count = pg_cur.fetchone()[0]

                logger.info(f"Table '{table}': SQLite Source = {sq_count:,} | PostgreSQL Total = {pg_count:,}")

        logger.info("🚀 Data successfully pushed to PostgreSQL database!")

    except Exception as e:
        pg_conn.rollback()
        logger.error(f"Push to PostgreSQL failed due to error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        sqlite_conn.close()
        pg_cur.close()
        pg_conn.close()

if __name__ == "__main__":
    default_sqlite = DEFAULT_SQLITE_PATH

    parser = argparse.ArgumentParser(description="Push SQLite data to PostgreSQL database (Dev or Prod)")
    parser.add_argument("--env", "--target", choices=["dev", "prod"], default="prod", help="Target environment: 'dev' (localhost:5433) or 'prod' (bryan-nas.gkhomenetwork.lan:5433) (default: prod)")
    parser.add_argument("--sqlite-path", default=default_sqlite, help=f"Path to SQLite database file (default: {default_sqlite})")
    parser.add_argument("--pg-url", default=None, help="Custom PostgreSQL connection URL (overrides --env setting)")
    args = parser.parse_args()

    target_pg_url = args.pg_url if args.pg_url else get_db_url(env_target=args.env)
    push_to_db(args.sqlite_path, target_pg_url)
