import os
import sys
import sqlite3
import logging
import argparse

try:
    import psycopg2
    from psycopg2 import extras
except ImportError:
    print("Error: psycopg2 is required. Run 'pip install psycopg2-binary'")
    sys.exit(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from server.core.DatabaseManager import DatabaseManager
import server.config as config

logging.basicConfig(level=logging.INFO, format="[Migration] %(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MigrateSQLiteToPostgres")

def migrate_database(sqlite_path: str, pg_url: str):
    logger.info(f"Starting migration from SQLite '{sqlite_path}' to PostgreSQL '{pg_url}'...")

    if not os.path.exists(sqlite_path):
        logger.error(f"SQLite file not found at: {sqlite_path}")
        sys.exit(1)

    # 1. Drop existing tables if present to ensure clean schema application
    temp_conn = psycopg2.connect(pg_url)
    temp_cur = temp_conn.cursor()
    temp_cur.execute("DROP TABLE IF EXISTS replays, maps, users, sessions CASCADE;")
    temp_conn.commit()
    temp_cur.close()
    temp_conn.close()

    # 2. Initialize PostgreSQL schema using DatabaseManager
    db_mgr = DatabaseManager(pg_url)
    db_mgr.initialize_schema()

    # 2. Connect to SQLite and PostgreSQL
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_cur = sqlite_conn.cursor()

    pg_conn = psycopg2.connect(pg_url)
    pg_cur = pg_conn.cursor()

    try:
        # Clear any partial test tables in PostgreSQL to ensure fresh clean migration
        pg_cur.execute("TRUNCATE TABLE replays, maps, users, sessions CASCADE;")
        pg_conn.commit()

        # --- Migrate USERS ---
        logger.info("Migrating table 'users'...")
        sqlite_cur.execute("SELECT username, osu_id FROM users")
        user_rows = sqlite_cur.fetchall()
        logger.info(f"Found {len(user_rows)} users in SQLite.")

        if user_rows:
            user_insert_sql = """
                INSERT INTO users (username, osu_id)
                VALUES %s
                ON CONFLICT (username) DO UPDATE SET osu_id = EXCLUDED.osu_id;
            """
            extras.execute_values(pg_cur, user_insert_sql, user_rows, page_size=1000)
            pg_conn.commit()

        # --- Migrate MAPS ---
        logger.info("Migrating table 'maps'...")
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
            map_insert_sql = """
                INSERT INTO maps (map_id, mapset_id, map_hash, title, artist, creator, version, embed, sr, coord_x, coord_y, projection_version)
                VALUES %s
                ON CONFLICT (map_hash) DO UPDATE SET
                    map_id = EXCLUDED.map_id,
                    mapset_id = EXCLUDED.mapset_id,
                    title = EXCLUDED.title,
                    artist = EXCLUDED.artist,
                    creator = EXCLUDED.creator,
                    version = EXCLUDED.version,
                    embed = EXCLUDED.embed,
                    sr = EXCLUDED.sr,
                    coord_x = EXCLUDED.coord_x,
                    coord_y = EXCLUDED.coord_y,
                    projection_version = EXCLUDED.projection_version;
            """
            extras.execute_values(pg_cur, map_insert_sql, map_rows, page_size=2000)
            pg_conn.commit()

        # --- Migrate REPLAYS ---
        logger.info("Migrating table 'replays'...")
        sqlite_cur.execute("""
            SELECT user_and_map_hash, username, replay_hash, map_hash, mods, accuracy, misses, max_combo, mastery_score
            FROM replays
        """)
        replay_rows = sqlite_cur.fetchall()
        logger.info(f"Found {len(replay_rows)} replays in SQLite.")

        if replay_rows:
            replay_insert_sql = """
                INSERT INTO replays (user_and_map_hash, username, replay_hash, map_hash, mods, accuracy, misses, max_combo, mastery_score)
                VALUES %s
                ON CONFLICT (user_and_map_hash) DO UPDATE SET
                    username = EXCLUDED.username,
                    replay_hash = EXCLUDED.replay_hash,
                    map_hash = EXCLUDED.map_hash,
                    mods = EXCLUDED.mods,
                    accuracy = EXCLUDED.accuracy,
                    misses = EXCLUDED.misses,
                    max_combo = EXCLUDED.max_combo,
                    mastery_score = EXCLUDED.mastery_score;
            """
            extras.execute_values(pg_cur, replay_insert_sql, replay_rows, page_size=5000)
            pg_conn.commit()

        # --- Migrate SESSIONS (if any) ---
        sqlite_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
        if sqlite_cur.fetchone():
            sqlite_cur.execute("SELECT session_id, osu_id, start_time, end_time FROM sessions")
            session_rows = sqlite_cur.fetchall()
            if session_rows:
                logger.info(f"Migrating {len(session_rows)} sessions...")
                session_insert_sql = """
                    INSERT INTO sessions (session_id, osu_id, start_time, end_time)
                    VALUES %s
                    ON CONFLICT (session_id) DO NOTHING;
                """
                extras.execute_values(pg_cur, session_insert_sql, session_rows, page_size=1000)
                pg_conn.commit()

        # --- Verification ---
        logger.info("\n========== MIGRATION VERIFICATION ==========")
        all_matched = True
        for table in ['users', 'maps', 'replays']:
            sqlite_cur.execute(f"SELECT COUNT(*) FROM {table}")
            sq_count = sqlite_cur.fetchone()[0]
            
            pg_cur.execute(f"SELECT COUNT(*) FROM {table}")
            pg_count = pg_cur.fetchone()[0]

            status = "MATCH" if sq_count == pg_count else "MISMATCH"
            if sq_count != pg_count:
                all_matched = False
            logger.info(f"Table '{table}': SQLite = {sq_count:,} | PostgreSQL = {pg_count:,} | [{status}]")

        if all_matched:
            logger.info("🎉 Database Migration completed with 100% data integrity!")
        else:
            logger.warning("⚠️ Migration finished with row count discrepancies.")

    except Exception as e:
        pg_conn.rollback()
        logger.error(f"Migration failed due to error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        sqlite_conn.close()
        pg_cur.close()
        pg_conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate SQLite pp.db to PostgreSQL")
    parser.add_argument("--sqlite-path", default=config.DB_FILE, help="Path to SQLite database file")
    parser.add_argument("--pg-url", default=config.DATABASE_URL, help="PostgreSQL connection string")
    args = parser.parse_args()

    migrate_database(args.sqlite_path, args.pg_url)
