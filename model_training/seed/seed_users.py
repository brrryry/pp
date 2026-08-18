import os
import sys
import sqlite3
import argparse
import time
import logging
from tqdm import tqdm
from osu import Client
from requests.exceptions import HTTPError

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="[User Seeder] %(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("seed_users")

def main():
    parser = argparse.ArgumentParser(description="Query the osu! API for unique usernames in replays and save mappings.")
    parser.add_argument(
        "--db_file",
        type=str,
        default=config.DB_FILE,
        help="SQLite database file path to read replays and seed users."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.55,
        help="Time delay (seconds) between API requests to respect the rate limit."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Maximum number of users to lookup in a single run (default: -1 for all)."
    )

    args = parser.parse_args()

    if not os.path.exists(args.db_file):
        logger.error(f"Database file not found at {args.db_file}")
        return

    # Connect to database and retrieve unique usernames from replays table
    conn = sqlite3.connect(args.db_file)
    cursor = conn.cursor()

    # Ensure replays table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='replays'")
    if not cursor.fetchone():
        logger.error("No 'replays' table found in the database. Run import_replays.py first.")
        conn.close()
        return

    # Create users table if not exists
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "osu_id INTEGER PRIMARY KEY UNIQUE, "
        "username TEXT UNIQUE NOT NULL"
        ")"
    )
    conn.commit()

    # Get distinct usernames from replays table (filtering out empty/null/blank ones)
    cursor.execute(
        "SELECT DISTINCT username FROM replays "
        "WHERE username IS NOT NULL AND username != ''"
    )
    all_replay_usernames = [row[0] for row in cursor.fetchall()]
    logger.info(f"Found {len(all_replay_usernames)} unique usernames in 'replays' table.")

    # Get already seeded usernames from users table
    cursor.execute("SELECT username FROM users")
    seeded_usernames = {row[0].lower() for row in cursor.fetchall()}
    logger.info(f"Already seeded {len(seeded_usernames)} users.")

    # Filter out already seeded usernames (case-insensitive check)
    usernames_to_lookup = [u for u in all_replay_usernames if u.lower() not in seeded_usernames]
    logger.info(f"Need to lookup {len(usernames_to_lookup)} new users.")

    if not usernames_to_lookup:
        logger.info("All users are already seeded!")
        conn.close()
        return

    # Enforce limit if specified
    if args.limit > 0 and len(usernames_to_lookup) > args.limit:
        logger.info(f"Limiting API queries to the first {args.limit} usernames.")
        usernames_to_lookup = usernames_to_lookup[:args.limit]

    # Initialize osu! API Client
    try:
        client = Client.from_credentials(
            config.osu_api_client_id,
            config.osu_api_client_secret,
            config.osu_api_redirect_uri
        )
    except Exception as e:
        logger.error(f"Failed to initialize osu! client: {e}")
        conn.close()
        return

    success_count = 0
    failed_count = 0
    deleted_users_count = 0
    deleted_replays_total = 0

    # Rate-limited query loop
    for i, username in enumerate(tqdm(usernames_to_lookup, desc="Querying osu! API")):
        # Delay to stay below rate limit (max 2 queries per second)
        if i > 0:
            time.sleep(args.delay)

        try:
            user = client.get_user(username, key='username')
            if user and user.id:
                cursor.execute(
                    "INSERT OR REPLACE INTO users (osu_id, username) VALUES (?, ?)",
                    (user.id, user.username)
                )
                conn.commit()
                success_count += 1
            else:
                logger.warning(f"User '{username}' lookup returned no matches. Deleting associated replays.")
                cursor.execute("DELETE FROM replays WHERE username = ?", (username,))
                deleted_rows = cursor.rowcount
                conn.commit()
                deleted_replays_total += deleted_rows
                deleted_users_count += 1
                failed_count += 1
        except HTTPError as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 404:
                logger.warning(f"User '{username}' not found on osu! (404). Deleting associated replays.")
                cursor.execute("DELETE FROM replays WHERE username = ?", (username,))
                deleted_rows = cursor.rowcount
                conn.commit()
                deleted_replays_total += deleted_rows
                deleted_users_count += 1
            else:
                logger.warning(f"Network error looking up '{username}': {e}")
            failed_count += 1
        except Exception as e:
            logger.warning(f"Failed to lookup '{username}': {e}")
            failed_count += 1

    conn.close()
    logger.info(
        f"User seeding complete! Successfully seeded {success_count} users. "
        f"Cleaned up {deleted_users_count} non-existent users, deleting {deleted_replays_total} associated replays."
    )

if __name__ == "__main__":
    main()