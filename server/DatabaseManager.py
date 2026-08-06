from core.portfolio import compute_map_portfolio_skills
import os
import sqlite3
import time
import logging
import json

logging.basicConfig(level=logging.INFO,
                    format="[Database] %(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_file):
        self.db_file = db_file
        self.initialize_schema()

    def init_db(self):
        """Wrapper alias for initialize_schema."""
        self.initialize_schema()

    def initialize_schema(self):
        """
        Initializes the database schemas for the 4 core tables:
        1. maps (metadata)
        2. map_portfolios (11-axis portfolio scores)
        3. map_stats (raw features for ML star rating prediction)
        4. replays (replay metrics, hits JSON data, and mechanical skill ratings)
        """
        db_dir = os.path.dirname(self.db_file)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Enable Foreign Keys support
            cursor.execute("PRAGMA foreign_keys = ON;")
            
            # 1. maps (metadata)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS maps (
                    map_id INT PRIMARY KEY,
                    mapset_id INTEGER,
                    map_hash TEXT UNIQUE NOT NULL,
                    title TEXT,
                    artist TEXT,
                    creator TEXT,
                    version TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_maps_hash ON maps(map_hash);")

            # 2. map_portfolios
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS map_portfolios (
                    map_id INT PRIMARY KEY,
                    SnapAim REAL,
                    FlowAim REAL,
                    Speed REAL,
                    Streaming REAL,
                    Stamina REAL,
                    Tech REAL,
                    FingerControl REAL,
                    Precision REAL,
                    Reading REAL,
                    VisualDensity REAL,
                    AimControl REAL,
                    FOREIGN KEY(map_id) REFERENCES maps(map_id) ON DELETE CASCADE
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_map_portfolios_map_id ON map_portfolios(map_id);")

            # 3. map_stats (raw features for star rating prediction)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS map_stats (
                    map_id INT PRIMARY KEY,
                    circle_size REAL,
                    overall_difficulty REAL,
                    hp_drain REAL,
                    approach_rate REAL,
                    slider_multiplier REAL,
                    total_objects INTEGER,
                    circles_ratio REAL,
                    sliders_ratio REAL,
                    spinners_ratio REAL,
                    duration_seconds REAL,
                    density_notes_per_sec REAL,
                    velocity_mean REAL,
                    velocity_std REAL,
                    velocity_median REAL,
                    velocity_p75 REAL,
                    velocity_p90 REAL,
                    velocity_p95 REAL,
                    velocity_p99 REAL,
                    distance_mean REAL,
                    distance_std REAL,
                    distance_median REAL,
                    distance_p75 REAL,
                    distance_p90 REAL,
                    distance_p95 REAL,
                    distance_p99 REAL,
                    time_delta_mean REAL,
                    time_delta_std REAL,
                    time_delta_median REAL,
                    time_delta_p10 REAL,
                    time_delta_p5 REAL,
                    time_delta_min REAL,
                    angle_mean REAL,
                    angle_std REAL,
                    angle_sharp_ratio REAL,
                    angle_wide_ratio REAL,
                    snap_aim_score REAL,
                    flow_aim_score REAL,
                    finger_control_score REAL,
                    streaming_score REAL,
                    visual_density_score REAL,
                    slider_complexity_score REAL,
                    combo_1_2_jump_ratio REAL,
                    mean_combo_length REAL,
                    std_combo_length REAL,
                    combo_cluster_0_ratio REAL,
                    combo_cluster_1_ratio REAL,
                    combo_cluster_2_ratio REAL,
                    combo_cluster_3_ratio REAL,
                    combo_cluster_4_ratio REAL,
                    combo_cluster_5_ratio REAL,
                    star_rating REAL,
                    FOREIGN KEY(map_id) REFERENCES maps(map_id) ON DELETE CASCADE
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_map_stats_map_id ON map_stats(map_id);")

            # Check if replays table needs upgrade (missing accuracy column)
            cursor.execute("PRAGMA table_info(replays);")
            columns = [col[1] for col in cursor.fetchall()]
            if columns and "accuracy" not in columns:
                logger.info("Upgrading replays table schema (dropping and recreating)...")
                cursor.execute("DROP TABLE IF EXISTS replays;")

            # 4. replays
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS replays (
                    replay_id INT PRIMARY KEY,
                    map_id INTEGER,
                    username TEXT,
                    replay_hash TEXT,
                    accuracy REAL,
                    unstable_rate REAL,
                    avg_aim_error_px REAL,
                    mods TEXT,
                    misses INTEGER,
                    total_notes INTEGER,
                    hits INTEGER,
                    hits_json TEXT,
                    mechanical_json TEXT,
                    FOREIGN KEY(map_id) REFERENCES maps(map_id) ON DELETE CASCADE
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replays_map ON replays(map_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replays_username ON replays(username);")

            # 5. api_scores_cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_scores_cache (
                    username TEXT PRIMARY KEY,
                    top_plays_json TEXT,
                    recent_plays_json TEXT,
                    cached_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Database schemas initialized successfully.")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database schemas: {e}")

    def find_mapset_by_id(self, mapset_id):
        """
        Checks if the mapset already exists in the database.
        Returns True if it exists, False otherwise.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM maps WHERE mapset_id = ? LIMIT 1", (mapset_id,))
            result = cursor.fetchone()
            conn.close()
            return result is not None
        except sqlite3.Error as e:
            logger.error(f"Database error in find_mapset_by_id: {e}")
            return False

    def find_map_by_md5(self, md5_hash):
        """
        Checks if a map with the given MD5 hash exists in the database.
        Returns True if it exists, False otherwise.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM maps WHERE map_hash = ? LIMIT 1", (md5_hash,))
            result = cursor.fetchone()
            conn.close()
            return result is not None
        except sqlite3.Error as e:
            logger.error(f"Database error in find_map_by_md5: {e}")
            return False

    def get_map_id_by_hash(self, md5_hash):
        """
        Retrieves the map_id corresponding to the given MD5 hash.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT map_id FROM maps WHERE map_hash = ?", (md5_hash,))
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None
        except sqlite3.Error as e:
            logger.error(f"Database error in get_map_id_by_hash: {e}")
            return None

    def add_map(self, features):
        """
        Adds a new map, its computed skill portfolio, and raw stats to the database.
        """
        from core.portfolio import compute_map_portfolio_skills
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            

            map_hash = features.get('map_hash') or features.get('file_hash') or features.get('beatmap_hash')
            if not map_hash:
                raise ValueError("features dictionary must contain a valid map_hash, file_hash, or beatmap_hash")
                
            import hashlib
            map_id = features.get('map_id') or features.get('beatmap_id') or (int(hashlib.md5(map_hash.encode()).hexdigest(), 16) % 100000000)
            mapset_id = features.get('mapset_id') or 0
            title = features.get('title', 'Unknown')
            artist = features.get('artist', 'Unknown')
            creator = features.get('creator', 'Unknown')
            version = features.get('version', 'Unknown')

            # 1. Insert into maps metadata table
            cursor.execute("""
                INSERT OR IGNORE INTO maps (map_id, mapset_id, map_hash, title, artist, creator, version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (map_id, mapset_id, map_hash, title, artist, creator, version))

            # 2. Compute and insert map_portfolio
            portfolio = compute_map_portfolio_skills(features)
            cursor.execute("""
                INSERT OR REPLACE INTO map_portfolios (map_id, SnapAim, FlowAim, Speed, Streaming, Stamina, Tech, FingerControl, Precision, Reading, VisualDensity, AimControl) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (map_id, portfolio['SnapAim'], portfolio['FlowAim'], portfolio['Speed'], portfolio['Streaming'], portfolio['Stamina'], portfolio['Tech'], portfolio['FingerControl'], portfolio['Precision'], portfolio['Reading'], portfolio['VisualDensity'], portfolio['AimControl']))

            # 3. Insert raw stats into map_stats
            stats_cols = [
                'circle_size', 'overall_difficulty', 'hp_drain', 'approach_rate', 'slider_multiplier',
                'total_objects', 'circles_ratio', 'sliders_ratio', 'spinners_ratio', 'duration_seconds',
                'density_notes_per_sec', 'velocity_mean', 'velocity_std', 'velocity_median',
                'velocity_p75', 'velocity_p90', 'velocity_p95', 'velocity_p99',
                'distance_mean', 'distance_std', 'distance_median',
                'distance_p75', 'distance_p90', 'distance_p95', 'distance_p99',
                'time_delta_mean', 'time_delta_std', 'time_delta_median',
                'time_delta_p10', 'time_delta_p5', 'time_delta_min',
                'angle_mean', 'angle_std', 'angle_sharp_ratio', 'angle_wide_ratio',
                'snap_aim_score', 'flow_aim_score', 'finger_control_score', 'streaming_score',
                'visual_density_score', 'slider_complexity_score', 'combo_1_2_jump_ratio',
                'mean_combo_length', 'std_combo_length', 'combo_cluster_0_ratio',
                'combo_cluster_1_ratio', 'combo_cluster_2_ratio', 'combo_cluster_3_ratio',
                'combo_cluster_4_ratio', 'combo_cluster_5_ratio', 'star_rating'
            ]
            
            placeholders = ', '.join(['?'] * (len(stats_cols) + 1))
            cols_str = ', '.join([f'"{col}"' for col in ['map_id'] + stats_cols])
            vals = [map_id] + [features.get(col, 0.0) for col in stats_cols]
            
            cursor.execute(f"""
                INSERT OR REPLACE INTO map_stats ({cols_str})
                VALUES ({placeholders})
            """, vals)

            conn.commit()
            conn.close()
            logger.info(f"Successfully processed and stored map: {title} [{version}] (map_id: {map_id})")
            return map_id
        except sqlite3.Error as e:
            logger.error(f"Failed to add map: {e}")
            return None

    def bulk_add_maps_from_df(self, df):
        """
        Bulk adds maps from a DataFrame.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            
            # bulk insert into maps
            maps_cols = ["map_id", "mapset_id", "map_hash", "title", "artist", "creator", "version"]
            maps_placeholders = ", ".join(["?"] * len(maps_cols))
            maps_data = df[maps_cols].values.tolist()
            cursor.executemany(f'INSERT OR IGNORE INTO maps ({ ", ".join(["`" + c + "`" for c in maps_cols])}) VALUES ({maps_placeholders})', maps_data)

            # compute map portfolios
            map_portfolios = []
            for _, row in df.iterrows():
                map_id = row['map_id']
                map_hash = row['map_hash']
                mapset_id = row['mapset_id']
                title = row['title']
                artist = row['artist']
                creator = row['creator']
                version = row['version']
                
                portfolio = compute_map_portfolio_skills(row)
                map_portfolios.append((map_id,) + tuple(portfolio.values()))
            
            # bulk insert into map portfolios
            portfolio_cols = ["map_id", "SnapAim", "FlowAim", "Speed", "Streaming", "Stamina", "Tech", "FingerControl", "Precision", "Reading", "VisualDensity", "AimControl"]
            portfolio_placeholders = ', '.join(['?'] * len(portfolio_cols))
            
            cursor.executemany(f"""
                INSERT OR REPLACE INTO map_portfolios (map_id, SnapAim, FlowAim, Speed, Streaming, Stamina, Tech, FingerControl, Precision, Reading, VisualDensity, AimControl)
                VALUES ({portfolio_placeholders})
            """, map_portfolios)

            # compute map stats
            stats_cols = [
                "map_id", "circle_size", "overall_difficulty", "hp_drain", "approach_rate",
                "slider_multiplier", "total_objects", "circles_ratio", "sliders_ratio", "spinners_ratio",
                "duration_seconds", "density_notes_per_sec", "velocity_mean", "velocity_std",
                "velocity_median", "velocity_p75", "velocity_p90", "velocity_p95", "velocity_p99",
                "distance_mean", "distance_std", "distance_median", "distance_p75", "distance_p90",
                "distance_p95", "distance_p99", "time_delta_mean", "time_delta_std", "time_delta_median",
                "time_delta_p10", "time_delta_p5", "time_delta_min", "angle_mean", "angle_std",
                "angle_sharp_ratio", "angle_wide_ratio", "snap_aim_score", "flow_aim_score",
                "finger_control_score", "streaming_score", "visual_density_score", "slider_complexity_score",
                "combo_1_2_jump_ratio", "mean_combo_length", "std_combo_length", "combo_cluster_0_ratio",
                "combo_cluster_1_ratio", "combo_cluster_2_ratio", "combo_cluster_3_ratio", "combo_cluster_4_ratio",
                "combo_cluster_5_ratio", "star_rating"
            ]

            # map the stats_cols to the df
            stats = df[stats_cols].to_records(index=False)
            
            # bulk insert into map stats
            stats_cols_str = ', '.join(["`" + c + "`" for c in stats_cols])
            stats_placeholders = ', '.join(['?'] * len(stats_cols))
            
            cursor.executemany(f"""
                INSERT OR REPLACE INTO map_stats ({stats_cols_str})
                VALUES ({stats_placeholders})
            """, stats)

            conn.commit()
            conn.close()
            logger.info(f"Successfully inserted {len(df)} maps into maps table.")
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to bulk insert into maps table: {e}")
            return False

    def delete_mapset_by_id(self, mapset_id):
        """
        Deletes a mapset and all associated difficulties, portfolios, stats, and replays from the database by its ID.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            
            # Since ON DELETE CASCADE is set on foreign keys referencing maps table,
            # we just need to delete from maps table. Deleting from maps will automatically
            # clear map_portfolios, map_stats, and replays via cascade!
            cursor.execute("DELETE FROM maps WHERE mapset_id = ?", (mapset_id,))
            
            conn.commit()
            conn.close()
            logger.info(f"Successfully deleted mapset with mapset_id {mapset_id} and all related records from the database.")
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to delete mapset from database: {e}")
            return False

    def add_replay(self, map_id, username, replay_hash, accuracy, unstable_rate, avg_aim_error_px, mods, misses, total_notes, hits, hits_json, mechanical_json=None):
        """
        Adds a new replay to the database with detailed performance metrics.
        """
        replay_id = f"{int(time.time())}-{username}-{replay_hash[:8]}"
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            
            # Standardize hits data input (ensure it is a string representation of json)
            if not isinstance(hits_json, str):
                hits_json = json.dumps(hits_json)
                
            if mechanical_json is not None and not isinstance(mechanical_json, str):
                mechanical_json = json.dumps(mechanical_json)

            cursor.execute("""
                INSERT OR REPLACE INTO replays (
                    replay_id, map_id, username, replay_hash, 
                    accuracy, unstable_rate, avg_aim_error_px, mods, 
                    misses, total_notes, hits, hits_json, mechanical_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                replay_id, map_id, username, replay_hash, 
                accuracy, unstable_rate, avg_aim_error_px, mods, 
                misses, total_notes, hits, hits_json, mechanical_json
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Successfully added replay with replay_id {replay_id} to the database.")
            return replay_id
        except sqlite3.Error as e:
            logger.error(f"Failed to add replay to database: {e}")
            return None

    def remove_replay(self, replay_id):
        """
        Removes a replay from the database by its ID.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM replays WHERE replay_id = ?", (replay_id,))
            conn.commit()
            conn.close()
            logger.info(f"Successfully removed replay with replay_id {replay_id} from the database.")
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to remove replay from database: {e}")
            return False
            
    def get_replay_hits(self, replay_file):
        """
        Retrieves the hits json for a replay by matching the replay filename prefix in replay_id or matching replay_hash.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            # Match replay filename prefix or exact replay_hash
            cursor.execute("SELECT hits_json FROM replays WHERE replay_id LIKE ? OR replay_hash = ? LIMIT 1", (f"%{replay_file}%", replay_file))
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None
        except sqlite3.Error as e:
            logger.error(f"Failed to get replay hits: {e}")
            return None

    def get_mechanical_skills(self, replay_file):
        """
        Retrieves mechanical skills json for a replay by matching filename or hash.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT mechanical_json FROM replays WHERE replay_id LIKE ? OR replay_hash = ? LIMIT 1", (f"%{replay_file}%", replay_file))
            row = cursor.fetchone()
            conn.close()
            return json.loads(row[0]) if row and row[0] else None
        except sqlite3.Error as e:
            logger.error(f"Failed to get mechanical skills: {e}")
            return None

    def get_map_portfolio(self, beatmap_hash):
        """
        Retrieves the 11-axis skills portfolio for a beatmap hash.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    p.SnapAim, p.FlowAim, p.Speed, p.Streaming, p.Stamina, p.Tech, 
                    p.FingerControl, p.Precision, p.Reading, p.VisualDensity, p.AimControl 
                FROM map_portfolios p
                JOIN maps m ON p.map_id = m.map_id
                WHERE m.map_hash = ?
            """, (beatmap_hash,))
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
        except sqlite3.Error as e:
            logger.error(f"Failed to get map portfolio: {e}")
            return None

    def get_unique_players(self):
        """
        Retrieves a list of all unique players who have submitted replays.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT username FROM replays ORDER BY username COLLATE NOCASE")
            players = [row[0] for row in cursor.fetchall()]
            conn.close()
            return players
        except sqlite3.Error as e:
            logger.error(f"Failed to get unique players: {e}")
            return []

    def get_player_replays(self, username):
        """
        Retrieves all plays for a specific player, including map metadata.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            # Row factory lets us return plays as list of dicts
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    r.replay_hash AS replay_file,
                    r.username AS player,
                    m.title AS map_title,
                    m.artist AS map_artist,
                    m.version AS difficulty_name,
                    m.creator AS map_creator,
                    m.map_hash AS beatmap_hash,
                    r.accuracy AS accuracy_percent,
                    r.unstable_rate,
                    r.avg_aim_error_px,
                    r.mods,
                    r.misses,
                    r.total_notes,
                    r.hits,
                    r.mechanical_json
                FROM replays r
                JOIN maps m ON r.map_id = m.map_id
                WHERE LOWER(r.username) = LOWER(?)
            """, (username,))
            plays = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return plays
        except sqlite3.Error as e:
            logger.error(f"Failed to get player replays for {username}: {e}")
            return []

    def get_map_leaderboard(self, beatmap_hash):
        """
        Queries scores leaderboard for a specific beatmap hash.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    r.replay_hash AS replay_file,
                    r.username AS player,
                    r.accuracy AS accuracy,
                    r.unstable_rate,
                    r.avg_aim_error_px AS avg_aim_error,
                    r.mods,
                    r.misses
                FROM replays r
                JOIN maps m ON r.map_id = m.map_id
                WHERE m.map_hash = ?
                ORDER BY r.accuracy DESC, r.unstable_rate ASC
            """, (beatmap_hash,))
            leaderboard = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return leaderboard
        except sqlite3.Error as e:
            logger.error(f"Failed to get map leaderboard: {e}")
            return []

    def get_cached_api_scores(self, username):
        """
        Retrieves cached API scores for a user if cached within the last 15 minutes.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            # Calculate 15 minutes ago
            cursor.execute("""
                SELECT top_plays_json, recent_plays_json FROM api_scores_cache 
                WHERE LOWER(username) = LOWER(?) 
                  AND cached_at >= datetime('now', '-15 minutes')
            """, (username,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return json.loads(row[0]), json.loads(row[1])
            return None
        except sqlite3.Error as e:
            logger.error(f"Failed to get cached API scores: {e}")
            return None

    def save_api_scores(self, username, top_plays, recent_plays):
        """
        Saves or replaces the API scores cache for a user.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO api_scores_cache (username, top_plays_json, recent_plays_json, cached_at)
                VALUES (?, ?, ?, datetime('now'))
            """, (
                username, 
                json.dumps(top_plays), 
                json.dumps(recent_plays)
            ))
            conn.commit()
            conn.close()
            logger.info(f"Successfully cached API scores for player: {username}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to save API scores cache: {e}")
            return False
