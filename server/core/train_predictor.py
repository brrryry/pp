import os
import sys
import sqlite3
import json
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add both server directory and parent directory to path to find config.py and core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from server.DatabaseManager import DatabaseManager

DB_PATH = "server/data/osu_profiler.db"
if not os.path.exists("server/data"):
    DB_PATH = "data/osu_profiler.db"

MODEL_PATHS = [
    "data/model_results/accuracy_predictor.joblib",
    "server/data/model_results/accuracy_predictor.joblib"
]

KEY_MAP = {
    "SnapAim": "Snap Aim",
    "FlowAim": "Flow Aim",
    "Speed": "Speed",
    "Streaming": "Streaming",
    "Stamina": "Stamina",
    "Tech": "Tech",
    "FingerControl": "Finger Control",
    "Precision": "Precision",
    "Reading": "Reading",
    "VisualDensity": "Visual Density",
    "AimControl": "Aim Control"
}

def weighted_stats(values, weights):
    """
    Computes weighted mean, weighted standard deviation, and weighted percentiles (10th, 50th, 90th).
    """
    values = np.array(values)
    weights = np.array(weights)
    if len(values) == 0:
        return {
            'mean': 95.0,
            'std': 2.0,
            'p10': 92.0,
            'p50': 95.0,
            'p90': 98.0
        }
    weights = weights / np.sum(weights)
    
    # Weighted mean
    mean = np.sum(values * weights)
    
    # Weighted variance/std
    variance = np.sum(weights * (values - mean) ** 2)
    std = np.sqrt(variance)
    
    # Weighted percentile helper
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumulative_weights = np.cumsum(sorted_weights)
    
    def get_percentile(p):
        idx = np.searchsorted(cumulative_weights, p / 100.0)
        idx = min(idx, len(sorted_values) - 1)
        return float(sorted_values[idx])
        
    return {
        'mean': float(mean),
        'std': float(std) if std > 0.0 else 1.0,
        'p10': get_percentile(10),
        'p50': get_percentile(50),
        'p90': get_percentile(90)
    }

def get_player_profile_features(username, db):
    """
    Computes a player's aggregated skill features from their cached top plays and local replays,
    incorporating recency weights, peak capabilities, and context degradation slopes.
    """
    profile = {
        'avg_accuracy': 95.0,
        'peak_accuracy': 98.0,
        'volatility_accuracy': 2.0,
        
        'avg_ur': 100.0,
        'peak_ur': 80.0,
        'volatility_ur': 15.0,
        
        'avg_aim_error': 15.0,
        'peak_aim_error': 10.0,
        'volatility_aim_error': 3.0,
        
        'aim_cs_slope': 1.5,
        'tap_density_slope': 5.0
    }
    
    # Initialize skills with default values (50.0)
    for skill in KEY_MAP.keys():
        profile[f'user_potential_{skill}'] = 50.0
        profile[f'user_mechanical_{skill}'] = 50.0

    # 1. Fetch local plays and compute mechanical features
    local_plays = db.get_player_replays(username)
    mechanical_skills_list = []
    
    if local_plays:
        accuracies = []
        urs = []
        aim_errors = []
        weights = []
        
        # Connect to DB to check map stats for slopes
        conn = sqlite3.connect(db.db_file)
        cursor = conn.cursor()
        
        cs_pairs = []
        density_pairs = []
        
        for i, play in enumerate(local_plays):
            # Recency weight: 30-play half-life style decay
            w = 0.95 ** (len(local_plays) - 1 - i)
            
            if 'mechanical_json' in play and play['mechanical_json']:
                try:
                    mech = json.loads(play['mechanical_json'])
                    mechanical_skills_list.append((mech, w))
                except Exception:
                    pass
                    
            if play.get('accuracy') is not None:
                accuracies.append(play['accuracy'])
                weights.append(w)
            if play.get('unstable_rate') is not None:
                urs.append(play['unstable_rate'])
            if play.get('avg_aim_error_px') is not None:
                aim_errors.append(play['avg_aim_error_px'])
                
            # Collect data for slopes
            map_id = play.get('map_id')
            if map_id is not None:
                cursor.execute("SELECT circle_size, density_notes_per_sec FROM map_stats WHERE map_id = ?", (map_id,))
                res = cursor.fetchone()
                if res:
                    cs, density = res
                    if play.get('avg_aim_error_px') is not None:
                        cs_pairs.append((cs, play['avg_aim_error_px']))
                    if play.get('unstable_rate') is not None:
                        density_pairs.append((density, play['unstable_rate']))
                        
        conn.close()
        
        # Fit CS-Aim slope
        if len(cs_pairs) >= 3:
            try:
                xs = np.array([p[0] for p in cs_pairs])
                ys = np.array([p[1] for p in cs_pairs])
                A = np.vstack([xs, np.ones(len(xs))]).T
                m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
                profile['aim_cs_slope'] = float(m)
            except Exception:
                pass
                
        # Fit Tapping Density-UR slope
        if len(density_pairs) >= 3:
            try:
                xs = np.array([p[0] for p in density_pairs])
                ys = np.array([p[1] for p in density_pairs])
                A = np.vstack([xs, np.ones(len(xs))]).T
                m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
                profile['tap_density_slope'] = float(m)
            except Exception:
                pass

        # Compute weighted stats
        if accuracies:
            w_acc = weights[:len(accuracies)]
            stats_acc = weighted_stats(accuracies, w_acc)
            profile['avg_accuracy'] = stats_acc['mean']
            profile['peak_accuracy'] = stats_acc['p90'] # 90th percentile is peak accuracy
            profile['volatility_accuracy'] = stats_acc['std']
            
        if urs:
            w_ur = weights[:len(urs)]
            stats_ur = weighted_stats(urs, w_ur)
            profile['avg_ur'] = stats_ur['mean']
            profile['peak_ur'] = stats_ur['p10'] # 10th percentile is lowest/best UR
            profile['volatility_ur'] = stats_ur['std']
            
        if aim_errors:
            w_aim = weights[:len(aim_errors)]
            stats_aim = weighted_stats(aim_errors, w_aim)
            profile['avg_aim_error'] = stats_aim['mean']
            profile['peak_aim_error'] = stats_aim['p10'] # 10th percentile is lowest/best aim error
            profile['volatility_aim_error'] = stats_aim['std']

        if mechanical_skills_list:
            for skill in KEY_MAP.keys():
                vals = [m[0][skill] for m in mechanical_skills_list if skill in m[0]]
                w_vals = [m[1] for m in mechanical_skills_list if skill in m[0]]
                if vals:
                    profile[f'user_mechanical_{skill}'] = weighted_stats(vals, w_vals)['mean']

    # 2. Fetch cached API top plays for potential skills
    cached_scores = db.get_cached_api_scores(username)
    top_plays = []
    if cached_scores:
        top_plays = cached_scores[0] # cached_scores returns (top_plays, recent_plays)

    if top_plays:
        # Top plays are weighted by ranking weight
        accuracies = [p['accuracy_percent'] for p in top_plays if 'accuracy_percent' in p]
        weights = [p.get('weight', 100.0) / 100.0 for p in top_plays if 'accuracy_percent' in p]
        if accuracies:
            stats_acc = weighted_stats(accuracies, weights)
            profile['avg_accuracy'] = stats_acc['mean']
            profile['peak_accuracy'] = stats_acc['p90']
            profile['volatility_accuracy'] = stats_acc['std']

        weighted_skills_sum = {k: 0.0 for k in KEY_MAP.keys()}
        weights_sum = 0.0
        for play in top_plays:
            h = play.get('beatmap_hash')
            if not h:
                continue
            skills = db.get_map_portfolio(h)
            if skills:
                acc = play.get('accuracy_percent', 95.0) / 100.0
                misses = play.get('misses', 0)
                perf_multiplier = (acc ** 2) * (0.95 ** misses)
                weight = (play.get('weight', 100.0) / 100.0) * perf_multiplier
                
                for k in KEY_MAP.keys():
                    if k in skills:
                        weighted_skills_sum[k] += skills[k] * weight
                weights_sum += weight

        if weights_sum > 0:
            for k in KEY_MAP.keys():
                profile[f'user_potential_{k}'] = weighted_skills_sum[k] / weights_sum

    return profile

def main():
    db = DatabaseManager(DB_PATH)
    
    # Find all unique players in the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Union list of players
    cursor.execute("""
        SELECT DISTINCT username FROM replays
        UNION
        SELECT DISTINCT username FROM api_scores_cache
    """)
    players = [r[0] for r in cursor.fetchall()]
    
    if not players:
        logger.warning("No players found in database. Cannot train accuracy predictor. Generating synthetic training database for startup verification...")
        # Create a tiny mock sample so the training succeeds
        players = ["MockPlayer"]
        
    logger.info(f"Loaded {len(players)} players for training.")
    
    # Load player profiles
    player_profiles = {}
    for p in players:
        player_profiles[p] = get_player_profile_features(p, db)
        
    # Compile dataset rows
    rows = []
    
    # 1. Fetch scores from api_scores_cache
    cursor.execute("SELECT username, top_plays_json, recent_plays_json FROM api_scores_cache")
    for username, top_json, recent_json in cursor.fetchall():
        scores = []
        if top_json:
            try: scores.extend(json.loads(top_json))
            except: pass
        if recent_json:
            try: scores.extend(json.loads(recent_json))
            except: pass
            
        for score in scores:
            h = score.get('beatmap_hash')
            if not h:
                continue
                
            # Query beatmap details and stats from DB
            cursor.execute("SELECT map_id FROM maps WHERE map_hash = ?", (h,))
            map_res = cursor.fetchone()
            if not map_res:
                continue
            map_id = map_res[0]
            
            # Fetch map stats and portfolio
            cursor.execute("SELECT * FROM map_stats WHERE map_id = ?", (map_id,))
            stat_res = cursor.fetchone()
            if not stat_res:
                continue
                
            cursor.execute("SELECT * FROM map_portfolios WHERE map_id = ?", (map_id,))
            port_res = cursor.fetchone()
            if not port_res:
                continue
                
            # Map stats details (OD, CS, AR, duration, total objects, density, star rating)
            # Fetch column names
            cursor.execute("PRAGMA table_info(map_stats)")
            stat_cols = [c[1] for c in cursor.fetchall()]
            stat_dict = dict(zip(stat_cols, stat_res))
            
            cursor.execute("PRAGMA table_info(map_portfolios)")
            port_cols = [c[1] for c in cursor.fetchall()]
            port_dict = dict(zip(port_cols, port_res))
            
            # Combine all features
            row = dict(player_profiles[username])
            
            # Map difficulty features
            row['map_cs'] = stat_dict.get('circle_size', 4.0)
            row['map_od'] = stat_dict.get('overall_difficulty', 8.0)
            row['map_ar'] = stat_dict.get('approach_rate', 9.0)
            row['map_hp'] = stat_dict.get('hp_drain', 5.0)
            row['map_star_rating'] = stat_dict.get('star_rating', 5.0)
            row['map_duration'] = stat_dict.get('duration_seconds', 120.0)
            row['map_density'] = stat_dict.get('density_notes_per_sec', 3.0)
            row['map_total_objects'] = stat_dict.get('total_objects', 500)
            row['map_sliders_ratio'] = stat_dict.get('sliders_ratio', 0.2)
            
            # Map portfolio demands
            for k in KEY_MAP.keys():
                row[f'map_demand_{k}'] = port_dict.get(k, 50.0)
                
            # Parse mods
            mods = score.get('mods', 'NoMod') or 'NoMod'
            mods_str = str(mods).upper()
            row['mod_DT'] = 1 if ('DT' in mods_str or 'NC' in mods_str) else 0
            row['mod_HR'] = 1 if 'HR' in mods_str else 0
            row['mod_EZ'] = 1 if 'EZ' in mods_str else 0
            row['mod_HD'] = 1 if 'HD' in mods_str else 0
            row['mod_FL'] = 1 if 'FL' in mods_str else 0
            row['mod_HT'] = 1 if 'HT' in mods_str else 0
            
            # Target
            row['accuracy'] = score.get('accuracy_percent', 95.0)
            
            rows.append(row)
            
    # 2. Fetch local plays from replays table
    cursor.execute("""
        SELECT username, map_id, accuracy, mods FROM replays
    """)
    for username, map_id, accuracy, mods in cursor.fetchall():
        if not accuracy or username not in player_profiles:
            continue
            
        cursor.execute("SELECT * FROM map_stats WHERE map_id = ?", (map_id,))
        stat_res = cursor.fetchone()
        if not stat_res:
            continue
            
        cursor.execute("SELECT * FROM map_portfolios WHERE map_id = ?", (map_id,))
        port_res = cursor.fetchone()
        if not port_res:
            continue
            
        cursor.execute("PRAGMA table_info(map_stats)")
        stat_cols = [c[1] for c in cursor.fetchall()]
        stat_dict = dict(zip(stat_cols, stat_res))
        
        cursor.execute("PRAGMA table_info(map_portfolios)")
        port_cols = [c[1] for c in cursor.fetchall()]
        port_dict = dict(zip(port_cols, port_res))
        
        row = dict(player_profiles[username])
        
        # Map difficulty features
        row['map_cs'] = stat_dict.get('circle_size', 4.0)
        row['map_od'] = stat_dict.get('overall_difficulty', 8.0)
        row['map_ar'] = stat_dict.get('approach_rate', 9.0)
        row['map_hp'] = stat_dict.get('hp_drain', 5.0)
        row['map_star_rating'] = stat_dict.get('star_rating', 5.0)
        row['map_duration'] = stat_dict.get('duration_seconds', 120.0)
        row['map_density'] = stat_dict.get('density_notes_per_sec', 3.0)
        row['map_total_objects'] = stat_dict.get('total_objects', 500)
        row['map_sliders_ratio'] = stat_dict.get('sliders_ratio', 0.2)
        
        # Map portfolio demands
        for k in KEY_MAP.keys():
            row[f'map_demand_{k}'] = port_dict.get(k, 50.0)
            
        # Parse mods
        mods_str = str(mods or 'NoMod').upper()
        row['mod_DT'] = 1 if ('DT' in mods_str or 'NC' in mods_str) else 0
        row['mod_HR'] = 1 if 'HR' in mods_str else 0
        row['mod_EZ'] = 1 if 'EZ' in mods_str else 0
        row['mod_HD'] = 1 if 'HD' in mods_str else 0
        row['mod_FL'] = 1 if 'FL' in mods_str else 0
        row['mod_HT'] = 1 if 'HT' in mods_str else 0
        
        # Target
        row['accuracy'] = accuracy
        
        rows.append(row)
        
    conn.close()
    
    # Fallback to synthetic data if we don't have enough plays to train
    if len(rows) < 10:
        logger.warning(f"Only {len(rows)} play records found. Generating synthetic dataset to establish predictor model architecture...")
        rows = []
        # Generate 100 random plays
        for i in range(100):
            row = {
                'avg_accuracy': np.random.uniform(92.0, 99.5),
                'avg_ur': np.random.uniform(70.0, 150.0),
                'avg_aim_error': np.random.uniform(8.0, 22.0),
            }
            # Add user profiles
            for k in KEY_MAP.keys():
                row[f'user_potential_{k}'] = np.random.uniform(30.0, 95.0)
                row[f'user_mechanical_{k}'] = np.random.uniform(30.0, 95.0)
                
            # Add map attributes
            row['map_cs'] = np.random.uniform(3.0, 5.5)
            row['map_od'] = np.random.uniform(6.0, 10.0)
            row['map_ar'] = np.random.uniform(8.0, 10.3)
            row['map_hp'] = np.random.uniform(4.0, 7.0)
            row['map_star_rating'] = np.random.uniform(4.0, 7.5)
            row['map_duration'] = np.random.uniform(60.0, 240.0)
            row['map_density'] = np.random.uniform(2.0, 6.5)
            row['map_total_objects'] = np.random.randint(200, 1200)
            row['map_sliders_ratio'] = np.random.uniform(0.05, 0.45)
            
            for k in KEY_MAP.keys():
                row[f'map_demand_{k}'] = np.random.uniform(25.0, 90.0)
                
            # Add mods
            row['mod_DT'] = np.random.choice([0, 1])
            row['mod_HR'] = np.random.choice([0, 1])
            row['mod_EZ'] = np.random.choice([0, 1])
            row['mod_HD'] = np.random.choice([0, 1])
            row['mod_FL'] = np.random.choice([0, 1])
            row['mod_HT'] = np.random.choice([0, 1])
            
            # Predict accuracy based on some rough rules so model has signal
            user_skill = (row['user_potential_Speed'] + row['user_potential_Precision']) / 2.0
            map_difficulty = (row['map_star_rating'] * 12.0)
            accuracy = 100.0 - max(0.5, (map_difficulty - user_skill) * 0.4 + np.random.normal(0, 1.5))
            row['accuracy'] = max(70.0, min(100.0, accuracy))
            rows.append(row)

    df_data = pd.DataFrame(rows)
    logger.info(f"Assembled training dataset with {len(df_data)} records.")
    
    # Prepare features
    X = df_data.drop(columns=['accuracy'])
    y = df_data['accuracy']
    
    # Save feature names list
    feature_cols = list(X.columns)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Train Random Forest Regressor
    logger.info("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    logger.info(f"Model trained. Validation Metrics: RMSE={rmse:.4f}, R2={r2:.4f}")
    
    # Save model and metadata
    model_package = {
        'model': model,
        'feature_cols': feature_cols,
        'metrics': {
            'rmse': rmse,
            'r2': r2
        }
    }
    
    for path in MODEL_PATHS:
        try:
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            logger.info(f"Saving predictor package to {path}...")
            joblib.dump(model_package, path)
        except Exception as ex:
            logger.warning(f"Could not save to path {path}: {ex}")
            
    logger.info("Accuracy predictor model training successfully completed.")

if __name__ == "__main__":
    main()
