import os
import sys
import sqlite3
import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config

DB_PATH = config.DB_FILE
MODEL_FOLDER = config.MODELS_DIR
EMBEDDING_SIZE = config.EMBEDDING_SIZE

def train_user_portfolios_als():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("📦 Step 1: Mapping clean user and map index dimensions...")
    
    # 1. Row Index Map: Contiguous users who have real replays
    cursor.execute("""
        SELECT DISTINCT u.osu_id FROM users u
        INNER JOIN replays r ON u.username = r.username
        WHERE r.mastery_score > 0 ORDER BY u.osu_id ASC
    """)
    all_users = [row[0] for row in cursor.fetchall()]
    osu_id_to_matrix_idx = {osu_id: idx for idx, osu_id in enumerate(all_users)}
    
    # 2. Column Index Map: Contiguous maps from the MAPS table that have been played
    cursor.execute("""
        SELECT DISTINCT m.map_hash FROM maps m
        INNER JOIN replays r ON m.map_hash = r.map_hash
        WHERE r.mastery_score > 0
    """)
    all_maps = [str(row[0]).strip().lower() for row in cursor.fetchall()]
    hash_to_matrix_idx = {b_hash: idx for idx, b_hash in enumerate(all_maps)}
    
    num_users = len(all_users)
    num_maps = len(all_maps)
    print(f"  Found {num_users} active unique users and {num_maps} unique maps.")

    print("\n📊 Step 2: Extracting real mastery scores and compiling Sparse Matrix...")
    
    # Extract the score logs. We use clean naked indexed inner joins.
    cursor.execute("""
        SELECT u.osu_id, m.map_hash, r.mastery_score 
        FROM replays r 
        INNER JOIN users u ON u.username = r.username
        INNER JOIN maps m ON r.map_hash = m.map_hash 
        WHERE r.mastery_score > 0
    """)
    
    matrix_rows = []
    matrix_cols = []
    matrix_data = []
    
    for osu_id, b_hash, mastery in cursor.fetchall():
        clean_hash = str(b_hash).strip().lower()
        if osu_id in osu_id_to_matrix_idx and clean_hash in hash_to_matrix_idx:
            matrix_rows.append(osu_id_to_matrix_idx[osu_id])
            matrix_cols.append(hash_to_matrix_idx[clean_hash])
            matrix_data.append(max(0.01, min(1.0, mastery))) # Clamp safely

    # Create the Compressed Sparse Row (CSR) matrix [Users x Maps]
    user_map_matrix = sparse.coo_matrix(
        (matrix_data, (matrix_rows, matrix_cols)), 
        shape=(num_users, num_maps)
    ).tocsr()

    print("\n🧠 Step 3: Loading pre-computed map embeddings from maps table column...")
    
    # Pre-populate your item factors with your real database embeds
    lstm_map_vectors = np.zeros((num_maps, EMBEDDING_SIZE), dtype=np.float32)
    cursor.execute("SELECT map_hash, embed FROM maps WHERE embed IS NOT NULL")
    
    for b_hash, embed_data in cursor.fetchall():
        clean_hash = str(b_hash).strip().lower()
        if clean_hash in hash_to_matrix_idx:
            target_col_idx = hash_to_matrix_idx[clean_hash]
            try:
                clean_text = str(embed_data).replace("[", "").replace("]", "").strip()
                vector_array = np.fromstring(clean_text, sep=",", dtype=np.float32)
                if vector_array.size == EMBEDDING_SIZE:
                    lstm_map_vectors[target_col_idx] = vector_array
            except Exception:
                pass
                
    conn.close()

    print("\n⚡ Step 4: Running Implicit Alternating Least Squares (ALS)...")
    
    # Initialize the high-performance implicit library model
    als_model = AlternatingLeastSquares(
        factors=EMBEDDING_SIZE,
        regularization=0.15,
        iterations=30,
        random_state=42
    )
    
    # THE ORIGINAL GUIDED TRICK: Overwrite and fix the maps coordinates [1]
    als_model.item_factors = lstm_map_vectors
    
    # Train the model. It automatically fits the user vectors to the map DNA [1]
    als_model.fit(user_map_matrix, show_progress=True)
    
    # Save to disk
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    with open(os.path.join(MODEL_FOLDER, "user_idx_map.pkl"), "wb") as f: pickle.dump(osu_id_to_matrix_idx, f)
    with open(os.path.join(MODEL_FOLDER, "map_idx_map.pkl"), "wb") as f: pickle.dump(hash_to_matrix_idx, f)
    np.save(os.path.join(MODEL_FOLDER, "user_vectors_mf.npy"), als_model.user_factors)
    np.save(os.path.join(MODEL_FOLDER, "item_vectors_mf.npy"), als_model.item_factors)
    print(f"💾 Model state variables safely preserved in {MODEL_FOLDER}/ (user_vectors_mf.npy & item_vectors_mf.npy).")

if __name__ == "__main__":
    train_user_portfolios_als()