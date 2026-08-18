import os
import sys
import sqlite3
import numpy as np
import pickle
import faiss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config

DB_PATH = config.DB_FILE
MODEL_FOLDER = config.MODELS_DIR
EMBEDDING_SIZE = config.EMBEDDING_SIZE

def get_player_history(osu_id):
    """
    Queries the database to find all map hashes the player has already played.
    We use this to prevent recommending maps they have already cleared.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # We join users and replays by username to locate the target player's history
    cursor.execute("""
        SELECT LOWER(TRIM(r.map_hash)) 
        FROM replays r
        INNER JOIN users u ON TRIM(LOWER(u.username)) = TRIM(LOWER(r.username))
        WHERE u.osu_id = ?
    """, (osu_id,))
    
    played_hashes = {row[0] for row in cursor.fetchall()}
    conn.close()
    return played_hashes


def load_vectors_from_db_column(map_idx_map):
    """
    Extracts the pre-computed LSTM embeddings straight from your maps table,
    aligning them perfectly with your matrix catalog indices.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    num_maps = len(map_idx_map)
    real_vectors = np.zeros((num_maps, EMBEDDING_SIZE), dtype=np.float32)
    
    cursor.execute("SELECT LOWER(TRIM(map_hash)), embed FROM maps WHERE embed IS NOT NULL")
    
    for b_hash, embed_data in cursor.fetchall():
        clean_hash = str(b_hash).strip().lower()
        if clean_hash in map_idx_map:
            target_col_idx = map_idx_map[clean_hash]
            try:
                clean_text = str(embed_data).replace("[", "").replace("]", "").strip()
                vector_array = np.fromstring(clean_text, sep=",", dtype=np.float32)
                if vector_array.size == EMBEDDING_SIZE:
                    real_vectors[target_col_idx] = vector_array
            except Exception:
                pass
                
    conn.close()
    return real_vectors


def recommend_maps_for_user(osu_id, k_suggestions=5, difficulty_tolerance=2.0):
    # 1. Load saved index mapping and ALS parameters from disk
    user_idx_path = f"{MODEL_FOLDER}/user_idx_map.pkl"
    map_idx_path = f"{MODEL_FOLDER}/map_idx_map.pkl"
    user_vec_path = f"{MODEL_FOLDER}/user_vectors_mf.npy"
    item_vec_path = f"{MODEL_FOLDER}/item_vectors_mf.npy"
    
    if not (os.path.exists(user_idx_path) and os.path.exists(map_idx_path) and os.path.exists(user_vec_path)):
        print("❌ Error: Missing trained model components in 'models/' folder.")
        return []

    with open(user_idx_path, "rb") as f:
        user_idx_map = pickle.load(f)
    with open(map_idx_path, "rb") as f:
        map_idx_map = pickle.load(f)
        
    user_factors = np.load(user_vec_path)
    matrix_idx_to_hash = {idx: b_hash for b_hash, idx in map_idx_map.items()}

    # 2. Check if the user exists in our matrix
    if osu_id not in user_idx_map:
        print(f"⚠️ User ID {osu_id} has no trained portfolio vector yet.")
        return []

    # 3. Extract target player vector
    user_row_idx = user_idx_map[osu_id]
    player_vector = user_factors[user_row_idx].reshape(1, -1).astype('float32')

    # 4. Load aligned item vectors (ALS item factors or fallback to map DNA)
    if os.path.exists(item_vec_path):
        item_vectors = np.load(item_vec_path).astype('float32')
        embedding_size = item_vectors.shape[1]
    else:
        item_vectors = load_vectors_from_db_column(map_idx_map)
        embedding_size = item_vectors.shape[1] if item_vectors.shape[1] > 0 else EMBEDDING_SIZE

    if player_vector.shape[1] != item_vectors.shape[1]:
        # Handle dimension fallback if user factors dimension differs from item vectors
        item_vectors = load_vectors_from_db_column(map_idx_map)
        embedding_size = item_vectors.shape[1]

    # 5. Fetch unplayed history set
    played_history = get_player_history(osu_id)

    # 6. FAISS search in aligned vector space
    faiss_index = faiss.IndexFlatIP(embedding_size)
    
    normalized_item_vectors = item_vectors.copy()
    faiss.normalize_L2(normalized_item_vectors)
    faiss_index.add(normalized_item_vectors)

    normalized_player_vector = player_vector.copy()
    if normalized_player_vector.shape[1] == embedding_size:
        faiss.normalize_L2(normalized_player_vector)
    else:
        normalized_player_vector = np.zeros((1, embedding_size), dtype=np.float32)
    
    similarities, target_indices = faiss_index.search(normalized_player_vector, 200)

    flat_indices = target_indices.flatten()
    flat_similarities = similarities.flatten()
    recommended_hashes = [matrix_idx_to_hash[idx] for idx in flat_indices if idx in matrix_idx_to_hash]
    
    # 7. Metadata pull and comfort star rating detection
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT AVG(m.sr) FROM maps m
        INNER JOIN replays r ON LOWER(TRIM(m.map_hash)) = LOWER(TRIM(r.map_hash))
        INNER JOIN users u ON u.username = r.username
        WHERE u.osu_id = ? AND r.mastery_score > 0.60
    """, (osu_id,))
    res = cursor.fetchone()[0]
    target_star_rating = res if res else 4.5
    if difficulty_tolerance is not None:
        print(f"📊 Auto-detected comfort baseline difficulty: {target_star_rating:.2f}★ (tolerance: ±{difficulty_tolerance:.2f}★)")
    else:
        print(f"📊 Unconstrained recommendation mode (No Star Rating restriction).")

    placeholders = ",".join(["?"] * len(recommended_hashes))
    query = f"""
        SELECT LOWER(TRIM(map_hash)), artist, title, version, sr 
        FROM maps 
        WHERE LOWER(TRIM(map_hash)) IN ({placeholders})
    """
    
    db_pool = {}
    try:
        cursor.execute(query, recommended_hashes)
        for b_hash, artist, title, version, stars in cursor.fetchall():
            db_pool[b_hash] = {
                'title': f"{artist} - {title} [{version}]",
                'stars': stars if stars else 0.0
            }
    finally:
        conn.close()

    # 8. Selection Corridor Filtering
    final_recommendations = []
    
    for rank in range(len(flat_indices)):
        matrix_idx = flat_indices[rank]
        b_hash = matrix_idx_to_hash.get(matrix_idx, None)
        
        if b_hash in db_pool:
            if b_hash in played_history:
                continue
                
            map_data = db_pool[b_hash]
            map_stars = map_data['stars']
            
            # Apply star restriction only if difficulty_tolerance is specified
            if difficulty_tolerance is None or abs(map_stars - target_star_rating) <= difficulty_tolerance:
                raw_cosine_score = flat_similarities[rank]
                match_percentage = (raw_cosine_score + 1.0) / 2.0 * 100.0
                
                final_recommendations.append(
                    f" 🌟 Style Match [{match_percentage:.1f}%] ({map_stars:.2f}★) ➔ {map_data['title']}"
                )
                
        if len(final_recommendations) == k_suggestions:
            break

    # 9. Terminal Layout Display
    mode_label = f"Unconstrained Mode" if difficulty_tolerance is None else f"Difficulty Corridor ±{difficulty_tolerance}★"
    print(f"\n🎯 TARGETED ALIGNED RECOMMENDATIONS FOR USER ID {osu_id} ({mode_label}):")
    print("-" * 75)
    if not final_recommendations:
        print("  No unplayed maps found matching your style.")
    for index, display_string in enumerate(final_recommendations):
        print(f" Rank {index+1}{display_string}")
    print("-" * 75)

    return final_recommendations

# --- Run Test Query ---
if __name__ == "__main__":
    try:
        # Pass an active osu! user ID from your table to execute tests
        recommend_maps_for_user(osu_id=11781698, k_suggestions=25)
    except Exception as e:
        print(f"Could not complete recommendation query: {e}")