import os
import sys
import numpy as np
import sqlite3
import pickle
import json
import umap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config

DB_PATH = config.DB_FILE
MODEL_FOLDER = config.MODELS_DIR
EMBEDDING_SIZE = config.EMBEDDING_SIZE

def project_all_maps():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Fetch all maps with embeddings
    cursor.execute("SELECT map_hash, embed FROM maps WHERE embed IS NOT NULL")
    rows = cursor.fetchall()

    if not rows:
        print("No maps with embeddings found.")
        return

    hashes = []
    embeddings = []
    for h, embed_json in rows:
        emb = np.array(json.loads(embed_json), dtype=np.float32)
        if emb.shape == (EMBEDDING_SIZE,):
            hashes.append(h)
            embeddings.append(emb)

    if len(embeddings) == 0:
        print("No valid embeddings found.")
        return

    X = np.array(embeddings)

    # 2. Fit UMAP (n_components=2 for 2D)
    print(f"Fitting UMAP on {len(X)} maps...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        verbose=True
    )
    coords = reducer.fit_transform(X)  # shape: (n_maps, 2)

    # 3. Save the UMAP model for future incremental updates
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    with open(os.path.join(MODEL_FOLDER, "umap_model.pkl"), "wb") as f:
        pickle.dump(reducer, f)

    # 4. Update the database with the 2D coordinates
    for idx, h in enumerate(hashes):
        x, y = float(coords[idx][0]), float(coords[idx][1])
        cursor.execute(
            "UPDATE maps SET coord_x = ?, coord_y = ?, projection_version = 1 WHERE map_hash = ?",
            (x, y, h)
        )

    conn.commit()
    conn.close()
    print(f"Projected {len(hashes)} maps to 2D.")

if __name__ == "__main__":
    project_all_maps()