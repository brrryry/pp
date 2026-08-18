import os
import sys
import sqlite3
import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import ranking_metrics_at_k
from collections import defaultdict
import random
import pickle
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config

DB_PATH = config.DB_FILE
MODEL_FOLDER = config.MODELS_DIR
EMBEDDING_SIZE = config.EMBEDDING_SIZE

def get_user_replays():
    """Fetch all replays: user_id, map_hash, mastery_score."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.osu_id, LOWER(TRIM(r.map_hash)), r.mastery_score
        FROM replays r
        INNER JOIN users u ON u.username = r.username
        WHERE r.mastery_score > 0
    """)
    data = cursor.fetchall()
    conn.close()
    return data

def split_replays_per_user(replays, test_ratio=0.2, holdout_best=False):
    """
    Splits replays per user.
    If holdout_best=True, puts the highest mastery replays into test.
    Else, random split.
    Returns dict: {user_id: {'train': [(map_hash, mastery)], 'test': [(map_hash, mastery)]}}
    """
    user_replays = defaultdict(list)
    for uid, h, score in replays:
        user_replays[uid].append((h, score))
    
    splits = {}
    for uid, items in user_replays.items():
        if holdout_best:
            # Sort by mastery descending
            items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
            n_test = max(1, int(len(items) * test_ratio))
            test_items = items_sorted[:n_test]
            train_items = items_sorted[n_test:]
        else:
            random.shuffle(items)
            n_test = max(1, int(len(items) * test_ratio))
            test_items = items[:n_test]
            train_items = items[n_test:]
        splits[uid] = {'train': train_items, 'test': test_items}
    return splits

def build_matrices(splits):
    """
    Build user-item matrices for both train and test splits using a unified map catalog.
    Returns: train_matrix, test_matrix, user_to_idx, map_to_idx
    """
    users = list(splits.keys())
    all_maps = set()
    for uid, data in splits.items():
        for h, _ in data['train']:
            all_maps.add(h)
        for h, _ in data['test']:
            all_maps.add(h)
            
    user_to_idx = {uid: i for i, uid in enumerate(users)}
    map_to_idx = {h: i for i, h in enumerate(all_maps)}
    
    def _create_csr(split_name):
        rows, cols, confs = [], [], []
        for uid, data in splits.items():
            row = user_to_idx[uid]
            for h, mastery in data[split_name]:
                if h in map_to_idx:
                    col = map_to_idx[h]
                    confidence = 1.0 + 3.0 * max(0.01, min(1.0, mastery))
                    rows.append(row)
                    cols.append(col)
                    confs.append(confidence)
        return sparse.coo_matrix((confs, (rows, cols)),
                                 shape=(len(users), len(all_maps))).tocsr()

    train_matrix = _create_csr('train')
    test_matrix = _create_csr('test')
    return train_matrix, test_matrix, user_to_idx, map_to_idx

def train_als(matrix, factors=32, reg=0.15, iterations=50):
    als = AlternatingLeastSquares(factors=factors, regularization=reg,
                                  iterations=iterations, random_state=42,
                                  use_gpu=False)
    als.fit(matrix, show_progress=False)
    return als

def evaluate_user(user_id, user_to_idx, map_to_idx, als, train_items, test_items, k=10):
    """Compute Recall@k and NDCG@k for a single user, with training items masked out."""
    if user_id not in user_to_idx:
        return 0.0, 0.0  # no training data
    user_row = user_to_idx[user_id]
    # Get user vector
    user_vec = als.user_factors[user_row]
    # Compute scores for all items
    item_vecs = als.item_factors
    scores = (item_vecs @ user_vec).copy()  # dot product

    # Action 1 Fix: Mask items the user already played in the training split
    train_indices = [map_to_idx[h] for h, _ in train_items if h in map_to_idx]
    if train_indices:
        scores[train_indices] = -np.inf

    # Sort descending
    top_indices = np.argsort(scores)[::-1][:k]
    # Convert to map hashes
    idx_to_map = {i: h for h, i in map_to_idx.items()}
    top_maps = [idx_to_map[i] for i in top_indices if i in idx_to_map]
    test_map_set = {h for h, _ in test_items}
    
    # Recall@k
    hits = sum(1 for h in top_maps if h in test_map_set)
    recall = hits / len(test_map_set) if test_map_set else 0.0
    
    # NDCG@k: position discount
    dcg = 0.0
    for rank, h in enumerate(top_maps, start=1):
        if h in test_map_set:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_ranks = list(range(1, min(len(test_map_set), k)+1))
    idcg = sum(1.0 / np.log2(r+1) for r in ideal_ranks)
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, ndcg

def main():
    # 1. Get all replays
    replays = get_user_replays()
    print(f"Total replays: {len(replays)}")
    
    # 2. Split per user (random 80/20, or set holdout_best=True)
    random.seed(42)
    splits = split_replays_per_user(replays, test_ratio=0.2, holdout_best=False)
    print(f"Split into train/test for {len(splits)} users.")
    
    # 3. Build unified training and testing matrices
    train_matrix, test_matrix, user_to_idx, map_to_idx = build_matrices(splits)
    print(f"Training matrix shape: {train_matrix.shape}, nnz={train_matrix.nnz}")
    print(f"Testing matrix shape:  {test_matrix.shape}, nnz={test_matrix.nnz}")
    
    # 4. Train ALS
    print("Training ALS...")
    als = train_als(train_matrix, factors=EMBEDDING_SIZE, reg=0.15, iterations=50)
    
    # 5. Custom Per-User Evaluation (with Train Item Masking)
    recalls = []
    ndcgs = []
    for uid, data in tqdm(splits.items(), desc="Evaluating users (masked train items)"):
        test_items = data['test']
        train_items = data['train']
        if not test_items:
            continue
        recall, ndcg = evaluate_user(uid, user_to_idx, map_to_idx, als, train_items, test_items, k=10)
        recalls.append(recall)
        ndcgs.append(ndcg)
    
    avg_recall = np.mean(recalls)
    avg_ndcg = np.mean(ndcgs)
    
    # 6. Built-in implicit.evaluation metrics
    print("\nCalculating built-in implicit.evaluation metrics...")
    implicit_metrics = ranking_metrics_at_k(als, train_matrix, test_matrix, K=10, show_progress=False)

    print(f"\n==========================================")
    print(f" Evaluation Results (K=10)")
    print(f"==========================================")
    print(f" Custom Evaluation (Masked Train Items):")
    print(f"   Average Recall@10: {avg_recall:.4f}")
    print(f"   Average NDCG@10:   {avg_ndcg:.4f}")
    print(f"\n Built-in implicit.evaluation:")
    print(f"   Precision@10:      {implicit_metrics['precision']:.4f}")
    print(f"   NDCG@10:           {implicit_metrics['ndcg']:.4f}")
    print(f"   MAP@10:            {implicit_metrics['map']:.4f}")
    print(f"   AUC@10:            {implicit_metrics['auc']:.4f}")
    print(f"==========================================\n")
    
    eval_dir = os.path.join(BASE_DIR, "models_eval")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "user_to_idx.pkl"), "wb") as f:
        pickle.dump(user_to_idx, f)
    with open(os.path.join(eval_dir, "map_to_idx.pkl"), "wb") as f:
        pickle.dump(map_to_idx, f)
    np.save(os.path.join(eval_dir, "user_factors.npy"), als.user_factors)
    np.save(os.path.join(eval_dir, "item_factors.npy"), als.item_factors)
    print(f"Model saved to {eval_dir}/")

if __name__ == "__main__":
    main()