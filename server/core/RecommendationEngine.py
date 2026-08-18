import numpy as np
import pickle
import os
import sqlite3
import logging
import json
from typing import Dict, List, Tuple
# pyrefly: ignore [missing-import]
from implicit.als import AlternatingLeastSquares
# pyrefly: ignore [missing-import]
import scipy.sparse as sparse

logging.basicConfig(level=logging.INFO, format="[RecommendationEngine] %(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self, db_manager, model_folder="models", embedding_size=128, reg=0.15, alpha=1.0):
        self.db = db_manager
        self.model_folder = model_folder
        self.embedding_size = embedding_size
        self.reg = reg
        self.alpha = alpha

        # Load existing model or initialise empty
        self.user_idx_map = {}       # osu_id -> int
        self.map_idx_map = {}        # map_hash -> int
        self.idx_to_hash = {}        # int -> map_hash
        self.user_factors = None     # np.ndarray (num_users, emb)
        self.item_factors = None     # np.ndarray (num_maps, emb)
        self.XtX_inv = None          # precomputed (emb, emb) for fast user update

        self._load()

    def _load(self):
        """Load model artifacts from disk, or initialise empty."""
        user_idx_path = os.path.join(self.model_folder, "user_idx_map.pkl")
        map_idx_path = os.path.join(self.model_folder, "map_idx_map.pkl")
        user_vec_path = os.path.join(self.model_folder, "user_vectors.npy")
        item_vec_path = os.path.join(self.model_folder, "item_vectors.npy")

        if all(os.path.exists(p) for p in [user_idx_path, map_idx_path, user_vec_path, item_vec_path]):
            with open(user_idx_path, "rb") as f:
                self.user_idx_map = pickle.load(f)
            with open(map_idx_path, "rb") as f:
                self.map_idx_map = pickle.load(f)
            self.user_factors = np.load(user_vec_path)
            self.item_factors = np.load(item_vec_path)
        else:
            # Empty start
            self.user_factors = np.empty((0, self.embedding_size), dtype=np.float32)
            self.item_factors = np.empty((0, self.embedding_size), dtype=np.float32)

        if self.item_factors.shape[0] == 0:
            self._sync_maps_from_db()

        self._rebuild_idx_to_hash()
        self._precompute_XtX_inv()

    def _rebuild_idx_to_hash(self):
        """Rebuild reverse map from index to hash."""
        self.idx_to_hash = {i: h for h, i in self.map_idx_map.items()}

    def _sync_maps_from_db(self):
        """Sync map embeddings from DatabaseManager into item_factors if empty or missing."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT map_hash, embed FROM maps WHERE embed IS NOT NULL AND embed != ''")
                rows = cursor.fetchall()

            if not rows:
                return

            hashes = []
            vecs = []
            for h, embed_json in rows:
                clean_h = str(h).strip().lower()
                if not embed_json or clean_h in self.map_idx_map:
                    continue
                try:
                    emb = np.array(json.loads(embed_json), dtype=np.float32)
                    if emb.shape == (self.embedding_size,):
                        hashes.append(clean_h)
                        vecs.append(emb)
                except Exception:
                    continue

            if hashes:
                start_idx = len(self.map_idx_map)
                for i, h in enumerate(hashes):
                    self.map_idx_map[h] = start_idx + i
                
                new_matrix = np.array(vecs, dtype=np.float32)
                if self.item_factors.shape[0] == 0:
                    self.item_factors = new_matrix
                else:
                    self.item_factors = np.vstack([self.item_factors, new_matrix])
                
                self._rebuild_idx_to_hash()
                self._save()
        except Exception as e:
            logger.error(f"Failed to sync maps from DB: {e}")

    def _precompute_XtX_inv(self):
        """Compute (X^T X + reg*I)^-1 once, where X = item_factors."""
        if self.item_factors.shape[0] > 0:
            XtX = self.item_factors.T @ self.item_factors + self.reg * np.eye(self.embedding_size)
            self.XtX_inv = np.linalg.inv(XtX)
        else:
            self.XtX_inv = np.linalg.inv(self.reg * np.eye(self.embedding_size))

    def _save(self):
        """Persist current model state to disk."""
        os.makedirs(self.model_folder, exist_ok=True)
        with open(os.path.join(self.model_folder, "user_idx_map.pkl"), "wb") as f:
            pickle.dump(self.user_idx_map, f)
        with open(os.path.join(self.model_folder, "map_idx_map.pkl"), "wb") as f:
            pickle.dump(self.map_idx_map, f)
        np.save(os.path.join(self.model_folder, "user_vectors.npy"), self.user_factors)
        np.save(os.path.join(self.model_folder, "item_vectors.npy"), self.item_factors)

    def _ensure_user(self, osu_id):
        """Add new user to mapping and factors if not present."""
        str_uid = str(osu_id)
        if str_uid not in self.user_idx_map:
            new_idx = len(self.user_idx_map)
            self.user_idx_map[str_uid] = new_idx
            new_vec = np.random.randn(self.embedding_size).astype(np.float32) * 0.01
            self.user_factors = np.vstack([self.user_factors, new_vec])

    def _ensure_map(self, map_hash):
        """Add new map to mapping and item factors if not present."""
        clean_h = str(map_hash).strip().lower()
        if clean_h not in self.map_idx_map:
            new_idx = len(self.map_idx_map)
            self.map_idx_map[clean_h] = new_idx
            new_vec = np.zeros(self.embedding_size, dtype=np.float32)
            self.item_factors = np.vstack([self.item_factors, new_vec])
            self._rebuild_idx_to_hash()
            self._precompute_XtX_inv()

    def add_map(self, map_hash, embedding):
        """Add a new map with its VAE embedding."""
        clean_h = str(map_hash).strip().lower()
        self._ensure_map(clean_h)
        idx = self.map_idx_map[clean_h]
        self.item_factors[idx] = embedding.astype(np.float32)
        self._rebuild_idx_to_hash()
        self._precompute_XtX_inv()
        self._save()

    def refresh_user(self, osu_id):
        """
        Recompute user factor online from all their replays (closed‑form ALS update).
        Matches matrix_factorization_engine.py ALS update formula.
        """
        str_uid = str(osu_id)
        self._ensure_user(str_uid)
        user_idx = self.user_idx_map[str_uid]

        replays = self.db.get_user_replays(str_uid)

        if not replays:
            self.user_factors[user_idx] = np.zeros(self.embedding_size, dtype=np.float32)
            self._save()
            return self.user_factors[user_idx]

        # Deduplicate replays by mapset_id (keep single highest-mastery play per beatmapset)
        best_replays_by_mapset = {}
        for r in replays:
            if isinstance(r, dict):
                map_hash = str(r.get('map_hash')).strip().lower()
                mastery = float(r.get('mastery_score', 0.0))
                ms_id = r.get('mapset_id')
            else:
                map_hash = str(r[0]).strip().lower()
                mastery = float(r[1]) if len(r) > 1 else 0.0
                ms_id = None

            if not ms_id and map_hash:
                map_meta = self.db.get_map_by_hash(map_hash)
                if map_meta:
                    ms_id = map_meta.get('mapset_id')

            key = ms_id if ms_id else map_hash
            if key not in best_replays_by_mapset or mastery > best_replays_by_mapset[key][1]:
                best_replays_by_mapset[key] = (r, mastery)

        replays_to_process = [item[0] for item in best_replays_by_mapset.values()]

        cols = []
        confidences = []
        mastery_weights = []
        for r in replays_to_process:
            if isinstance(r, dict):
                map_hash = str(r.get('map_hash')).strip().lower()
                mastery = float(r.get('mastery_score', 0.0))
            else:
                map_hash = str(r[0]).strip().lower()
                mastery = float(r[1]) if len(r) > 1 else 0.0
            if not map_hash or map_hash not in self.map_idx_map:
                continue
            col = self.map_idx_map[map_hash]
            # Quadratic mastery weighting: low mastery (<0.2) gets negligible weight (~0.01-0.04)
            mastery_w = max(0.0, min(1.0, mastery)) ** 2
            confidence = 1.0 + self.alpha * mastery_w
            cols.append(col)
            confidences.append(confidence)
            mastery_weights.append(mastery_w)

        if not cols:
            self.user_factors[user_idx] = np.zeros(self.embedding_size, dtype=np.float32)
            self._save()
            return self.user_factors[user_idx]

        X = self.item_factors[cols]
        c = np.array(confidences, dtype=np.float32)
        w = np.array(mastery_weights, dtype=np.float32)

        XtCX = X.T @ (c[:, None] * X) + self.reg * np.eye(self.embedding_size)
        RHS = X.T @ c
        try:
            user_factor = np.linalg.solve(XtCX, RHS)
        except np.linalg.LinAlgError:
            user_factor = np.linalg.lstsq(XtCX, RHS, rcond=None)[0]

        # Blend ALS vector with mastery-weighted centroid of played map embeddings
        total_w = w.sum()
        if total_w > 0:
            centroid = (w[:, None] * X).sum(axis=0) / total_w
            c_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            u_norm = user_factor / (np.linalg.norm(user_factor) + 1e-8)
            user_factor = 0.65 * u_norm + 0.35 * c_norm

        self.user_factors[user_idx] = user_factor.astype(np.float32)
        self._save()

        # Invalidate user recs cache in Redis
        if self.db and getattr(self.db, 'redis_client', None):
            try:
                for k in self.db.redis_client.scan_iter(f"cache:recs:{str_uid}:*"):
                    self.db.redis_client.delete(k)
                for k in self.db.redis_client.scan_iter(f"cache:endpoint:recs:{str_uid}:*"):
                    self.db.redis_client.delete(k)
                self.db.redis_client.delete(f"cache:comfort_sr:{str_uid}")
            except Exception:
                pass

        return user_factor

    def get_user_comfort_sr(self, osu_id, default_sr=4.5):
        """
        Auto-detect comfort star rating from user's played replays with high mastery score (> 0.60).
        Matches model_training/recommender.py logic (cached in Redis).
        """
        str_uid = str(osu_id).strip().lower()
        redis_key = f"cache:comfort_sr:{str_uid}"

        if self.db and getattr(self.db, 'redis_client', None):
            try:
                cached = self.db.redis_client.get(redis_key)
                if cached is not None:
                    return float(cached)
            except Exception:
                pass

        val = default_sr
        try:
            p = "%s" if getattr(self.db, "is_postgres", False) else "?"
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT AVG(m.sr) FROM maps m
                    INNER JOIN replays r ON LOWER(TRIM(m.map_hash)) = LOWER(TRIM(r.map_hash))
                    INNER JOIN users u ON u.username = r.username
                    WHERE (CAST(u.osu_id AS TEXT) = {p} OR u.username = {p}) AND r.mastery_score > 0.60
                """, (str_uid, str_uid))
                row = cursor.fetchone()
                if row and row[0] is not None:
                    val = float(row[0])
        except Exception:
            pass

        if self.db and getattr(self.db, 'redis_client', None):
            try:
                self.db.redis_client.set(redis_key, str(val), ex=3600)
            except Exception:
                pass

        return val

    def fit_global_als(self, iterations=30, regularization=0.15):
        """
        Fits global ALS model across all database replays.
        Matches model_training/matrix_factorization_engine.py logic.
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT DISTINCT u.osu_id FROM users u INNER JOIN replays r ON u.username = r.username WHERE r.mastery_score > 0 ORDER BY u.osu_id ASC")
                all_users = [row[0] for row in cursor.fetchall()]
                osu_id_to_idx = {str(uid): i for i, uid in enumerate(all_users)}

                cursor.execute("SELECT DISTINCT m.map_hash FROM maps m INNER JOIN replays r ON m.map_hash = r.map_hash WHERE r.mastery_score > 0")
                all_maps = [str(row[0]).strip().lower() for row in cursor.fetchall()]
                hash_to_idx = {h: i for i, h in enumerate(all_maps)}
                num_users, num_maps = len(all_users), len(all_maps)

                if num_users == 0 or num_maps == 0:
                    return

                cursor.execute("SELECT u.osu_id, m.map_hash, r.mastery_score FROM replays r INNER JOIN users u ON u.username = r.username INNER JOIN maps m ON r.map_hash = m.map_hash WHERE r.mastery_score > 0")
                rows, cols, data = [], [], []
                for osu_id, b_hash, mastery in cursor.fetchall():
                    clean_hash = str(b_hash).strip().lower()
                    str_uid = str(osu_id)
                    if str_uid in osu_id_to_idx and clean_hash in hash_to_idx:
                        rows.append(osu_id_to_idx[str_uid])
                        cols.append(hash_to_idx[clean_hash])
                        confidence = 1.0 + self.alpha * max(0.01, min(1.0, mastery))
                        data.append(confidence)

            user_map = sparse.coo_matrix((data, (rows, cols)), shape=(num_users, num_maps)).tocsr()

            als = AlternatingLeastSquares(
                factors=self.embedding_size,
                regularization=regularization,
                iterations=iterations,
                random_state=42,
                use_gpu=False
            )
            als.fit(user_map, show_progress=False)

            self.user_idx_map = osu_id_to_idx
            self.map_idx_map = hash_to_idx
            self.user_factors = als.user_factors.astype(np.float32)
            self.item_factors = als.item_factors.astype(np.float32)
            self._rebuild_idx_to_hash()
            self._precompute_XtX_inv()
            self._save()
            logger.info(f"Successfully fitted global ALS model on {num_users} users and {num_maps} maps.")
        except Exception as e:
            logger.error(f"Failed to fit global ALS model: {e}")

    def get_user_recommendations(self, osu_id, k=10, exclude_played=True, difficulty_tolerance=2.0):
        """
        Generate top‑k map recommendations for a user (cached in Redis).
        Supports optional difficulty tolerance around comfort baseline SR (default: ±2.0★).
        """
        str_uid = str(osu_id).strip().lower()
        redis_key = f"cache:recs:{str_uid}:{k}:{exclude_played}:{difficulty_tolerance}"

        if self.db and getattr(self.db, 'redis_client', None):
            try:
                cached = self.db.redis_client.get(redis_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        if str_uid not in self.user_idx_map:
            self.refresh_user(str_uid)

        user_idx = self.user_idx_map[str_uid]
        user_vec = self.user_factors[user_idx]

        if self.item_factors.shape[0] == 0:
            return []

        # L2-normalize factors for Cosine Similarity matching model_training/recommender.py
        item_norms = self.item_factors / (np.linalg.norm(self.item_factors, axis=1, keepdims=True) + 1e-8)
        user_norm = user_vec / (np.linalg.norm(user_vec) + 1e-8)

        scores = item_norms @ user_norm
        top_indices = np.argsort(scores)[::-1]

        if exclude_played:
            raw_replays = self.db.get_user_replays(str_uid) or []
            played_hashes = {str(r['map_hash'] if isinstance(r, dict) else r[0]).strip().lower() for r in raw_replays}
        else:
            played_hashes = set()

        if not hasattr(self, 'idx_to_hash') or len(self.idx_to_hash) != len(self.map_idx_map):
            self._rebuild_idx_to_hash()

        comfort_sr = self.get_user_comfort_sr(str_uid) if difficulty_tolerance else None

        seen_mapsets = set()
        recommendations = []
        for idx in top_indices:
            map_hash = self.idx_to_hash.get(idx)
            if map_hash and map_hash not in played_hashes:
                raw_score = float(scores[idx])

                map_meta = self.db.get_map_by_hash(map_hash) or {}
                mapset_id = map_meta.get('mapset_id')

                # Deduplicate: max 1 map per beatmapset in recommendation output
                if mapset_id and mapset_id in seen_mapsets:
                    continue

                if comfort_sr is not None and map_meta:
                    sr = map_meta.get('sr', 4.5)
                    if abs(sr - comfort_sr) > difficulty_tolerance:
                        continue

                if mapset_id:
                    seen_mapsets.add(mapset_id)

                recommendations.append({
                    'map_hash': map_hash,
                    'score': raw_score,
                    'comfort_sr': comfort_sr
                })
                if len(recommendations) >= k:
                    break

        if self.db and getattr(self.db, 'redis_client', None):
            try:
                self.db.redis_client.set(redis_key, json.dumps(recommendations), ex=900)
            except Exception:
                pass

        return recommendations

    def get_embed_by_hash(self, map_hash):
        clean_h = str(map_hash).strip().lower()
        if clean_h in self.map_idx_map:
            idx = self.map_idx_map[clean_h]
            return self.item_factors[idx]
        else:
            map_data = self.db.get_map_data_by_hash(clean_h) if hasattr(self.db, 'get_map_data_by_hash') else (self.db.get_map_by_hash(clean_h) if hasattr(self.db, 'get_map_by_hash') else None)
            if map_data and map_data.get('embed'):
                try:
                    emb = np.array(json.loads(map_data['embed']), dtype=np.float32)
                    self.add_map(clean_h, emb)
                    return emb
                except Exception:
                    return None
            else:
                return None

    def get_influential_plays_batch(self, osu_id, target_map_hashes):
        """
        Vectorized batch calculation of the most influential played map for a set of recommended maps.
        Runs in ~5ms using single NumPy matrix multiplication.
        """
        if not target_map_hashes:
            return {}

        str_uid = str(osu_id)
        replays = self.db.get_user_replays(str_uid) or []
        if not replays:
            return {}

        played_hashes = []
        played_vecs = []
        played_weights = []
        played_metas = {}

        for r in replays:
            map_hash = str(r.get('map_hash') if isinstance(r, dict) else r[0]).strip().lower()
            mastery = r.get('mastery_score', 1.0) if isinstance(r, dict) else (r[1] if len(r) > 1 else 1.0)
    
            if not map_hash:
                continue

            vec = self.get_embed_by_hash(map_hash)
            if vec is None:
                continue

            norm = np.linalg.norm(vec)
            if norm == 0:
                continue

            played_hashes.append(map_hash)
            played_vecs.append(vec / norm)
            weight = 1.0 + self.alpha * (max(0.0, min(1.0, float(mastery))) ** 2)
            played_weights.append(weight)

            if isinstance(r, dict) and r.get('title'):
                played_metas[map_hash] = r

        if not played_vecs:
            return {}

        P = np.array(played_vecs, dtype=np.float32)        # (M, 128)
        W = np.array(played_weights, dtype=np.float32)     # (M,)

        target_vecs = []
        valid_targets = []

        for th in target_map_hashes:
            clean_th = str(th).strip().lower()
            vec = self.get_embed_by_hash(clean_th)
            if vec is not None:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    target_vecs.append(vec / norm)
                    valid_targets.append(clean_th)

        if not target_vecs:
            return {}

        T = np.array(target_vecs, dtype=np.float32)        # (N, 128)

        # Matrix multiply: S = T @ P.T -> shape (N, M)
        S = T @ P.T
        I = S * W[None, :]

        result = {}
        for i, th in enumerate(valid_targets):
            row_scores = I[i]
            sorted_j = np.argsort(row_scores)[::-1]

            top_plays = []
            seen_mapset_ids = set()
            for j in sorted_j:
                sim = float(S[i, j])
                h = played_hashes[j]
                meta = played_metas.get(h, {})
                if not meta or meta.get("title") in (None, "Unknown Title", "Unknown"):
                    db_map = self.db.get_map_by_hash(h)
                    if db_map:
                        meta = db_map

                ms_id = meta.get("mapset_id") if meta else None
                if ms_id and ms_id in seen_mapset_ids:
                    continue
                if ms_id:
                    seen_mapset_ids.add(ms_id)

                top_plays.append({
                    "map_hash": h,
                    "similarity": round(max(0.0, min(1.0, sim)), 4),
                    "title": meta.get("title", "Unknown Title"),
                    "artist": meta.get("artist", "Unknown Artist"),
                    "difficulty": meta.get("difficulty", meta.get("version", "Normal")),
                    "star_rating": float(meta.get("sr") or meta.get("star_rating") or 0.0),
                    "beatmap_id": meta.get("beatmap_id", meta.get("map_id")),
                    "mods": meta.get("mods", 0)
                })
                if len(top_plays) >= 3:
                    break

            result[th] = top_plays

        return result

    def get_influential_play_for_map(self, osu_id, target_map_hash):
        batch_res = self.get_influential_plays_batch(osu_id, [target_map_hash])
        plays = batch_res.get(str(target_map_hash).strip().lower(), [])
        return plays[0] if plays else None