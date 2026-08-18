import unittest
import os
import sys
import numpy as np

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVER_DIR)

from core.DatabaseManager import DatabaseManager
from core.RecommendationEngine import RecommendationEngine

class TestMapsetDeduplication(unittest.TestCase):
    def setUp(self):
        self.test_db_path = os.path.join(SERVER_DIR, "data", "test_dedup.db")
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        os.environ["DATABASE_URL"] = f"sqlite:///{self.test_db_path}"
        self.db = DatabaseManager(db_target=self.test_db_path)
        self.engine = RecommendationEngine(db_manager=self.db)
        
        # Reset model maps for isolated unit testing
        self.engine.map_idx_map = {}
        self.engine.item_factors = np.empty((0, 128), dtype=np.float32)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except Exception:
                pass

    def test_recommendation_output_mapset_deduplication(self):
        # Insert 4 maps belonging to the SAME mapset (mapset_id 100) and 2 maps from mapsets 200 & 201
        for i in range(4):
            emb = np.random.randn(128).astype(np.float32)
            self.db.add_map(
                mapset_id=100,
                map_hash=f"hash_set100_{i}",
                embed=emb.tolist(),
                title="Song A",
                version=f"Diff {i}"
            )
            self.engine.add_map(f"hash_set100_{i}", emb)

        for i in range(2):
            emb = np.random.randn(128).astype(np.float32)
            self.db.add_map(
                mapset_id=200 + i,
                map_hash=f"hash_set200_{i}",
                embed=emb.tolist(),
                title=f"Song B{i}",
                version=f"Diff {i}"
            )
            self.engine.add_map(f"hash_set200_{i}", emb)

        # Set dummy user factor
        self.engine.user_idx_map["user1"] = 0
        self.engine.user_factors = np.ones((1, 128), dtype=np.float32)
        self.engine._rebuild_idx_to_hash()

        recs = self.engine.get_user_recommendations("user1", k=10, exclude_played=False)

        # Check mapsets in recommendations
        mapsets_seen = []
        for r in recs:
            meta = self.db.get_map_by_hash(r['map_hash'])
            if meta and meta.get('mapset_id'):
                mapsets_seen.append(meta['mapset_id'])

        # Since mapset 100 has 4 difficulties, only 1 should be recommended!
        self.assertEqual(mapsets_seen.count(100), 1)

    def test_refresh_user_mapset_deduplication(self):
        # Add 3 replays from mapset 100 and 1 replay from mapset 200
        for i in range(3):
            h = f"hash_user_set100_{i}"
            emb = np.random.randn(128).astype(np.float32)
            self.db.add_map(mapset_id=100, map_hash=h, embed=emb.tolist(), title="Song A")
            self.engine.add_map(h, emb)
            self.db.add_replay(username="user1", map_hash=h, mastery_score=0.5 + i * 0.1)

        self.db.add_user("999", "user1")
        user_vec = self.engine.refresh_user("999")
        self.assertIsNotNone(user_vec)
        self.assertEqual(len(user_vec), 128)

if __name__ == "__main__":
    unittest.main()
