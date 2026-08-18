import unittest
import os
import sys
import numpy as np

# Add server directory to Python path
SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVER_DIR)

from core.Projector import MapProjector
from core.DatabaseManager import DatabaseManager

class TestUMAPProjection(unittest.TestCase):
    def setUp(self):
        self.test_db_path = os.path.join(SERVER_DIR, "data", "test_umap.db")
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        os.environ["DATABASE_URL"] = f"sqlite:///{self.test_db_path}"
        self.db = DatabaseManager(db_target=self.test_db_path)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except Exception:
                pass

    def test_projector_transform(self):
        projector = MapProjector()
        dummy_embed = np.random.randn(128).astype(np.float32)
        cx, cy = projector.transform(dummy_embed)
        self.assertIsInstance(cx, float)
        self.assertIsInstance(cy, float)
        self.assertFalse(np.isnan(cx))
        self.assertFalse(np.isnan(cy))

    def test_add_map_with_coordinates(self):
        map_hash = "test_umap_hash_001"
        embed = np.random.randn(128).tolist()
        self.db.add_map(
            mapset_id=1001,
            map_hash=map_hash,
            embed=embed,
            title="Test Map Title",
            coord_x=12.34,
            coord_y=-56.78
        )
        map_data = self.db.get_map_by_hash(map_hash)
        self.assertIsNotNone(map_data)
        self.assertAlmostEqual(map_data['coord_x'], 12.34, places=2)
        self.assertAlmostEqual(map_data['coord_y'], -56.78, places=2)

    def test_backfill_missing_coordinates(self):
        # Insert 5 maps with embeddings but 0.0 coordinates
        for i in range(5):
            h = f"test_backfill_hash_{i}"
            emb = np.random.randn(128).tolist()
            self.db.add_map(
                mapset_id=2000 + i,
                map_hash=h,
                embed=emb,
                title=f"Backfill Map {i}",
                coord_x=0.0,
                coord_y=0.0
            )

        projector = MapProjector()
        updated_count = self.db.backfill_missing_coordinates(projector=projector)
        self.assertEqual(updated_count, 5)

        for i in range(5):
            h = f"test_backfill_hash_{i}"
            m = self.db.get_map_by_hash(h)
            self.assertTrue(m['coord_x'] != 0.0 or m['coord_y'] != 0.0)

if __name__ == "__main__":
    unittest.main()
