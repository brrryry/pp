import os
import sys
import unittest
import json
import numpy as np

# Add parent directory to path to find main and DatabaseManager
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import server.main

class TestPredictor(unittest.TestCase):
    
    def test_prediction_logic(self):
        # Trigger model load
        loaded = server.main.load_accuracy_predictor_model()
        self.assertTrue(loaded, "Failed to load accuracy predictor model.")
        self.assertIsNotNone(server.main.accuracy_predictor_package)
        
        # Add a mock map to the database so endpoint has data
        features = {
            'file_hash': 'mockhash1234567890',
            'mapset_id': 9999,
            'title': 'Test Song',
            'artist': 'Test Artist',
            'creator': 'Test Creator',
            'version': 'Insane',
            'circle_size': 4.0,
            'overall_difficulty': 8.0,
            'hp_drain': 5.0,
            'approach_rate': 9.0,
            'slider_multiplier': 1.4,
            'total_objects': 100
        }
        server.main.db.add_map(features)
        
        # Call predict_accuracy function directly
        res = server.main.predict_accuracy(username="MockPlayer", beatmap_hash="mockhash1234567890", mods="DT")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["username"], "MockPlayer")
        self.assertEqual(res["beatmap_hash"], "mockhash1234567890")
        self.assertIn("predicted_accuracy", res)
        self.assertTrue(50.0 <= res["predicted_accuracy"] <= 100.0)

if __name__ == "__main__":
    unittest.main()
