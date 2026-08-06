import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
import sqlite3
import json
import time
from server.DatabaseManager import DatabaseManager

class TestDatabaseManager(unittest.TestCase):
    
    def setUp(self):
        # Use a temporary disk file to avoid SQLite connection closure wiping in-memory schemas
        self.db_file = "data/test_osu_profiler.db"
        if os.path.exists(self.db_file):
            try:
                os.remove(self.db_file)
            except OSError:
                pass
        self.db = DatabaseManager(self.db_file)
        
    def tearDown(self):
        if os.path.exists(self.db_file):
            try:
                os.remove(self.db_file)
            except OSError:
                pass

    def test_schema_initialization(self):
        # Verify the core tables exist in the schema
        conn = sqlite3.connect(self.db.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        self.assertIn("maps", tables)
        self.assertIn("map_portfolios", tables)
        self.assertIn("map_stats", tables)
        self.assertIn("replays", tables)
        self.assertIn("api_scores_cache", tables)

    def test_add_and_retrieve_map(self):
        # Mock map features
        features = {
            'file_hash': 'abcdef1234567890',
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
        
        # Insert
        map_id = self.db.add_map(features)
        self.assertIsNotNone(map_id)
        
        # Verify metadata exists in maps table
        self.assertTrue(self.db.find_mapset_by_id(9999))
        self.assertTrue(self.db.find_map_by_md5('abcdef1234567890'))
        self.assertEqual(self.db.get_map_id_by_hash('abcdef1234567890'), map_id)

    def test_add_and_retrieve_replay(self):
        # 1. Add map first to satisfy Foreign Key constraint
        features = {
            'file_hash': 'hash123',
            'mapset_id': 8888,
            'title': 'Song',
            'artist': 'Artist',
            'creator': 'Creator',
            'version': 'Normal'
        }
        map_id = self.db.add_map(features)
        
        # 2. Add replay linked to map_id
        hits_data = [{'target_time': 1000, 'aim_distance': 12.3, 'hit': True, 'score': 300}]
        mech_skills = {'SnapAim': 75.0, 'Speed': 60.0}
        
        replay_id = self.db.add_replay(
            map_id=map_id,
            username='Peppy',
            replay_hash='replay_file_name.osr',
            accuracy=98.5,
            unstable_rate=120.0,
            avg_aim_error_px=11.2,
            mods='HardRock',
            misses=1,
            total_notes=100,
            hits=99,
            hits_json=hits_data,
            mechanical_json=mech_skills
        )
        self.assertIsNotNone(replay_id)
        
        # 3. Retrieve and assert
        hits_str = self.db.get_replay_hits('replay_file_name.osr')
        self.assertIsNotNone(hits_str)
        retrieved_hits = json.loads(hits_str)
        self.assertEqual(retrieved_hits[0]['aim_distance'], 12.3)
        
        retrieved_mech = self.db.get_mechanical_skills('replay_file_name.osr')
        self.assertEqual(retrieved_mech['SnapAim'], 75.0)

    def test_cascade_delete(self):
        # 1. Add map
        features = {
            'file_hash': 'cascade_hash',
            'mapset_id': 7777,
            'title': 'Song',
            'artist': 'Artist',
            'creator': 'Creator',
            'version': 'Hard'
        }
        map_id = self.db.add_map(features)
        
        # 2. Add replay
        self.db.add_replay(
            map_id=map_id,
            username='Player',
            replay_hash='cascade_replay',
            accuracy=95.0,
            unstable_rate=180.0,
            avg_aim_error_px=14.5,
            mods='NoMod',
            misses=3,
            total_notes=200,
            hits=197,
            hits_json='[]'
        )
        
        # 3. Delete mapset
        self.db.delete_mapset_by_id(7777)
        
        # 4. Verify cascade delete cleared the replay
        self.assertIsNone(self.db.get_replay_hits('cascade_replay'))

    def test_queries_and_helpers(self):
        # Add a map
        map_features = {
            'file_hash': 'query_hash_abc',
            'mapset_id': 1111,
            'title': 'Test Queries',
            'artist': 'Query Artist',
            'creator': 'Query Creator',
            'version': 'Easy'
        }
        map_id = self.db.add_map(map_features)

        # Add two replays for different players
        self.db.add_replay(
            map_id=map_id, username='Cookiezi', replay_hash='rep1.osr',
            accuracy=99.2, unstable_rate=85.4, avg_aim_error_px=5.2,
            mods='Hidden+DoubleTime', misses=0, total_notes=500, hits=500,
            hits_json='[]'
        )
        self.db.add_replay(
            map_id=map_id, username='WhiteCat', replay_hash='rep2.osr',
            accuracy=100.0, unstable_rate=65.2, avg_aim_error_px=3.1,
            mods='HardRock', misses=0, total_notes=500, hits=500,
            hits_json='[]'
        )

        # Test unique players listing
        players = self.db.get_unique_players()
        self.assertEqual(len(players), 2)
        self.assertIn('Cookiezi', players)
        self.assertIn('WhiteCat', players)

        # Test player replays query
        cookiezi_plays = self.db.get_player_replays('Cookiezi')
        self.assertEqual(len(cookiezi_plays), 1)
        self.assertEqual(cookiezi_plays[0]['map_title'], 'Test Queries')
        self.assertEqual(cookiezi_plays[0]['accuracy_percent'], 99.2)

        # Test map leaderboard
        # Test map leaderboard
        leaderboard = self.db.get_map_leaderboard('query_hash_abc')
        self.assertEqual(len(leaderboard), 2)
        # Should be ordered by accuracy DESC
        self.assertEqual(leaderboard[0]['player'], 'WhiteCat')
        self.assertEqual(leaderboard[1]['player'], 'Cookiezi')

        # Test get_map_portfolio
        portfolio = self.db.get_map_portfolio('query_hash_abc')
        self.assertIsNotNone(portfolio)
        self.assertIn('SnapAim', portfolio)

    def test_api_scores_cache(self):
        top_mock = [{'title': 'Top Play 1', 'pp': 500}]
        recent_mock = [{'title': 'Recent Failure', 'pp': None}]

        # Save cache
        success = self.db.save_api_scores('YouLikeCats', top_mock, recent_mock)
        self.assertTrue(success)

        # Retrieve cache (should be valid)
        cache = self.db.get_cached_api_scores('YouLikeCats')
        self.assertIsNotNone(cache)
        retrieved_top, retrieved_recent = cache
        self.assertEqual(retrieved_top[0]['title'], 'Top Play 1')
        self.assertEqual(retrieved_recent[0]['title'], 'Recent Failure')

        # Retrieve cache with case insensitivity
        cache_case = self.db.get_cached_api_scores('youlikecats')
        self.assertIsNotNone(cache_case)
