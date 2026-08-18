import os
import sys
import unittest
import uuid
from unittest.mock import MagicMock, patch

# Ensure root server directory is in sys.path
SERVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from osrparse import Replay
import config
from core.DatabaseManager import DatabaseManager
from core.ReplayIngestor import ReplayIngestor, calculate_mastery, process_replay_file_job


class TestReplayIngestion(unittest.TestCase):

    def setUp(self):
        # Use isolated test DB file
        self.test_db_file = os.path.join(SERVER_DIR, "data", "test_replay_ingestion.db")
        if os.path.exists(self.test_db_file):
            try:
                os.remove(self.test_db_file)
            except OSError:
                pass

        self.db = DatabaseManager(self.test_db_file)
        self.replay_file = os.path.join(SERVER_DIR, "new_replay.osr")

        # Parse test replay details
        self.assertTrue(os.path.exists(self.replay_file), f"Test replay file {self.replay_file} does not exist")
        self.parsed_replay = Replay.from_path(self.replay_file)
        self.replay_hash = self.parsed_replay.replay_hash
        self.map_hash = self.parsed_replay.beatmap_hash
        self.username = self.parsed_replay.username

    def tearDown(self):
        # Clean up database after tests
        if os.path.exists(self.test_db_file):
            try:
                os.remove(self.test_db_file)
            except OSError:
                pass

    def test_add_and_delete_replay_and_map_database(self):
        """Test adding a map and replay to DatabaseManager, verifying existence, and deleting both."""
        # 1. Ensure replay and map do not exist initially
        self.assertFalse(self.db.find_replay_by_hash(self.replay_hash))
        self.assertFalse(self.db.find_map_by_md5(self.map_hash))

        # 2. Add map to DB
        map_id = self.db.add_map(mapset_id=12345, map_hash=self.map_hash, title="Test Song")
        self.assertIsNotNone(map_id)
        self.assertTrue(self.db.find_map_by_md5(self.map_hash))

        # 3. Add replay to DB
        mods = self.parsed_replay.mods.value if hasattr(self.parsed_replay.mods, 'value') else int(self.parsed_replay.mods)
        misses = self.parsed_replay.count_miss
        total = self.parsed_replay.count_300 + self.parsed_replay.count_100 + self.parsed_replay.count_50 + misses
        accuracy = (300 * self.parsed_replay.count_300 + 100 * self.parsed_replay.count_100 + 50 * self.parsed_replay.count_50) / (300 * total) if total > 0 else 0.0
        max_combo = self.parsed_replay.max_combo
        mastery = calculate_mastery(accuracy, misses, total, max_combo)

        self.db.add_replay(
            username=self.username,
            replay_hash=self.replay_hash,
            map_hash=self.map_hash,
            mods=mods,
            accuracy=accuracy,
            misses=misses,
            max_combo=max_combo,
            mastery_score=mastery
        )

        # 4. Verify replay was added
        self.assertTrue(self.db.find_replay_by_hash(self.replay_hash))

        # 5. Delete replay and map from DB
        removed_replay = self.db.remove_replay(self.replay_hash)
        self.assertTrue(removed_replay)
        self.assertFalse(self.db.find_replay_by_hash(self.replay_hash))

        removed_map = self.db.remove_map_by_hash(self.map_hash)
        self.assertTrue(removed_map)
        self.assertFalse(self.db.find_map_by_md5(self.map_hash))

    def test_replay_ingestor_enqueue(self):
        """Test ReplayIngestor enqueuing a replay job using mock Redis."""
        mock_redis = MagicMock()
        mock_queue = MagicMock()

        mock_beatmap_ingestor = MagicMock()
        mock_recommendation_engine = MagicMock()

        ingestor = ReplayIngestor(
            db_manager=self.db,
            beatmap_ingestor=mock_beatmap_ingestor,
            recommendation_engine=mock_recommendation_engine
        )
        ingestor.redis = mock_redis
        ingestor.queue = mock_queue

        # Try to ingest replay
        job_id = ingestor.ingest_replay(self.replay_file)
        self.assertTrue(job_id.startswith("replay_"))

        # Verify job was enqueued and status set in Redis
        mock_queue.enqueue.assert_called_once()
        mock_redis.hset.assert_called_once()
        mock_redis.expire.assert_called_once()

    def test_process_replay_job_existing_beatmap(self):
        """Test running process_replay_file_job for an existing beatmap, verifying DB insertion and deletion of both replay and map."""
        mock_redis = MagicMock()
        mock_osu_client = MagicMock()
        mock_beatmap_ingestor = MagicMock()
        mock_recommendation_engine = MagicMock()

        # Add map first
        self.db.add_map(mapset_id=12345, map_hash=self.map_hash)

        test_worker_state = {
            'db': self.db,
            'redis': mock_redis,
            'osu_client': mock_osu_client,
            'beatmap_ingestor': mock_beatmap_ingestor,
            'recommendation_engine': mock_recommendation_engine
        }

        job_id = f"replay_{uuid.uuid4().hex[:8]}"

        with patch('core.ReplayIngestor.get_worker_state', return_value=test_worker_state):
            process_replay_file_job(self.replay_file, job_id)

        # Verify replay was inserted into test DB
        self.assertTrue(self.db.find_replay_by_hash(self.replay_hash))

        # Delete the ingested replay and map
        self.assertTrue(self.db.remove_replay(self.replay_hash))
        self.assertFalse(self.db.find_replay_by_hash(self.replay_hash))

        self.assertTrue(self.db.delete_mapset_by_id(12345))
        self.assertFalse(self.db.find_map_by_md5(self.map_hash))

    def test_process_replay_job_missing_beatmap_ingestion(self):
        """Test process_replay_file_job when beatmap is NOT in DB: resolves mapset_id, triggers beatmap_ingestor, inserts replay/map, and cleans up both."""
        mock_redis = MagicMock()
        mock_osu_client = MagicMock()
        mock_beatmap_ingestor = MagicMock()
        mock_recommendation_engine = MagicMock()

        test_worker_state = {
            'db': self.db,
            'redis': mock_redis,
            'osu_client': mock_osu_client,
            'beatmap_ingestor': mock_beatmap_ingestor,
            'recommendation_engine': mock_recommendation_engine
        }

        job_id = f"replay_{uuid.uuid4().hex[:8]}"

        # Mock beatmap NOT being in DB initially
        with patch('core.ReplayIngestor.get_worker_state', return_value=test_worker_state), \
             patch('core.ReplayIngestor._resolve_mapset_id', return_value=88888) as mock_resolve, \
             patch('core.ReplayIngestor.wait_for_beatmap', return_value=True) as mock_wait:

            process_replay_file_job(self.replay_file, job_id)

            # Verify mapset_id was resolved via API
            mock_resolve.assert_called_once_with(mock_osu_client, self.map_hash)

            # Verify beatmap ingestion was triggered with resolved mapset_id
            mock_beatmap_ingestor.ingest_mapset.assert_called_once_with(88888)

            # Verify worker waited for beatmap ready signal
            mock_wait.assert_called_once_with(mock_redis, 88888, timeout=120)

        # Verify replay was inserted into test DB
        self.assertTrue(self.db.find_replay_by_hash(self.replay_hash))

        # Add mock map and test deleting map by hash and mapset_id
        self.db.add_map(mapset_id=88888, map_hash=self.map_hash)
        self.assertTrue(self.db.find_map_by_md5(self.map_hash))

        # Delete replay
        self.assertTrue(self.db.remove_replay(self.replay_hash))
        self.assertFalse(self.db.find_replay_by_hash(self.replay_hash))

        # Delete map
        self.assertTrue(self.db.delete_mapset_by_id(88888))
        self.assertFalse(self.db.find_map_by_md5(self.map_hash))


if __name__ == '__main__':
    unittest.main()
