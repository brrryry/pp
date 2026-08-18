import os
import sys
import json
import shutil
import unittest
from datetime import datetime

# Add project directories to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MODEL_TRAINING_DIR = os.path.join(PROJECT_ROOT, "model_training")
SERVER_DIR = os.path.join(PROJECT_ROOT, "server")

if MODEL_TRAINING_DIR not in sys.path:
    sys.path.insert(0, MODEL_TRAINING_DIR)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from run_pipeline import evaluate_quality_gate, archive_model_artifacts, run_pipeline

class TestRetrainPipeline(unittest.TestCase):

    def test_quality_gate(self):
        """Test Quality Gate validation logic with passing and failing losses."""
        self.assertTrue(evaluate_quality_gate(val_loss=0.10, max_val_loss=0.15))
        self.assertTrue(evaluate_quality_gate(val_loss=0.15, max_val_loss=0.15))
        self.assertFalse(evaluate_quality_gate(val_loss=0.20, max_val_loss=0.15))

    def test_archive_artifacts(self):
        """Test timestamped archiving directory creation and metrics.json contents."""
        timestamp_str = "test_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = archive_model_artifacts(
            timestamp_str=timestamp_str,
            val_loss=0.12,
            passed_gate=True,
            dry_run=False
        )

        self.assertTrue(os.path.exists(archive_dir))
        metrics_path = os.path.join(archive_dir, "metrics.json")
        self.assertTrue(os.path.exists(metrics_path))

        with open(metrics_path, "r") as f:
            data = json.load(f)
            self.assertEqual(data["timestamp"], timestamp_str)
            self.assertEqual(data["val_loss"], 0.12)
            self.assertTrue(data["quality_gate_passed"])

        # Clean up test archive directory
        shutil.rmtree(archive_dir, ignore_errors=True)

    def test_dry_run_pipeline(self):
        """Test end-to-end dry-run pipeline execution."""
        try:
            run_pipeline(
                env_target="dev",
                auto_promote=False,
                max_val_loss=0.15,
                skip_download=True,
                dry_run=True
            )
        except Exception as e:
            self.fail(f"run_pipeline dry-run threw an unexpected exception: {e}")

if __name__ == "__main__":
    unittest.main()
