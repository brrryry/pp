"""
Model Training Configuration Module.
Centralized environment variables and model hyperparameter configuration.
"""

import os
from dotenv import load_dotenv

# Load .env file if available
try:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data & Logging Paths
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
MAPS_DIR = os.environ.get("MAPS_DIR", os.path.join(DATA_DIR, "maps"))
REPLAYS_DIR = os.environ.get("REPLAYS_DIR", os.path.join(DATA_DIR, "replays"))
LOGS_DIR = os.environ.get("LOGS_DIR", os.path.join(DATA_DIR, "logs"))
DB_FILE = os.environ.get("DB_FILE") or os.environ.get("SQLITE_DB_PATH") or os.path.join(DATA_DIR, "pp.db")

# Model Storage Paths
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", os.path.join(MODELS_DIR, "map_embedder"))
ALS_MODEL_PATH = os.environ.get("ALS_MODEL_PATH", os.path.join(MODELS_DIR, "als_model"))
UMAP_MODEL_PATH = os.environ.get("UMAP_MODEL_PATH", os.path.join(MODELS_DIR, "umap_model"))

# osu! API v2 Credentials
OSU_API_CLIENT_ID = int(os.environ.get("OSU_API_CLIENT_ID") or os.environ.get("OSU_CLIENT_ID") or 0)
OSU_API_CLIENT_SECRET = os.environ.get("OSU_API_CLIENT_SECRET") or os.environ.get("OSU_CLIENT_SECRET") or ""
OSU_API_REDIRECT_URI = os.environ.get("OSU_API_REDIRECT_URI") or os.environ.get("OSU_REDIRECT_URI") or "http://localhost:4000"

# Beatmap Mirrors
MIRRORS = [
    ("Nerinyan", "https://api.nerinyan.moe/d/"),
    ("OsuDirect", "https://osu.direct/api/d/"),
    ("Sayobot", "https://txy1.sayobot.cn/beatmaps/download/full/")
]

# Neural Network & Training Hyperparameters
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", 9))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 128))
EMBEDDING_SIZE = int(os.environ.get("EMBEDDING_SIZE", 32))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
EPOCHS = int(os.environ.get("EPOCHS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))

# Backward Compatibility Exports
default_db_file = DB_FILE
db_file = DB_FILE
osu_api_client_id = OSU_API_CLIENT_ID
osu_api_client_secret = OSU_API_CLIENT_SECRET
osu_api_redirect_uri = OSU_API_REDIRECT_URI
mirrors = MIRRORS
embedding_model_path = EMBEDDING_MODEL_PATH
als_model_path = ALS_MODEL_PATH
umap_model_path = UMAP_MODEL_PATH