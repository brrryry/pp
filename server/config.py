"""
Server Configuration Module.
Loads settings from environment variables (.env file supported) with safe, sensible defaults.
"""

import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Load .env file if python-dotenv is installed and file exists
if load_dotenv is not None:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)

# System Configuration
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"))

# Base Directory Definition
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Environment & Server Settings
ENVIRONMENT = os.environ.get("ENVIRONMENT") or os.environ.get("ENV") or os.environ.get("APP_ENV") or "development"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
SERVER_HOST = os.environ.get("SERVER_HOST") or os.environ.get("HOST") or "127.0.0.1"
SERVER_PORT = int(os.environ.get("SERVER_PORT") or os.environ.get("PORT") or 8000)

# Data & Logging Paths
MAPS_PATH = os.environ.get("MAPS_PATH", os.path.join(BASE_DIR, "data", "maps"))
REPLAYS_PATH = os.environ.get("REPLAYS_PATH", os.path.join(BASE_DIR, "data", "replays"))
LOG_PATH = os.environ.get("LOG_PATH", os.path.join(BASE_DIR, "data", "logs"))
DB_FILE = os.environ.get("DB_FILE") or os.environ.get("SQLITE_DB_PATH") or os.path.join(BASE_DIR, "data", "pp.db")

# Beatmap Download Mirrors
DEFAULT_MIRRORS = [
    ("Nerinyan", "https://api.nerinyan.moe/d/"),
    ("OsuDirect", "https://osu.direct/api/d/"),
    ("Sayobot", "https://txy1.sayobot.cn/beatmaps/download/full/")
]
MIRRORS = DEFAULT_MIRRORS

# Database & Cache Connection Services
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://ppuser:pppassword@postgres:5432/ppdb")

# osu! API v2 Credentials
OSU_API_CLIENT_ID = int(os.environ.get("OSU_API_CLIENT_ID") or os.environ.get("OSU_CLIENT_ID") or 0)
OSU_API_CLIENT_SECRET = os.environ.get("OSU_API_CLIENT_SECRET") or os.environ.get("OSU_CLIENT_SECRET") or ""
OSU_API_REDIRECT_URI = os.environ.get("OSU_API_REDIRECT_URI") or os.environ.get("OSU_REDIRECT_URI") or "http://localhost:4000"

# VAE & Recommendation Model Paths & Parameters
EMBEDDER_FILE_PATH = os.environ.get("EMBEDDER_FILE_PATH", os.path.join(BASE_DIR, "embedder", "best_vae_contrastive.pth"))
RECOMMENDATION_ENGINE_FOLDER = os.environ.get("RECOMMENDATION_ENGINE_FOLDER", os.path.join(BASE_DIR, "models"))

INPUT_SIZE = int(os.environ.get("INPUT_SIZE", 13))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 128))
EMBEDDING_SIZE = int(os.environ.get("EMBEDDING_SIZE", 128))
DROPOUT = float(os.environ.get("DROPOUT", 0.2))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
EPOCHS = int(os.environ.get("EPOCHS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))

# Backward Compatibility Aliases (Lowercase)
maps_path = MAPS_PATH
replays_path = REPLAYS_PATH
log_path = LOG_PATH
mirrors = MIRRORS
redis_url = REDIS_URL
database_url = DATABASE_URL
db_file = DB_FILE
osu_api_client_id = OSU_API_CLIENT_ID
osu_api_client_secret = OSU_API_CLIENT_SECRET
osu_api_redirect_uri = OSU_API_REDIRECT_URI
embedder_file_path = EMBEDDER_FILE_PATH
recommendation_engine_folder = RECOMMENDATION_ENGINE_FOLDER
input_size = INPUT_SIZE
hidden_size = HIDDEN_SIZE
embedding_size = EMBEDDING_SIZE
dropout = DROPOUT
server_host = SERVER_HOST
server_port = SERVER_PORT
environment = ENVIRONMENT
