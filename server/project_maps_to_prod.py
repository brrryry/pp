import os
import sys
import json
import pickle
import logging
import argparse
import numpy as np
from urllib.parse import urlparse

try:
    import psycopg2
    from psycopg2 import extras
except ImportError:
    print("Error: psycopg2 is required to run this script. Run 'pip install psycopg2-binary'")
    sys.exit(1)

try:
    import umap
except ImportError:
    umap = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

logging.basicConfig(level=logging.INFO, format="[Project Maps] %(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ProjectMaps")

def get_db_url(env_target: str = "prod", host: str = None, port: int = None) -> str:
    """
    Constructs PostgreSQL connection URL for 'dev' or 'prod' environment using credentials from server/.env.
    """
    env_paths = [
        os.path.join(SCRIPT_DIR, ".env"),
        os.path.join(PARENT_DIR, "server", ".env"),
        os.path.join(PARENT_DIR, ".env"),
        os.path.join(os.getcwd(), "server", ".env"),
        os.path.join(os.getcwd(), ".env")
    ]
    
    env_db_url = None
    env_password = None
    env_user = None
    env_dbname = None

    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DATABASE_URL="):
                        env_db_url = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("POSTGRES_PASSWORD="):
                        env_password = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("POSTGRES_USER="):
                        env_user = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("POSTGRES_DB="):
                        env_dbname = line.split("=", 1)[1].strip().strip('"').strip("'")

    user = env_user or "ppuser"
    password = env_password or "pppassword"
    dbname = env_dbname or "ppdb"

    if env_db_url:
        try:
            parsed = urlparse(env_db_url)
            if parsed.username:
                user = parsed.username
            if parsed.password:
                password = parsed.password
            if parsed.path and parsed.path != '/':
                dbname = parsed.path.lstrip('/')
        except Exception as e:
            logger.warning(f"Failed to parse DATABASE_URL from .env: {e}")

    if env_target.lower() == "dev":
        target_host = host or "localhost"
        target_port = port or 5433
    else:
        target_host = host or "bryan-nas.gkhomenetwork.lan"
        target_port = port or 5433

    return f"postgresql://{user}:{password}@{target_host}:{target_port}/{dbname}"

def project_all_maps(pg_url: str, model_folder: str = "models", embedding_size: int = 128):
    logger.info(f"Connecting to PostgreSQL at '{pg_url}'...")

    try:
        conn = psycopg2.connect(pg_url)
        cursor = conn.cursor()
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL at '{pg_url}': {e}")
        sys.exit(1)

    try:
        # 1. Fetch all maps with non-empty embeddings from PostgreSQL
        logger.info("Fetching map embeddings from database...")
        cursor.execute("SELECT map_hash, embed FROM maps WHERE embed IS NOT NULL AND embed != ''")
        rows = cursor.fetchall()

        if not rows:
            logger.warning("No maps with embeddings found in database.")
            return

        hashes = []
        embeddings = []

        for h, embed_raw in rows:
            if not embed_raw:
                continue
            try:
                if isinstance(embed_raw, str):
                    emb_list = json.loads(embed_raw)
                elif isinstance(embed_raw, (list, tuple)):
                    emb_list = embed_raw
                else:
                    emb_list = None

                if emb_list:
                    emb = np.array(emb_list, dtype=np.float32)
                    if emb.shape == (embedding_size,):
                        hashes.append(h)
                        embeddings.append(emb)
            except Exception as e:
                logger.debug(f"Failed to parse embedding for map_hash {h}: {e}")

        if len(embeddings) == 0:
            logger.warning("No valid embeddings of expected dimension found.")
            return

        X = np.array(embeddings, dtype=np.float32)
        n_samples = len(X)
        logger.info(f"Found {n_samples} valid map embeddings.")

        # 2. Fit UMAP (or PCA fallback)
        reducer = None
        use_umap = False
        
        if n_samples >= 15 and umap is not None:
            try:
                n_neighbors = min(30, n_samples - 1)
                logger.info(f"Fitting UMAP on {n_samples} maps (n_neighbors={n_neighbors}, metric='cosine')...")
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42,
                    verbose=True
                )
                coords = reducer.fit_transform(X)
                use_umap = True
            except Exception as e:
                logger.warning(f"UMAP fit failed: {e}. Falling back to PCA.")

        if not use_umap and PCA is not None:
            n_components = min(2, n_samples)
            logger.info(f"Fitting PCA fallback reducer on {n_samples} maps...")
            reducer = PCA(n_components=n_components, random_state=42)
            coords = reducer.fit_transform(X)

        # 3. Save the trained UMAP model
        os.makedirs(model_folder, exist_ok=True)
        umap_path = os.path.join(model_folder, "umap_model.pkl")
        projector_path = os.path.join(model_folder, "umap_projector.pkl")

        with open(umap_path, "wb") as f:
            pickle.dump(reducer, f)
        logger.info(f"Saved UMAP model to '{umap_path}'")

        with open(projector_path, "wb") as f:
            pickle.dump(reducer, f)
        logger.info(f"Saved projector model to '{projector_path}'")

        # 4. Update PostgreSQL database with the 2D coordinates
        logger.info(f"Updating 2D coordinates for {len(hashes)} maps in PostgreSQL...")
        update_data = []
        for idx, h in enumerate(hashes):
            cx = float(coords[idx][0])
            cy = float(coords[idx][1]) if coords.shape[1] > 1 else 0.0
            update_data.append((cx, cy, 1, h))

        update_sql = """
            UPDATE maps
            SET coord_x = %s, coord_y = %s, projection_version = %s
            WHERE map_hash = %s;
        """
        extras.execute_batch(cursor, update_sql, update_data, page_size=2000)
        conn.commit()

        logger.info(f"🎉 Successfully projected {len(hashes)} maps to 2D coordinates in PostgreSQL!")

    except Exception as e:
        conn.rollback()
        logger.error(f"Map projection failed due to error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project map embeddings to 2D coordinates in PostgreSQL (Dev or Prod)")
    parser.add_argument("--env", "--target", choices=["dev", "prod"], default="prod", help="Target environment: 'dev' (localhost:5433) or 'prod' (bryan-nas.gkhomenetwork.lan:5433) (default: prod)")
    parser.add_argument("--pg-url", default=None, help="Custom PostgreSQL connection URL (overrides --env setting)")
    parser.add_argument("--model-folder", default="models", help="Folder to save trained UMAP model (default: models)")
    parser.add_argument("--embedding-size", type=int, default=128, help="Embedding dimension (default: 128)")
    args = parser.parse_args()

    target_pg_url = args.pg_url if args.pg_url else get_db_url(env_target=args.env)
    project_all_maps(target_pg_url, args.model_folder, args.embedding_size)
