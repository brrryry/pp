import os
import shutil
from osu import Client
import threading
from dataclasses import dataclass
import requests
import zipfile
from io import BytesIO
import logging
# pyrefly: ignore [missing-import]
from redis import Redis
# pyrefly: ignore [missing-import]
from rq import Queue, Retry
import uuid
import time
import hashlib
import torch

import config
from core.DatabaseManager import DatabaseManager
from core.ml_parser import osu_to_ml_sequence
from core.Embedder import CNNEmbedder, MapVAE
from core.Projector import MapProjector

logging.basicConfig(level=logging.INFO,
format='[BeatmapIngestor] %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class IngestionJob:
    mapset_id: int

class BeatmapIngestor:
    def __init__(self, db_manager, embed_model, osu_client, mirrors,
                 redis_url=None,
                 max_concurrent=3, requests_per_second=20):
        self.db_manager = db_manager
        self.embed_model = embed_model
        self.osu_client = osu_client
        self.mirrors = mirrors
        if redis_url is None:
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        print(f"BeatmapIngestor using Redis URL: {redis_url}")
        self.redis = Redis.from_url(redis_url)
        self.queue = Queue('beatmap_ingestion', connection=self.redis, default_timeout=600)
        self.max_concurrent = max_concurrent
        self.requests_per_second = requests_per_second

    def ingest_mapset(self, mapset_id):
        # Deduplication lock
        lock_key = f"lock:beatmap:{mapset_id}"
        if not self.redis.set(lock_key, "1", nx=True, ex=300):
            logger.info(f"Mapset {mapset_id} already being processed.")
            return None

        try:
            # Double-check DB
            if self.db_manager.find_mapset_by_id(mapset_id):
                logger.info(f"Mapset {mapset_id} already exists in DB.")
                self.redis.hset(f"job:beatmap_{mapset_id}", "status", "ready")
                self.redis.publish("beatmap_events", f"ready:{mapset_id}")
                return f"beatmap_{mapset_id}"

            job_id = f"beatmap_{mapset_id}_{uuid.uuid4().hex[:8]}"
            job = self.queue.enqueue(
                self._process_job,
                mapset_id,
                job_id=job_id,
                job_timeout=600,
                retry=Retry(max=3, interval=[10, 30, 60]),
                description=f"Ingest mapset {mapset_id}"
            )
            # Store initial status
            self.redis.hset(f"job:{job.id}", mapping={
                "status": "queued",
                "mapset_id": mapset_id,
                "created_at": time.time()
            })
            self.redis.expire(f"job:{job.id}", 86400)
            logger.info(f"Enqueued job {job.id} for mapset {mapset_id}")
        except Exception as e:
            self.redis.delete(lock_key)
            raise

    @staticmethod
    def _process_job(mapset_id):
        db = DatabaseManager(db_target=config.DB_FILE)
        embedder = MapVAE(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            embedding_size=config.EMBEDDING_SIZE,
            dropout=config.DROPOUT
        )
        if os.path.exists(config.EMBEDDER_FILE_PATH):
            embedder.load_state_dict(torch.load(config.EMBEDDER_FILE_PATH, map_location=torch.device('cpu')))
        embedder.eval()

        try:
            redis_client = Redis.from_url(config.REDIS_URL)
            redis_client.ping()
        except Exception:
            redis_client = Redis.from_url("redis://localhost:6379/0")

        mirrors = config.MIRRORS
        status_key = f"job:beatmap_{mapset_id}"
        redis_client.hset(status_key, "status", "downloading")

        try:
            for mirror_name, mirror_url in mirrors:
                download_url = f"{mirror_url}{mapset_id}"
                try:
                    response = requests.get(download_url, timeout=15)
                except Exception as req_err:
                    logger.warning(f"Download request failed for mirror {mirror_name}: {req_err}")
                    continue

                if response.status_code == 200:
                    try:
                        mapset_dir = os.path.join(config.MAPS_PATH, str(mapset_id))
                        os.makedirs(mapset_dir, exist_ok=True)

                        with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
                            osu_files = [f for f in zip_file.namelist() if f.endswith('.osu')]
                            if not osu_files:
                                continue

                            for osu_filename in osu_files:
                                osu_bytes = zip_file.read(osu_filename)
                                map_hash = hashlib.md5(osu_bytes).hexdigest()

                                temp_map_path = os.path.join(mapset_dir, os.path.basename(osu_filename))
                                with open(temp_map_path, "wb") as f_out:
                                    f_out.write(osu_bytes)

                                res = osu_to_ml_sequence(temp_map_path)
                                seq = res[0] if isinstance(res, tuple) else res
                                seq_tensor = seq.unsqueeze(0) if seq.dim() == 2 else seq
                                if seq_tensor.size(2) < config.input_size:
                                    pad = torch.zeros((seq_tensor.size(0), seq_tensor.size(1), config.input_size - seq_tensor.size(2)), dtype=torch.float32)
                                    seq_tensor = torch.cat([seq_tensor, pad], dim=-1)

                                seq_len = torch.tensor([seq_tensor.size(1)], dtype=torch.long)
                                with torch.no_grad():
                                    mu, _ = embedder.encoder(seq_tensor, seq_len)
                                    embed = mu.squeeze(0).numpy()

                                projector = MapProjector()
                                coord_x, coord_y = projector.transform(embed)

                                title, artist, creator, version = "Unknown Title", "Unknown Artist", "Unknown Creator", "Normal"
                                parsed_map_id = None
                                for line in osu_bytes.decode('utf-8', errors='ignore').splitlines():
                                    line_s = line.strip()
                                    if line_s.startswith("Title:"):
                                        title = line_s[6:].strip() or title
                                    elif line_s.startswith("Artist:"):
                                        artist = line_s[7:].strip() or artist
                                    elif line_s.startswith("Creator:"):
                                        creator = line_s[8:].strip() or creator
                                    elif line_s.startswith("Version:"):
                                        version = line_s[8:].strip() or version
                                    elif line_s.startswith("BeatmapID:"):
                                        try:
                                            parsed_map_id = int(line_s[10:].strip())
                                        except ValueError:
                                            pass

                                db.add_map(
                                    map_id=parsed_map_id,
                                    mapset_id=mapset_id,
                                    map_hash=map_hash,
                                    embed=embed,
                                    title=title,
                                    artist=artist,
                                    creator=creator,
                                    version=version,
                                    coord_x=coord_x,
                                    coord_y=coord_y
                                )

                                if map_hash:
                                    redis_client.set(f"beatmap:hash:{map_hash}", "ready", ex=3600)
                                    redis_client.publish("beatmap_hash_events", f"ready:{map_hash}")

                        redis_client.hset(status_key, "status", "ready")
                        redis_client.publish("beatmap_events", f"ready:{mapset_id}")
                        logger.info(f"Successfully processed mapset {mapset_id}")
                        shutil.rmtree(mapset_dir, ignore_errors=True)
                        return
                    except Exception as e:
                        logger.error(f"Failed to process mapset {mapset_id}: {e}")
                        continue
                else:
                    logger.warning(f"Failed to download mapset {mapset_id} from mirror {mirror_name}: {response.status_code}")
                    continue

            redis_client.hset(status_key, mapping={"status": "failed", "error": "All mirrors failed"})
            raise Exception("All mirrors failed")
        except Exception as e:
            redis_client.hset(status_key, mapping={"status": "failed", "error": str(e)})
            raise