import os
import zipfile
import logging
import requests
import hashlib
from filelock import FileLock, Timeout  # Installation: pip install filelock
import pandas as pd

import config
from core.parser import parse_osu_file
from core.features import extract_map_features

logging.basicConfig(level=logging.INFO, format="[Ingestion Pipeline] %(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
mirrors = config.mirrors

class BeatmapIngestionPipeline:
    def __init__(self, dbManager, maps_dir):
        self.dbManager = dbManager
        self.maps_dir = maps_dir

    def __download_mapset(self, mapset_id):
        """Downloads the mapset .osz file safely and atomically using a temporary file."""
        osz_path = os.path.join(self.maps_dir, f"{mapset_id}.osz")
        
        for mirror_name, mirror_url in mirrors:
            try:
                logger.info(f"Attempting to download beatmapset {mapset_id} from {mirror_name}...")
                response = requests.get(f"{mirror_url}{mapset_id}", timeout=10)
                
                if response.status_code == 200:
                    tmp_osz_path = f"{osz_path}.tmp"
                    with open(tmp_osz_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Atomic file swap prevents concurrent readers from opening a partial download
                    os.replace(tmp_osz_path, osz_path)
                    logger.info(f"Successfully downloaded beatmapset {mapset_id} from {mirror_name}.")
                    return True
                else:
                    logger.warning(f"Failed to download from {mirror_name}: HTTP {response.status_code}")
            except requests.RequestException as e:
                logger.error(f"Error downloading from {mirror_name}: {e}")
        
        logger.error(f"All mirrors failed for beatmapset {mapset_id}.")
        return False

    def __extract_and_parse_maps(self, mapset_id):
        """Extracts .osu files, renames them by MD5 hash, parses them, and saves to the DB."""
        osz_path = os.path.join(self.maps_dir, f"{mapset_id}.osz")
        if not os.path.exists(osz_path):
            logger.error(f"OSZ file for mapset {mapset_id} does not exist at {osz_path}.")
            return 0

        logger.info(f"Extracting maps from {osz_path}...")
        count = 0
        
        try:
            with zipfile.ZipFile(osz_path) as zip_file:
                osu_files = [f for f in zip_file.namelist() if f.endswith('.osu')]
                if not osu_files:
                    logger.warning(f"No .osu files found in archive: {osz_path}")
                    return 0

                for osu_file in osu_files:
                    content = zip_file.read(osu_file)
                    
                    # Compute MD5 hash of raw bytes (aligns natively with .osr replay structures)
                    file_hash = hashlib.md5(content).hexdigest()
                    new_filename = f"{file_hash}.osu"
                    new_path = os.path.join(self.maps_dir, new_filename)
                    
                    # Deduplication: Skip writing if the file already exists on disk
                    if not os.path.exists(new_path):
                        tmp_path = f"{new_path}.tmp"
                        with open(tmp_path, 'wb') as f:
                            f.write(content)
                        os.replace(tmp_path, new_path)
                    
                    # Parse and extract properties using imported parser
                    parsed = parse_osu_file(new_path)
                    if parsed is None:
                        logger.warning(f"Failed to parse {new_filename}. Skipping.")
                        continue
                    
                    extracted_features = extract_map_features(parsed)
                    if extracted_features is None:
                        logger.warning(f"Failed to extract features from {new_filename}. Skipping.")
                        continue
                    
                    # Inject mapping data into the features dictionary
                    extracted_features['file_hash'] = file_hash
                    extracted_features['mapset_id'] = mapset_id
                    
                    # Push to DB (Ensure dbManager handles write locks/retries internally)
                    self.dbManager.add_map(extracted_features)
                    count += 1
                    
            logger.info(f"Extracted and processed {count} maps from {osz_path}.")
            return count
            
        except zipfile.BadZipFile:
            logger.error(f"Bad ZIP file: {osz_path}")
            return 0
        except Exception as e:
            logger.error(f"Error extracting {osz_path}: {e}")
            return 0

    def run(self, mapset_id, downloaded=False):
        """Execution wrapper managing cross-process safety hooks and checkpoints."""
        lock_path = os.path.join(self.maps_dir, f"{mapset_id}.lock")
        lock = FileLock(lock_path, timeout=60)  # 60-second timeout
        try:
            # Prevent multiple instances from working on the same mapset simultaneously
            with lock:
                # Double-check database after acquiring lock to see if a prior worker completed it
                if not self.dbManager.find_mapset_by_id(mapset_id):
                    if self.__download_mapset(mapset_id):
                        self.__extract_and_parse_maps(mapset_id)
                        logger.info(f"Successfully ingested mapset {mapset_id}.")
                    else:
                        logger.error(f"Failed to download mapset {mapset_id}.")
                        raise Exception(f"Failed to download mapset {mapset_id}.")
                else:
                    logger.info(f"Mapset {mapset_id} already exists in the database.")
        except Timeout:
            logger.error(f"Pipeline timed out waiting for lock on mapset {mapset_id}.")
        finally:
            # Clean up the lock file safely once worker context clears
            try:
                os.remove(lock_path)
            except OSError:
                pass

    def _process_single_osu_file(self, args):
        """
        Worker function executed in parallel. 
        Accepts a tuple of arguments to handle tracking and processing.
        """
        filename, maps_dir = args
        if not filename.endswith(".osu"):
            return None
            
        file_path = os.path.join(maps_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            parsed = parse_osu_file(file_path)
            if parsed is None:
                logger.warning(f"Failed to parse {file_path}. Skipping.")
                return None
                
            extracted_features = extract_map_features(parsed)
            extracted_features['map_hash'] = hashlib.md5(content.encode()).hexdigest()
            extracted_features['star_rating'] = parsed["difficulty"]["star_rating"]
            return extracted_features
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None


    # Inside your main class:
    def bulk_ingest(self):
        from tqdm import tqdm
        import multiprocessing
        
        # 1. Gather files and respect your 10-file testing cap
        all_files = os.listdir(self.maps_dir)
        # Remove or comment out the next line when moving to production
        #all_files = all_files[:10] 
        
        # 2. Prepare arguments for the worker pool (filename, directory)
        tasks = [(filename, self.maps_dir) for filename in all_files]
        results = []

        # 3. Initialize Pool and map tasks using imap_unordered for a live progress bar
        # (Using os.cpu_count() - 1 leaves a core free for OS tasks)
        num_workers = max(1, (os.cpu_count() or 2) - 1)
        
        logger.info(f"Starting parallel ingestion with {num_workers} workers...")
        with multiprocessing.Pool(processes=num_workers) as pool:
            # pool.imap_unordered yields results instantly as they finish
            for res in tqdm(
                pool.imap_unordered(self._process_single_osu_file, tasks), 
                total=len(tasks), 
                desc="Ingesting maps"
            ):
                if res is not None:
                    results.append(res)

        # 4. Construct DataFrame efficiently all at once (avoids slow ._append loops)
        if results:
            df = pd.DataFrame(results)
            self.dbManager.bulk_add_maps_from_df(df)
            logger.info(f"Successfully bulk ingested {len(df)} maps.")
        else:
            logger.warning("No valid maps were parsed.")
        
        def parse_osu_file(self, file_path):
            return parse_osu_file(file_path)
                    
        
