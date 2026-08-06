import os
import sys
import zipfile
import logging
import argparse
import datetime
import re
from concurrent.futures import ThreadPoolExecutor

import config



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f"{config.log_path}/ingest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    filemode='w'
)
logger = logging.getLogger(__name__)

def unzip_osz(osz_path, target_dir):
    """
    Unzips all .osu files from an .osz file and renames them to {beatmapset_id}_{idx}.osu.
    """
    filename = os.path.basename(osz_path)
    beatmapset_id = os.path.splitext(filename)[0]
    
    # Try to extract leading numbers as the beatmapset_id
    match = re.match(r'^(\d+)', beatmapset_id)
    if match:
        beatmapset_id = match.group(1)
        
    try:
        with zipfile.ZipFile(osz_path, 'r') as zip_ref:
            # Get list of .osu files
            osu_files = [f for f in zip_ref.namelist() if f.endswith('.osu')]
            if not osu_files:
                logger.warning(f"No .osu files found in archive: {filename}")
                return 0
                
            count = 0
            for idx, osu_file in enumerate(osu_files):
                content = zip_ref.read(osu_file)
                
                # Format: {beatmapset_id}_{idx}.osu
                new_filename = f"{beatmapset_id}_{idx}.osu"
                new_path = os.path.join(target_dir, new_filename)
                
                # Write to target directory
                with open(new_path, 'wb') as f:
                    f.write(content)
                count += 1
                
            logger.info(f"Extracted {count} maps from {filename} -> {beatmapset_id}_*.osu")
            return count
    except zipfile.BadZipFile:
        logger.error(f"Bad ZIP file: {filename}")
        return 0
    except Exception as e:
        logger.error(f"Error extracting {filename}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Ingest local .osz files into the maps directory.")
    parser.add_argument(
        "--source_dir", 
        type=str, 
        default="D:/pp/maps", 
        help="Directory containing .osz files."
    )
    parser.add_argument(
        "--target_dir", 
        type=str, 
        default=None, 
        help="Target directory to place extracted .osu files. Defaults to config.maps_path."
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=4, 
        help="Number of worker threads for parallel extraction."
    )
    args = parser.parse_args()
    
    source_dir = args.source_dir
    target_dir = args.target_dir if args.target_dir else config.maps_path
    
    if not os.path.exists(source_dir):
        logger.error(f"Source directory '{source_dir}' does not exist.")
        sys.exit(1)
        
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Source Directory: {source_dir}")
    logger.info(f"Target Directory: {target_dir}")
    
    # List all .osz files
    osz_files = [
        os.path.join(source_dir, f) 
        for f in os.listdir(source_dir) 
        if f.endswith('.osz')
    ]
    
    total_files = len(osz_files)
    if total_files == 0:
        logger.warning(f"No .osz files found in '{source_dir}'.")
        return
        
    logger.info(f"Found {total_files} .osz files to process.")
    
    # Try importing tqdm for a progress bar
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        
    total_extracted = 0
    
    if has_tqdm:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(unzip_osz, osz_path, target_dir) 
                for osz_path in osz_files
            ]
            for future in tqdm(futures, desc="Extracting maps"):
                total_extracted += future.result()
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(unzip_osz, osz_path, target_dir) 
                for osz_path in osz_files
            ]
            for idx, future in enumerate(futures):
                total_extracted += future.result()
                if (idx + 1) % 50 == 0 or idx + 1 == total_files:
                    logger.info(f"Processed {idx + 1}/{total_files} files...")
                    
    logger.info(f"Successfully finished unzipping. Extracted {total_extracted} .osu map files to '{target_dir}'.")

if __name__ == "__main__":
    main()