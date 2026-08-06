import glob
import os
import logging
from main import ReplayWatcherHandler

# Configure logging to console and file
from logger_setup import setup_logger
logger = setup_logger("import_replays")

def print(*args, **kwargs):
    import builtins
    builtins.print(*args, **kwargs)
    msg = " ".join(str(arg) for arg in args)
    logger.info(msg)

def main():
    exports_dir = r"C:\Users\thisi\AppData\Roaming\osu\exports"
    if not os.path.exists(exports_dir):
        print(f"Error: Exports directory not found at {exports_dir}")
        return
        
    files = glob.glob(os.path.join(exports_dir, "*.osr"))
    print(f"Found {len(files)} replay files in exports directory.")
    
    # Initialize the watcher handler to trigger processing logic
    handler = ReplayWatcherHandler()
    
    for i, file_path in enumerate(files, 1):
        filename = os.path.basename(file_path)
        print(f"\n[{i}/{len(files)}] Processing {filename}...")
        try:
            handler.process_new_replay(file_path)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            
    print("\n[SUCCESS] Replay import process completed!")

if __name__ == "__main__":
    main()
