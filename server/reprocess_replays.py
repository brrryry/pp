import os
import sys
import pandas as pd
from main import process_replay_data, get_beatmap_path, scan_local_maps, REPLAYS_DIR, SUMMARY_CSV
from osrparse import Replay
from logger_setup import setup_logger

logger = setup_logger("reprocess_replays")

def print(*args, **kwargs):
    import builtins
    builtins.print(*args, **kwargs)
    msg = " ".join(str(arg) for arg in args)
    logger.info(msg)

def main():
    print("Scanning local beatmaps...")
    scan_local_maps()
    
    if not os.path.exists(REPLAYS_DIR):
        print(f"Error: Replays directory '{REPLAYS_DIR}' does not exist.")
        return
        
    osr_files = [f for f in os.listdir(REPLAYS_DIR) if f.endswith(".osr")]
    print(f"Found {len(osr_files)} replays in database to reprocess.")
    
    results = []
    
    for i, filename in enumerate(osr_files, 1):
        print(f"[{i}/{len(osr_files)}] Reprocessing {filename}...")
        replay_path = os.path.join(REPLAYS_DIR, filename)
        try:
            replay = Replay.from_path(replay_path)
            h = replay.beatmap_hash
            osu_path = get_beatmap_path(h)
            
            if not osu_path or not os.path.exists(osu_path):
                print(f"  Warning: Map hash {h} not found locally. Skipping.")
                continue
                
            analysis = process_replay_data(replay_path, osu_path)
            if analysis:
                results.append(analysis)
                print(f"  Success: Acc = {analysis['accuracy_percent']:.2f}% | UR = {analysis['unstable_rate']:.1f}")
            else:
                print(f"  Failed: process_replay_data returned None.")
        except Exception as e:
            print(f"  Error: {e}")
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv(SUMMARY_CSV, index=False)
        print(f"\n[SUCCESS] Successfully reprocessed {len(results)} replays and updated {SUMMARY_CSV}!")
    else:
        print("\nNo replays were successfully reprocessed.")

if __name__ == "__main__":
    main()
