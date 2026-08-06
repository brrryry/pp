import os
import sys
import time
import json
import shutil
import hashlib
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from osrparse import Replay

# Load client config
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "server_url": "http://127.0.0.1:8000",
    "osu_path": "",  # Auto-detected if blank
    "songs_path": "",  # Auto-detected if blank
    "watch_lazer": True,
    "watch_stable": True
}

config = DEFAULT_CONFIG.copy()
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            config.update(json.load(f))
    except Exception as e:
        print(f"Error loading config.json, using defaults: {e}")
else:
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
    except Exception as e:
        print(f"Could not save default config.json: {e}")

SERVER_URL = config["server_url"]

# Local maps indexing
local_songs_cache = {}
SONGS_CACHE_FILE = "songs_cache.json"

def compute_md5(file_path):
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception:
        return None

def scan_songs_folder(songs_path):
    global local_songs_cache
    local_songs_cache = {}
    
    if os.path.exists(SONGS_CACHE_FILE):
        try:
            with open(SONGS_CACHE_FILE, "r") as f:
                local_songs_cache = json.load(f)
            print(f"Loaded {len(local_songs_cache)} cached local song paths.")
        except Exception:
            pass
            
    if not os.path.exists(songs_path):
        print(f"Songs path not found: {songs_path}")
        return
        
    print(f"Scanning Songs directory for new beatmaps: {songs_path}...")
    new_maps = 0
    # Walk and index .osu files
    for root, _, files in os.walk(songs_path):
        for file in files:
            if file.endswith('.osu'):
                full_path = os.path.join(root, file)
                # Quick stats verification
                try:
                    mtime = os.path.getmtime(full_path)
                    size = os.path.getsize(full_path)
                except Exception:
                    continue
                
                # Check cache validity
                cache_entry = local_songs_cache.get(full_path)
                if cache_entry and cache_entry.get('mtime') == mtime and cache_entry.get('size') == size:
                    continue
                    
                h = compute_md5(full_path)
                if h:
                    local_songs_cache[h] = full_path
                    # Also store by path for cache check
                    local_songs_cache[full_path] = {'mtime': mtime, 'size': size}
                    new_maps += 1
                    
    if new_maps > 0:
        try:
            # Clean path check entries before saving to keep JSON light
            light_cache = {k: v for k, v in local_songs_cache.items() if not k.startswith('/') and not (len(k) > 2 and k[1] == ':')}
            with open(SONGS_CACHE_FILE, "w") as f:
                json.dump(light_cache, f, indent=4)
            print(f"Scanned Songs. Indexed {new_maps} new beatmaps.")
        except Exception as e:
            print(f"Failed to save songs cache: {e}")

# Try locating directories
appdata = os.getenv('APPDATA')
localappdata = os.getenv('LOCALAPPDATA')

lazer_export_dir = os.path.join(appdata, "osu", "exports") if appdata else None
stable_r_dir = os.path.join(localappdata, "osu!", "Data", "r") if localappdata else None
stable_songs_dir = config.get("songs_path")

if not stable_songs_dir and localappdata:
    stable_songs_dir = os.path.join(localappdata, "osu!", "Songs")

# Scan local songs
if stable_songs_dir and os.path.exists(stable_songs_dir):
    scan_songs_folder(stable_songs_dir)

class CompanionWatcherHandler(FileSystemEventHandler):
    def process_replay(self, replay_path):
        time.sleep(0.5) # Wait for file write to complete
        
        if not os.path.exists(replay_path) or os.path.getsize(replay_path) == 0:
            return
            
        print(f"\n[Companion] New replay detected: {os.path.basename(replay_path)}")
        try:
            replay = Replay.from_path(replay_path)
            h = replay.beatmap_hash
            print(f"[Companion] Replay beatmap hash: {h}")
            
            # Find corresponding .osu locally
            osu_path = local_songs_cache.get(h)
            files = {}
            
            # Read replay file bytes
            with open(replay_path, "rb") as f_rep:
                files["file"] = (os.path.basename(replay_path), f_rep.read(), "application/octet-stream")
                
            if osu_path and os.path.exists(osu_path):
                print(f"[Companion] Found local beatmap match: {os.path.basename(osu_path)}")
                with open(osu_path, "rb") as f_osu:
                    files["osu_file"] = (os.path.basename(osu_path), f_osu.read(), "text/plain")
            else:
                print(f"[Companion] Corresponding beatmap not found in local Songs. Server will lookup via API.")
                
            # Send to server
            print(f"[Companion] Uploading play to central server at {SERVER_URL}...")
            response = requests.post(f"{SERVER_URL}/api/analyze", files=files, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get("analysis", {})
                print(f"⭐ [Success] Replay analyzed! Accuracy: {analysis.get('accuracy_percent', 0.0):.2f}%, UR: {analysis.get('unstable_rate', 0.0):.1f}")
            else:
                print(f"❌ [Error] Server rejected upload with status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ [Error] Failed to process replay: {e}")

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.osr'):
            return
        self.process_replay(event.src_path)

    def on_moved(self, event):
        if event.is_directory or not event.dest_path.endswith('.osr'):
            return
        self.process_replay(event.dest_path)

def start_companion():
    observers = []
    
    if config["watch_lazer"] and lazer_export_dir and os.path.exists(lazer_export_dir):
        print(f"Watching Osu! Lazer exports: {lazer_export_dir}")
        obs = Observer()
        obs.schedule(CompanionWatcherHandler(), path=lazer_export_dir, recursive=False)
        obs.start()
        observers.append(obs)
        
    if config["watch_stable"] and stable_r_dir and os.path.exists(stable_r_dir):
        print(f"Watching Osu! Stable replay cache: {stable_r_dir}")
        obs = Observer()
        obs.schedule(CompanionWatcherHandler(), path=stable_r_dir, recursive=False)
        obs.start()
        observers.append(obs)
        
    if not observers:
        print("Error: No valid Osu! directories found to watch. Please update config.json manually.")
        sys.exit(1)
        
    print("\n[Companion] Watcher is active! Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for obs in observers:
            obs.stop()
        for obs in observers:
            obs.join()
        print("\n[Companion] Stopped successfully.")

if __name__ == "__main__":
    start_companion()
