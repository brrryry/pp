import os, sqlite3

# 1. Get first 3 hashes from the DB
conn = sqlite3.connect('../data/osu_profiler.db')
db_hashes = [row[0] for row in conn.execute("SELECT map_hash FROM maps LIMIT 3").fetchall()]
conn.close()

# 2. Get first 3 files from your disk folder
disk_files = os.listdir('../data/maps')[:3] if os.path.exists('../data/maps') else ["Folder not found!"]

print("--- DIAGNOSTIC RESULTS ---")
print(f"Total files on disk: {len(os.listdir('../data/maps')) if os.path.exists('../data/maps') else 0}")
print(f"Sample DB values:   {db_hashes}")
print(f"Sample Disk files: {disk_files}")
if db_hashes and disk_files:
    print(f"DB String Length:   {len(str(db_hashes[0]))} chars")
    print(f"Disk String Length: {len(disk_files[0])} chars")
