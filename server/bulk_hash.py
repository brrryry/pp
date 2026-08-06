import os
import hashlib
import random


folder = "data/maps"

files = []
for root, dirs, files in os.walk(folder):
    for file in files:
        if file.endswith(".osu"):
            files.append(os.path.join(root, file))

# test one file
test_file = random.choice(files)
with open(test_file, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()
hash = hashlib.md5(content.encode()).hexdigest()
print(f"Hash of {test_file}: {hash}")
print(f"Length of {test_file}: {len(content)}")
print(f"Size of {test_file}: {os.path.getsize(test_file)} bytes")
exit()


for file in files:
    # read file content
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    hash = hashlib.md5(content.encode()).hexdigest()
    os.rename(file, os.path.join(folder, hash + ".osu"))
    print(f"Renamed {file} to {hash + '.osu'}")

