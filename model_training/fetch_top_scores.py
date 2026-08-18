import os
import sys
from osu import Client
import osu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
import sqlite3
from seed.seed_replays import calculate_mastery

client = Client.from_credentials(
    config.OSU_API_CLIENT_ID,
    config.OSU_API_CLIENT_SECRET,
    config.OSU_API_REDIRECT_URI
)

def fetch_top_scores(osu_id, top_limit=100, recent_limit=50):
    best_scores = client.get_user_scores(user=osu_id, mode="osu", type="best", limit=top_limit) or []
    recent_scores = client.get_user_scores(user=osu_id, mode="osu", type="recent", limit=recent_limit) or []
    scores = list(best_scores) + list(recent_scores)
    for score in scores:
        username = score.user.username
        accuracy = score.accuracy

        # convert mods to bitmask
        mods_int = 0

        for mod in score.mods:
            if mod.mod.name == "Classic":
                continue
            mods_int |= mod.mod.value

        # get map
        beatmap = client.get_beatmap(score.beatmap.id)
        map_hash = beatmap.checksum

        misses = score.statistics.miss if score.statistics.miss is not None else 0

        max_combo = score.max_combo if score.max_combo is not None else 0

        # store in database (use NULL for replay_hash)
        conn = sqlite3.connect(config.DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO replays (username, replay_hash, map_hash, mods, accuracy, misses, max_combo, mastery_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (username, None, map_hash, mods_int, accuracy, misses, score.max_combo, calculate_mastery(accuracy, misses, score.beatmap.total_length, score.max_combo))
        )
        conn.commit()
        conn.close()
        print(f"Added replay: {username} - {score.beatmap.id}")


        
    
if __name__ == "__main__":
    fetch_top_scores(11781698)
    
