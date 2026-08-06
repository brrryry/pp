import os
import json
import logging
import sqlite3
import pandas as pd
import numpy as np
from osrparse import Replay, Mod

import config
from server.core.parser import parse_osu_file
from server.core.features import extract_map_features
from server.core.portfolio import compute_mechanical_portfolio

logging.basicConfig(level=logging.INFO, format="[Replay Pipeline] %(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ReplayAnalysisPipeline:
    def __init__(self, dbManager, replays_dir, maps_dir=None):
        self.dbManager = dbManager
        self.replays_dir = replays_dir
        self.maps_dir = maps_dir or config.maps_path

    def analyze_replay(self, replay_path):
        """
        Parses and analyzes a replay file, matches clicks to hit objects,
        computes mechanical skills, and persists both raw hits and mechanical
        profile scores into the database.
        """
        if not os.path.exists(replay_path):
            logger.error(f"Replay file not found: {replay_path}")
            return None

        try:
            logger.info(f"Starting replay analysis for {replay_path}...")
            replay = Replay.from_path(replay_path)
            beatmap_hash = replay.beatmap_hash
            username = replay.username

            # 1. Retrieve map_id from database
            map_id = self.dbManager.get_map_id_by_hash(beatmap_hash)
            if not map_id:
                logger.error(f"Beatmap with hash {beatmap_hash} not found in database. Ingest the map set first.")
                return None

            # 2. Locate .osu file
            osu_map_path = os.path.join(self.maps_dir, f"{beatmap_hash}.osu")
            if not os.path.exists(osu_map_path):
                logger.error(f"Beatmap file not found on disk at {osu_map_path}")
                return None

            # 3. Parse the map
            parsed_map = parse_osu_file(osu_map_path)
            if parsed_map is None:
                logger.error(f"Failed to parse beatmap: {osu_map_path}")
                return None

            # 4. Extract clicks from replay data
            current_time = 0
            clicks = []
            prev_keys = 0
            for event in replay.replay_data:
                current_time += event.time_delta
                keys_val = event.keys.value
                for mask in (1, 2, 4, 8):
                    if (keys_val & mask) and not (prev_keys & mask):
                        clicks.append((current_time, event.x, event.y, mask))
                prev_keys = keys_val

            # Get difficulty settings and mods
            od = float(parsed_map['difficulty'].get('OverallDifficulty', 8.0))
            cs = float(parsed_map['difficulty'].get('CircleSize', 4.0))
            
            speed_multiplier = 1.0
            if Mod.HardRock in replay.mods:
                od = min(10.0, od * 1.4)
                cs = min(10.0, cs * 1.3)
            elif Mod.Easy in replay.mods:
                od = od * 0.5
                cs = cs * 0.5
                
            if Mod.DoubleTime in replay.mods or Mod.Nightcore in replay.mods:
                speed_multiplier = 1.5
            elif Mod.HalfTime in replay.mods:
                speed_multiplier = 0.75

            # Calculate hit windows and radius
            window_300 = 80.0 - 6.0 * od
            window_100 = 140.0 - 8.0 * od
            window_50 = 200.0 - 10.0 * od
            radius = 54.4 - 4.48 * cs

            hit_objects = parsed_map.get('hit_objects', [])
            clicks_left = clicks.copy()
            hit_results = []
            window = 150.0  # Search window

            # Match clicks to hit objects
            for obj in hit_objects:
                if obj['type'] == 'spinner':
                    continue
                target_time = obj['time']
                target_x = obj['x']
                target_y = obj['y']
                
                best_click = None
                best_idx = -1
                min_diff = float('inf')
                
                for i, click in enumerate(clicks_left):
                    click_time = click[0]
                    diff = abs(click_time - target_time)
                    if diff < window and diff < min_diff:
                        min_diff = diff
                        best_click = click
                        best_idx = i
                        
                if best_click is not None:
                    click_time, click_x, click_y, click_mask = best_click
                    timing_offset = click_time - target_time
                    real_offset = timing_offset / speed_multiplier
                    aim_distance = ((click_x - target_x)**2 + (click_y - target_y)**2)**0.5
                    dx = click_x - target_x
                    dy = click_y - target_y
                    
                    is_hit = False
                    score = 0
                    if aim_distance <= radius and abs(real_offset) <= window_50:
                        is_hit = True
                        if abs(real_offset) <= window_300:
                            score = 300
                        elif abs(real_offset) <= window_100:
                            score = 100
                        else:
                            score = 50
                    
                    hit_results.append({
                        'target_time': target_time,
                        'target_x': target_x,
                        'target_y': target_y,
                        'type': obj['type'],
                        'hit': is_hit,
                        'score': score,
                        'timing_offset': timing_offset,
                        'aim_distance': aim_distance,
                        'dx': dx,
                        'dy': dy
                    })
                    clicks_left.pop(best_idx)
                else:
                    hit_results.append({
                        'target_time': target_time,
                        'target_x': target_x,
                        'target_y': target_y,
                        'type': obj['type'],
                        'hit': False,
                        'score': 0,
                        'timing_offset': None,
                        'aim_distance': None,
                        'dx': None,
                        'dy': None
                    })

            df_hits = pd.DataFrame(hit_results)
            if df_hits.empty:
                logger.warning(f"No hit results generated for replay {replay_path}")
                return None

            hits_only = df_hits[df_hits['hit'] == True]
            total_notes = len(df_hits)
            hits_count = len(hits_only)
            misses_count = total_notes - hits_count
            
            # Accuracy
            count_300 = len(df_hits[df_hits['score'] == 300])
            count_100 = len(df_hits[df_hits['score'] == 100])
            count_50 = len(df_hits[df_hits['score'] == 50])
            accuracy = 0.0
            if total_notes > 0:
                accuracy = (300 * count_300 + 100 * count_100 + 50 * count_50) / (300 * total_notes) * 100.0

            # Offsets and errors
            avg_offset = hits_only['timing_offset'].mean() if hits_count > 0 else 0.0
            abs_offset = hits_only['timing_offset'].abs().mean() if hits_count > 0 else 0.0
            std_offset = hits_only['timing_offset'].std() if hits_count > 1 else 0.0
            unstable_rate = std_offset * 10.0 if not pd.isna(std_offset) else 0.0
            avg_aim_error = hits_only['aim_distance'].mean() if hits_count > 0 else 0.0
            std_aim_error = hits_only['aim_distance'].std() if hits_count > 1 else 0.0

            # Compute mechanical skills
            mechanical_skills = None
            try:
                mechanical_skills = compute_mechanical_portfolio(hit_results, parsed_map.get('difficulty', {}))
            except Exception as e:
                logger.error(f"Error computing mechanical portfolio: {e}")

            # 5. Save to database using dbManager
            replay_filename = os.path.basename(replay_path)
            hits_json = df_hits.to_json(orient='records')
            
            # Format mods as a string
            mods_str = "+".join([str(m).replace("Mod.", "") for m in replay.mods]) if replay.mods else "NoMod"
            
            replay_id = self.dbManager.add_replay(
                map_id=map_id,
                username=username,
                replay_hash=replay_filename,  # using filename as hash/identifier
                accuracy=accuracy,
                unstable_rate=unstable_rate,
                avg_aim_error_px=avg_aim_error,
                mods=mods_str,
                misses=int(misses_count),
                total_notes=int(total_notes),
                hits=int(hits_count),
                hits_json=hits_json,
                mechanical_json=mechanical_skills
            )

            if replay_id:
                logger.info(f"Replay analysis complete. Replay ID: {replay_id}")
                
            return {
                'replay_id': replay_id,
                'accuracy': accuracy,
                'unstable_rate': unstable_rate,
                'avg_aim_error': avg_aim_error,
                'mechanical_skills': mechanical_skills
            }
        except Exception as e:
            logger.error(f"Error executing replay analysis pipeline: {e}")
            return None