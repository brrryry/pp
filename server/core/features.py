import numpy as np

def compute_note_features(hit_objects):
    """
    Computes local, note-level geometric and rhythmic features from a list of hit objects.
    """
    if len(hit_objects) < 2:
        return []
        
    # Annotate combo_index on all hit objects
    combo = 1
    for idx, obj in enumerate(hit_objects):
        if idx == 0 or (obj.get('raw_type', 0) & 4):
            combo = 1
        else:
            combo += 1
        obj['combo_index'] = combo
        
    note_features = []
    
    for i in range(1, len(hit_objects)):
        prev = hit_objects[i-1]
        curr = hit_objects[i]
        
        # 1. Spacing: distance from prev end position to curr start position
        dx = curr['x'] - prev['end_x']
        dy = curr['y'] - prev['end_y']
        distance = (dx**2 + dy**2)**0.5
        
        # 2. Time Delta: time from prev end to curr start
        # If time delta is negative or zero (e.g. overlapping objects), set a minimum of 1ms
        time_delta = curr['time'] - prev['end_time']
        if time_delta <= 0:
            time_delta = 1.0
            
        # 3. Aim Velocity (pixels per millisecond)
        velocity = distance / time_delta
        
        # 4. Note Info
        note_type = curr['type']
        
        note_feat = {
            'time': curr['time'],
            'type': note_type,
            'distance': distance,
            'time_delta': time_delta,
            'velocity': velocity,
            'dx': dx,
            'dy': dy,
            'prev_combo': prev['combo_index'],
            'curr_combo': curr['combo_index']
        }
        
        # 5. Angle: angle between movement vector (prev_prev -> prev) and (prev -> curr)
        # We need at least 3 objects to compute an angle
        if i >= 2:
            prev_prev = hit_objects[i-2]
            
            # Vector 1: prev_prev end to prev start
            v1_x = prev['x'] - prev_prev['end_x']
            v1_y = prev['y'] - prev_prev['end_y']
            
            # Vector 2: prev end to curr start (which is dx, dy)
            v2_x = dx
            v2_y = dy
            
            len_v1 = (v1_x**2 + v1_y**2)**0.5
            len_v2 = (v2_x**2 + v2_y**2)**0.5
            
            if len_v1 > 0 and len_v2 > 0:
                dot_product = v1_x * v2_x + v1_y * v2_y
                cos_theta = dot_product / (len_v1 * len_v2)
                # Clamp cos_theta to prevent floating point issues out of [-1, 1]
                cos_theta = max(-1.0, min(1.0, cos_theta))
                angle_rad = np.arccos(cos_theta)
                angle_deg = np.degrees(angle_rad)
            else:
                angle_deg = np.nan
        else:
            angle_deg = np.nan
            
        note_feat['angle'] = angle_deg
        note_features.append(note_feat)
        
    return note_features

def extract_map_features(parsed_map):
    """
    Aggregates note-level features and combines them with map difficulty settings
    to create a single feature dictionary for the entire map.
    """
    metadata = parsed_map.get('metadata', {})
    difficulty = parsed_map.get('difficulty', {})
    hit_objects = parsed_map.get('hit_objects', [])
    
    if len(hit_objects) < 10:
        return None
        
    # Calculate note-level features
    note_feats = compute_note_features(hit_objects)
    if not note_feats:
        return None
        
    # Extract arrays for statistical aggregation
    velocities = np.array([n['velocity'] for n in note_feats])
    distances = np.array([n['distance'] for n in note_feats])
    time_deltas = np.array([n['time_delta'] for n in note_feats])
    angles = np.array([n['angle'] for n in note_feats if not np.isnan(n['angle'])])
    
    # Calculate object type counts
    total_objects = len(hit_objects)
    circles = sum(1 for h in hit_objects if h['type'] == 'circle')
    sliders = sum(1 for h in hit_objects if h['type'] == 'slider')
    spinners = sum(1 for h in hit_objects if h['type'] == 'spinner')
    
    # Total song length in seconds (from first to last object)
    first_time = hit_objects[0]['time']
    last_time = hit_objects[-1]['end_time']
    duration_sec = (last_time - first_time) / 1000.0 if last_time > first_time else 1.0
    
    # Calculate density (objects per second)
    density = total_objects / duration_sec
    
    # Angle categories
    sharp_angles = 0
    wide_angles = 0
    if len(angles) > 0:
        sharp_angles = np.sum(angles < 90.0) / len(angles)
        wide_angles = np.sum(angles >= 120.0) / len(angles)
        
    # Calculate combo stats
    combo_lengths = []
    current_len = 0
    for idx, obj in enumerate(hit_objects):
        if idx > 0 and (obj.get('raw_type', 0) & 4):
            combo_lengths.append(current_len)
            current_len = 1
        else:
            current_len += 1
    if current_len > 0:
        combo_lengths.append(current_len)
        
    mean_combo_length = np.mean(combo_lengths) if combo_lengths else 0.0
    std_combo_length = np.std(combo_lengths) if combo_lengths else 0.0
    
    # 1-2 Jumps ratio: alternating 1->2 or 2->1 combo indexes with above-median spacing
    med_dist = np.median(distances) if len(distances) > 0 else 0.0
    one_two_jump_count = 0
    for nf in note_feats:
        is_one_two = (nf['prev_combo'] == 1 and nf['curr_combo'] == 2) or (nf['prev_combo'] == 2 and nf['curr_combo'] == 1)
        if is_one_two and nf['distance'] > med_dist:
            one_two_jump_count += 1
            
    ratio_of_1_2_jumps = one_two_jump_count / len(note_feats) if len(note_feats) > 0 else 0.0
        
    # Build the map feature dict
    map_features = {
        # Metadata / Targets
        'title': metadata.get('Title', 'Unknown'),
        'artist': metadata.get('Artist', 'Unknown'),
        'version': metadata.get('Version', 'Unknown'),
        'creator': metadata.get('Creator', 'Unknown'),
        'map_id': metadata.get('BeatmapID', None),
        'mapset_id': metadata.get("BeatmapSetID", None),
        
        # Difficulty parameters (CS, HP, OD, AR, SV)
        'circle_size': difficulty.get('CircleSize', 4.0),
        'overall_difficulty': difficulty.get('OverallDifficulty', 8.0),
        'hp_drain': difficulty.get('HPDrainRate', 5.0),
        'approach_rate': difficulty.get('ApproachRate', 9.0),
        'slider_multiplier': difficulty.get('SliderMultiplier', 1.4),
        
        # Object statistics
        'total_objects': total_objects,
        'circles_ratio': circles / total_objects,
        'sliders_ratio': sliders / total_objects,
        'spinners_ratio': spinners / total_objects,
        'duration_seconds': duration_sec,
        'density_notes_per_sec': density,
        
        # Velocity statistics (aim speed)
        'velocity_mean': np.mean(velocities),
        'velocity_std': np.std(velocities),
        'velocity_median': np.median(velocities),
        'velocity_p75': np.percentile(velocities, 75),
        'velocity_p90': np.percentile(velocities, 90),
        'velocity_p95': np.percentile(velocities, 95),
        'velocity_p99': np.percentile(velocities, 99),
        
        # Spacing statistics (jump sizes)
        'distance_mean': np.mean(distances),
        'distance_std': np.std(distances),
        'distance_median': np.median(distances),
        'distance_p75': np.percentile(distances, 75),
        'distance_p90': np.percentile(distances, 90),
        'distance_p95': np.percentile(distances, 95),
        'distance_p99': np.percentile(distances, 99),
        
        # Rhythm statistics (BPM / speed)
        'time_delta_mean': np.mean(time_deltas),
        'time_delta_std': np.std(time_deltas),
        'time_delta_median': np.median(time_deltas),
        'time_delta_p10': np.percentile(time_deltas, 10),  # short time gaps = fast clicking
        'time_delta_p5': np.percentile(time_deltas, 5),
        'time_delta_min': np.min(time_deltas),
        
        # Angle statistics (flow / visual style)
        'angle_mean': np.mean(angles) if len(angles) > 0 else 0.0,
        'angle_std': np.std(angles) if len(angles) > 0 else 0.0,
        'angle_sharp_ratio': sharp_angles,
        'angle_wide_ratio': wide_angles,
        
        # === NEW: Advanced skill features ===
        
        # Snap Aim: sharp-angle jumps with above-median spacing
        'snap_aim_score': _compute_snap_aim(note_feats, angles, distances),
        
        # Flow Aim: wide-arc flow patterns with spacing
        'flow_aim_score': _compute_flow_aim(note_feats, angles, distances),
        
        # Finger Control: rhythm irregularity (coefficient of variation)
        'finger_control_score': float(np.std(time_deltas) / np.mean(time_deltas)) if np.mean(time_deltas) > 0 else 0.0,
        
        # Streaming: sustained fast tapping consistency
        'streaming_score': _compute_streaming(time_deltas, total_objects),
        
        # Visual Density: overlapping notes on screen
        'visual_density_score': _compute_visual_density(hit_objects, difficulty.get('ApproachRate', 9.0)),
        
        # Slider Complexity: aim control from slider paths
        'slider_complexity_score': _compute_slider_complexity(hit_objects, difficulty.get('SliderMultiplier', 1.4)),
        
        # Combo structure statistics
        'combo_1_2_jump_ratio': ratio_of_1_2_jumps,
        'mean_combo_length': mean_combo_length,
        'std_combo_length': std_combo_length,
        
        # Combo Cluster Ratios (Unsupervised pattern types)
        'combo_cluster_0_ratio': _classify_combos_ratios(note_feats, hit_objects)[0],
        'combo_cluster_1_ratio': _classify_combos_ratios(note_feats, hit_objects)[1],
        'combo_cluster_2_ratio': _classify_combos_ratios(note_feats, hit_objects)[2],
        'combo_cluster_3_ratio': _classify_combos_ratios(note_feats, hit_objects)[3],
        'combo_cluster_4_ratio': _classify_combos_ratios(note_feats, hit_objects)[4],
        'combo_cluster_5_ratio': _classify_combos_ratios(note_feats, hit_objects)[5]
    }
    
    return map_features


def _compute_snap_aim(note_feats, angles, distances):
    """Sharp-angle (< 90°) jumps with above-median spacing."""
    if len(angles) == 0 or len(distances) == 0:
        return 0.0
    med_dist = float(np.median(distances))
    # note_feats[i] corresponds to angles starting from index 0 for those with valid angles
    snap_count = 0
    total_with_angles = 0
    for nf in note_feats:
        a = nf.get('angle')
        if a is not None and not np.isnan(a):
            total_with_angles += 1
            if a < 90.0 and nf['distance'] > med_dist:
                snap_count += 1
    return snap_count / total_with_angles if total_with_angles > 0 else 0.0


def _compute_flow_aim(note_feats, angles, distances):
    """Wide-arc (>= 120°) flow patterns with above-median spacing."""
    if len(angles) == 0 or len(distances) == 0:
        return 0.0
    med_dist = float(np.median(distances))
    flow_count = 0
    total_with_angles = 0
    for nf in note_feats:
        a = nf.get('angle')
        if a is not None and not np.isnan(a):
            total_with_angles += 1
            if a >= 120.0 and nf['distance'] > med_dist:
                flow_count += 1
    return flow_count / total_with_angles if total_with_angles > 0 else 0.0


def _compute_streaming(time_deltas, total_objects):
    """Sustained fast tapping: longest streak of consecutive notes with time_delta < 120ms."""
    if len(time_deltas) == 0 or total_objects == 0:
        return 0.0
    threshold = 120.0  # ~125 BPM 1/4
    max_streak = 0
    current_streak = 0
    for td in time_deltas:
        if td < threshold:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    # Normalize: longest streak relative to total objects, capped at 1.0
    return min(1.0, max_streak / max(1, total_objects) * 3.0)


def _compute_visual_density(hit_objects, ar):
    """Average note overlap count based on AR preempt time and spatial proximity."""
    if len(hit_objects) < 2:
        return 0.0
    # AR to preempt time (ms)
    if ar > 5.0:
        preempt = 1200.0 - 750.0 * (ar - 5.0) / 5.0
    else:
        preempt = 1200.0 + 600.0 * (5.0 - ar) / 5.0
    
    # Sample up to 500 notes to keep computation fast
    sample_indices = range(0, len(hit_objects), max(1, len(hit_objects) // 500))
    overlap_counts = []
    proximity_scores = []
    
    for i in sample_indices:
        obj = hit_objects[i]
        t = obj['time']
        overlap = 0
        close_count = 0
        for j in range(max(0, i - 30), min(len(hit_objects), i + 30)):
            if i == j:
                continue
            other = hit_objects[j]
            if abs(other['time'] - t) < preempt:
                overlap += 1
                dx = other['x'] - obj['x']
                dy = other['y'] - obj['y']
                dist = (dx**2 + dy**2)**0.5
                if dist < 60.0:  # Spatially close = stacked/overlapping
                    close_count += 1
        overlap_counts.append(overlap)
        proximity_scores.append(close_count)
    
    avg_overlap = np.mean(overlap_counts) if overlap_counts else 0.0
    avg_proximity = np.mean(proximity_scores) if proximity_scores else 0.0
    # Normalize: typical maps have 3-8 overlap, dense maps 15+
    return min(1.0, (avg_overlap / 12.0) * 0.6 + (avg_proximity / 5.0) * 0.4)


def _compute_slider_complexity(hit_objects, slider_multiplier):
    """Average control point count per slider, factored by slider velocity."""
    sliders = [h for h in hit_objects if h.get('type') == 'slider']
    if not sliders:
        return 0.0
    total_cp = sum(len(s.get('slider_control_points', [])) for s in sliders)
    avg_cp = total_cp / len(sliders)
    # Normalize: 2 control points is simple, 5+ is complex
    complexity = min(1.0, (avg_cp - 2.0) / 4.0) if avg_cp > 2.0 else 0.0
    # Factor in slider velocity
    sv_factor = min(1.0, slider_multiplier / 2.5)
    return complexity * 0.7 + sv_factor * 0.3


combo_clusters = None

def _load_combo_clusters():
    global combo_clusters
    if combo_clusters is not None:
        return True
    try:
        import json
        import os
        path = "data/model_results/combo_clusters.json"
        if not os.path.exists(path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base_dir, "..", "data", "model_results", "combo_clusters.json")
        if not os.path.exists(path):
            path = os.path.join("server", "data", "model_results", "combo_clusters.json")
            
        if os.path.exists(path):
            with open(path, "r") as f:
                combo_clusters = json.load(f)
            return True
    except Exception:
        pass
    return False

def _classify_combos_ratios(note_feats, hit_objects):
    """Groups note features into combos, standardizes them, and assigns them to K-Means clusters."""
    if not _load_combo_clusters():
        return [0.0] * 6
        
    try:
        # Group note features into combos
        combos = []
        current_combo_notes = []
        for nf in note_feats:
            current_combo_notes.append(nf)
            if nf['curr_combo'] == 1:
                if len(current_combo_notes) >= 2:
                    combos.append(current_combo_notes)
                current_combo_notes = []
        if len(current_combo_notes) >= 2:
            combos.append(current_combo_notes)
            
        if not combos:
            return [0.0] * 6
            
        # Extract features for each combo group
        cluster_counts = [0] * 6
        mean = np.array(combo_clusters["scaler_mean"])
        scale = np.array(combo_clusters["scaler_scale"])
        centers = np.array(combo_clusters["cluster_centers"])
        
        for combo in combos:
            spacings = [n['distance'] for n in combo]
            time_deltas = [n['time_delta'] for n in combo]
            velocities = [n['velocity'] for n in combo]
            angles = [n['angle'] for n in combo if not np.isnan(n['angle'])]
            sliders = sum(1 for n in combo if n['type'] == 'slider')
            
            feat = np.array([
                float(len(combo) + 1),
                float(np.mean(spacings)),
                float(np.std(spacings)) if len(spacings) > 1 else 0.0,
                float(np.mean(time_deltas)),
                float(np.mean(angles)) if angles else 90.0,
                float(sliders / len(combo)),
                float(np.percentile(velocities, 95))
            ])
            
            # Standardize
            feat_scaled = (feat - mean) / scale
            
            # Find closest cluster centroid (Euclidean distance)
            dists = np.sum((centers - feat_scaled) ** 2, axis=1)
            best_cluster = int(np.argmin(dists))
            cluster_counts[best_cluster] += 1
            
        # Compute ratios
        total_combos = sum(cluster_counts)
        if total_combos > 0:
            return [count / total_combos for count in cluster_counts]
    except Exception:
        pass
    return [0.0] * 6
