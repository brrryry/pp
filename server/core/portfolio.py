import numpy as np

def compute_map_portfolio_skills(feats):
    """
    Computes the 11-axis map skill portfolio from raw extracted map features.
    
    Args:
        feats (dict): Dictionary of raw features (e.g. density_notes_per_sec, velocity_p95, etc.)
        
    Returns:
        dict: 11 skill levels (0-100), calibrated so a typical 5-star map scores ~40-60 on each axis.
    """
    # 1. Snap Aim — sharp-angle jumps with high spacing
    snap_raw = feats.get('snap_aim_score', 0.0) or 0.0
    dist_p95 = feats.get('distance_p95', 100.0) or 100.0
    snap_aim = max(5.0, min(100.0, snap_raw * 80.0 + (dist_p95 / 512.0) * 50.0))
    
    # 2. Flow Aim — wide sweeping arcs with spacing
    flow_raw = feats.get('flow_aim_score', 0.0) or 0.0
    dist_p90 = feats.get('distance_p90', 80.0) or 80.0
    flow_aim = max(5.0, min(100.0, flow_raw * 65.0 + (dist_p90 / 350.0) * 35.0))
    
    # 3. Speed — raw tapping speed (density + velocity)
    density = feats.get('density_notes_per_sec', 3.0) or 3.0
    vel_p95 = feats.get('velocity_p95', 1.0) or 1.0
    speed = max(5.0, min(100.0, (density / 10.0) * 55.0 + (vel_p95 / 3.0) * 45.0))
    
    # 4. Streaming — sustained consistent burst tapping
    stream_raw = feats.get('streaming_score', 0.0) or 0.0
    td_p5 = feats.get('time_delta_p5', 200.0) or 200.0
    streaming = max(5.0, min(100.0, stream_raw * 70.0 + max(0.0, (120.0 - td_p5) / 120.0) * 50.0))
    
    # 5. Stamina — endurance over long maps
    duration = feats.get('duration_seconds', 120.0) or 120.0
    total_objects = feats.get('total_objects', 500) or 500
    stamina = max(5.0, min(100.0, (duration / 400.0) * 35.0 + (total_objects / 2000.0) * 35.0 + speed * 0.20 + streaming * 0.10))
    
    # 6. Tech — slider-heavy, unconventional rhythm structure
    slider_ratio = feats.get('sliders_ratio', 0.2) or 0.2
    angle_std = feats.get('angle_std', 30.0) or 30.0
    tech = max(5.0, min(100.0, (slider_ratio * 50.0) + (angle_std / 70.0) * 50.0))
    
    # 7. Finger Control — irregular rhythm changes
    # fc_raw is coefficient of variation (std/mean) of time_deltas, typically 0.3-1.5
    fc_raw = feats.get('finger_control_score', 0.0) or 0.0
    td_std = feats.get('time_delta_std', 50.0) or 50.0
    finger_control = max(5.0, min(100.0, fc_raw * 35.0 + (td_std / 500.0) * 40.0))
    
    # 8. Precision — CS + OD tightness
    cs = feats.get('circle_size', 4.0) or 4.0
    od = feats.get('overall_difficulty', 8.0) or 8.0
    precision = max(5.0, min(100.0, ((cs - 2.0) / 5.0) * 50.0 + (od / 10.0) * 50.0))
    
    # 9. Reading — visual parsing difficulty (AR extremes + density)
    ar = feats.get('approach_rate', 9.0) or 9.0
    # Both very low AR (<7) and very high AR (>9.5) with high density are hard to read
    ar_difficulty = max(0.0, (10.3 - ar) / 3.0) if ar > 8.0 else max(0.0, (8.0 - ar) / 4.0) * 1.5
    vd_raw = feats.get('visual_density_score', 0.0) or 0.0
    reading = max(5.0, min(100.0, ar_difficulty * 35.0 + (density / 8.0) * 35.0 + vd_raw * 30.0))
    
    # 10. Visual Density — overlapping notes on screen
    visual_density = max(5.0, min(100.0, vd_raw * 45.0 + (density / 8.0) * 35.0 + ar_difficulty * 20.0))
    
    # 11. Aim Control — slider path precision
    sc_raw = feats.get('slider_complexity_score', 0.0) or 0.0
    sv = feats.get('slider_multiplier', 1.4) or 1.4
    aim_control = max(5.0, min(100.0, sc_raw * 35.0 + (sv / 3.0) * 25.0 + (cs / 7.0) * 25.0 + slider_ratio * 15.0))
    
    return {
        "SnapAim": round(snap_aim, 1),
        "FlowAim": round(flow_aim, 1),
        "Speed": round(speed, 1),
        "Streaming": round(streaming, 1),
        "Stamina": round(stamina, 1),
        "Tech": round(tech, 1),
        "FingerControl": round(finger_control, 1),
        "Precision": round(precision, 1),
        "Reading": round(reading, 1),
        "VisualDensity": round(visual_density, 1),
        "AimControl": round(aim_control, 1)
    }


def compute_mechanical_portfolio(hit_results, difficulty):
    """
    Computes 11-axis mechanical skill scores from per-note replay data.
    Each axis filters notes by pattern type, then measures execution quality.
    
    Args:
        hit_results: list of dicts with target_x/y, timing_offset, aim_distance, hit, score, type
        difficulty:  dict with CircleSize, OverallDifficulty, ApproachRate
    
    Returns:
        dict with 11 mechanical skill scores (0-100), or None if insufficient data
    """
    if len(hit_results) < 10:
        return None
    
    cs = float(difficulty.get('CircleSize', 4.0))
    od = float(difficulty.get('OverallDifficulty', 8.0))
    ar = float(difficulty.get('ApproachRate', 9.0))
    radius = 54.4 - 4.48 * cs
    window_100 = 140.0 - 8.0 * od
    window_50 = 200.0 - 10.0 * od
    preempt = (1200 - 750 * (ar - 5) / 5) if ar > 5 else (1200 + 600 * (5 - ar) / 5)
    
    # Enrich each hit result with pattern context (distance, time_delta, angle)
    merged = []
    for i, hr in enumerate(hit_results):
        entry = dict(hr)
        if i > 0:
            prev = hit_results[i - 1]
            dx_map = entry['target_x'] - prev['target_x']
            dy_map = entry['target_y'] - prev['target_y']
            entry['distance'] = (dx_map**2 + dy_map**2)**0.5
            entry['time_delta'] = entry['target_time'] - prev['target_time']
            if entry['time_delta'] <= 0:
                entry['time_delta'] = 1.0
            entry['velocity'] = entry['distance'] / entry['time_delta']
        else:
            entry['distance'] = 0.0
            entry['time_delta'] = 0.0
            entry['velocity'] = 0.0
        
        # Angle requires 3 consecutive notes
        if i >= 2:
            prev = hit_results[i - 1]
            prev_prev = hit_results[i - 2]
            v1_x = prev['target_x'] - prev_prev['target_x']
            v1_y = prev['target_y'] - prev_prev['target_y']
            v2_x = entry['target_x'] - prev['target_x']
            v2_y = entry['target_y'] - prev['target_y']
            len_v1 = (v1_x**2 + v1_y**2)**0.5
            len_v2 = (v2_x**2 + v2_y**2)**0.5
            if len_v1 > 0 and len_v2 > 0:
                cos_theta = (v1_x * v2_x + v1_y * v2_y) / (len_v1 * len_v2)
                cos_theta = max(-1.0, min(1.0, cos_theta))
                entry['angle'] = float(np.degrees(np.arccos(cos_theta)))
            else:
                entry['angle'] = None
        else:
            entry['angle'] = None
        merged.append(entry)
    
    # Pre-compute shared thresholds
    distances = [m['distance'] for m in merged if m['distance'] > 0]
    med_dist = float(np.median(distances)) if distances else 0.0
    time_deltas = [m['time_delta'] for m in merged if m['time_delta'] > 0]
    td_p25 = float(np.percentile(time_deltas, 25)) if time_deltas else 100.0
    
    # --- Scoring helpers ---
    def aim_score(subset):
        """Hit rate × aim precision on a filtered note subset."""
        if not subset:
            return 50.0
        hits = [n for n in subset if n.get('hit')]
        hit_rate = len(hits) / len(subset)
        if not hits:
            return 5.0
        mean_aim = float(np.mean([n['aim_distance'] for n in hits]))
        return max(5.0, min(100.0, 100.0 * (1.0 - mean_aim / radius) * hit_rate))
    
    def timing_score(subset, window):
        """Hit rate × timing precision on a filtered note subset."""
        if not subset:
            return 50.0
        hits = [n for n in subset if n.get('hit') and n.get('timing_offset') is not None]
        hit_rate = len(hits) / len(subset)
        if not hits:
            return 5.0
        mean_abs_offset = float(np.mean([abs(n['timing_offset']) for n in hits]))
        return max(5.0, min(100.0, 100.0 * hit_rate * (1.0 - mean_abs_offset / window)))
    
    # 1. Snap Aim: sharp-angle jumps with above-median spacing
    snap_notes = [m for m in merged if m.get('angle') is not None
                  and m['angle'] < 90 and m['distance'] > med_dist]
    snap_aim = aim_score(snap_notes)
    
    # 2. Flow Aim: wide-arc patterns with spacing
    flow_notes = [m for m in merged if m.get('angle') is not None
                  and m['angle'] >= 120 and m['distance'] > med_dist]
    flow_aim = aim_score(flow_notes)
    
    # 3. Speed: fastest quartile of notes
    speed_notes = [m for m in merged if m['time_delta'] > 0 and m['time_delta'] < td_p25]
    speed = timing_score(speed_notes, window_50)
    
    # 4. Streaming: consecutive fast-note runs of 6+
    stream_notes = []
    current_run = []
    for m in merged:
        if 0 < m['time_delta'] < 150:
            current_run.append(m)
        else:
            if len(current_run) >= 6:
                stream_notes.extend(current_run)
            current_run = []
    if len(current_run) >= 6:
        stream_notes.extend(current_run)
    streaming = timing_score(stream_notes, window_100)
    
    # 5. Stamina: Q4 vs Q1 accuracy ratio
    quarter = len(merged) // 4
    if quarter > 0:
        q1 = merged[:quarter]
        q4 = merged[-quarter:]
        q1_acc = sum(1 for n in q1 if n.get('hit')) / len(q1)
        q4_acc = sum(1 for n in q4 if n.get('hit')) / len(q4)
        stamina = max(5.0, min(100.0, 100.0 * (q4_acc / q1_acc))) if q1_acc > 0 else 50.0
    else:
        stamina = 50.0
    
    # 6. Tech: slider notes + irregular rhythm transitions
    tech_notes = []
    for i, m in enumerate(merged):
        is_slider = m.get('type') == 'slider'
        is_irregular = False
        if i > 0 and merged[i - 1]['time_delta'] > 0 and m['time_delta'] > 0:
            ratio = m['time_delta'] / merged[i - 1]['time_delta']
            is_irregular = abs(ratio - 1.0) > 0.3
        if is_slider or is_irregular:
            tech_notes.append(m)
    tech = aim_score(tech_notes)
    
    # 7. Finger Control: rhythm change points
    fc_notes = []
    for i, m in enumerate(merged):
        if i > 0 and merged[i - 1]['time_delta'] > 0 and m['time_delta'] > 0:
            ratio = m['time_delta'] / merged[i - 1]['time_delta']
            if abs(ratio - 1.0) > 0.25:
                fc_notes.append(m)
    finger_control = timing_score(fc_notes, window_100)
    
    # 8. Precision: global timing + aim + 300 rate
    all_hits = [m for m in merged if m.get('hit')]
    if all_hits:
        ur = float(np.std([n['timing_offset'] for n in all_hits])) * 10.0
        aim_component = 1.0 - float(np.mean([n['aim_distance'] for n in all_hits])) / radius
        rate_300 = sum(1 for n in merged if n.get('score') == 300) / len(merged)
        precision = max(5.0, min(100.0,
            50.0 * max(0.0, 1.0 - ur / 200.0) + 30.0 * max(0.0, aim_component) + 20.0 * rate_300 * 100.0
        ))
    else:
        precision = 5.0
    
    # 9. Reading: notes with high visual overlap (sliding window, O(n*k))
    reading_notes = []
    for i, m in enumerate(merged):
        overlap_count = 0
        # Scan backward within preempt window
        j = i - 1
        while j >= 0 and m['target_time'] - merged[j]['target_time'] < preempt:
            spatial_dist = ((m['target_x'] - merged[j]['target_x'])**2 +
                           (m['target_y'] - merged[j]['target_y'])**2)**0.5
            if spatial_dist < 3 * radius:
                overlap_count += 1
            j -= 1
        # Scan forward within preempt window
        j = i + 1
        while j < len(merged) and merged[j]['target_time'] - m['target_time'] < preempt:
            spatial_dist = ((m['target_x'] - merged[j]['target_x'])**2 +
                           (m['target_y'] - merged[j]['target_y'])**2)**0.5
            if spatial_dist < 3 * radius:
                overlap_count += 1
            j += 1
        if overlap_count >= 2:
            reading_notes.append(m)
    reading = timing_score(reading_notes, window_100)
    
    # 10. Visual Density: 3+ notes visible simultaneously
    vd_notes = []
    for i, m in enumerate(merged):
        simultaneous = 0
        j = i - 1
        while j >= 0 and m['target_time'] - merged[j]['target_time'] < preempt:
            simultaneous += 1
            j -= 1
        j = i + 1
        while j < len(merged) and merged[j]['target_time'] - m['target_time'] < preempt:
            simultaneous += 1
            j += 1
        if simultaneous >= 3:
            vd_notes.append(m)
    visual_density = aim_score(vd_notes)
    
    # 11. Aim Control: slider heads only
    slider_notes = [m for m in merged if m.get('type') == 'slider']
    aim_control = aim_score(slider_notes)
    
    return {
        'SnapAim': round(snap_aim, 1),
        'FlowAim': round(flow_aim, 1),
        'Speed': round(speed, 1),
        'Streaming': round(streaming, 1),
        'Stamina': round(stamina, 1),
        'Tech': round(tech, 1),
        'FingerControl': round(finger_control, 1),
        'Precision': round(precision, 1),
        'Reading': round(reading, 1),
        'VisualDensity': round(visual_density, 1),
        'AimControl': round(aim_control, 1)
    }
