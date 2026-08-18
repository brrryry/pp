import torch
import numpy as np
from core.parser import parse_osu_file

def osu_to_ml_sequence(map_path_file, max_seq_len=2000):
    """
    Transforms the output dictionary of parse_osu_file into a uniform,
    normalized machine learning matrix of shape [sequence_length, 9].
    
    Features per hitobject row:
    0: delta_x (Normalized -1.0 to 1.0)
    1: delta_y (Normalized -1.0 to 1.0)
    2: delta_t (Normalized time gap)
    3: is_circle (0.0 or 1.0)
    4: is_slider (0.0 or 1.0)
    5: is_spinner (0.0 or 1.0)
    6: slider_velocity (Normalized pixels/ms)
    7: slider_linearity (Displacement over path length: 0.0 to 1.0)
    8: is_bezier (0.0 or 1.0)
    """
    
    parsed_data = parse_osu_file(map_path_file)
    hit_objects = parsed_data.get('hit_objects', [])
    if not hit_objects:
        return torch.zeros((max_seq_len, 9), dtype=torch.float32)
        
    sequence = []

    # get difficulty stats
    diff = parsed_data.get('difficulty', {})
    circle_size = diff.get('CircleSize', 4.0) / 10.0
    approach_rate = diff.get('ApproachRate', 4.0) / 10.0
    overall_difficulty = diff.get('OverallDifficulty', 4.0) / 10.0
    hp_drain_rate = diff.get('HPDrainRate', 4.0) / 10.0
    
    
    # Initialize previous positions at the first object's start parameters
    # to avoid a massive artificial jump from (0,0) at the start of the map
    prev_end_x = hit_objects[0]['x'] if hit_objects else 256.0
    prev_end_y = hit_objects[0]['y'] if hit_objects else 192.0
    prev_end_time = hit_objects[0]['time'] if hit_objects else 0.0
    
    for obj in hit_objects:
        # 1. Raw spatial and temporal shifts
        raw_dx = obj['x'] - prev_end_x
        raw_dy = obj['y'] - prev_end_y
        raw_dt = obj['time'] - prev_end_time
        
        # 2. Normalize spatial inputs relative to the osu! playfield bounds (512 x 384)
        delta_x = raw_dx / 512.0
        delta_y = raw_dy / 384.0
        
        # 3. Normalize time gaps (Cap at 2000ms for long breaks to avoid gradient skewing, then scale)
        delta_t = min(raw_dt, 2000.0) / 1000.0
        
        # 4. Extract basic one-hot type identifiers
        is_circle = 1.0 if obj['type'] == 'circle' else 0.0
        is_slider = 1.0 if obj['type'] == 'slider' else 0.0
        is_spinner = 1.0 if obj['type'] == 'spinner' else 0.0
        
        # 5. Initialize advanced slider structural metrics
        slider_velocity = 0.0
        slider_linearity = 1.0  # 1.0 means perfectly straight or standard circle
        is_bezier = 0.0
        
        if obj['type'] == 'slider':
            duration = obj['end_time'] - obj['time']
            length = obj.get('slider_length', 0.0)
            
            # Slider Velocity (SV) Calculation & scaling
            if duration > 0:
                # Typical SV yields ~0.5 to 3.0 pixels/ms. We scale by 2.0 to keep around a 0-1 range.
                slider_velocity = (length / duration) / 2.0
                
            # Slider Linearity Calculation (Beeline distance / slider length)
            dx = obj['end_x'] - obj['x']
            dy = obj['end_y'] - obj['y']
            straight_line_dist = (dx**2 + dy**2)**0.5
            
            if length > 0:
                slider_linearity = clamp(straight_line_dist / length, 0.0, 1.0)
                
            # Curve format checking
            if obj.get('slider_curve_type') == 'B':
                is_bezier = 1.0
                
        # Consolidate into our complete 13-dimensional slice
        feature_vector = [
            delta_x, delta_y, delta_t, 
            is_circle, is_slider, is_spinner,
            slider_velocity, slider_linearity, is_bezier,
            circle_size, approach_rate, hp_drain_rate, overall_difficulty
        ]
        sequence.append(feature_vector)
        
        # Update pointer assignments to the current note's output trail
        prev_end_x = obj['end_x']
        prev_end_y = obj['end_y']
        prev_end_time = obj['end_time']
        
    # Convert sequence list to a rigid NumPy array
    seq_arr = np.array(sequence, dtype=np.float32)
    curr_len = seq_arr.shape[0]
    
    # Force alignment with max_seq_len to create valid PyTorch batches
    if curr_len > max_seq_len:
        # Trim extreme marathon maps down to the maximum sequence size limit
        seq_arr = seq_arr[:max_seq_len, :]
    elif curr_len < max_seq_len:
        # Pad shorter maps out with pure 0 rows
        pad_width = max_seq_len - curr_len
        seq_arr = np.pad(seq_arr, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
        
    return torch.from_numpy(seq_arr), max(curr_len, max_seq_len)

def clamp(n, minn, maxn):
    return max(min(n, maxn), minn)
