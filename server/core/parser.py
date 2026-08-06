import os
import re
import rosu_pp_py as rosu

def interpolate_slider_path(control_points, target_length):
    """
    Approximates the slider end point by walking along the control points
    and interpolating at target_length.
    """
    if not control_points:
        return (256, 192)
    if len(control_points) < 2:
        return control_points[0]
    
    accum_len = 0.0
    curr_pt = control_points[0]
    for i in range(1, len(control_points)):
        next_pt = control_points[i]
        dx = next_pt[0] - curr_pt[0]
        dy = next_pt[1] - curr_pt[1]
        seg_len = (dx**2 + dy**2)**0.5
        if seg_len == 0:
            continue
        if accum_len + seg_len >= target_length:
            rem_len = target_length - accum_len
            t = rem_len / seg_len
            return (curr_pt[0] + dx * t, curr_pt[1] + dy * t)
        accum_len += seg_len
        curr_pt = next_pt
    
    return control_points[-1]

def get_star_rating(file_path):
    map_data = rosu.Beatmap(path=file_path)
    diff_attrs = rosu.Difficulty().calculate(map_data)
    return diff_attrs.stars


def parse_osu_file(file_path):
    """
    Parses a single .osu file and returns a structured dictionary.
    """
    metadata = {}
    difficulty = {}
    general = {}
    timing_points = []
    hit_objects = []
    
    current_section = None
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            # Check for section header
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                continue
                
            if current_section == "General":
                if ':' in line:
                    key, val = line.split(':', 1)
                    general[key.strip()] = val.strip()
            elif current_section == "Metadata":
                if ':' in line:
                    key, val = line.split(':', 1)
                    metadata[key.strip()] = val.strip()
            elif current_section == "Difficulty":
                if ':' in line:
                    key, val = line.split(':', 1)
                    try:
                        difficulty[key.strip()] = float(val.strip())
                    except ValueError:
                        difficulty[key.strip()] = val.strip()
            elif current_section == "TimingPoints":
                # Format: time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        t_point = {
                            'time': float(parts[0]),
                            'beat_length': float(parts[1]),
                            'uninherited': int(parts[6]) if len(parts) >= 7 else 1
                        }
                        timing_points.append(t_point)
                    except ValueError:
                        pass
            elif current_section == "HitObjects":
                # Format: x,y,time,type,hitSound,objectParams...
                parts = line.split(',')
                if len(parts) >= 5:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        time = float(parts[2])
                        obj_type = int(parts[3])
                        
                        # Determine object type from bitmask
                        # Bit 0: Circle, Bit 1: Slider, Bit 3: Spinner
                        is_circle = bool(obj_type & 1)
                        is_slider = bool(obj_type & 2)
                        is_spinner = bool(obj_type & 8)
                        
                        hit_obj = {
                            'x': x,
                            'y': y,
                            'time': time,
                            'raw_type': obj_type,
                            'end_x': x,
                            'end_y': y,
                            'end_time': time
                        }
                        
                        if is_circle:
                            hit_obj['type'] = 'circle'
                        elif is_slider:
                            hit_obj['type'] = 'slider'
                            # Parse slider parameters: curveType|curvePoints,slides,length
                            slider_info = parts[5].split('|')
                            curve_type = slider_info[0]
                            
                            # Parse control points
                            control_points = [(x, y)]
                            for pt in slider_info[1:]:
                                if ':' in pt:
                                    try:
                                        pt_parts = pt.split(':')
                                        control_points.append((float(pt_parts[0]), float(pt_parts[1])))
                                    except ValueError:
                                        pass
                            
                            slides = int(parts[6]) if len(parts) >= 7 else 1
                            length = float(parts[7]) if len(parts) >= 8 else 0.0
                            
                            hit_obj['slider_curve_type'] = curve_type
                            hit_obj['slider_control_points'] = control_points
                            hit_obj['slider_slides'] = slides
                            hit_obj['slider_length'] = length
                            
                            # End point approximation
                            end_pt = interpolate_slider_path(control_points, length)
                            # If slides is even, the slider ends back at the start coordinate
                            if slides % 2 == 0:
                                hit_obj['end_x'] = x
                                hit_obj['end_y'] = y
                            else:
                                hit_obj['end_x'] = end_pt[0]
                                hit_obj['end_y'] = end_pt[1]
                                
                        elif is_spinner:
                            hit_obj['type'] = 'spinner'
                            # Spinner end time is the 6th parameter (index 5)
                            try:
                                hit_obj['end_time'] = float(parts[5])
                            except (IndexError, ValueError):
                                pass
                            hit_obj['end_x'] = 256.0
                            hit_obj['end_y'] = 192.0
                        else:
                            hit_obj['type'] = 'unknown'
                            
                        hit_objects.append(hit_obj)
                    except ValueError:
                        pass
                        
    # Sort lists to ensure order
    timing_points.sort(key=lambda t: t['time'])
    hit_objects.sort(key=lambda h: h['time'])
    
    # Calculate durations for sliders based on timing points
    slider_mult = difficulty.get('SliderMultiplier', 1.0)
    
    for i, obj in enumerate(hit_objects):
        if obj['type'] == 'slider':
            t = obj['time']
            length = obj.get('slider_length', 0.0)
            slides = obj.get('slider_slides', 1)
            
            # Find the active uninherited timing point (defines base beat length)
            active_uninherited = None
            for tp in timing_points:
                if tp['time'] <= t and tp['uninherited'] == 1:
                    active_uninherited = tp
                elif tp['time'] > t:
                    break
            
            # Find the active timing point (defines velocity multiplier)
            active_tp = None
            for tp in timing_points:
                if tp['time'] <= t:
                    active_tp = tp
                elif tp['time'] > t:
                    break
            
            base_beat_length = active_uninherited['beat_length'] if active_uninherited else 1000.0
            
            # Slider velocity multiplier
            sv_multiplier = 1.0
            if active_tp:
                # If beat_length is negative, it's a relative multiplier: -100 / beat_length
                if active_tp['beat_length'] < 0:
                    sv_multiplier = -100.0 / active_tp['beat_length']
            
            # Duration calculation in milliseconds
            # duration = (length * base_beat_length) / (100 * SliderMultiplier * sv_multiplier)
            # Then multiply by number of slides
            if slider_mult > 0 and sv_multiplier > 0:
                slide_dur = (length * base_beat_length) / (100.0 * slider_mult * sv_multiplier)
                total_dur = slide_dur * slides
                obj['end_time'] = t + total_dur
            else:
                obj['end_time'] = t

    difficulty["star_rating"] = get_star_rating(file_path)
                
    return {
        'general': general,
        'metadata': metadata,
        'difficulty': difficulty,
        'timing_points': timing_points,
        'hit_objects': hit_objects
    }

if __name__ == "__main__":
    # Test script on a sample file
    import sys
    test_file = "maps/10003_0.osu"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        
    if os.path.exists(test_file):
        try:
            res = parse_osu_file(test_file)
            print("Successfully parsed Osu file!")
            print(f"Title: {res['metadata'].get('Title')}")
            print(f"Artist: {res['metadata'].get('Artist')}")
            print(f"Version: {res['metadata'].get('Version')}")
            print(f"Difficulty Settings: {res['difficulty']}")
            print(f"Total Hit Objects: {len(res['hit_objects'])}")
            circles = sum(1 for h in res['hit_objects'] if h['type'] == 'circle')
            sliders = sum(1 for h in res['hit_objects'] if h['type'] == 'slider')
            spinners = sum(1 for h in res['hit_objects'] if h['type'] == 'spinner')
            print(f"  Circles: {circles}, Sliders: {sliders}, Spinners: {spinners}")
            if res['hit_objects']:
                print(f"First object: {res['hit_objects'][0]}")
        except Exception as e:
            print(f"Error parsing: {e}")
    else:
        print(f"Test file not found at: {test_file}")
