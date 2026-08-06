import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set up logging to console and file
from logger_setup import setup_logger
logger = setup_logger("cluster_combos")

def print(*args, **kwargs):
    import builtins
    builtins.print(*args, **kwargs)
    msg = " ".join(str(arg) for arg in args)
    logger.info(msg)

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser import parse_osu_file
from features import compute_note_features

def extract_combos_from_map(file_path):
    try:
        parsed = parse_osu_file(file_path)
        hit_objects = parsed.get('hit_objects', [])
        if len(hit_objects) < 20:
            return []
            
        note_feats = compute_note_features(hit_objects)
        if not note_feats:
            return []
            
        # Group note features by their combo boundaries
        # Note features have 'prev_combo' and 'curr_combo'
        combos = []
        current_combo_notes = []
        
        for nf in note_feats:
            current_combo_notes.append(nf)
            # If the next note starts a new combo (curr_combo == 1), close the current combo group
            if nf['curr_combo'] == 1:
                if len(current_combo_notes) >= 2:
                    combos.append(current_combo_notes)
                current_combo_notes = []
        if len(current_combo_notes) >= 2:
            combos.append(current_combo_notes)
            
        # Calculate features for each combo group
        combo_features = []
        for combo in combos:
            spacings = [n['distance'] for n in combo]
            time_deltas = [n['time_delta'] for n in combo]
            velocities = [n['velocity'] for n in combo]
            angles = [n['angle'] for n in combo if not np.isnan(n['angle'])]
            sliders = sum(1 for n in combo if n['type'] == 'slider')
            
            combo_features.append({
                'length': float(len(combo) + 1),  # Transition count + 1 = objects count
                'mean_spacing': float(np.mean(spacings)),
                'std_spacing': float(np.std(spacings)) if len(spacings) > 1 else 0.0,
                'mean_time_delta': float(np.mean(time_deltas)),
                'mean_angle': float(np.mean(angles)) if angles else 90.0,
                'slider_ratio': float(sliders / len(combo)),
                'max_velocity': float(np.percentile(velocities, 95))
            })
            
        return combo_features
    except Exception:
        return []

def main():
    maps_dir = "data/maps"
    output_path = "data/model_results/combo_clusters.json"
    
    print("Sampling maps to collect combo training data...")
    if not os.path.exists(maps_dir):
        print(f"Error: Maps directory '{maps_dir}' does not exist.")
        return
        
    osu_files = [f for f in os.listdir(maps_dir) if f.endswith(".osu")]
    if not osu_files:
        print("No .osu files found.")
        return
        
    # Sample up to 400 maps to collect enough combos (~5,000+ combos)
    np.random.seed(42)
    sample_files = np.random.choice(osu_files, min(400, len(osu_files)), replace=False)
    
    all_combos = []
    for f in sample_files:
        path = os.path.join(maps_dir, f)
        all_combos.extend(extract_combos_from_map(path))
        
    if len(all_combos) < 100:
        print(f"Too few combos collected ({len(all_combos)}). Cannot train clustering.")
        return
        
    print(f"Collected {len(all_combos)} combos for clustering.")
    df = pd.DataFrame(all_combos)
    
    # Standardize the features
    feature_names = ['length', 'mean_spacing', 'std_spacing', 'mean_time_delta', 'mean_angle', 'slider_ratio', 'max_velocity']
    X = df[feature_names].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train K-Means (K=6 pattern categories)
    print("Training K-Means (K=6 clusters)...")
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Dynamically verify and map cluster names to indices based on centroid values
    centers = kmeans.cluster_centers_
    cluster_names = [""] * 6
    unassigned = list(range(6))
    
    # 1. Long Streams / Stamina: Centroid with the absolute highest 'length' (index 0)
    idx_long = int(np.argmax(centers[:, 0]))
    cluster_names[idx_long] = "Long Streams / Stamina"
    unassigned.remove(idx_long)
    
    # 2. Pause/Break / Rhythmic Gaps: Centroid with the highest 'time_delta' (index 3) among remaining
    idx_gap = int(np.argmax([centers[i, 3] if i in unassigned else -999 for i in range(6)]))
    cluster_names[idx_gap] = "Pause/Break / Gaps"
    unassigned.remove(idx_gap)
    
    # 3. Complex Sliders / Tech: Centroid with the highest 'slider_ratio' (index 5) among remaining
    idx_sliders = int(np.argmax([centers[i, 5] if i in unassigned else -999 for i in range(6)]))
    cluster_names[idx_sliders] = "Complex Sliders"
    unassigned.remove(idx_sliders)
    
    # 4. Fast/Snappy Jumps: Centroid with highest 'spacing' (index 1) among remaining
    idx_jumps = int(np.argmax([centers[i, 1] if i in unassigned else -999 for i in range(6)]))
    cluster_names[idx_jumps] = "Fast/Snappy Jumps"
    unassigned.remove(idx_jumps)
    
    # 5. Raw Streams / Bursts: Centroid with the lowest 'spacing' (index 1) among remaining
    idx_streams = int(np.argmin([centers[i, 1] if i in unassigned else 999 for i in range(6)]))
    cluster_names[idx_streams] = "Raw Streams / Bursts"
    unassigned.remove(idx_streams)
    
    # 6. Normal Jumps / Flow: The remaining cluster
    idx_flow = unassigned[0]
    cluster_names[idx_flow] = "Normal Jumps / Flow"
    
    print("\n--- Centroid Mapping Verification ---")
    for i in range(6):
        c = centers[i]
        print(f"Cluster {i} -> {cluster_names[i]}: length={c[0]:.2f}, spacing={c[1]:.2f}, time_delta={c[3]:.2f}, slider_ratio={c[5]:.2f}")
        
    cluster_data = {
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "cluster_centers": centers.tolist(),
        "cluster_names": cluster_names
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cluster_data, f, indent=4)
        
    print(f"\n[SUCCESS] Saved combo cluster centers to {output_path}")

if __name__ == "__main__":
    main()
