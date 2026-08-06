import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Set up logging to console and file
from ..logger_setup import setup_logger
logger = setup_logger("evaluate_clustering")

def print(*args, **kwargs):
    import builtins
    builtins.print(*args, **kwargs)
    msg = " ".join(str(arg) for arg in args)
    logger.info(msg)

# Add parent directory to path to import local parser and features
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser import parse_osu_file
from features import compute_note_features
from cluster_combos import extract_combos_from_map

def main():
    maps_dir = "data/maps"
    output_json = "data/model_results/clustering_evaluation.json"
    output_dir = "data/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    print("Evaluating combo clustering architectures...")
    if not os.path.exists(maps_dir):
        print(f"Error: Maps directory '{maps_dir}' does not exist.")
        return
        
    osu_files = [f for f in os.listdir(maps_dir) if f.endswith(".osu")]
    if not osu_files:
        print("No .osu files found.")
        return
        
    # Sample 400 maps to collect enough combos (~5,000+ combos)
    np.random.seed(42)
    sample_files = np.random.choice(osu_files, min(400, len(osu_files)), replace=False)
    
    print(f"Sampling {len(sample_files)} maps to collect combo features...")
    all_combos = []
    for f in sample_files:
        path = os.path.join(maps_dir, f)
        all_combos.extend(extract_combos_from_map(path))
        
    print(f"Collected {len(all_combos)} combos for analysis.")
    df = pd.DataFrame(all_combos)
    
    # Standardize the features
    feature_names = ['length', 'mean_spacing', 'std_spacing', 'mean_time_delta', 'mean_angle', 'slider_ratio', 'max_velocity']
    X = df[feature_names].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Range of K to test
    k_range = list(range(2, 11))
    
    # 1. K-Means Evaluation
    print("\nEvaluating K-Means...")
    kmeans_inertia = []
    kmeans_silhouette = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        kmeans_inertia.append(float(kmeans.inertia_))
        
        # Calculate silhouette score on a sample of 2000 points to keep calculation fast
        sample_indices = np.random.choice(len(X_scaled), min(2000, len(X_scaled)), replace=False)
        sil = float(silhouette_score(X_scaled[sample_indices], labels[sample_indices]))
        kmeans_silhouette.append(sil)
        print(f"  K={k:2d}: Inertia (WCSS) = {kmeans.inertia_:.2f}, Silhouette = {sil:.4f}")
        
    # 2. GMM Evaluation (Bayesian & Akaike Information Criteria)
    print("\nEvaluating Gaussian Mixture Models (GMM)...")
    gmm_bic = []
    gmm_aic = []
    
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=2)
        gmm.fit(X_scaled)
        bic = float(gmm.bic(X_scaled))
        aic = float(gmm.aic(X_scaled))
        gmm_bic.append(bic)
        gmm_aic.append(aic)
        print(f"  K={k:2d}: BIC = {bic:.2f}, AIC = {aic:.2f}")
        
    # 3. DBSCAN Exploration
    print("\nExploring DBSCAN (density-based)...")
    dbscan_runs = []
    for eps in [0.3, 0.5, 0.7, 1.0]:
        for min_samples in [5, 10, 15]:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = float(n_noise / len(labels))
            dbscan_runs.append({
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": int(n_clusters),
                "noise_ratio": noise_ratio
            })
            print(f"  eps={eps:.1f}, min_samples={min_samples:2d}: clusters={n_clusters:2d}, noise={noise_ratio*100:.1f}%")
            
    # Export results to JSON
    evaluation_data = {
        "k_range": k_range,
        "kmeans": {
            "inertia": kmeans_inertia,
            "silhouette": kmeans_silhouette
        },
        "gmm": {
            "bic": gmm_bic,
            "aic": gmm_aic
        },
        "dbscan": dbscan_runs
    }
    
    with open(output_json, "w") as f:
        json.dump(evaluation_data, f, indent=4)
    print(f"\nSaved evaluation metrics to {output_json}")
    
    # 4. Generate Plot: K-Means Elbow Curve & Silhouette
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (WCSS)', color=color)
    ax1.plot(k_range, kmeans_inertia, marker='o', color=color, linewidth=2, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Coefficient', color=color)
    ax2.plot(k_range, kmeans_silhouette, marker='s', linestyle='--', color=color, linewidth=2, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('K-Means Clustering Analysis: Elbow Method & Silhouette Scores')
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "clustering_elbow.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved K-Means evaluation plot to {plot_path}")
    
    # 5. Generate Plot: GMM BIC vs AIC
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, gmm_bic, marker='o', color='crimson', linewidth=2, label='BIC (Bayesian Info Criterion)')
    plt.plot(k_range, gmm_aic, marker='x', linestyle='--', color='teal', linewidth=2, label='AIC (Akaike Info Criterion)')
    plt.xlabel('Number of Components (K)')
    plt.ylabel('Score')
    plt.title('GMM Information Criteria (Lower is Better)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    gmm_plot_path = os.path.join(output_dir, "gmm_evaluation.png")
    plt.savefig(gmm_plot_path, dpi=150)
    plt.close()
    print(f"Saved GMM evaluation plot to {gmm_plot_path}")
    
    print("\n[SUCCESS] Clustering architecture evaluation complete!")

if __name__ == "__main__":
    main()
