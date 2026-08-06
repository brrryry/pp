import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

DATASET_CSV = "data/processed_dataset.csv"
OUTPUT_DIR = "data/visualizations"

def main():
    if not os.path.exists(DATASET_CSV):
        print(f"Error: Dataset file '{DATASET_CSV}' not found. Please compile the dataset first.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading dataset from {DATASET_CSV}...")
    df = pd.read_csv(DATASET_CSV)
    print(f"Loaded dataset with {len(df)} records.")
    
    # 1. Target Difficulty Distribution
    print("Generating Star Rating distribution plot...")
    plt.figure()
    sns.histplot(data=df, x="star_rating", kde=True, bins=25, color="magenta")
    plt.title("Distribution of Map Difficulties (Star Rating)")
    plt.xlabel("Star Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "star_rating_distribution.png"), dpi=150)
    plt.close()
    
    # 2. Key Correlation Analysis
    print("Generating Feature Correlation plot...")
    feature_cols = [
        'circle_size', 'overall_difficulty', 'density_notes_per_sec',
        'velocity_mean', 'velocity_p95', 'velocity_p99',
        'distance_mean', 'distance_p95', 'distance_p99',
        'time_delta_median', 'angle_sharp_ratio', 'angle_wide_ratio', 'approach_rate',
        'star_rating'
    ]
    # Filter to columns that exist in the dataframe
    corr_cols = [col for col in feature_cols if col in df.columns]
    
    plt.figure(figsize=(12, 10))
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix: Geometric Features vs. Star Rating")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_correlation.png"), dpi=150)
    plt.close()
    
    # 3. Density vs. Star Rating Scatter Plot
    print("Generating Density vs. Star Rating scatter plot...")
    plt.figure()
    sns.scatterplot(
        data=df, 
        x="density_notes_per_sec", 
        y="star_rating", 
        hue="circle_size", 
        palette="viridis", 
        alpha=0.7
    )
    plt.title("Note Density vs. Star Rating")
    plt.xlabel("Note Density (Objects per Second)")
    plt.ylabel("Star Rating")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "density_vs_star_rating.png"), dpi=150)
    plt.close()
    
    # 4. Spacing vs. Speed vs. Star Rating
    print("Generating Velocity vs. Spacing scatter plot...")
    plt.figure()
    sns.scatterplot(
        data=df, 
        x="distance_mean", 
        y="velocity_mean", 
        hue="star_rating", 
        palette="magma", 
        alpha=0.8
    )
    plt.title("Average Spacing vs. Average Aim Velocity")
    plt.xlabel("Mean Spacing (Osu! Pixels)")
    plt.ylabel("Mean Aim Velocity (Pixels/ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "spacing_vs_velocity.png"), dpi=150)
    plt.close()
    
    print(f"\nAll visualizations successfully saved to '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()
