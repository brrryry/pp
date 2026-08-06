import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging to console and file
from logger_setup import setup_logger
logger = setup_logger("train")

def print(*args, **kwargs):
    import builtins
    builtins.print(*args, **kwargs)
    msg = " ".join(str(arg) for arg in args)
    logger.info(msg)

DATASET_DB = "data/osu_profiler.db"
TABLE_NAME = "beatmaps"
OUTPUT_DIR = "data/model_results"

def main():
    if not os.path.exists(DATASET_DB):
        print(f"Error: Dataset file '{DATASET_DB}' not found. Please compile the dataset first (run dataset.py).")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading dataset from {DATASET_DB}...")
    conn = sqlite3.connect(DATASET_DB)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    
    # Define features to exclude (metadata and target column)
    exclude_cols = ['title', 'artist', 'version', 'creator', 'filename', 'beatmapset_id', 'star_rating', 'approach_rate', 'circle_size', 'overall_difficulty', 'hp_drain']
    
    # Feature columns (everything that is numeric and not in exclude list)
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols]
    y = df['star_rating']
    
    # Drop rows with NaN targets (if any)
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Impute any missing values in features (e.g. if angles were nan)
    X = X.fillna(X.mean())
    
    print(f"Total samples: {len(X)}")
    print(f"Number of features: {len(feature_cols)}")
    print("Features list:", feature_cols)
    
    # Split into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining model: {name}...")
        # Ridge benefits from scaling, tree-based models don't require it but it doesn't hurt
        if name == "Ridge Regression":
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
        # Evaluate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        results[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")
        
        # Plot actual vs predicted
        plt.figure()
        sns.scatterplot(x=y_test, y=predictions, alpha=0.5, color="purple")
        # Line of perfect prediction
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.title(f"{name}: Actual vs. Predicted Difficulty")
        plt.xlabel("Actual Star Rating")
        plt.ylabel("Predicted Star Rating")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_actual_vs_predicted.png"), dpi=150)
        plt.close()

    # 4. Feature Importance for Random Forest
    rf = models["Random Forest Regressor"]
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top 15 features
    top_n = min(15, len(feature_cols))
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=importances[indices[:top_n]], 
        y=[feature_cols[i] for i in indices[:top_n]],
        palette="viridis"
    )
    plt.title("Random Forest Top Feature Importances for Map Difficulty")
    plt.xlabel("Relative Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "random_forest_feature_importances.png"), dpi=150)
    plt.close()
    
    # Output metrics table
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_metrics.csv"))
    print("\nModel Comparison Table:")
    print(metrics_df)
    
    # Save Ridge Regression weights, scaler statistics, and feature names as JSON for secure, dependency-free inference
    import json
    ridge_model = models["Ridge Regression"]
    model_params = {
        "feature_names": feature_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "ridge_coef": ridge_model.coef_.tolist(),
        "ridge_intercept": float(ridge_model.intercept_)
    }
    with open(os.path.join(OUTPUT_DIR, "model_parameters.json"), "w") as f:
        json.dump(model_params, f, indent=4)
    print("\n[SUCCESS] Saved Ridge Regression model parameters to data/model_results/model_parameters.json")
    
if __name__ == "__main__":
    main()
