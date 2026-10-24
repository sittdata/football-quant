"""Regression Models for Expected Total Goals Prediction (Week 13).

Implements and compares:
1. Linear Regression (Baseline, สไลด์หน้า 76-77)
2. Support Vector Regression - SVR with RBF Kernel (สไลด์หน้า 81-83)
3. Decision Tree Regressor (สไลด์หน้า 78)
4. Random Forest Regressor (สไลด์หน้า 79)
5. Gradient Boosting / XGBoost Regressor (สไลด์หน้า 80)
6. LightGBM Regressor (สไลด์หน้า 63)

Evaluates using Week 11 & 13 Metrics:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score (Coefficient of Determination)
- Saves best model via pickle (สไลด์หน้า 83)
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
import xgboost as xgb

SYS_PATH_PREP = Path(__file__).resolve().parents[1] / "01_data_prep_feature_selection"
if str(SYS_PATH_PREP) not in sys.path:
    sys.path.insert(0, str(SYS_PATH_PREP))

from prepare_data import load_and_clean_data
from feature_selection import get_selected_features

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_evaluate_regressors(
    df: pd.DataFrame,
    target_col: str = "target_total_goals",
    test_size: float = 0.20,
    random_state: int = 42
) -> pd.DataFrame:
    """Train multiple regression models and produce comparison metrics."""
    print(f"\n=======================================================")
    print(f"[Week 13] Training Regression Models for: '{target_col}'")
    print(f"=======================================================")

    features = get_selected_features(df, target_col=target_col, top_n=15)
    X = df[features].values
    y = df[target_col].values

    # Train / Test Split (สไลด์หน้า 45-46, 73, 83)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scaling with RobustScaler (สไลด์หน้า 15, 83)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define All Coursework Regressors
    models = {
        "Linear Regression": LinearRegression(),
        "SVR (RBF Kernel)": SVR(C=1.5, epsilon=0.2),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_state),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_state),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_state, verbose=-1)
    }

    results = []
    best_model_name = None
    best_rmse = float("inf")
    best_model_obj = None
    best_preds = None

    print(f"\nEvaluating on {len(X_train)} Train samples, {len(X_test)} Test samples:")
    for name, model in models.items():
        # Train model (สไลด์หน้า 73-83)
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Metrics (สไลด์หน้า 43-44, 74)
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_test_pred)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        overfit_gap = train_rmse - rmse  # train vs test error gap (สไลด์หน้า 45)

        results.append({
            "Algorithm": name,
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE (Test)": round(rmse, 4),
            "RMSE (Train)": round(train_rmse, 4),
            "R² Score": round(r2, 4),
            "Overfit Gap": round(abs(overfit_gap), 4)
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model_obj = model
            best_preds = y_test_pred

    results_df = pd.DataFrame(results).sort_values("RMSE (Test)")
    print("\n[Regression Model Comparison Table]:")
    print(results_df.to_string(index=False))

    # Save Best Model with pickle (สไลด์หน้า 83)
    save_path = MODELS_DIR / "best_total_goals_regressor.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({"model": best_model_obj, "scaler": scaler, "features": features}, f)
    print(f"\n  - Saved Best Model ({best_model_name}, RMSE={best_rmse:.4f}) to: {save_path}")

    # Plot Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    x_indices = np.arange(len(results_df))
    plt.bar(x_indices - 0.2, results_df["RMSE (Test)"], width=0.4, label="RMSE (Lower is Better)", color="#e67e22")
    plt.bar(x_indices + 0.2, results_df["MAE"], width=0.4, label="MAE (Lower is Better)", color="#3498db")
    plt.xticks(x_indices, results_df["Algorithm"], rotation=25, ha="right")
    plt.title(f"Comparison of Regression Models for {target_col}")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart_path = REPORTS_FIG_DIR / "regression_comparison.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"  - Saved comparison chart to: {chart_path}")

    # Plot Actual vs Predicted for Best Model (สไลด์หน้า 74, 82)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_preds, alpha=0.35, color="#2c3e50", edgecolors="none")
    ideal_line = np.linspace(min(y_test), max(y_test), 100)
    plt.plot(ideal_line, ideal_line, color="#e74c3c", linestyle="--", label="Ideal Prediction (y = x)")
    plt.title(f"Best Regressor ({best_model_name}): Actual vs Predicted {target_col}")
    plt.xlabel("Actual Goals")
    plt.ylabel("Predicted Goals")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    pred_path = REPORTS_FIG_DIR / "best_regression_actual_vs_pred.png"
    plt.savefig(pred_path, dpi=200)
    plt.close()
    print(f"  - Saved Actual vs Predicted chart to: {pred_path}")

    # Export comparison table to CSV
    results_csv = REPORTS_DIR / "regression_results.csv"
    results_df.to_csv(results_csv, index=False)
    return results_df


if __name__ == "__main__":
    df = load_and_clean_data()
    train_and_evaluate_regressors(df, target_col="target_total_goals")
