"""Feature Selection and Correlation Analysis (Week 10).

Implements:
1. Filter Method: Pearson Correlation Matrix (.corr()) with Targets
2. Embedded Method 1: Lasso (L1 Regularization) Selection (สไลด์หน้า 21, 38)
3. Embedded Method 2: Random Forest Feature Importance (สไลด์หน้า 21)
4. Visualizations: Correlation Heatmap and Feature Importance Bar Chart (สไลด์หน้า 25-26)
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler

from prepare_data import load_and_clean_data

REPORTS_FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)


CANDIDATE_FEATURES = [
    "ah_open", "pool_fav_open", "pool_dog_open", "draw_ft_odds",
    "draw_ht_odds", "ou_line_open", "ou_over_price", "ou_under_price",
    "ah_close", "ou_line_close", "pool_fav_close", "pool_dog_close", "draw_ft_close",
    "btts_open", "goal_line_half_open",
    "rule_match", "rule_near", "rule_dist", "draw_ht_high", "draw_ht_vhigh", "draw_ft_match",
    "home_gf_avg", "home_ga_avg", "home_total_avg", "home_over_rate",
    "away_gf_avg", "away_ga_avg", "away_total_avg", "away_over_rate",
    "goals_avg_diff", "expected_total", "is_weekend"
]


def run_correlation_analysis(df: pd.DataFrame, target_col: str = "target_total_goals") -> pd.Series:
    """Compute Pearson Correlation between features and target (สไลด์หน้า 20, 31-36)."""
    print(f"\n[Week 10 - Filter Method] Computing Pearson Correlation with '{target_col}'...")
    cols = [c for c in CANDIDATE_FEATURES if c in df.columns] + [target_col]
    corr_matrix = df[cols].corr()
    target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)

    print("  Top 5 Positive Correlations:")
    print(target_corr.head(5).to_string())
    print("\n  Top 5 Negative Correlations:")
    print(target_corr.tail(5).to_string())

    # Plot top correlated features
    top_features = target_corr.abs().sort_values(ascending=False).head(15).index.tolist()
    plt.figure(figsize=(10, 6))
    top_corr = target_corr.loc[top_features]
    colors = ["#2ecc71" if val >= 0 else "#e74c3c" for val in top_corr.values]
    plt.barh(top_corr.index[::-1], top_corr.values[::-1], color=colors[::-1])
    plt.title(f"Top 15 Feature Correlations with {target_col} (Filter Method)")
    plt.xlabel("Correlation Coefficient (r)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path = REPORTS_FIG_DIR / f"correlation_{target_col}.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"  - Saved correlation plot to: {plot_path}")
    return target_corr


def run_lasso_selection(df: pd.DataFrame, target_col: str = "target_total_goals", alpha: float = 0.005) -> list[str]:
    """Perform Lasso (L1 Regularization) Feature Selection (สไลด์หน้า 21, 38)."""
    print(f"\n[Week 10 - Embedded Method 1] Running L1 Regularization for '{target_col}'...")
    features = [c for c in CANDIDATE_FEATURES if c in df.columns]
    X = MinMaxScaler().fit_transform(df[features].values)
    y = df[target_col].values

    is_discrete = df[target_col].nunique() <= 5
    if is_discrete:
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        base_lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=42)
        n_classes = len(np.unique(y))
        if n_classes > 2:
            l1_model = OneVsRestClassifier(base_lr)
            l1_model.fit(X, y)
            coef_vals = np.mean(np.abs(np.array([est.coef_[0] for est in l1_model.estimators_])), axis=0)
        else:
            l1_model = base_lr
            l1_model.fit(X, y)
            coef_vals = np.abs(l1_model.coef_[0])
        coef = pd.Series(coef_vals, index=features)
    else:
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X, y)
        coef = pd.Series(lasso.coef_, index=features)

    selected = coef[coef.abs() > 1e-4].sort_values(ascending=False)
    if selected.empty:
        selected = coef.abs().sort_values(ascending=False).head(10)

    print(f"  - Selected {len(selected)} / {len(features)} active features.")
    print("  Top Selected Features by L1 Weight:")
    print(selected.head(10).to_string())

    if not selected.empty:
        plt.figure(figsize=(10, 6))
        selected.head(15).plot(kind="bar", color="#3498db")
        plt.title(f"Top Features Selected by L1 Regularization for {target_col}")
        plt.ylabel("L1 Coefficient Magnitude")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plot_path = REPORTS_FIG_DIR / f"lasso_selection_{target_col}.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"  - Saved L1 selection plot to: {plot_path}")

    return selected.index.tolist()


def run_random_forest_importance(df: pd.DataFrame, target_col: str = "target_total_goals") -> pd.Series:
    """Calculate Random Forest Feature Importance (สไลด์หน้า 21)."""
    print(f"\n[Week 10 - Embedded Method 2] Computing Random Forest Feature Importance for '{target_col}'...")
    features = [c for c in CANDIDATE_FEATURES if c in df.columns]
    X = df[features].values
    y = df[target_col].values

    is_discrete = df[target_col].nunique() <= 5
    if is_discrete:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    print("  Top 10 Most Important Features:")
    print(importances.head(10).to_string())

    plt.figure(figsize=(10, 6))
    importances.head(15)[::-1].plot(kind="barh", color="#9b59b6")
    plt.title(f"Random Forest Feature Importance for {target_col}")
    plt.xlabel("Importance Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path = REPORTS_FIG_DIR / f"rf_importance_{target_col}.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"  - Saved Random Forest Importance plot to: {plot_path}")
    return importances


def get_selected_features(df: pd.DataFrame, target_col: str = "target_total_goals", top_n: int = 15) -> list[str]:
    """Combine Correlation, Lasso, and Random Forest rankings to select optimal features."""
    rf_imp = run_random_forest_importance(df, target_col)
    lasso_feats = run_lasso_selection(df, target_col)
    run_correlation_analysis(df, target_col)

    # Intersection of top RF and Lasso features, prioritized by RF importance
    ranked = [f for f in rf_imp.index if f in lasso_feats or f in rf_imp.index[:top_n]]
    final_features = ranked[:top_n]
    print(f"\n[Feature Selection Complete] Final Selected {len(final_features)} Features:")
    for i, f in enumerate(final_features, 1):
        print(f"  {i}. {f}")
    return final_features


if __name__ == "__main__":
    df = load_and_clean_data()
    get_selected_features(df, target_col="target_total_goals", top_n=15)
