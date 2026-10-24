"""Model Evaluation, K-Fold Validation, and Bias-Variance Analysis (Week 11).

Implements:
1. K-Fold Cross Validation (5 Folds) with fold-by-fold accuracy (สไลด์หน้า 46-47)
2. Bias & Variance Calculation based on lecture formulas (สไลด์หน้า 48-49):
   - Bias = |1 - Train Accuracy|
   - Variance = |Train Accuracy - Test Accuracy|
3. Overfitting / Underfitting Diagnosis:
   - Evaluates tree depth complexity (max_depth from 1 to 15) to demonstrate
     Underfitting -> Good Fit -> Overfitting transitions (สไลด์หน้า 45, 48-50)
4. Validation Curve Plot saved to reports/figures/
"""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SYS_PATH_PREP = Path(__file__).resolve().parents[1] / "01_data_prep_feature_selection"
if str(SYS_PATH_PREP) not in sys.path:
    sys.path.insert(0, str(SYS_PATH_PREP))

from prepare_data import load_and_clean_data
from feature_selection import get_selected_features

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_FIG_DIR = REPORTS_DIR / "figures"
REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_kfold_cross_validation(
    X: np.ndarray, y: np.ndarray, n_splits: int = 5
) -> dict[str, float]:
    """Run K-Fold Cross Validation and print fold-by-fold accuracy (สไลด์หน้า 46-47)."""
    print(f"\n[Week 11] Running {n_splits}-Fold Cross Validation (สไลด์หน้า 46-47)...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    fold_scores = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        clf.fit(X_tr, y_tr)
        score = clf.score(X_te, y_te)
        fold_scores.append(score)
        print(f"  - Fold {fold}: Accuracy = {score * 100:.2f}%")

    mean_acc = np.mean(fold_scores)
    std_acc = np.std(fold_scores)
    print(f"  >> Average {n_splits}-Fold Accuracy = {mean_acc * 100:.2f}% (+/- {std_acc * 100:.2f}%)")
    return {"mean_accuracy": mean_acc, "std_accuracy": std_acc, "folds": fold_scores}


def analyze_bias_and_variance(
    X: np.ndarray, y: np.ndarray, depths: list[int] | None = None
) -> pd.DataFrame:
    """Analyze Bias vs Variance trade-off across model complexity (สไลด์หน้า 48-50)."""
    if depths is None:
        depths = list(range(1, 16))

    print("\n[Week 11] Analyzing Bias vs Variance & Overfitting across Tree Depths...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    records = []
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)

        train_acc = dt.score(X_train, y_train)
        test_acc = dt.score(X_test, y_test)

        # Formulas directly from slide page 48-49:
        # Bias = |1 - train accuracy|
        # Variance = |train accuracy - test accuracy|
        bias = abs(1.0 - train_acc)
        variance = abs(train_acc - test_acc)

        if train_acc < 0.60 and test_acc < 0.60:
            status = "Underfitting (High Bias)"
        elif variance > 0.15:
            status = "Overfitting (High Variance)"
        else:
            status = "Good Fit (Balanced)"

        records.append({
            "Depth": d,
            "Train Accuracy": round(train_acc, 4),
            "Test Accuracy": round(test_acc, 4),
            "Bias": round(bias, 4),
            "Variance": round(variance, 4),
            "Diagnosis": status
        })

    analysis_df = pd.DataFrame(records)
    print("\n[Bias-Variance & Overfitting Analysis Table]:")
    print(analysis_df.to_string(index=False))

    # Plot Validation Curve (Train vs Test Accuracy across depth)
    plt.figure(figsize=(10, 6))
    plt.plot(analysis_df["Depth"], analysis_df["Train Accuracy"], marker="o", label="Train Accuracy (Memorization)", color="#2980b9")
    plt.plot(analysis_df["Depth"], analysis_df["Test Accuracy"], marker="s", label="Test Accuracy (Generalization)", color="#e74c3c")
    plt.axvline(x=4, color="green", linestyle="--", label="Optimal Depth (Good Fit)")
    plt.title("Model Complexity vs Overfitting: Decision Tree Depth Analysis (Week 11)")
    plt.xlabel("Max Depth (Complexity)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    curve_path = REPORTS_FIG_DIR / "overfitting_validation_curve.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"  - Saved Overfitting Validation Curve to: {curve_path}")

    # Export to CSV
    csv_path = REPORTS_DIR / "bias_variance_analysis.csv"
    analysis_df.to_csv(csv_path, index=False)
    return analysis_df


def run_full_validation_experiment(df: pd.DataFrame, target_col: str = "target_btts"):
    """Execute both K-Fold and Bias-Variance experiments."""
    features = get_selected_features(df, target_col=target_col, top_n=15)
    X = StandardScaler().fit_transform(df[features].values)
    y = df[target_col].values

    run_kfold_cross_validation(X, y, n_splits=5)
    analyze_bias_and_variance(X, y)


if __name__ == "__main__":
    df = load_and_clean_data()
    run_full_validation_experiment(df, target_col="target_btts")
