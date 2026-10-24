"""Classification Models for Football Prediction (Week 11 & 12).

Supports both:
1. Binary Classification: BTTS (Both Teams To Score) & AH Cover
2. Multi-class Classification: 1X2 Match Result (Home=2, Draw=1, Away=0)

Implements and compares coursework classifiers (สไลด์หน้า 70):
- Logistic Regression (สไลด์หน้า 52-53)
- K-Nearest Neighbors - KNN (สไลด์หน้า 53-54)
- Gaussian Naive Bayes (สไลด์หน้า 55)
- Decision Tree Classifier (สไลด์หน้า 56)
- Random Forest Classifier (สไลด์หน้า 57)
- Support Vector Machine - SVC (สไลด์หน้า 65-72)
- AdaBoost (สไลด์หน้า 58-60)
- XGBoost (สไลด์หน้า 61-62)
- LightGBM (สไลด์หน้า 63-64)

Evaluates using Week 11 Metrics:
- Accuracy, Precision, Recall, Specificity, F1-Score
- ROC-AUC Curve & Score (สไลด์หน้า 42-43)
- Confusion Matrix Heatmap (สไลด์หน้า 41)
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
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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


def train_and_evaluate_classifiers(
    df: pd.DataFrame,
    target_col: str = "target_btts",
    is_multiclass: bool = False,
    test_size: float = 0.20,
    random_state: int = 42
) -> pd.DataFrame:
    """Train multiple classifiers and output performance metrics table (สไลด์หน้า 70)."""
    task_type = "Multi-class (1X2)" if is_multiclass else "Binary Classification"
    print(f"\n=======================================================")
    print(f"[Week 12] Training Classifiers ({task_type}) for: '{target_col}'")
    print(f"=======================================================")

    features = get_selected_features(df, target_col=target_col, top_n=15)
    X = df[features].values
    y = df[target_col].values

    # Train/Test Split (สไลด์หน้า 45-46, 71)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Candidate Classifiers from DSA.pdf
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state, n_jobs=-1),
        "SVM (RBF Kernel)": SVC(kernel="rbf", C=1.0, probability=True, random_state=random_state),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=random_state),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=random_state, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=random_state, verbose=-1)
    }

    results = []
    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None
    best_y_pred = None
    best_y_proba = None

    roc_data = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Probabilities
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)
        else:
            y_proba = None

        # Metrics calculation (สไลด์หน้า 41-43)
        acc = accuracy_score(y_test, y_pred)
        
        if is_multiclass:
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            spec = 0.0
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr") if y_proba is not None else 0.0
        else:
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            auc = roc_auc_score(y_test, y_proba[:, 1]) if y_proba is not None else 0.0
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_data[name] = (fpr, tpr, auc)

        results.append({
            "Algorithm": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "Specificity": round(spec, 4) if not is_multiclass else "-",
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model
            best_y_pred = y_pred
            best_y_proba = y_proba

    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
    print("\n[Classification Model Comparison Table] (สไลด์หน้า 70):")
    print(results_df.to_string(index=False))

    # Save Best Model with pickle (สไลด์หน้า 71)
    save_path = MODELS_DIR / f"best_classifier_{target_col}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({"model": best_model_obj, "scaler": scaler, "features": features}, f)
    print(f"\n  - Saved Best Classifier ({best_model_name}, F1={best_f1:.4f}) to: {save_path}")

    # Plot Confusion Matrix for Best Model (สไลด์หน้า 41)
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, best_y_pred)
    if is_multiclass:
        labels = ["Away (0)", "Draw (1)", "Home (2)"]
    else:
        labels = ["Negative (0)", "Positive (1)"]
        
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix: Best Model ({best_model_name})\nTarget: {target_col}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    cm_path = REPORTS_FIG_DIR / f"confusion_matrix_{target_col}.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"  - Saved Confusion Matrix heatmap to: {cm_path}")

    # Plot ROC Curves for Binary Classification (สไลด์หน้า 42-43)
    if not is_multiclass and roc_data:
        plt.figure(figsize=(9, 7))
        for name, (fpr, tpr, auc_val) in roc_data.items():
            plt.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC = {auc_val:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Guess (AUC = 0.50)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Recall / Sensitivity)")
        plt.title(f"Receiver Operating Characteristic (ROC) Curves for {target_col}")
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        roc_path = REPORTS_FIG_DIR / f"roc_curves_{target_col}.png"
        plt.savefig(roc_path, dpi=200)
        plt.close()
        print(f"  - Saved ROC Curves plot to: {roc_path}")

    # Export results to CSV
    results_csv = REPORTS_DIR / f"classification_results_{target_col}.csv"
    results_df.to_csv(results_csv, index=False)
    return results_df


if __name__ == "__main__":
    df = load_and_clean_data()
    # 1. Binary: BTTS
    train_and_evaluate_classifiers(df, target_col="target_btts", is_multiclass=False)
    # 2. Multi-class: 1X2
    train_and_evaluate_classifiers(df, target_col="target_1x2", is_multiclass=True)
