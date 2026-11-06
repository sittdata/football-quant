"""Multi-class modeling and evaluation for 1X2 Football Match Outcomes.

Models trained and compared:
1. Bookmaker Market Odds Baseline (Normalized Implied Probability)
2. Multinomial Logistic Regression (L2 regularized)
3. Random Forest Classifier
4. Gradient Boosting Classifier (HistGradientBoosting / XGBoost)

Metrics:
- Accuracy
- Macro F1-Score
- Multi-class Log Loss (Cross-Entropy)
- Multi-class Brier Score
- Confusion Matrix
- Feature Importance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS_NO_ODDS = [
    "home_form_pts",
    "home_gf_avg",
    "home_ga_avg",
    "home_gd_avg",
    "home_rest_days",
    "away_form_pts",
    "away_gf_avg",
    "away_ga_avg",
    "away_gd_avg",
    "away_rest_days",
    "diff_form_pts",
    "diff_gd_avg",
    "diff_rest_days",
]

FEATURE_COLS_WITH_ODDS = FEATURE_COLS_NO_ODDS + [
    "market_p_home",
    "market_p_draw",
    "market_p_away",
]

CLASS_NAMES = ["Away", "Draw", "Home"]  # Corresponding to 0, 1, 2


@dataclass
class ModelResult:
    name: str
    accuracy: float
    macro_f1: float
    log_loss_val: float
    brier_score: float
    confusion_mat: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute multi-class Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2"""
    n_classes = y_prob.shape[1]
    y_one_hot = np.zeros_like(y_prob)
    for i, label in enumerate(y_true):
        if 0 <= int(label) < n_classes:
            y_one_hot[i, int(label)] = 1.0
    return float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))


def evaluate_predictions(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> ModelResult:
    """Calculate standard academic classification metrics."""
    y_pred = np.argmax(y_prob, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    ll = float(log_loss(y_true, y_prob, labels=[0, 1, 2]))
    brier = multiclass_brier_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    return ModelResult(
        name=name,
        accuracy=acc,
        macro_f1=f1,
        log_loss_val=ll,
        brier_score=brier,
        confusion_mat=cm,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def run_1x2_experiment(
    df: pd.DataFrame,
    include_market_odds: bool = True,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end 1X2 experiment with temporal train/test split."""
    # Filter only rows with valid target
    valid_df = df[df["target_1x2"].notna()].copy()
    valid_df["target_1x2"] = valid_df["target_1x2"].astype(int)

    feature_cols = FEATURE_COLS_WITH_ODDS if include_market_odds else FEATURE_COLS_NO_ODDS
    available_cols = [c for c in feature_cols if c in valid_df.columns]

    # Chronological Split
    split_idx = int(len(valid_df) * train_ratio)
    train_df = valid_df.iloc[:split_idx]
    test_df = valid_df.iloc[split_idx:]

    X_train = train_df[available_cols].values
    y_train = train_df["target_1x2"].values

    X_test = test_df[available_cols].values
    y_test = test_df["target_1x2"].values

    results: List[ModelResult] = []

    # 1. Market Baseline (if odds columns exist and test has them)
    if "market_p_home" in test_df.columns and test_df["market_p_home"].notna().any():
        market_probs = np.column_stack([
            test_df["market_p_away"].fillna(0.33).values,
            test_df["market_p_draw"].fillna(0.33).values,
            test_df["market_p_home"].fillna(0.34).values,
        ])
        # Re-normalize to guarantee sum to 1
        market_probs = market_probs / market_probs.sum(axis=1, keepdims=True)
        results.append(evaluate_predictions("Market Implied Baseline", y_test, market_probs))

    # 2. Multinomial Logistic Regression Pipeline
    logreg_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=seed)),
    ])
    logreg_pipe.fit(X_train, y_train)
    p_logreg = logreg_pipe.predict_proba(X_test)
    results.append(evaluate_predictions("Multinomial Logistic Regression", y_test, p_logreg))

    # 3. Random Forest Classifier
    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed)),
    ])
    rf_pipe.fit(X_train, y_train)
    p_rf = rf_pipe.predict_proba(X_test)
    results.append(evaluate_predictions("Random Forest", y_test, p_rf))

    # 4. HistGradientBoosting Classifier
    hgb_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=seed)),
    ])
    hgb_pipe.fit(X_train, y_train)
    p_hgb = hgb_pipe.predict_proba(X_test)
    results.append(evaluate_predictions("HistGradientBoosting", y_test, p_hgb))

    # Feature Importance from Random Forest
    rf_model = rf_pipe.named_steps["model"]
    importances = pd.Series(rf_model.feature_importances_, index=available_cols).sort_values(ascending=False)

    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "features_used": available_cols,
        "results": results,
        "feature_importances": importances,
        "test_df": test_df,
    }
