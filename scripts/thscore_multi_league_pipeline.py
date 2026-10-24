"""Thscore Multi-League Ingestion & Machine Learning Pipeline.

Fetches real matches across hundreds of global leagues from Thscore API,
extracts Half-Time scores, Corner kicks, Cards, and 1X2 outcomes,
and trains Machine Learning models (Week 11-14) on the multi-league dataset.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Fix console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fbpredict.config import Config
from fbpredict.db import init_db
from fbpredict.providers.thscore_api import ThscoreApiProvider

REPORTS_DIR = ROOT_DIR / "dsa_project" / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "dsa_project" / "models"
DATA_DIR = ROOT_DIR / "dsa_project" / "data"

for p in [FIGURES_DIR, MODELS_DIR, DATA_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def fetch_and_build_thscore_dataset(dates: list[str]) -> pd.DataFrame:
    """Pull matches across all leagues for specified dates from Thscore API."""
    cfg = Config.from_env()
    con = init_db(cfg.db_path)
    provider = ThscoreApiProvider(config=cfg, con=con)

    all_matches = []
    print(f"\n[Thscore Ingestion] Fetching global matches for dates: {dates}")
    for d in dates:
        print(f"  - Requesting schedule for {d}...")
        try:
            matches = provider.get_schedule_by_date(d)
            all_matches.extend(matches)
            print(f"    Fetched {len(matches)} matches.")
        except Exception as e:
            print(f"    Failed for {d}: {e}")
        time.sleep(2.5)  # Rate limit courtesy delay

    # Also fetch today's live scores
    time.sleep(2.0)
    try:
        print("  - Requesting today's live scores...")
        live = provider.get_livescores()
        all_matches.extend(live)
        print(f"    Fetched {len(live)} live matches.")
    except Exception as e:
        print(f"    Failed live fetch: {e}")

    # Deduplicate matches by match_id
    seen_ids = set()
    unique_matches = []
    for m in all_matches:
        if m.match_id not in seen_ids:
            seen_ids.add(m.match_id)
            unique_matches.append(m)

    print(f"[Thscore Ingestion] Total unique matches collected: {len(unique_matches)}")

    # Convert to DataFrame
    rows = []
    for m in unique_matches:
        # Keep completed matches with valid scores
        if m.status == -1 and m.home_goals is not None and m.away_goals is not None:
            total_goals = m.home_goals + m.away_goals
            goal_diff = m.home_goals - m.away_goals
            ht_total = (m.ht_home or 0) + (m.ht_away or 0) if m.ht_home is not None else None
            total_corners = (m.home_corner or 0) + (m.away_corner or 0) if m.home_corner is not None else None
            
            # 1X2 result: 2 = Home, 1 = Draw, 0 = Away
            result_1x2 = 2 if goal_diff > 0 else (1 if goal_diff == 0 else 0)

            rows.append({
                "match_id": m.match_id,
                "league": m.league,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "home_goals": m.home_goals,
                "away_goals": m.away_goals,
                "total_goals": total_goals,
                "goal_diff": goal_diff,
                "ht_home": m.ht_home,
                "ht_away": m.ht_away,
                "ht_total_goals": ht_total,
                "has_ht_goal": 1 if (ht_total is not None and ht_total > 0) else 0,
                "home_corner": m.home_corner,
                "away_corner": m.away_corner,
                "total_corners": total_corners,
                "home_red": m.home_red or 0,
                "away_red": m.away_red or 0,
                "total_reds": (m.home_red or 0) + (m.away_red or 0),
                "home_yellow": m.home_yellow or 0,
                "away_yellow": m.away_yellow or 0,
                "btts": 1 if (m.home_goals > 0 and m.away_goals > 0) else 0,
                "result_1x2": result_1x2
            })

    df = pd.DataFrame(rows)
    print(f"[Thscore Ingestion] Filtered {len(df)} finished matches with complete scores.")
    print(f"  - Unique global leagues covered: {df['league'].nunique()}")
    
    out_csv = DATA_DIR / "thscore_multi_league_matches.csv"
    df.to_csv(out_csv, index=False)
    print(f"  - Saved dataset to: {out_csv}")
    return df


def train_corner_regression(df: pd.DataFrame):
    """Train regression models to predict Total Corners across all leagues (Week 13)."""
    # Filter rows with valid corner data
    corner_df = df.dropna(subset=["total_corners"]).copy()
    if len(corner_df) < 50:
        print("[Corner Model] Insufficient corner data rows.")
        return

    print(f"\n=======================================================")
    print(f"[Week 13] Training Corner Regressors on {len(corner_df)} Multi-League Matches")
    print(f"=======================================================")

    # Feature Engineering for Corners
    # League-level corner tendencies
    league_corner_avg = corner_df.groupby("league")["total_corners"].transform("mean")
    corner_df["league_corner_avg"] = league_corner_avg.fillna(corner_df["total_corners"].mean())

    league_goals_avg = corner_df.groupby("league")["total_goals"].transform("mean")
    corner_df["league_goals_avg"] = league_goals_avg.fillna(corner_df["total_goals"].mean())

    features = ["league_corner_avg", "league_goals_avg", "home_red", "away_red", "total_reds"]
    X = corner_df[features].values
    y = corner_df["total_corners"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    regressors = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost Regressor": xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    }

    results = []
    best_rmse = float("inf")
    best_model = None

    for name, model in regressors.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2 Score": round(r2, 4)
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    results_df = pd.DataFrame(results)
    print("\n[Corner Regression Comparison Table]:")
    print(results_df.to_string(index=False))

    # Save best corner model
    import pickle
    with open(MODELS_DIR / "thscore_corner_regressor.pkl", "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler, "features": features}, f)

    # Plot Corner actual vs predicted
    plt.figure(figsize=(8, 5))
    y_best_pred = best_model.predict(X_test_scaled)
    plt.scatter(y_test, y_best_pred, alpha=0.4, color="#16a085")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="#c0392b", linestyle="--")
    plt.title("Thscore Multi-League: Actual vs Predicted Total Corners")
    plt.xlabel("Actual Corners")
    plt.ylabel("Predicted Corners")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "thscore_corners_actual_vs_pred.png", dpi=200)
    plt.close()


def train_ht_goal_classification(df: pd.DataFrame):
    """Train classification models to predict First-Half Goal (O/U 0.5 HT) (Week 12)."""
    ht_df = df.dropna(subset=["has_ht_goal"]).copy()
    print(f"\n=======================================================")
    print(f"[Week 12] Training First-Half Goal Classifier (HT Over 0.5) on {len(ht_df)} Matches")
    print(f"=======================================================")

    # Feature engineering: league-level HT rate
    league_ht_rate = ht_df.groupby("league")["has_ht_goal"].transform("mean")
    ht_df["league_ht_rate"] = league_ht_rate.fillna(ht_df["has_ht_goal"].mean())

    league_goals_avg = ht_df.groupby("league")["total_goals"].transform("mean")
    ht_df["league_goals_avg"] = league_goals_avg.fillna(ht_df["total_goals"].mean())

    features = ["league_ht_rate", "league_goals_avg", "home_red", "away_red"]
    X = ht_df[features].values
    y = ht_df["has_ht_goal"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss")
    }

    results = []
    best_f1 = -1.0
    best_name = None
    best_model = None
    best_pred = None

    for name, model in classifiers.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model
            best_pred = y_pred

    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
    print("\n[First-Half Goal Classifier Comparison Table]:")
    print(results_df.to_string(index=False))

    # Save best HT model
    import pickle
    with open(MODELS_DIR / "thscore_ht_goal_classifier.pkl", "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler, "features": features}, f)

    # Plot Confusion Matrix for HT Goal Prediction
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, best_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["No HT Goal (0)", "HT Goal (1)"], yticklabels=["No HT Goal (0)", "HT Goal (1)"])
    plt.title(f"Confusion Matrix: Best HT Model ({best_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "thscore_ht_goal_confusion_matrix.png", dpi=200)
    plt.close()


def run_pipeline():
    # Dates to fetch (past 4 days of finished matches across hundreds of leagues)
    dates = ["2026-10-21", "2026-10-22", "2026-10-23"]
    df = fetch_and_build_thscore_dataset(dates)

    # Train new models
    train_corner_regression(df)
    train_ht_goal_classification(df)
    print("\n[DONE] Thscore Multi-League Models Trained & Reports Generated Successfully!")


if __name__ == "__main__":
    run_pipeline()
