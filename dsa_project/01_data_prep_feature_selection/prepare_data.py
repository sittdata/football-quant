"""Data Preparation and Feature Engineering (Week 10).

Implements:
1. Data Cleaning (handling missing values, dropping duplicates, capping outliers)
2. Date Transformation (year, month, day, day of week, hour from kickoff_at)
3. Target Engineering:
   - Regression: total_goals, goal_diff
   - Binary Classification: ah_cover, ou_over, btts
   - Multi-class Classification: result_1x2 (Home=2, Draw=1, Away=0)
4. Feature Scaling (MinMaxScaler, RobustScaler, StandardScaler)
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT_DIR / "data" / "preprocessed_matches.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_CSV = OUT_DIR / "clean_dsa_dataset.csv"


def load_and_clean_data(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    """Load raw dataset and perform data cleaning and target creation."""
    print(f"[Week 10] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    initial_rows = len(df)
    
    # 1. Drop duplicate matches
    df = df.drop_duplicates(subset=["home_team", "away_team", "kickoff_at"])
    print(f"  - Dropped duplicates: {initial_rows - len(df)} rows removed.")

    # 2. Filter rows with valid goals
    df = df.dropna(subset=["home_goals", "away_goals"]).copy()
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    # 3. Create Target Variables (Classification & Regression)
    # Regression targets
    df["target_total_goals"] = df["home_goals"] + df["away_goals"]
    df["target_goal_diff"] = df["home_goals"] - df["away_goals"]

    # Binary Classification targets
    df["target_btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)
    df["target_ah_cover"] = df["ah_label"].fillna(0).astype(int)
    df["target_ou_over"] = df["ou_label"].fillna(0).astype(int)

    # Multi-class Classification target (1X2: Home=2, Draw=1, Away=0)
    conditions = [
        df["home_goals"] > df["away_goals"],
        df["home_goals"] == df["away_goals"],
        df["home_goals"] < df["away_goals"]
    ]
    choices = [2, 1, 0]  # 2: Home Win, 1: Draw, 0: Away Win
    df["target_1x2"] = np.select(conditions, choices, default=1)

    # 4. Date Transformation (สไลด์หน้า 12, 17)
    df["kickoff_dt"] = pd.to_datetime(df["kickoff_at"], errors="coerce")
    df["match_year"] = df["kickoff_dt"].dt.year.fillna(2025).astype(int)
    df["match_month"] = df["kickoff_dt"].dt.month.fillna(1).astype(int)
    df["match_day"] = df["kickoff_dt"].dt.day.fillna(1).astype(int)
    df["match_dayofweek"] = df["kickoff_dt"].dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = df["match_dayofweek"].isin([5, 6]).astype(int)

    # 5. Core Numerical Features for ML
    numeric_features = [
        "ah_open", "pool_fav_open", "pool_dog_open", "draw_ft_odds",
        "draw_ht_odds", "ou_line_open", "ou_over_price", "ou_under_price",
        "ah_close", "ou_line_close", "pool_fav_close", "pool_dog_close", "draw_ft_close",
        "btts_open", "goal_line_half_open",
        "rule_match", "rule_near", "rule_dist", "draw_ht_high", "draw_ht_vhigh", "draw_ft_match",
        "home_gf_avg", "home_ga_avg", "home_total_avg", "home_over_rate",
        "away_gf_avg", "away_ga_avg", "away_total_avg", "away_over_rate",
        "goals_avg_diff", "expected_total"
    ]

    # Fill missing values with median for numeric columns (สไลด์หน้า 12)
    for col in numeric_features:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0.0)
        else:
            df[col] = 0.0

    print(f"  - Final Cleaned Dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    df.to_csv(CLEAN_CSV, index=False)
    print(f"  - Saved cleaned dataset to: {CLEAN_CSV}")
    return df


def get_feature_matrix(df: pd.DataFrame, feature_cols: list[str], scaler_type: str = "minmax"):
    """Scale feature matrix using MinMaxScaler, RobustScaler, or StandardScaler (สไลด์หน้า 13-15)."""
    X = df[feature_cols].copy().values
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


if __name__ == "__main__":
    load_and_clean_data()
