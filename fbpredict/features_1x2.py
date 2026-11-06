"""Simple 1X2 feature engineering for academic Data Science projects.

Features:
- Rolling team form (points per match, goals scored, goals conceded) over last N matches.
- Strict .shift(1) to prevent any data leakage.
- Rest days between matches.
- Market implied probabilities from opening 1X2 odds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_target_1x2(df: pd.DataFrame) -> pd.Series:
    """Classify match outcome into 1X2 label:
    - 2: Home Win ('H')
    - 1: Draw ('D')
    - 0: Away Win ('A')
    """
    home_g = pd.to_numeric(df["home_goals"], errors="coerce")
    away_g = pd.to_numeric(df["away_goals"], errors="coerce")

    conditions = [
        home_g > away_g,
        home_g == away_g,
        home_g < away_g,
    ]
    choices = [2, 1, 0]  # 2: Home, 1: Draw, 0: Away
    return pd.Series(np.select(conditions, choices, default=np.nan), index=df.index)


def compute_market_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Convert opening 1X2 odds into normalized implied probabilities (vig removed proportionally)."""
    out = pd.DataFrame(index=df.index)

    # Check if 1X2 odds columns exist
    # Can be pool_fav_open / draw_ft_odds / pool_dog_open, or standard 1x2 odds
    fav_side = df["fav_side"] if "fav_side" in df.columns else pd.Series(index=df.index, dtype=object)
    pool_fav = pd.to_numeric(df["pool_fav_open"], errors="coerce") if "pool_fav_open" in df.columns else pd.Series(np.nan, index=df.index)
    pool_dog = pd.to_numeric(df["pool_dog_open"], errors="coerce") if "pool_dog_open" in df.columns else pd.Series(np.nan, index=df.index)
    draw_odds = pd.to_numeric(df["draw_ft_odds"], errors="coerce") if "draw_ft_odds" in df.columns else pd.Series(np.nan, index=df.index)

    # If standard 1x2 odds are directly present
    odds_h = pd.to_numeric(df["odds_1x2_home"], errors="coerce") if "odds_1x2_home" in df.columns else pd.Series(np.nan, index=df.index)
    odds_d = pd.to_numeric(df["odds_1x2_draw"], errors="coerce") if "odds_1x2_draw" in df.columns else pd.Series(np.nan, index=df.index)
    odds_a = pd.to_numeric(df["odds_1x2_away"], errors="coerce") if "odds_1x2_away" in df.columns else pd.Series(np.nan, index=df.index)

    # If not present, reconstruct from pool_fav / pool_dog / fav_side
    if odds_h.isna().all() and hasattr(pool_fav, "notna") and pool_fav.notna().any():
        is_home_fav = fav_side == "home"
        odds_h = np.where(is_home_fav, pool_fav, pool_dog)
        odds_a = np.where(is_home_fav, pool_dog, pool_fav)
        odds_d = draw_odds

    raw_p_h = 1.0 / odds_h
    raw_p_d = 1.0 / odds_d
    raw_p_a = 1.0 / odds_a

    overround = raw_p_h + raw_p_d + raw_p_a
    valid = overround > 0

    out["market_p_home"] = np.where(valid, raw_p_h / overround, np.nan)
    out["market_p_draw"] = np.where(valid, raw_p_d / overround, np.nan)
    out["market_p_away"] = np.where(valid, raw_p_a / overround, np.nan)

    return out


def build_1x2_features(df: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
    """Build a clean feature matrix for 1X2 prediction.

    Guarantees zero future leakage by sorting by kickoff date and applying
    .shift(1) before computing any rolling statistics.
    """
    clean = df.copy()
    if "kickoff_at" in clean.columns:
        clean["kickoff_at"] = pd.to_datetime(clean["kickoff_at"], utc=True)
        clean = clean.sort_values("kickoff_at").reset_index(drop=True)

    # 1. Target
    clean["target_1x2"] = compute_target_1x2(clean)

    # 2. Market baseline probabilities
    market_probs = compute_market_probabilities(clean)
    for col in market_probs.columns:
        clean[col] = market_probs[col]

    # 3. Build team-level rolling match logs
    # Deconstruct each match into two team-perspective rows
    home_rows = pd.DataFrame({
        "match_idx": clean.index,
        "team": clean["home_team"],
        "kickoff_at": clean["kickoff_at"],
        "is_home": 1,
        "gf": pd.to_numeric(clean["home_goals"], errors="coerce"),
        "ga": pd.to_numeric(clean["away_goals"], errors="coerce"),
    })
    away_rows = pd.DataFrame({
        "match_idx": clean.index,
        "team": clean["away_team"],
        "kickoff_at": clean["kickoff_at"],
        "is_home": 0,
        "gf": pd.to_numeric(clean["away_goals"], errors="coerce"),
        "ga": pd.to_numeric(clean["home_goals"], errors="coerce"),
    })

    team_logs = pd.concat([home_rows, away_rows], ignore_index=True)
    team_logs = team_logs.sort_values(["team", "kickoff_at"]).reset_index(drop=True)

    # Points earned
    team_logs["pts"] = np.where(
        team_logs["gf"] > team_logs["ga"], 3,
        np.where(team_logs["gf"] == team_logs["ga"], 1, 0)
    )
    team_logs["gd"] = team_logs["gf"] - team_logs["ga"]

    # Rest days
    team_logs["prev_kickoff"] = team_logs.groupby("team")["kickoff_at"].shift(1)
    team_logs["rest_days"] = (team_logs["kickoff_at"] - team_logs["prev_kickoff"]).dt.total_seconds() / 86400.0
    team_logs["rest_days"] = team_logs["rest_days"].clip(lower=1.0, upper=30.0)

    # STRICT SHIFT(1) BEFORE ROLLING TO AVOID DATA LEAKAGE
    grouped = team_logs.groupby("team")

    # Shifted history
    team_logs["shift_pts"] = grouped["pts"].shift(1)
    team_logs["shift_gf"] = grouped["gf"].shift(1)
    team_logs["shift_ga"] = grouped["ga"].shift(1)
    team_logs["shift_gd"] = grouped["gd"].shift(1)

    # Rolling averages over previous matches
    rolling = team_logs.groupby("team")[["shift_pts", "shift_gf", "shift_ga", "shift_gd"]].rolling(
        window=rolling_window, min_periods=1
    ).mean().reset_index(level=0, drop=True)

    team_logs["form_pts_avg"] = rolling["shift_pts"]
    team_logs["gf_avg"] = rolling["shift_gf"]
    team_logs["ga_avg"] = rolling["shift_ga"]
    team_logs["gd_avg"] = rolling["shift_gd"]

    # Merge back to matches
    home_feats = team_logs[team_logs["is_home"] == 1].set_index("match_idx")
    away_feats = team_logs[team_logs["is_home"] == 0].set_index("match_idx")

    clean["home_form_pts"] = home_feats["form_pts_avg"]
    clean["home_gf_avg"] = home_feats["gf_avg"]
    clean["home_ga_avg"] = home_feats["ga_avg"]
    clean["home_gd_avg"] = home_feats["gd_avg"]
    clean["home_rest_days"] = home_feats["rest_days"].fillna(7.0)

    clean["away_form_pts"] = away_feats["form_pts_avg"]
    clean["away_gf_avg"] = away_feats["gf_avg"]
    clean["away_ga_avg"] = away_feats["ga_avg"]
    clean["away_gd_avg"] = away_feats["gd_avg"]
    clean["away_rest_days"] = away_feats["rest_days"].fillna(7.0)

    # Differentials (Home vs Away)
    clean["diff_form_pts"] = clean["home_form_pts"] - clean["away_form_pts"]
    clean["diff_gd_avg"] = clean["home_gd_avg"] - clean["away_gd_avg"]
    clean["diff_rest_days"] = clean["home_rest_days"] - clean["away_rest_days"]

    return clean
