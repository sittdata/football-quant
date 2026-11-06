"""Unit tests for simple 1X2 feature extraction and model training."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from fbpredict.features_1x2 import build_1x2_features, compute_target_1x2
from fbpredict.model_1x2 import run_1x2_experiment


def _make_dummy_matches() -> pd.DataFrame:
    rows = []
    teams = ["Arsenal", "Chelsea", "Liverpool", "Man City"]
    base_date = pd.Timestamp("2025-01-01", tz="UTC")

    idx = 0
    for round_num in range(15):
        # 2 matches per round
        d = base_date + pd.Timedelta(days=round_num * 7)
        # Match 1
        rows.append({
            "match_id": f"m_{idx}",
            "kickoff_at": d.isoformat(),
            "home_team": teams[0],
            "away_team": teams[1],
            "home_goals": 2 if round_num % 2 == 0 else 1,
            "away_goals": 1 if round_num % 2 == 0 else 2,
            "fav_side": "home",
            "pool_fav_open": 1.80,
            "pool_dog_open": 4.20,
            "draw_ft_odds": 3.60,
        })
        idx += 1
        # Match 2
        rows.append({
            "match_id": f"m_{idx}",
            "kickoff_at": (d + pd.Timedelta(hours=2)).isoformat(),
            "home_team": teams[2],
            "away_team": teams[3],
            "home_goals": 1,
            "away_goals": 1,
            "fav_side": "home",
            "pool_fav_open": 2.10,
            "pool_dog_open": 3.30,
            "draw_ft_odds": 3.40,
        })
        idx += 1

    return pd.DataFrame(rows)


def test_compute_target_1x2():
    df = pd.DataFrame({
        "home_goals": [2, 1, 0],
        "away_goals": [1, 1, 3],
    })
    targets = compute_target_1x2(df)
    assert targets.tolist() == [2, 1, 0]  # Home=2, Draw=1, Away=0


def test_build_1x2_features_no_leakage():
    df = _make_dummy_matches()
    prepared = build_1x2_features(df, rolling_window=3)

    # First match of Arsenal (index 0) must have NaN or filled defaults for rolling form (never current match goals)
    arsenal_first = prepared[(prepared["home_team"] == "Arsenal")].iloc[0]
    assert pd.isna(arsenal_first["home_form_pts"])  # no prior matches

    # Check target exists
    assert "target_1x2" in prepared.columns
    assert len(prepared) == len(df)


def test_run_1x2_experiment():
    df = _make_dummy_matches()
    prepared = build_1x2_features(df, rolling_window=3)
    exp_res = run_1x2_experiment(prepared, include_market_odds=True, train_ratio=0.7)

    assert exp_res["n_train"] > 0
    assert exp_res["n_test"] > 0
    assert len(exp_res["results"]) >= 3  # Market, LogReg, RF, HGB

    for res in exp_res["results"]:
        assert 0.0 <= res.accuracy <= 1.0
        assert res.log_loss_val >= 0.0
        assert res.confusion_mat.shape == (3, 3)

    assert not exp_res["feature_importances"].empty
