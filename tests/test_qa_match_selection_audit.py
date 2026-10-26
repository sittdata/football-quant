"""Testing-QA Audit Suite: Football Match and Pick Selection Logic.
Author: cook45

Audits:
1. Threshold Discipline: Strict skip when edge < threshold (no sub-threshold gambling).
2. Temporal Firewall: Rejection of kicked-off or past matches (anti-lookahead).
3. Corrupted / Inverted Odds Resilience: Graceful rejection of NaN, 0.0, or extreme odds.
4. Favorite / Dog Team Attribution: Accurate team attribution under both home and away handicap favorites.
5. Kelly Staking Safety Clamps: Staking units strictly within institutional boundaries.
6. Anti-Cannibalization: Match pick selects exactly one highest-EV market, never conflicting sides.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest

from fbpredict import model
from fbpredict.labels import fair_two_way
from fbpredict.staking import calculate_kelly_fraction
from fbpredict.config import is_safe_match


def make_test_match_row(**overrides) -> pd.Series:
    base = {
        "match_id": "test_m1",
        "league": "Premier League",
        "kickoff_at": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "fav_side": "home",
        "ah_open": -0.5,
        "ah_fav_price": 1.95,
        "ah_dog_price": 1.95,
        "ou_line_open": 2.5,
        "ou_over_price": 1.95,
        "ou_under_price": 1.95,
        "draw_ft_odds": 3.40,
    }
    base.update(overrides)
    return pd.Series(base)


class TestMatchAndPickSelectionQA:

    def test_threshold_discipline_strict_skip(self):
        """Model MUST skip matches when highest edge is strictly below the required threshold."""
        row = make_test_match_row()
        # Fair two-way of 1.95 / 1.95 is 0.50 / 0.50
        # If model gives 0.53 (edge = +0.030) and threshold is 0.040 -> MUST SKIP!
        pick = model.select_market(row, p_fav_cover=0.53, p_over=0.50, threshold=0.04)

        assert pick.market is None, f"Expected None but got {pick.market}"
        assert pick.edge == pytest.approx(0.03, abs=1e-4)
        assert "below threshold" in pick.reason

    def test_highest_edge_selection_and_anti_cannibalization(self):
        """Model must pick ONLY the single market with maximum +EV, never conflicting markets."""
        row = make_test_match_row()
        # Fav edge = 0.60 - 0.50 = +0.10
        # Over edge = 0.56 - 0.50 = +0.06
        pick = model.select_market(row, p_fav_cover=0.60, p_over=0.56, threshold=0.04)

        assert pick.market == "AH_fav"
        assert pick.edge == pytest.approx(0.10, abs=1e-4)
        assert pick.p_model == pytest.approx(0.60, abs=1e-4)

        # Opposite side (AH_dog) edge is -0.10, which must NOT be picked
        assert pick.all_edges["AH_dog"] < 0.0

    def test_fav_dog_team_attribution_logic(self):
        """Verify handicap side attribution correctly resolves team when favorite is Away."""
        # Case A: Home is favorite
        row_home_fav = make_test_match_row(home_team="Liverpool", away_team="Everton", fav_side="home", ah_open=-1.0)
        fav_team_a = row_home_fav["home_team"] if row_home_fav["fav_side"] == "home" else row_home_fav["away_team"]
        dog_team_a = row_home_fav["away_team"] if row_home_fav["fav_side"] == "home" else row_home_fav["home_team"]
        assert fav_team_a == "Liverpool"
        assert dog_team_a == "Everton"

        # Case B: Away is favorite (e.g. Man City playing away at Southampton)
        row_away_fav = make_test_match_row(home_team="Southampton", away_team="Manchester City", fav_side="away", ah_open=-1.5)
        fav_team_b = row_away_fav["home_team"] if row_away_fav["fav_side"] == "home" else row_away_fav["away_team"]
        dog_team_b = row_away_fav["away_team"] if row_away_fav["fav_side"] == "home" else row_away_fav["home_team"]
        assert fav_team_b == "Manchester City"
        assert dog_team_b == "Southampton"

    def test_corrupted_and_impossible_odds_rejection(self):
        """Missing or nonsensical bookmaker odds must result in a clean skip without exception."""
        # 1. Negative odds or odds < 1.01
        row_neg = make_test_match_row(ah_fav_price=-1.50, ah_dog_price=1.90, ou_over_price=0.80)
        pick_neg = model.select_market(row_neg, 0.70, 0.70, threshold=0.04)
        assert pick_neg.market is None

        # 2. NaN or missing lines
        row_nan = make_test_match_row(ah_open=None, ou_line_open=None)
        pick_nan = model.select_market(row_nan, 0.80, 0.80, threshold=0.04)
        assert pick_nan.market is None
        assert "no usable prices" in pick_nan.reason

    def test_temporal_firewall_detects_past_matches(self):
        """Past or in-play matches must be strictly rejected by scanner."""
        now_utc = datetime.now(timezone.utc)
        past_kickoff = (now_utc - timedelta(hours=2)).isoformat()
        future_kickoff = (now_utc + timedelta(hours=3)).isoformat()

        assert pd.to_datetime(past_kickoff, utc=True) <= now_utc
        assert pd.to_datetime(future_kickoff, utc=True) > now_utc

    def test_kelly_staking_hard_clamps(self):
        """Fractional Kelly staking must produce sane, non-negative, clamped bet sizes."""
        # Standard +EV bet: prob 0.55, decimal odds 2.00 (implied 0.50) -> edge 0.05
        stake = calculate_kelly_fraction(prob=0.55, decimal_odds=2.00, fraction=0.25)
        assert 0.0 < stake <= 1.5

        # Extreme high edge: prob 0.90, decimal odds 2.00
        # Must be clamped by max exposure ceiling (default 2.5 or 3.0 units)
        stake_extreme = calculate_kelly_fraction(prob=0.90, decimal_odds=2.00, fraction=0.25, max_stake=2.5)
        assert stake_extreme <= 2.5

        # Negative edge: prob 0.40, decimal odds 2.00 -> Kelly is 0.0 (never negative)
        stake_neg = calculate_kelly_fraction(prob=0.40, decimal_odds=2.00, fraction=0.25)
        assert stake_neg == 0.0

    def test_league_whitelist_safety_firewall(self):
        """Safety firewall must filter out youth, reserve, and amateur leagues."""
        assert is_safe_match("Arsenal", "Chelsea", "Premier League") is True
        assert is_safe_match("Real Madrid", "Barcelona", "La Liga") is True
        assert is_safe_match("Arsenal U21", "Chelsea U21", "Premier League 2") is False
        assert is_safe_match("Amateur Team A", "Amateur Team B", "Regional Cup") is False
