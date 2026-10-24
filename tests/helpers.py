"""Shared fixtures for the test suite."""

from __future__ import annotations

from typing import Any

import pandas as pd

DEFAULTS: dict[str, Any] = {
    "match_id": "m0",
    "provider": "test",
    "league": "Test League",
    "season": "2025",
    "kickoff_at": "2025-01-01T12:00:00+00:00",
    "home_team": "Alpha",
    "away_team": "Beta",
    "fav_side": "home",
    "ah_open": -0.5,
    "ah_fav_price": 1.90,
    "ah_dog_price": 1.90,
    "pool_fav_open": 1.80,
    "pool_dog_open": 4.20,
    "draw_ft_odds": 3.60,
    "draw_ht_odds": 2.40,
    "ou_line_open": 2.5,
    "ou_over_price": 1.90,
    "ou_under_price": 1.90,
    "ht_home": 1,
    "ht_away": 0,
    "home_goals": 2,
    "away_goals": 1,
    "odds_source": "test",
    "result_source": "test",
    "odds_fetched_at": "2025-01-01T00:00:00+00:00",
    "result_fetched_at": "2025-01-01T15:00:00+00:00",
    "is_synthetic": 0,
}


def make_match(**overrides: Any) -> dict[str, Any]:
    return {**DEFAULTS, **overrides}


def make_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
