"""QA Stress and Edge-Case Test Suite for Dynamic Player Form & Lineup Engine.
Author: cook45

Covers:
1. Zero division protection when player stats are all 0.0.
2. Hard clamp boundaries enforcement [0.60, 1.40] under extreme values.
3. Strict data leakage prevention (future matches are never read).
4. Missing player fallback to static baseline without crashing.
5. Monte Carlo probability distribution coherence under extreme lineup shock.
"""

import sqlite3
import pytest
from fbpredict.db import init_db
from fbpredict.player_form import (
    compute_positional_form_factor,
    calculate_starter_shock_factor,
    get_player_dynamic_impact,
    MIN_FORM_FACTOR,
    MAX_FORM_FACTOR,
)
from fbpredict.lineup_engine import (
    calculate_lineup_lambdas,
    evaluate_match_with_lineups,
)


def test_zero_division_resilience():
    """Verify zero stats or all-zero baselines do not raise ZeroDivisionError."""
    # All zeros
    l5_zero = [{"xg": 0.0, "xa": 0.0, "key_passes": 0, "rating": 0.0} for _ in range(5)]
    season_zero = [{"xg": 0.0, "xa": 0.0, "key_passes": 0, "rating": 0.0} for _ in range(20)]

    for pos in ["FW", "MF", "DF", "GK", "UNKNOWN"]:
        factor = compute_positional_form_factor(l5_zero, season_zero, position=pos)
        assert MIN_FORM_FACTOR <= factor <= MAX_FORM_FACTOR


def test_extreme_hyper_form_clamping():
    """Verify insane outlier numbers (e.g. 10.0 xG/game) are strictly clamped to MAX_FORM_FACTOR."""
    l5_insane = [{"xg": 10.0, "xa": 5.0} for _ in range(5)]
    season_normal = [{"xg": 0.5, "xa": 0.2} for _ in range(20)]

    factor = compute_positional_form_factor(l5_insane, season_normal, position="FW")
    assert factor == MAX_FORM_FACTOR


def test_strict_temporal_leakage_barrier():
    """Verify matches occurring at or after match_date are completely invisible to the engine."""
    con = init_db(":memory:")
    cur = con.cursor()

    # Match before kickoff: bad form (0 goals, 0.1 xG)
    cur.execute("""
        INSERT INTO player_match_stats (
            player_name, team, match_date, position, minutes, is_starter, goals, xg, xa, key_passes, rating
        ) VALUES ('Phil Foden', 'Manchester City', '2026-10-15', 'MF', 90, 1, 0, 0.10, 0.05, 1, 6.2)
    """)

    # FUTURE match (e.g. tomorrow or next week): hat-trick (3 goals, 2.0 xG)
    cur.execute("""
        INSERT INTO player_match_stats (
            player_name, team, match_date, position, minutes, is_starter, goals, xg, xa, key_passes, rating
        ) VALUES ('Phil Foden', 'Manchester City', '2026-10-30', 'MF', 90, 1, 3, 2.00, 0.80, 6, 9.8)
    """)

    # Evaluate on game day 2026-10-20 (must NOT see 2026-10-30 data)
    impact = get_player_dynamic_impact(
        con=con,
        player_name="Phil Foden",
        team="Manchester City",
        match_date="2026-10-20",
        base_attack_mult=0.85,
    )

    # Only 1 past data point should be registered
    assert impact["data_points"] == 1
    # Form factor must reflect the low xG match, not the future hat-trick
    assert impact["form_factor"] <= 1.0


def test_unknown_player_graceful_fallback():
    """Verify an unknown or unindexed player falls back to base tier multiplier without error."""
    con = init_db(":memory:")

    impact = get_player_dynamic_impact(
        con=con,
        player_name="Completely Unknown Player XYZ",
        team="Unknown FC",
        match_date="2026-10-25",
        base_attack_mult=0.94,
        base_defense_mult=1.04,
    )

    assert impact["data_points"] == 0
    assert impact["form_factor"] == 1.0
    assert impact["adj_attack_mult"] == 0.94
    assert impact["adj_defense_mult"] == 1.04


def test_monte_carlo_extreme_lineup_shock_coherence():
    """Verify Monte Carlo simulation produces coherent probabilities even under catastrophic 4-star absence."""
    res = evaluate_match_with_lineups(
        home_team="Manchester City",
        away_team="Real Madrid",
        base_lambda_home=2.20,
        base_lambda_away=1.40,
        home_absences=["Erling Haaland", "Rodri", "Kevin De Bruyne", "Ederson"],
        away_absences=[],
        n_sims=5000,
        seed=123,
    )

    post = res["post_lineup"]
    # Probabilities must be legitimate
    assert 0.0 < post["home_win_prob"] < 1.0
    assert 0.0 < post["draw_prob"] < 1.0
    assert 0.0 < post["away_win_prob"] < 1.0
    assert pytest.approx(post["home_win_prob"] + post["draw_prob"] + post["away_win_prob"], abs=1e-3) == 1.0
    # Over/under probabilities must be bounded
    assert 0.0 < post["ou_over_prob"] < 1.0
    assert 0.0 < post["ah_fav_cover_prob"] < 1.0
    # Fair odds must not be negative or infinite
    assert post["fair_over_odds"] > 1.0
    assert post["fair_ah_fav_odds"] > 1.0
