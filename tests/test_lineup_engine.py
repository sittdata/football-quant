"""Unit tests for Football Lineup Watcher & xG Lambda Penalty Engine."""

import pytest
from fbpredict.lineup_engine import (
    lookup_player_tier,
    calculate_lineup_lambdas,
    evaluate_match_with_lineups,
    TIER_MULTIPLIERS,
)


def test_player_tier_classification():
    """Verify star classifications for elite strikers, playmakers, and defenders."""
    assert lookup_player_tier("Erling Haaland") == "TIER_1_STRIKER"
    assert lookup_player_tier("Mohamed Salah") == "TIER_1_STRIKER"
    assert lookup_player_tier("Rodri") == "TIER_1_PLAYMAKER"
    assert lookup_player_tier("Kevin De Bruyne") == "TIER_1_PLAYMAKER"
    assert lookup_player_tier("Virgil van Dijk") == "TIER_1_DEFENDER_GK"
    assert lookup_player_tier("Alisson Becker") == "TIER_1_DEFENDER_GK"
    assert lookup_player_tier("Random Squad Player") == "REGULAR_STARTER"


def test_lineup_lambda_penalties():
    """Missing key striker drops team lambda; missing defender boosts opponent lambda."""
    base_lh = 2.0
    base_la = 1.0

    # Home missing Haaland (Striker: 0.78 attack)
    adj_lh, adj_la, intel = calculate_lineup_lambdas(
        base_lambda_home=base_lh,
        base_lambda_away=base_la,
        home_absences=["Erling Haaland"],
        away_absences=[],
    )

    # Home lambda should drop from 2.0 to ~1.56
    assert adj_lh < 1.60
    assert adj_lh > 1.50
    # Away lambda unchanged
    assert adj_la == 1.0


def test_defensive_absence_penalty():
    """Missing key centerback increases opponent's expected goals."""
    base_lh = 1.5
    base_la = 1.2

    # Away team missing Van Dijk (Defender: 1.18 defense penalty)
    adj_lh, adj_la, intel = calculate_lineup_lambdas(
        base_lambda_home=base_lh,
        base_lambda_away=base_la,
        home_absences=[],
        away_absences=["Virgil van Dijk"],
    )

    # Home team gets higher xG against weakened away defense
    assert adj_lh > base_lh
    assert intel["delta_home_xg"] > 0.20


def test_monte_carlo_lineup_shift_bounds():
    """Monte Carlo with lineup adjustments preserves probability bounds and reflects market shift."""
    res = evaluate_match_with_lineups(
        home_team="Manchester City",
        away_team="Arsenal",
        base_lambda_home=1.8,
        base_lambda_away=1.2,
        home_absences=["Erling Haaland", "Rodri"],
        away_absences=[],
        n_sims=5000,
        seed=42,
    )

    # City missing both Haaland and Rodri should significantly drop home win probability
    assert res["delta_home_win_prob"] < -0.05
    # Over probability should also drop
    assert res["delta_over_prob"] < 0.0

    post = res["post_lineup"]
    assert 0.0 <= post["home_win_prob"] <= 1.0
    assert 0.0 <= post["draw_prob"] <= 1.0
    assert 0.0 <= post["away_win_prob"] <= 1.0
    total_p = post["home_win_prob"] + post["draw_prob"] + post["away_win_prob"]
    assert pytest.approx(total_p, abs=1e-3) == 1.0


def test_dynamic_lineup_with_database_form():
    """Verify lineup engine dynamically adjusts lambda penalties using DB player stats."""
    import sqlite3
    from fbpredict.db import init_db

    con = init_db(":memory:")
    cur = con.cursor()

    # Case 1: In-form Saka (scored 4 goals, high xG, started all 5)
    for i in range(1, 6):
        cur.execute("""
            INSERT INTO player_match_stats (
                player_name, team, match_date, position, minutes, is_starter, goals, xg, xa, key_passes, rating
            ) VALUES ('Bukayo Saka', 'Arsenal', ?, 'FW', 90, 1, 1, 0.85, 0.40, 3, 8.4)
        """, (f"2026-10-0{i}",))

    # Static vs Dynamic comparison
    _, _, static_intel = calculate_lineup_lambdas(
        base_lambda_home=1.80,
        base_lambda_away=1.00,
        home_absences=["Bukayo Saka"],
        away_absences=[],
    )

    _, _, dynamic_intel = calculate_lineup_lambdas(
        base_lambda_home=1.80,
        base_lambda_away=1.00,
        home_absences=["Bukayo Saka"],
        away_absences=[],
        home_team="Arsenal",
        match_date="2026-10-25",
        con=con,
    )

    # In-form Saka missing should inflict equal or greater drop in home lambda than static default
    assert dynamic_intel["adj_lambda_home"] <= static_intel["adj_lambda_home"]
    assert dynamic_intel["home_absences"][0]["form"]["starter_shock"] == 1.0

