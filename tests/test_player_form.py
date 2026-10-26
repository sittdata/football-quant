"""Tests for Dynamic Positional Form and Starter Shock Calculator."""

import sqlite3
import pytest
from fbpredict.db import init_db
from fbpredict.player_form import (
    compute_positional_form_factor,
    calculate_starter_shock_factor,
    get_player_dynamic_impact,
)


def test_positional_form_factor_fw():
    """FW form factor evaluates xG+xA per game, bounded in [0.60, 1.40]."""
    # In-form striker: recent 1.20 vs season 0.60 -> ratio 2.0 -> clamped to 1.40
    l5 = [{"xg": 0.90, "xa": 0.30} for _ in range(5)]
    season = [{"xg": 0.45, "xa": 0.15} for _ in range(20)]
    factor = compute_positional_form_factor(l5, season, position="FW")
    assert factor == 1.40

    # Slump striker: recent 0.10 vs season 0.80 -> ratio 0.125 -> clamped to 0.60
    l5_slump = [{"xg": 0.05, "xa": 0.05} for _ in range(5)]
    season_high = [{"xg": 0.60, "xa": 0.20} for _ in range(20)]
    factor_slump = compute_positional_form_factor(l5_slump, season_high, position="FW")
    assert factor_slump == 0.60


def test_positional_form_factor_mf():
    """MF form factor evaluates Key Passes per game."""
    l5 = [{"key_passes": 4} for _ in range(5)]
    season = [{"key_passes": 2} for _ in range(15)]
    factor = compute_positional_form_factor(l5, season, position="MF")
    assert factor == 1.40  # 4/2 = 2.0 -> clamped to 1.40


def test_starter_shock_factor():
    """Infrequent starters produce lower shock to market betting lines."""
    # Starter all 5 games -> shock factor 1.0 (full impact)
    assert calculate_starter_shock_factor(5) == 1.0
    # Started only 1 game -> shock factor shrinks to 0.20
    assert calculate_starter_shock_factor(1) == 0.20
    # Never started -> minimal baseline shock 0.10
    assert calculate_starter_shock_factor(0) == 0.10


def test_get_player_dynamic_impact_from_db():
    """Integration test fetching player history from SQLite and computing net impact."""
    con = init_db(":memory:")
    cur = con.cursor()

    # Populate 5 past matches for Salah (all started, high xG)
    for i in range(1, 6):
        cur.execute("""
            INSERT INTO player_match_stats (
                player_name, team, match_date, position, minutes, is_starter, xg, xa, key_passes, rating
            ) VALUES ('Mohamed Salah', 'Liverpool', ?, 'FW', 90, 1, 0.80, 0.30, 3, 8.2)
        """, (f"2026-10-0{i}",))

    # Match before kickoff date 2026-10-25
    impact = get_player_dynamic_impact(
        con=con,
        player_name="Mohamed Salah",
        team="Liverpool",
        match_date="2026-10-25",
        base_attack_mult=0.78,
    )

    assert impact["starter_shock"] == 1.0
    assert impact["form_factor"] >= 1.0
    # Adjusted attack penalty should reflect high importance of in-form star
    assert impact["adj_attack_mult"] < 0.85
