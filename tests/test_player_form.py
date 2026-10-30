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


def test_get_player_dynamic_impact_raw_tuple_rows():
    """Test compatibility when connection row_factory is NOT sqlite3.Row (raw tuples)."""
    import sqlite3
    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE player_match_stats (
            player_name TEXT, team TEXT, match_date TEXT, position TEXT,
            minutes INTEGER, is_starter INTEGER, xg REAL, xa REAL,
            key_passes INTEGER, rating REAL
        )
    """)
    cur.execute("""
        INSERT INTO player_match_stats VALUES
        ('Erling Haaland', 'Man City', '2026-10-01', 'FW', 90, 1, 1.2, 0.1, 2, 8.5)
    """)
    con.commit()

    # Must not raise ValueError on dict(r) with raw tuples
    impact = get_player_dynamic_impact(
        con=con,
        player_name="Erling Haaland",
        team="Man City",
        match_date="2026-10-10",
        base_attack_mult=0.75,
    )
    assert impact["data_points"] == 1
    assert impact["form_factor"] > 0


def test_get_player_dynamic_impact_all_null_metrics():
    """Verify NULL database metrics (xg=None, xa=None, rating=None) do not crash player form."""
    import sqlite3
    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE player_match_stats (
            player_name TEXT, team TEXT, match_date TEXT, position TEXT,
            minutes INTEGER, is_starter INTEGER, xg REAL, xa REAL,
            key_passes INTEGER, rating REAL
        )
    """)
    cur.execute("""
        INSERT INTO player_match_stats VALUES
        ('Bukayo Saka', 'Arsenal', '2026-10-01', 'FW', 90, 1, NULL, NULL, NULL, NULL)
    """)
    con.commit()

    impact = get_player_dynamic_impact(
        con=con,
        player_name="Bukayo Saka",
        team="Arsenal",
        match_date="2026-10-10",
        base_attack_mult=0.80,
    )
    assert impact["data_points"] == 1
    # All-NULL metrics must stay neutral, not collapse to the 0.60 worst-form floor.
    assert impact["form_factor"] == 1.0
    # starter_shock 0.20 (1 of 5 starts) * base penalty 0.20 -> adj 0.96
    assert impact["adj_attack_mult"] == 0.96


def test_positional_form_factor_all_null_metrics_neutral():
    """A record set with no usable metric values is neutral (1.0), not worst form."""
    l5 = [{"xg": None, "xa": None} for _ in range(5)]
    season = [{"xg": None, "xa": None} for _ in range(20)]
    assert compute_positional_form_factor(l5, season, position="FW") == 1.0

    l5_mf = [{"key_passes": None} for _ in range(5)]
    season_mf = [{"key_passes": None} for _ in range(20)]
    assert compute_positional_form_factor(l5_mf, season_mf, position="MF") == 1.0


def test_positional_form_factor_real_zero_still_worst_form():
    """Genuine zero output (no NULLs) must still hit the 0.60 floor."""
    l5 = [{"xg": 0.0, "xa": 0.0} for _ in range(5)]
    season = [{"xg": 0.60, "xa": 0.20} for _ in range(20)]
    assert compute_positional_form_factor(l5, season, position="FW") == 0.60
