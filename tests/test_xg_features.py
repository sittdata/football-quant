"""Unit tests for xG Feature Engineering and Rolling Metrics.
Author: cook45
"""

import sqlite3
import pytest
from fbpredict.xg_features import (
    normalize_team_name,
    compute_team_rolling_xg_metrics,
    build_match_xg_features,
)


def test_normalize_team_name():
    assert normalize_team_name("Manchester City") == "manchester city"
    assert normalize_team_name("Man City") == "manchester city"
    assert normalize_team_name("Tottenham Hotspur") == "tottenham"
    assert normalize_team_name("Spurs") == "tottenham"
    assert normalize_team_name("Arsenal FC") == "arsenal"
    assert normalize_team_name("PSG") == "paris saint germain"


def test_compute_team_rolling_xg_metrics():
    # In-memory test SQLite DB
    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE understat_matches (
            match_id TEXT, league TEXT, season INTEGER, match_date TEXT,
            home_team TEXT, away_team TEXT, home_goals INTEGER, away_goals INTEGER,
            home_xg REAL, away_xg REAL, home_npxg REAL, away_npxg REAL,
            home_ppda REAL, away_ppda REAL, home_deep INTEGER, away_deep INTEGER,
            home_xpts REAL, away_xpts REAL
        )
        """
    )
    # Insert 3 past matches for Arsenal
    cur.execute(
        """
        INSERT INTO understat_matches VALUES
        ('1', 'EPL', 2024, '2024-08-10', 'Arsenal', 'Wolves', 2, 0, 1.80, 0.40, 1.80, 0.40, 8.5, 14.0, 8, 2, 2.2, 0.5),
        ('2', 'EPL', 2024, '2024-08-17', 'Aston Villa', 'Arsenal', 0, 2, 0.60, 1.60, 0.60, 1.60, 12.0, 9.0, 3, 7, 0.7, 2.1),
        ('3', 'EPL', 2024, '2024-08-24', 'Arsenal', 'Brighton', 1, 1, 1.40, 1.20, 1.40, 1.20, 9.5, 11.0, 6, 5, 1.4, 1.3)
        """
    )
    con.commit()

    metrics = compute_team_rolling_xg_metrics(con, "Arsenal", cutoff_date="2024-08-30", window=3)
    assert metrics["matches_count"] == 3
    # Expected xG for: (1.80 + 1.60 + 1.40) / 3 = 1.60
    assert metrics["rolling_xg_for"] == 1.60
    # Expected xGA: (0.40 + 0.60 + 1.20) / 3 = 0.7333
    assert round(metrics["rolling_xg_against"], 2) == 0.73
    # Goals scored: 2 + 2 + 1 = 5. xG = 4.8. Luck = (5 - 4.8) / 3 = +0.067
    assert round(metrics["luck_overperformance"], 2) == 0.07


def test_build_match_xg_features():
    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE understat_matches (
            match_id TEXT, league TEXT, season INTEGER, match_date TEXT,
            home_team TEXT, away_team TEXT, home_goals INTEGER, away_goals INTEGER,
            home_xg REAL, away_xg REAL, home_npxg REAL, away_npxg REAL,
            home_ppda REAL, away_ppda REAL, home_deep INTEGER, away_deep INTEGER,
            home_xpts REAL, away_xpts REAL
        )
        """
    )
    cur.execute(
        """
        INSERT INTO understat_matches VALUES
        ('1', 'EPL', 2024, '2024-08-10', 'Arsenal', 'Everton', 2, 0, 2.0, 0.5, 2.0, 0.5, 8.0, 15.0, 8, 2, 2.5, 0.3),
        ('2', 'EPL', 2024, '2024-08-10', 'Chelsea', 'Fulham', 1, 2, 1.0, 1.5, 1.0, 1.5, 11.0, 9.0, 4, 6, 1.0, 1.7)
        """
    )
    con.commit()

    features = build_match_xg_features(con, home_team="Arsenal", away_team="Chelsea", cutoff_date="2024-08-20")
    assert "xg_diff_net" in features
    assert "ppda_clash" in features
    assert "luck_spread" in features
    # Home xG: 2.0, Away xG: 1.0 -> xg_diff_net = 1.0
    assert features["xg_diff_net"] == 1.0
    # Home PPDA: 8.0, Away PPDA: 11.0 -> ppda_clash = -3.0
    assert features["ppda_clash"] == -3.0


def test_enrich_dataframe_with_xg():
    import pandas as pd
    from fbpredict.xg_features import enrich_dataframe_with_xg

    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE understat_matches (
            match_id TEXT, league TEXT, season INTEGER, match_date TEXT,
            home_team TEXT, away_team TEXT, home_goals INTEGER, away_goals INTEGER,
            home_xg REAL, away_xg REAL, home_npxg REAL, away_npxg REAL,
            home_ppda REAL, away_ppda REAL, home_deep INTEGER, away_deep INTEGER,
            home_xpts REAL, away_xpts REAL
        )
        """
    )
    cur.execute(
        """
        INSERT INTO understat_matches VALUES
        ('1', 'EPL', 2024, '2024-08-10', 'Arsenal', 'Everton', 2, 0, 2.0, 0.5, 2.0, 0.5, 8.0, 15.0, 8, 2, 2.5, 0.3)
        """
    )
    con.commit()

    df = pd.DataFrame([
        {"home_team": "Arsenal", "away_team": "Everton", "kickoff_at": "2024-08-20 15:00:00"}
    ])
    enriched = enrich_dataframe_with_xg(df, con=con)
    assert "xg_diff_net" in enriched.columns
    assert "ppda_clash" in enriched.columns
    assert "luck_spread" in enriched.columns
    assert len(enriched) == 1

