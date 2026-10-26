"""Tests for Understat Player Match History Provider."""

import sqlite3
import pytest
from fbpredict.db import init_db
from fbpredict.providers.understat_player import (
    parse_understat_player_history,
    save_player_match_stats,
)


def test_parse_understat_player_history():
    """Verify parsing of Understat matchesData structure into player records."""
    mock_payload = [
        {
            "goals": "1",
            "shots": "4",
            "xG": "0.78",
            "time": "85",
            "position": "FW",
            "xA": "0.25",
            "assists": "0",
            "key_passes": "2",
            "date": "2026-10-10 15:00:00",
            "h_team": "Manchester City",
            "a_team": "Fulham",
            "h_a": "h",
        },
        {
            "goals": "0",
            "shots": "1",
            "xG": "0.12",
            "time": "25",
            "position": "Sub",
            "xA": "0.05",
            "assists": "0",
            "key_passes": "0",
            "date": "2026-10-03 17:30:00",
            "h_team": "Newcastle",
            "a_team": "Manchester City",
            "h_a": "a",
        },
    ]

    records = parse_understat_player_history(
        player_id="8260",
        player_name="Erling Haaland",
        team="Manchester City",
        raw_matches=mock_payload,
    )

    assert len(records) == 2
    rec0 = records[0]
    assert rec0["player_name"] == "Erling Haaland"
    assert rec0["xg"] == 0.78
    assert rec0["is_starter"] == 1
    assert rec0["position"] == "FW"

    rec1 = records[1]
    assert rec1["is_starter"] == 0
    assert rec1["position"] == "FW"  # normalized primary position


def test_save_player_match_stats():
    """Verify storing records into player_match_stats database."""
    con = init_db(":memory:")
    mock_records = [
        {
            "player_id": "8260",
            "player_name": "Erling Haaland",
            "team": "Manchester City",
            "match_date": "2026-10-10",
            "position": "FW",
            "minutes": 85,
            "is_starter": 1,
            "goals": 1,
            "assists": 0,
            "shots": 4,
            "xg": 0.78,
            "xa": 0.25,
            "key_passes": 2,
            "rating": 7.8,
            "clean_sheet": 0,
        }
    ]

    inserted = save_player_match_stats(con, mock_records)
    assert inserted == 1

    cur = con.cursor()
    cur.execute("SELECT xg, key_passes FROM player_match_stats WHERE player_name = 'Erling Haaland'")
    row = cur.fetchone()
    assert row[0] == 0.78
    assert row[1] == 2
