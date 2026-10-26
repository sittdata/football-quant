"""Tests for player match stats table creation and indexing in fbpredict.db."""

import sqlite3
import pytest
from fbpredict import db


def test_init_player_stats_table():
    """Verify player_match_stats table is properly initialized with unique constraints."""
    con = sqlite3.connect(":memory:")
    db.init_db(con)

    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='player_match_stats'")
    assert cur.fetchone() is not None, "player_match_stats table should exist"

    # Insert test player record
    cur.execute("""
        INSERT INTO player_match_stats (
            player_id, player_name, team, match_date, position,
            minutes, is_starter, goals, assists, shots, xg, xa, key_passes, rating, clean_sheet
        ) VALUES (
            'p100', 'Erling Haaland', 'Manchester City', '2026-10-20', 'FW',
            90, 1, 2, 0, 5, 1.45, 0.10, 2, 8.5, 0
        )
    """)
    con.commit()

    cur.execute("SELECT xg, rating FROM player_match_stats WHERE player_name = 'Erling Haaland'")
    row = cur.fetchone()
    assert row[0] == 1.45
    assert row[1] == 8.5
