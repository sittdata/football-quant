"""Understat Player Match History Provider.
Author: cook45

Fetches and parses player-level match history from Understat (xG, xA, key passes,
minutes played, starter flag) to power position-specific dynamic form calculation.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def parse_understat_player_history(
    player_id: str,
    player_name: str,
    team: str,
    raw_matches: List[Dict[str, Any]],
    default_position: str = "FW",
) -> List[Dict[str, Any]]:
    """Parse raw Understat player match dictionaries into normalized schema records."""
    records: List[Dict[str, Any]] = []

    # Determine primary non-sub position if available
    detected_pos = default_position
    for m in raw_matches:
        pos = m.get("position", "")
        if pos and pos != "Sub":
            detected_pos = pos
            break

    for m in raw_matches:
        raw_date = m.get("date", "")
        # Format date as YYYY-MM-DD
        match_date = raw_date.split(" ")[0] if " " in raw_date else raw_date

        pos = m.get("position", detected_pos)
        is_sub = (pos == "Sub")
        actual_pos = detected_pos if is_sub else pos

        minutes = int(m.get("time", 0))
        is_starter = 0 if is_sub or minutes < 45 else 1

        goals = int(m.get("goals", 0))
        assists = int(m.get("assists", 0))
        shots = int(m.get("shots", 0))
        key_passes = int(m.get("key_passes", 0))
        xg = float(m.get("xG", 0.0))
        xa = float(m.get("xA", 0.0))

        # Heuristic base rating: 6.5 + (goals*1.0) + (assists*0.5) + (xg*0.4) + (key_passes*0.2)
        rating = round(min(10.0, max(5.0, 6.5 + (goals * 1.0) + (assists * 0.5) + (xg * 0.4) + (key_passes * 0.2))), 1)

        records.append({
            "player_id": str(player_id),
            "player_name": player_name,
            "team": team,
            "match_date": match_date,
            "position": actual_pos,
            "minutes": minutes,
            "is_starter": is_starter,
            "goals": goals,
            "assists": assists,
            "shots": shots,
            "xg": round(xg, 3),
            "xa": round(xa, 3),
            "key_passes": key_passes,
            "rating": rating,
            "clean_sheet": 0,
        })

    return records


def save_player_match_stats(con: sqlite3.Connection, records: List[Dict[str, Any]]) -> int:
    """Insert or replace player match stats into player_match_stats table."""
    cur = con.cursor()
    inserted = 0
    for r in records:
        cur.execute(
            """
            INSERT INTO player_match_stats (
                player_id, player_name, team, match_date, position,
                minutes, is_starter, goals, assists, shots, xg, xa, key_passes, rating, clean_sheet
            ) VALUES (
                :player_id, :player_name, :team, :match_date, :position,
                :minutes, :is_starter, :goals, :assists, :shots, :xg, :xa, :key_passes, :rating, :clean_sheet
            )
            ON CONFLICT(player_name, team, match_date) DO UPDATE SET
                minutes=excluded.minutes,
                is_starter=excluded.is_starter,
                goals=excluded.goals,
                assists=excluded.assists,
                shots=excluded.shots,
                xg=excluded.xg,
                xa=excluded.xa,
                key_passes=excluded.key_passes,
                rating=excluded.rating
            """,
            r,
        )
        inserted += 1
    con.commit()
    return inserted


def fetch_and_save_player(con: sqlite3.Connection, player_id: str, player_name: str, team: str) -> int:
    """Fetch live player HTML from Understat, parse matchesData, and save to database."""
    url = f"https://understat.com/player/{player_id}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    try:
        html = urllib.request.urlopen(req, timeout=10.0).read().decode("utf-8")
        match = re.search(r"var matchesData\s*=\s*JSON\.parse\('([^']+)'\)", html)
        if not match:
            return 0
        raw_json_str = match.group(1).encode("utf-8").decode("unicode_escape")
        raw_matches = json.loads(raw_json_str)
        records = parse_understat_player_history(player_id, player_name, team, raw_matches)
        return save_player_match_stats(con, records)
    except Exception as exc:
        logger.warning(f"Could not fetch Understat player {player_name} ({player_id}): {exc}")
        return 0
