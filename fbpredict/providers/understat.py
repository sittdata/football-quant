"""Understat xG, xA, PPDA, and Deep Passing Quantitative Ingestion Engine.
Author: cook45

Extracts advanced expected metrics from Understat API:
- xG (Expected Goals) & xGA (Expected Goals Against)
- npxG (Non-Penalty xG) & npxGD (Goal Difference)
- PPDA (Passes Per Defensive Action - Pressing Intensity metric)
- Deep Completions (Passes into opposition penalty box / 20 yards)
- xPTS (Expected Points based on simulated match outcomes)
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def calculate_ppda_ratio(ppda: Optional[Dict[str, Any]]) -> float:
    """Calculate PPDA = Attacking Passes / Defensive Actions."""
    if not ppda or not isinstance(ppda, dict):
        return 0.0
    att = float(ppda.get("att", 0.0))
    defense = float(ppda.get("def", 0.0))
    if defense <= 0:
        return att
    return round(att / defense, 2)


@dataclass
class UnderstatMatchRecord:
    match_id: str
    league: str
    season: int
    match_date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_xg: float
    away_xg: float
    home_npxg: float
    away_npxg: float
    home_ppda: float
    away_ppda: float
    home_deep: int
    away_deep: int
    home_xpts: float
    away_xpts: float

    def to_tuple(self) -> tuple:
        return (
            self.match_id,
            self.league,
            self.season,
            self.match_date,
            self.home_team,
            self.away_team,
            self.home_goals,
            self.away_goals,
            self.home_xg,
            self.away_xg,
            self.home_npxg,
            self.away_npxg,
            self.home_ppda,
            self.away_ppda,
            self.home_deep,
            self.away_deep,
            self.home_xpts,
            self.away_xpts,
        )


def parse_understat_league_payload(
    payload: Dict[str, Any], league: str, season: int
) -> List[UnderstatMatchRecord]:
    """Parse dates and teams history from getLeagueData payload."""
    dates = payload.get("dates", [])
    teams = payload.get("teams", {})

    # Map team history by (team_title, date) -> stats
    history_map: Dict[tuple[str, str], Dict[str, Any]] = {}
    for team_id, team_info in teams.items():
        title = team_info.get("title", "")
        for h in team_info.get("history", []):
            d = h.get("date", "")
            if d:
                history_map[(title, d)] = h

    records: List[UnderstatMatchRecord] = []
    for m in dates:
        if not m.get("isResult", False):
            continue

        match_id = str(m.get("id"))
        dt = m.get("datetime", "")
        h_title = m.get("h", {}).get("title", "")
        a_title = m.get("a", {}).get("title", "")
        goals = m.get("goals", {})
        xg = m.get("xG", {})

        h_goals = int(goals.get("h", 0))
        a_goals = int(goals.get("a", 0))
        h_xg = round(float(xg.get("h", 0.0)), 4)
        a_xg = round(float(xg.get("a", 0.0)), 4)

        # Lookup deep metrics from team history
        h_hist = history_map.get((h_title, dt), {})
        a_hist = history_map.get((a_title, dt), {})

        h_npxg = round(float(h_hist.get("npxG", h_xg)), 4)
        a_npxg = round(float(a_hist.get("npxG", a_xg)), 4)
        h_ppda = calculate_ppda_ratio(h_hist.get("ppda"))
        a_ppda = calculate_ppda_ratio(a_hist.get("ppda"))
        h_deep = int(h_hist.get("deep", 0))
        a_deep = int(a_hist.get("deep", 0))
        h_xpts = round(float(h_hist.get("xpts", 0.0)), 4)
        a_xpts = round(float(a_hist.get("xpts", 0.0)), 4)

        rec = UnderstatMatchRecord(
            match_id=match_id,
            league=league,
            season=season,
            match_date=dt,
            home_team=h_title,
            away_team=a_title,
            home_goals=h_goals,
            away_goals=a_goals,
            home_xg=h_xg,
            away_xg=a_xg,
            home_npxg=h_npxg,
            away_npxg=a_npxg,
            home_ppda=h_ppda,
            away_ppda=a_ppda,
            home_deep=h_deep,
            away_deep=a_deep,
            home_xpts=h_xpts,
            away_xpts=a_xpts,
        )
        records.append(rec)

    return records


def fetch_understat_league_data(league: str, season: int, timeout: int = 15) -> Dict[str, Any]:
    """Fetch raw getLeagueData JSON from understat.com."""
    url = f"https://understat.com/getLeagueData/{league}/{season}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "Accept-Encoding": "gzip, deflate",
        "Referer": f"https://understat.com/league/{league}/{season}",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))


def init_understat_table(con: sqlite3.Connection) -> None:
    """Ensure understat_matches table exists."""
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS understat_matches (
            match_id TEXT NOT NULL,
            league TEXT NOT NULL,
            season INTEGER NOT NULL,
            match_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_goals INTEGER NOT NULL,
            away_goals INTEGER NOT NULL,
            home_xg REAL NOT NULL,
            away_xg REAL NOT NULL,
            home_npxg REAL NOT NULL,
            away_npxg REAL NOT NULL,
            home_ppda REAL NOT NULL,
            away_ppda REAL NOT NULL,
            home_deep INTEGER NOT NULL,
            away_deep INTEGER NOT NULL,
            home_xpts REAL NOT NULL,
            away_xpts REAL NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (league, season, match_id)
        )
        """
    )
    con.commit()


def store_understat_matches(con: sqlite3.Connection, records: List[UnderstatMatchRecord]) -> int:
    """Insert or update matches into understat_matches table."""
    init_understat_table(con)
    cur = con.cursor()
    cur.executemany(
        """
        INSERT INTO understat_matches (
            match_id, league, season, match_date, home_team, away_team,
            home_goals, away_goals, home_xg, away_xg, home_npxg, away_npxg,
            home_ppda, away_ppda, home_deep, away_deep, home_xpts, away_xpts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(league, season, match_id) DO UPDATE SET
            home_goals=excluded.home_goals,
            away_goals=excluded.away_goals,
            home_xg=excluded.home_xg,
            away_xg=excluded.away_xg,
            home_npxg=excluded.home_npxg,
            away_npxg=excluded.away_npxg,
            home_ppda=excluded.home_ppda,
            away_ppda=excluded.away_ppda,
            home_deep=excluded.home_deep,
            away_deep=excluded.away_deep,
            home_xpts=excluded.home_xpts,
            away_xpts=excluded.away_xpts,
            updated_at=CURRENT_TIMESTAMP
        """,
        [r.to_tuple() for r in records],
    )
    con.commit()
    return len(records)
