"""Quantitative Feature Engineering using Understat xG, xGA, and PPDA Metrics.
Author: cook45

Computes predictive rolling metrics:
- Rolling xG / xGA differentials (Box threat vs Defensive leaks)
- npxGD (Non-Penalty xG Differential spread)
- PPDA Clash (High-press intensity vs Press-vulnerability)
- Luck / Variance Indicator (Goals Scored - xG) for Mean Reversion trading
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict

# Canonical team aliases mapping
TEAM_ALIASES = {
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "spurs": "tottenham",
    "tottenham hotspur": "tottenham",
    "arsenal fc": "arsenal",
    "chelsea fc": "chelsea",
    "liverpool fc": "liverpool",
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "inter milan": "inter",
    "internazionale": "inter",
    "ac milan": "milan",
    "bayern munich": "bayern munich",
    "bayern munchen": "bayern munich",
    "bayer leverkusen": "bayer leverkusen",
    "rb leipzig": "rb leipzig",
    "atletico madrid": "atletico madrid",
    "real madrid": "real madrid",
    "barcelona": "barcelona",
}


def normalize_team_name(name: str) -> str:
    """Normalize team names across multiple data providers."""
    if not name:
        return ""
    clean = name.strip().lower()
    # Strip common noise
    clean = re.sub(r"\b(fc|cf|afc|sc|ac|ssc|as|us)\b", "", clean)
    clean = re.sub(r"[^\w\s]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    return TEAM_ALIASES.get(clean, clean)


_CACHE_CON_ID: Any = None
_TEAM_MATCH_MAP: Dict[str, List[Dict[str, Any]]] = {}


def _get_team_match_map(con: sqlite3.Connection) -> Dict[str, List[Dict[str, Any]]]:
    global _CACHE_CON_ID, _TEAM_MATCH_MAP
    if _CACHE_CON_ID == id(con) and _TEAM_MATCH_MAP:
        return _TEAM_MATCH_MAP

    cur = con.cursor()
    try:
        cur.execute(
            """
            SELECT match_date, home_team, away_team,
                   home_goals, away_goals,
                   home_xg, away_xg, home_npxg, away_npxg,
                   home_ppda, away_ppda, home_deep, away_deep,
                   home_xpts, away_xpts
            FROM understat_matches
            ORDER BY match_date ASC
            """
        )
        rows = cur.fetchall()
    except Exception:
        return {}

    match_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        h_norm = normalize_team_name(r[1])
        a_norm = normalize_team_name(r[2])
        dt = r[0]

        h_entry = {
            "date": dt,
            "goals_for": r[3],
            "goals_against": r[4],
            "xg_for": r[5],
            "xg_against": r[6],
            "npxg_for": r[7],
            "npxg_against": r[8],
            "ppda": r[9],
            "deep": r[11],
        }
        a_entry = {
            "date": dt,
            "goals_for": r[4],
            "goals_against": r[3],
            "xg_for": r[6],
            "xg_against": r[5],
            "npxg_for": r[8],
            "npxg_against": r[7],
            "ppda": r[10],
            "deep": r[12],
        }

        match_map.setdefault(h_norm, []).append(h_entry)
        match_map.setdefault(a_norm, []).append(a_entry)

    _CACHE_CON_ID = id(con)
    _TEAM_MATCH_MAP = match_map
    return _TEAM_MATCH_MAP


def compute_team_rolling_xg_metrics(
    con: sqlite3.Connection,
    team_name: str,
    cutoff_date: str,
    window: int = 5,
) -> Dict[str, float]:
    """Compute rolling xG, xGA, PPDA, and luck overperformance for a team."""
    norm_name = normalize_team_name(team_name)
    m_map = _get_team_match_map(con)

    entries = m_map.get(norm_name)
    if not entries:
        for k, v in m_map.items():
            if norm_name in k or k in norm_name:
                entries = v
                break

    if not entries:
        return {
            "matches_count": 0,
            "rolling_xg_for": 1.25,
            "rolling_xg_against": 1.25,
            "rolling_npxg_diff": 0.0,
            "rolling_ppda": 11.0,
            "rolling_deep": 5.0,
            "luck_overperformance": 0.0,
        }

    past_matches = [e for e in entries if e["date"] < cutoff_date][-window:]
    n = len(past_matches)
    if n == 0:
        return {
            "matches_count": 0,
            "rolling_xg_for": 1.25,
            "rolling_xg_against": 1.25,
            "rolling_npxg_diff": 0.0,
            "rolling_ppda": 11.0,
            "rolling_deep": 5.0,
            "luck_overperformance": 0.0,
        }

    tot_xg_for = sum(m["xg_for"] for m in past_matches)
    tot_xg_against = sum(m["xg_against"] for m in past_matches)
    tot_npxg_diff = sum((m["npxg_for"] - m["npxg_against"]) for m in past_matches)
    tot_ppda = sum(m["ppda"] for m in past_matches)
    tot_deep = sum(m["deep"] for m in past_matches)
    tot_luck = sum((m["goals_for"] - m["xg_for"]) for m in past_matches)

    return {
        "matches_count": n,
        "rolling_xg_for": round(tot_xg_for / n, 4),
        "rolling_xg_against": round(tot_xg_against / n, 4),
        "rolling_npxg_diff": round(tot_npxg_diff / n, 4),
        "rolling_ppda": round(tot_ppda / n, 2),
        "rolling_deep": round(tot_deep / n, 2),
        "luck_overperformance": round(tot_luck / n, 4),
    }


def build_match_xg_features(
    con: sqlite3.Connection,
    home_team: str,
    away_team: str,
    cutoff_date: str,
    window: int = 5,
) -> Dict[str, float]:
    """Compute differential xG / PPDA / Luck features for a match."""
    h = compute_team_rolling_xg_metrics(con, home_team, cutoff_date, window=window)
    a = compute_team_rolling_xg_metrics(con, away_team, cutoff_date, window=window)

    xg_diff_net = round(h["rolling_xg_for"] - a["rolling_xg_for"], 4)
    xga_diff_net = round(h["rolling_xg_against"] - a["rolling_xg_against"], 4)
    npxgd_spread = round(h["rolling_npxg_diff"] - a["rolling_npxg_diff"], 4)
    ppda_clash = round(h["rolling_ppda"] - a["rolling_ppda"], 2)
    luck_spread = round(h["luck_overperformance"] - a["luck_overperformance"], 4)

    return {
        "home_rolling_xg": h["rolling_xg_for"],
        "away_rolling_xg": a["rolling_xg_for"],
        "home_rolling_xga": h["rolling_xg_against"],
        "away_rolling_xga": a["rolling_xg_against"],
        "xg_diff_net": xg_diff_net,
        "xga_diff_net": xga_diff_net,
        "npxgd_spread": npxgd_spread,
        "ppda_clash": ppda_clash,
        "luck_spread": luck_spread,
        "home_ppda": h["rolling_ppda"],
        "away_ppda": a["rolling_ppda"],
        "home_luck": h["luck_overperformance"],
        "away_luck": a["luck_overperformance"],
    }


def enrich_dataframe_with_xg(
    df: Any,
    con: sqlite3.Connection | None = None,
    window: int = 5,
) -> Any:
    """Enrich a matches DataFrame with rolling xG, xGA, PPDA and Luck features."""
    import pandas as pd
    from pathlib import Path

    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    should_close = False
    if con is None:
        db_path = Path(__file__).resolve().parent.parent / "data" / "football.db"
        if not db_path.exists():
            return df
        con = sqlite3.connect(db_path)
        should_close = True

    try:
        feature_rows = []
        for _, row in df.iterrows():
            h_team = str(row.get("home_team", ""))
            a_team = str(row.get("away_team", ""))
            cutoff = str(row.get("kickoff_at", "2099-01-01"))
            feats = build_match_xg_features(con, h_team, a_team, cutoff, window=window)
            feature_rows.append(feats)

        feat_df = pd.DataFrame(feature_rows, index=df.index)
        # Avoid column collision if columns already exist
        cols_to_add = [c for c in feat_df.columns if c not in df.columns]
        return pd.concat([df, feat_df[cols_to_add]], axis=1)
    finally:
        if should_close:
            con.close()

