"""Dynamic Positional Form & Starter Shock Engine.
Author: cook45

Calculates rolling 5-match form factors vs season baseline segmented by position:
- FW: xG + xA per match (Goal production)
- MF: Key Passes per match (Chance creation)
- DF / GK: Match Rating / Clean Sheets (Defensive solidity)

Combines positional form with Starter Shock Factor (whether the player has been starting)
to dynamically scale Poisson lambda penalties for match lineup simulations.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Strict safety bounds to prevent sample noise explosions
MIN_FORM_FACTOR = 0.60
MAX_FORM_FACTOR = 1.40


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def _metric_means(records: List[Dict[str, Any]], keys: tuple[str, ...]) -> List[float]:
    """Sum the given metric keys per record, skipping records where every key is missing."""
    values = []
    for r in records:
        present = []
        for key in keys:
            raw = r.get(key)
            if raw is None:
                continue
            try:
                f = float(raw)
            except (ValueError, TypeError):
                continue
            if math.isfinite(f):
                present.append(f)
        if present:
            values.append(sum(present))
    return values


def compute_positional_form_factor(
    l5_records: List[Dict[str, Any]],
    season_records: List[Dict[str, Any]],
    position: str = "FW",
) -> float:
    """Calculate bounded form ratio (Last 5 matches vs season average) by position."""
    if not l5_records or not season_records:
        return 1.0

    pos = (position or "FW").upper()

    if pos in ("FW", "ST", "F", "FORWARD"):
        # Metric: xG + xA
        l5_values = _metric_means(l5_records, ("xg", "xa"))
        season_values = _metric_means(season_records, ("xg", "xa"))
        if not l5_values or not season_values:
            return 1.0
        l5_val = sum(l5_values) / len(l5_values)
        season_val = sum(season_values) / len(season_values)
        baseline = max(0.05, season_val)
        ratio = l5_val / baseline
    elif pos in ("MF", "M", "MIDFIELDER", "CAM", "CM", "CDM"):
        # Metric: Key Passes
        l5_values = _metric_means(l5_records, ("key_passes",))
        season_values = _metric_means(season_records, ("key_passes",))
        if not l5_values or not season_values:
            return 1.0
        l5_val = sum(l5_values) / len(l5_values)
        season_val = sum(season_values) / len(season_values)
        baseline = max(0.5, season_val)
        ratio = l5_val / baseline
    else:
        # DF / GK: Match Rating
        l5_val = sum(_safe_float(r.get("rating"), 6.5) for r in l5_records) / len(l5_records)
        season_val = sum(_safe_float(r.get("rating"), 6.5) for r in season_records) / len(season_records)
        baseline = max(5.0, season_val)
        ratio = l5_val / baseline

    return round(max(MIN_FORM_FACTOR, min(MAX_FORM_FACTOR, ratio)), 2)


def calculate_starter_shock_factor(starts_last_5: int) -> float:
    """Calculate market surprise factor based on starter frequency in last 5 games.

    If a player started all 5 games, their absence causes 100% full market shock (1.0).
    If a player has been regularly benched (e.g. started 0 or 1 game), the market
    has already priced in their absence, so the shock penalty drops towards 0.10.
    """
    starts = max(0, min(5, int(starts_last_5)))
    if starts >= 5:
        return 1.0
    if starts == 4:
        return 0.85
    if starts == 3:
        return 0.65
    if starts == 2:
        return 0.40
    if starts == 1:
        return 0.20
    return 0.10


def get_player_dynamic_impact(
    con: sqlite3.Connection,
    player_name: str,
    team: str,
    match_date: str,
    base_attack_mult: float = 0.78,
    base_defense_mult: float = 1.00,
) -> Dict[str, Any]:
    """Retrieve historical match stats strictly before match_date and compute dynamic multipliers."""
    cur = con.cursor()

    # Query all matches for this player before match_date (strictly no future leakage)
    cur.execute(
        """
        SELECT position, is_starter, xg, xa, key_passes, rating, minutes
        FROM player_match_stats
        WHERE player_name = ? AND match_date < ?
        ORDER BY match_date DESC
        """,
        (player_name, match_date),
    )
    rows = cur.fetchall()

    if not rows:
        # No historical data in DB -> return base multipliers directly
        return {
            "player_name": player_name,
            "form_factor": 1.0,
            "starter_shock": 1.0,
            "adj_attack_mult": base_attack_mult,
            "adj_defense_mult": base_defense_mult,
            "data_points": 0,
        }

    if hasattr(rows[0], "keys"):
        records = [dict(r) for r in rows]
    elif cur.description:
        cols = [d[0] for d in cur.description]
        records = [dict(zip(cols, r)) for r in rows]
    else:
        records = [dict(r) for r in rows]
    l5 = records[:5]
    season = records

    pos = l5[0].get("position", "FW")
    starts_l5 = sum(1 for r in l5 if r.get("is_starter", 0) == 1)

    form_factor = compute_positional_form_factor(l5, season, position=pos)
    starter_shock = calculate_starter_shock_factor(starts_l5)

    # Scaling penalty:
    # Base penalty delta = (1.0 - base_attack_mult)
    # Effective penalty = base_delta * form_factor * starter_shock
    base_atk_penalty = 1.0 - base_attack_mult
    effective_atk_penalty = base_atk_penalty * form_factor * starter_shock
    adj_attack_mult = round(max(0.50, min(1.0, 1.0 - effective_atk_penalty)), 3)

    base_def_penalty = base_defense_mult - 1.0
    effective_def_penalty = base_def_penalty * form_factor * starter_shock
    adj_defense_mult = round(max(1.0, 1.0 + effective_def_penalty), 3)

    return {
        "player_name": player_name,
        "position": pos,
        "starts_last_5": starts_l5,
        "form_factor": form_factor,
        "starter_shock": starter_shock,
        "adj_attack_mult": adj_attack_mult,
        "adj_defense_mult": adj_defense_mult,
        "data_points": len(records),
    }
