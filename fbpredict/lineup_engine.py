"""Football 60-Minute Lineup Watcher & xG Lambda Penalty Engine.
Author: cook45

Monitors confirmed starting lineups 60 minutes before kickoff.
Quantifies the absence of Tier 1 Goalscorers, Playmakers, and Key Defenders/GK,
dynamically adjusting Poisson attack/defense lambdas before running Monte Carlo simulation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "lineup_cache"

# Quantitative Lambda Penalty Multipliers for Absences
# attack_multiplier: shrinks the team's expected goals
# defense_penalty_multiplier: increases the opponent's expected goals
TIER_MULTIPLIERS = {
    "TIER_1_STRIKER": {"attack": 0.78, "defense": 1.00},     # -22% team goal threat
    "TIER_1_PLAYMAKER": {"attack": 0.85, "defense": 1.08},   # -15% threat, +8% counters conceded
    "TIER_1_DEFENDER_GK": {"attack": 1.00, "defense": 1.18}, # +18% goals conceded without anchor
    "REGULAR_STARTER": {"attack": 0.94, "defense": 1.04},    # minor rotation
}

# Empirical Database of High-Impact Football Stars (lowercase for robust lookup)
KEY_PLAYERS_DB: Dict[str, str] = {
    # Strikers / Goalscorers
    "haaland": "TIER_1_STRIKER",
    "salah": "TIER_1_STRIKER",
    "kane": "TIER_1_STRIKER",
    "mbappe": "TIER_1_STRIKER",
    "lewandowski": "TIER_1_STRIKER",
    "saka": "TIER_1_STRIKER",
    "palmer": "TIER_1_STRIKER",
    "son": "TIER_1_STRIKER",
    "isak": "TIER_1_STRIKER",
    "watkins": "TIER_1_STRIKER",
    "vinicius": "TIER_1_STRIKER",
    "martinez": "TIER_1_STRIKER",
    "osimhen": "TIER_1_STRIKER",
    "gyokeres": "TIER_1_STRIKER",

    # Playmakers / Anchors
    "rodri": "TIER_1_PLAYMAKER",
    "de bruyne": "TIER_1_PLAYMAKER",
    "odegaard": "TIER_1_PLAYMAKER",
    "fernandes": "TIER_1_PLAYMAKER",
    "wirtz": "TIER_1_PLAYMAKER",
    "pedri": "TIER_1_PLAYMAKER",
    "musiala": "TIER_1_PLAYMAKER",
    "bellingham": "TIER_1_PLAYMAKER",
    "rice": "TIER_1_PLAYMAKER",
    "modric": "TIER_1_PLAYMAKER",

    # Goalkeepers & Defenders
    "alisson": "TIER_1_DEFENDER_GK",
    "ederson": "TIER_1_DEFENDER_GK",
    "courtois": "TIER_1_DEFENDER_GK",
    "raya": "TIER_1_DEFENDER_GK",
    "van dijk": "TIER_1_DEFENDER_GK",
    "saliba": "TIER_1_DEFENDER_GK",
    "gabriel": "TIER_1_DEFENDER_GK",
    "dias": "TIER_1_DEFENDER_GK",
    "rudiger": "TIER_1_DEFENDER_GK",
    "bastoni": "TIER_1_DEFENDER_GK",
    "militão": "TIER_1_DEFENDER_GK",
}


@dataclass
class LineupAbsence:
    player_name: str
    team: str
    tier: str
    status: str = "CONFIRMED_OUT"  # or 'BENCHED', 'DOUBTFUL'


def lookup_player_tier(player_name: str) -> str:
    import re
    clean = player_name.lower().strip()
    # Check exact full name or word boundary match, prioritizing longer names
    for key in sorted(KEY_PLAYERS_DB.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(key) + r"\b", clean):
            return KEY_PLAYERS_DB[key]
    return "REGULAR_STARTER"


def calculate_lineup_lambdas(
    base_lambda_home: float,
    base_lambda_away: float,
    home_absences: List[str | Dict[str, Any]],
    away_absences: List[str | Dict[str, Any]],
    home_team: str = "",
    away_team: str = "",
    match_date: Optional[str] = None,
    con: Optional[sqlite3.Connection] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Calculate lineup-adjusted Poisson lambdas based on confirmed absences and dynamic form."""
    from .player_form import get_player_dynamic_impact

    home_atk_mult = 1.0
    home_def_mult = 1.0
    away_atk_mult = 1.0
    away_def_mult = 1.0

    parsed_home_absences = []
    for item in home_absences:
        name = item if isinstance(item, str) else item.get("player", "")
        tier = lookup_player_tier(name)
        base_mults = TIER_MULTIPLIERS.get(tier, TIER_MULTIPLIERS["REGULAR_STARTER"])

        if con is not None and match_date:
            dyn = get_player_dynamic_impact(
                con=con,
                player_name=name,
                team=home_team,
                match_date=match_date,
                base_attack_mult=base_mults["attack"],
                base_defense_mult=base_mults["defense"],
            )
            atk_m = dyn["adj_attack_mult"]
            def_m = dyn["adj_defense_mult"]
            form_info = {"form_factor": dyn["form_factor"], "starter_shock": dyn["starter_shock"]}
        else:
            atk_m = base_mults["attack"]
            def_m = base_mults["defense"]
            form_info = {"form_factor": 1.0, "starter_shock": 1.0}

        home_atk_mult *= atk_m
        home_def_mult *= def_m
        parsed_home_absences.append({
            "player": name,
            "tier": tier,
            "impact": {"attack": atk_m, "defense": def_m},
            "form": form_info,
        })

    parsed_away_absences = []
    for item in away_absences:
        name = item if isinstance(item, str) else item.get("player", "")
        tier = lookup_player_tier(name)
        base_mults = TIER_MULTIPLIERS.get(tier, TIER_MULTIPLIERS["REGULAR_STARTER"])

        if con is not None and match_date:
            dyn = get_player_dynamic_impact(
                con=con,
                player_name=name,
                team=away_team,
                match_date=match_date,
                base_attack_mult=base_mults["attack"],
                base_defense_mult=base_mults["defense"],
            )
            atk_m = dyn["adj_attack_mult"]
            def_m = dyn["adj_defense_mult"]
            form_info = {"form_factor": dyn["form_factor"], "starter_shock": dyn["starter_shock"]}
        else:
            atk_m = base_mults["attack"]
            def_m = base_mults["defense"]
            form_info = {"form_factor": 1.0, "starter_shock": 1.0}

        away_atk_mult *= atk_m
        away_def_mult *= def_m
        parsed_away_absences.append({
            "player": name,
            "tier": tier,
            "impact": {"attack": atk_m, "defense": def_m},
            "form": form_info,
        })

    # Home adjusted lambda = Home Attack * Away Defense Conceded
    adj_lambda_home = max(0.20, round(base_lambda_home * home_atk_mult * away_def_mult, 3))
    # Away adjusted lambda = Away Attack * Home Defense Conceded
    adj_lambda_away = max(0.20, round(base_lambda_away * away_atk_mult * home_def_mult, 3))

    intel = {
        "base_lambda_home": base_lambda_home,
        "base_lambda_away": base_lambda_away,
        "adj_lambda_home": adj_lambda_home,
        "adj_lambda_away": adj_lambda_away,
        "delta_home_xg": round(adj_lambda_home - base_lambda_home, 3),
        "delta_away_xg": round(adj_lambda_away - base_lambda_away, 3),
        "home_absences": parsed_home_absences,
        "away_absences": parsed_away_absences,
    }

    return adj_lambda_home, adj_lambda_away, intel


def evaluate_match_with_lineups(
    home_team: str,
    away_team: str,
    base_lambda_home: float,
    base_lambda_away: float,
    home_absences: List[str | Dict[str, Any]],
    away_absences: List[str | Dict[str, Any]],
    ah_line: float = -0.5,
    ou_line: float = 2.5,
    n_sims: int = 10000,
    seed: int = 42,
    match_date: Optional[str] = None,
    con: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """Recalculate Monte Carlo probabilities with lineup adjustments."""
    from .xg_monte_carlo import evaluate_fixture_monte_carlo

    # Pre-lineup baseline
    base_sim = evaluate_fixture_monte_carlo(
        lambda_home=base_lambda_home,
        lambda_away=base_lambda_away,
        ah_line=ah_line,
        ou_line=ou_line,
        n_sims=n_sims,
        seed=seed,
    )

    # Post-lineup adjusted
    adj_lh, adj_la, intel = calculate_lineup_lambdas(
        base_lambda_home,
        base_lambda_away,
        home_absences,
        away_absences,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        con=con,
    )

    adj_sim = evaluate_fixture_monte_carlo(
        lambda_home=adj_lh,
        lambda_away=adj_la,
        ah_line=ah_line,
        ou_line=ou_line,
        n_sims=n_sims,
        seed=seed,
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "intel": intel,
        "pre_lineup": base_sim.to_dict(),
        "post_lineup": adj_sim.to_dict(),
        "delta_home_win_prob": round(adj_sim.home_win_prob - base_sim.home_win_prob, 4),
        "delta_over_prob": round(adj_sim.ou_over_prob - base_sim.ou_over_prob, 4),
    }
