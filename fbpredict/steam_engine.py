"""Quantitative Steam & Line Movement Engine (Smart Money / CLV Tracker).
Author: cook45

Detects institutional market pressure and Closing Line Value (CLV):
1. Handicap Velocity: Shift between Opening and Live/Closing Asian lines.
2. Implied Probability Drift: Vig-free margin-stripped probability changes.
3. Reverse Line Movement (RLM): Detecting syndicate action opposing public bias.
4. Signal Synergy: Validates whether ML model edge aligns with Sharp Money.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np

from fbpredict import labels


def _safe_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


@dataclass
class SteamSignal:
    # Asian Handicap
    open_ah: float | None
    current_ah: float | None
    ah_line_delta: float  # current_ah - open_ah
    fav_prob_drift: float  # fair_fav_current - fair_fav_open
    dog_prob_drift: float  # fair_dog_current - fair_dog_open

    # Over/Under
    open_ou: float | None
    current_ou: float | None
    ou_line_delta: float  # current_ou - open_ou
    over_prob_drift: float  # fair_over_current - fair_over_open
    under_prob_drift: float  # fair_under_current - fair_under_open

    # Classification
    steam_direction: Literal["STEAM_FAV", "STEAM_DOG", "STEAM_OVER", "STEAM_UNDER", "NEUTRAL"]
    is_rlm: bool  # True if Reverse Line Movement detected
    steam_strength: float  # Normalized 0.0 to 1.0 confidence score
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "open_ah": self.open_ah,
            "current_ah": self.current_ah,
            "ah_line_delta": round(self.ah_line_delta, 2),
            "fav_prob_drift": round(self.fav_prob_drift, 4),
            "dog_prob_drift": round(self.dog_prob_drift, 4),
            "open_ou": self.open_ou,
            "current_ou": self.current_ou,
            "ou_line_delta": round(self.ou_line_delta, 2),
            "over_prob_drift": round(self.over_prob_drift, 4),
            "under_prob_drift": round(self.under_prob_drift, 4),
            "steam_direction": self.steam_direction,
            "is_rlm": self.is_rlm,
            "steam_strength": round(self.steam_strength, 2),
            "description": self.description,
        }


def calculate_ah_drift(
    open_fav_odds: float | None,
    open_dog_odds: float | None,
    current_fav_odds: float | None,
    current_dog_odds: float | None,
) -> tuple[float, float]:
    """Calculate margin-stripped probability drift for Asian Handicap."""
    try:
        f_open_fav, f_open_dog = (
            labels.fair_two_way(open_fav_odds, open_dog_odds)
            if (open_fav_odds and open_dog_odds and open_fav_odds > 1.0 and open_dog_odds > 1.0)
            else (0.5, 0.5)
        )
        f_cur_fav, f_cur_dog = (
            labels.fair_two_way(current_fav_odds, current_dog_odds)
            if (current_fav_odds and current_dog_odds and current_fav_odds > 1.0 and current_dog_odds > 1.0)
            else (0.5, 0.5)
        )
        return (f_cur_fav - f_open_fav), (f_cur_dog - f_open_dog)
    except Exception:
        return 0.0, 0.0


def calculate_ou_drift(
    open_over_odds: float | None,
    open_under_odds: float | None,
    current_over_odds: float | None,
    current_under_odds: float | None,
) -> tuple[float, float]:
    """Calculate margin-stripped probability drift for Over/Under."""
    try:
        f_open_over, f_open_under = (
            labels.fair_two_way(open_over_odds, open_under_odds)
            if (open_over_odds and open_under_odds and open_over_odds > 1.0 and open_under_odds > 1.0)
            else (0.5, 0.5)
        )
        f_cur_over, f_cur_under = (
            labels.fair_two_way(current_over_odds, current_under_odds)
            if (current_over_odds and current_under_odds and current_over_odds > 1.0 and current_under_odds > 1.0)
            else (0.5, 0.5)
        )
        return (f_cur_over - f_open_over), (f_cur_under - f_open_under)
    except Exception:
        return 0.0, 0.0


def analyze_market_steam(
    open_ah: float | None = None,
    current_ah: float | None = None,
    open_fav_price: float | None = 1.95,
    open_dog_price: float | None = 1.95,
    current_fav_price: float | None = 1.95,
    current_dog_price: float | None = 1.95,
    open_ou: float | None = 2.5,
    current_ou: float | None = 2.5,
    open_over_price: float | None = 1.95,
    open_under_price: float | None = 1.95,
    current_over_price: float | None = 1.95,
    current_under_price: float | None = 1.95,
    drift_threshold: float = 0.025,  # 2.5% vig-free implied probability shift
) -> SteamSignal:
    """Analyze price movement and classify steam / smart money velocity."""
    op_ah = _safe_float(open_ah)
    cu_ah = _safe_float(current_ah)
    op_ou = _safe_float(open_ou)
    cu_ou = _safe_float(current_ou)

    # 1. Line Deltas
    ah_line_delta = (cu_ah - op_ah) if (cu_ah is not None and op_ah is not None) else 0.0
    ou_line_delta = (cu_ou - op_ou) if (cu_ou is not None and op_ou is not None) else 0.0

    # 2. Probability Drifts
    fav_drift, dog_drift = calculate_ah_drift(
        open_fav_price, open_dog_price, current_fav_price, current_dog_price
    )
    over_drift, under_drift = calculate_ou_drift(
        open_over_price, open_under_price, current_over_price, current_under_price
    )

    # 3. Reverse Line Movement (RLM) Detection:
    # If AH handicap became easier for favorite (e.g. from -0.75 to -0.50), ah_line_delta > 0
    # Or dog price collapsed despite favorite being market public choice
    is_rlm = False
    if ah_line_delta > 0.10 or dog_drift > 0.035:
        is_rlm = True

    # 4. Steam Direction & Strength Classification
    steam_direction: Literal["STEAM_FAV", "STEAM_DOG", "STEAM_OVER", "STEAM_UNDER", "NEUTRAL"] = "NEUTRAL"
    strength = 0.0
    desc_parts = []

    # Check Asian Handicap Steam
    if ah_line_delta < -0.10 or fav_drift >= drift_threshold:
        steam_direction = "STEAM_FAV"
        strength = max(0.0, min(1.0, abs(ah_line_delta) * 2.0 + fav_drift * 10.0))
        desc_parts.append(f"Favorite Steam (AH Δ: {ah_line_delta:+.2f}, Drift: {fav_drift*100:+.1f}%)")
    elif ah_line_delta > 0.10 or dog_drift >= drift_threshold:
        steam_direction = "STEAM_DOG"
        strength = max(0.0, min(1.0, abs(ah_line_delta) * 2.0 + dog_drift * 10.0))
        desc_parts.append(f"Underdog Steam (AH Δ: {ah_line_delta:+.2f}, Drift: {dog_drift*100:+.1f}%)")

    # Check Over/Under Steam (if AH was neutral or OU drift is higher)
    if ou_line_delta > 0.10 or over_drift >= drift_threshold:
        ou_str = max(0.0, min(1.0, abs(ou_line_delta) * 2.0 + over_drift * 10.0))
        if ou_str > strength:
            steam_direction = "STEAM_OVER"
            strength = ou_str
            desc_parts = [f"Over Steam (Line Δ: {ou_line_delta:+.2f}, Drift: {over_drift*100:+.1f}%)"]
    elif ou_line_delta < -0.10 or under_drift >= drift_threshold:
        ou_str = max(0.0, min(1.0, abs(ou_line_delta) * 2.0 + under_drift * 10.0))
        if ou_str > strength:
            steam_direction = "STEAM_UNDER"
            strength = ou_str
            desc_parts = [f"Under Steam (Line Δ: {ou_line_delta:+.2f}, Drift: {under_drift*100:+.1f}%)"]

    if not desc_parts:
        desc = "Market Balanced (No significant line drift)"
    else:
        desc = "; ".join(desc_parts)
        if is_rlm and steam_direction == "STEAM_DOG":
            desc = "⚡ Reverse Line Movement! " + desc

    return SteamSignal(
        open_ah=op_ah,
        current_ah=cu_ah,
        ah_line_delta=ah_line_delta,
        fav_prob_drift=fav_drift,
        dog_prob_drift=dog_drift,
        open_ou=op_ou,
        current_ou=cu_ou,
        ou_line_delta=ou_line_delta,
        over_prob_drift=over_drift,
        under_prob_drift=under_drift,
        steam_direction=steam_direction,
        is_rlm=is_rlm,
        steam_strength=strength,
        description=desc,
    )


def evaluate_steam_confirmation(
    model_pick: str,
    steam_signal: SteamSignal,
) -> dict[str, Any]:
    """Check if model pick matches institutional steam direction.

    Returns:
        is_confirmed: True if ML pick aligns with Sharp Money movement.
        conviction_tier: "TIER_1_STEAM_CONFIRMED" (60%+ expected win rate) or "TIER_2_MODEL_ONLY"
    """
    mkt = str(model_pick).strip()
    s_dir = steam_signal.steam_direction

    is_confirmed = False
    if mkt == "AH_dog" and s_dir == "STEAM_DOG":
        is_confirmed = True
    elif mkt == "AH_fav" and s_dir == "STEAM_FAV":
        is_confirmed = True
    elif mkt == "OU_over" and s_dir == "STEAM_OVER":
        is_confirmed = True
    elif mkt == "OU_under" and s_dir == "STEAM_UNDER":
        is_confirmed = True

    tier = "TIER_1_STEAM_CONFIRMED" if is_confirmed else "TIER_2_MODEL_ONLY"

    return {
        "is_confirmed": is_confirmed,
        "conviction_tier": tier,
        "steam_direction": s_dir,
        "is_rlm": steam_signal.is_rlm,
        "steam_strength": steam_signal.steam_strength,
        "badge_label": "🔥 STEAM CONFIRMED" if is_confirmed else ("⚡ RLM ALERT" if steam_signal.is_rlm else None),
        "description": steam_signal.description,
    }


def compute_synergy_kelly(
    base_stake: float,
    market_pick: str,
    steam_signal: SteamSignal | None,
    min_stake: float = 0.5,
    max_stake: float = 2.5,
) -> float:
    """Dynamically scales Kelly staking according to Sharp Money steam alignment.

    - Confirmed (Model aligns with Sharp Steam): 1.25x scaling factor (Max Alpha)
    - Opposed (Model opposes Sharp Steam): 0.50x scaling factor (Capital Defense)
    - Neutral / None: 1.00x base stake
    """
    if base_stake is None or base_stake <= 0:
        return 0.0
    if steam_signal is None or steam_signal.steam_direction == "NEUTRAL":
        return round(float(max(min_stake, min(max_stake, base_stake))), 2)

    mkt = str(market_pick).strip()
    s_dir = steam_signal.steam_direction

    is_confirmed = False
    is_opposed = False

    if mkt == "AH_dog":
        if s_dir == "STEAM_DOG":
            is_confirmed = True
        elif s_dir == "STEAM_FAV":
            is_opposed = True
    elif mkt == "AH_fav":
        if s_dir == "STEAM_FAV":
            is_confirmed = True
        elif s_dir == "STEAM_DOG":
            is_opposed = True
    elif mkt == "OU_over":
        if s_dir == "STEAM_OVER":
            is_confirmed = True
        elif s_dir == "STEAM_UNDER":
            is_opposed = True
    elif mkt == "OU_under":
        if s_dir == "STEAM_UNDER":
            is_confirmed = True
        elif s_dir == "STEAM_OVER":
            is_opposed = True

    if is_confirmed:
        multiplier = 1.25
    elif is_opposed:
        multiplier = 0.50
    else:
        multiplier = 1.00

    scaled = base_stake * multiplier
    return round(float(max(min_stake, min(max_stake, scaled))), 2)

