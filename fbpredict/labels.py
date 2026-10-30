"""Outcome / payout maths for the two markets.

Implements the formulas from README.md section 4, but with two deliberate
hardening changes:

* quarter lines are compared with a tolerance instead of ``==`` on raw floats
  (the brief's own note about floating point),
* a *profit* helper applies the real decimal odds, so the backtest can report
  profit after the bookmaker's margin rather than at even money.

Units and conventions
---------------------
``ah`` is the handicap **on the favourite** and is negative when the favourite
gives goals (``-0.75``). ``goal_diff_fav`` is ``favourite_goals - underdog_goals``.

``unit`` is the fraction of the stake won or lost at even money:

===== ====================================
unit  meaning
===== ====================================
+1.0  full win
+0.5  half win  (quarter line)
 0.0  push      (whole-number line landed exactly)
-0.5  half loss (quarter line)
-1.0  full loss
===== ====================================
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

Side = Literal["over", "under"]
Market = Literal["AH_fav", "AH_dog", "OU_over", "OU_under"]

_QUARTER_TOL = 1e-9


def _as_quarter(value: float) -> float:
    """Snap ``value`` to the nearest 0.25 and verify it really was one."""
    number = float(value)
    if not math.isfinite(number):
        # round() on NaN/inf raises a bare TypeError, which is useless to a
        # caller. Say what is actually wrong instead.
        raise ValueError(
            f"{value!r} is not a finite number; the label formulas cannot settle it"
        )
    snapped = round(number * 4.0) / 4.0
    if abs(snapped - number) > _QUARTER_TOL:
        raise ValueError(
            f"{value!r} is not a quarter-line value; the label formulas only "
            "handle whole, half and quarter handicaps/totals"
        )
    return snapped


# --------------------------------------------------------------------------- #
# unit outcomes
# --------------------------------------------------------------------------- #
def ah_unit(goal_diff_fav: float, ah: float) -> float:
    """Outcome in stake units for backing the **favourite** on handicap ``ah``.

    ``ah`` is negative when the favourite gives goals, e.g. ``-0.75``.
    """
    x = _as_quarter(float(goal_diff_fav) + float(ah))
    if x > 0.25:
        return 1.0
    if x == 0.25:
        return 0.5
    if x == 0.0:
        return 0.0
    if x == -0.25:
        return -0.5
    return -1.0


def ah_unit_dog(goal_diff_fav: float, ah: float) -> float:
    """Outcome in stake units for backing the **underdog** side."""
    return ah_unit(-float(goal_diff_fav), -float(ah))


def ou_unit(total_goals: float, line: float, side: Side = "over") -> float:
    """Outcome in stake units for the over/under market."""
    total = _as_quarter(float(total_goals))
    line = _as_quarter(float(line))
    x = total - line if side == "over" else line - total
    x = _as_quarter(x)
    if x >= 0.5:
        return 1.0
    if x == 0.25:
        return 0.5
    if x == 0.0:
        return 0.0
    if x == -0.25:
        return -0.5
    return -1.0


# --------------------------------------------------------------------------- #
# profit with real decimal odds
# --------------------------------------------------------------------------- #
def unit_to_profit(unit: float, decimal_odds: float, stake: float = 1.0) -> float:
    """Convert a unit outcome into money using the price actually taken.

    A win returns ``stake * (decimal_odds - 1)``; a half win returns half of
    that; a push returns the stake (i.e. profit 0); a loss costs the stake.
    """
    if decimal_odds is None or not math.isfinite(float(decimal_odds)) or decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds!r}")
    if unit > 0:
        return float(unit) * (float(decimal_odds) - 1.0) * float(stake)
    if unit < 0:
        return float(unit) * float(stake)
    return 0.0


# --------------------------------------------------------------------------- #
# binary labels for the classifiers (pushes are excluded, never guessed)
# --------------------------------------------------------------------------- #
def ah_binary(goal_diff_fav: float, ah: float) -> int | None:
    """``1`` favourite covers, ``0`` does not, ``None`` for an exact push.

    Half-win counts as a cover and half-loss as a miss - only the exact push on
    a whole-number line is dropped, and the report states how many were dropped.
    """
    x = _as_quarter(float(goal_diff_fav) + float(ah))
    if x > 0:
        return 1
    if x < 0:
        return 0
    return None


def ou_binary(total_goals: float, line: float, side: Side = "over") -> int | None:
    """``1`` the chosen side wins, ``0`` it loses, ``None`` for an exact push."""
    total = _as_quarter(float(total_goals))
    line = _as_quarter(float(line))
    x = total - line if side == "over" else line - total
    x = _as_quarter(x)
    if x > 0:
        return 1
    if x < 0:
        return 0
    return None


# --------------------------------------------------------------------------- #
# odds -> fair probabilities (bookmaker margin removed)
# --------------------------------------------------------------------------- #
def _safe_inverse(decimal_odds: float) -> float:
    if decimal_odds is None:
        raise ValueError("decimal odds is required")
    odds = float(decimal_odds)
    if not math.isfinite(odds) or odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds!r}")
    return 1.0 / odds


def fair_two_way_array(
    odds_a: object, odds_b: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised removal of the bookmaker margin for a two-way market.

    This is the **single** implementation of "invert, then normalise by the
    sum". The scalar helpers below delegate here, so the rule used to pick bets
    (`model.select_market`) and the rule used to report the market baseline
    (`features.add_fair_probs`) cannot drift apart.

    Returns ``(fair_a, fair_b, overround)``. Entries with unusable odds are NaN.
    """
    a = np.asarray(odds_a, dtype="float64")
    b = np.asarray(odds_b, dtype="float64")
    valid = np.isfinite(a) & np.isfinite(b) & (a > 1.0) & (b > 1.0)

    safe_a = np.where(valid, a, 1.0)
    safe_b = np.where(valid, b, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_a = np.where(valid, 1.0 / safe_a, np.nan)
        raw_b = np.where(valid, 1.0 / safe_b, np.nan)
        total = raw_a + raw_b
        return raw_a / total, raw_b / total, total - 1.0


def fair_three_way_array(
    odds_a: object, odds_b: object, odds_c: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised removal of the bookmaker margin for a three-way market.

    The **single** implementation of "invert, then normalise by the sum" for a
    1X2 book. The scalar helper below and the rule engine's no-vig conversion
    both delegate here, so the margin rule cannot drift between them.

    Returns ``(fair_a, fair_b, fair_c, overround)``. Entries with unusable odds
    are NaN - the row is skipped rather than guessed.
    """
    a = np.asarray(odds_a, dtype="float64")
    b = np.asarray(odds_b, dtype="float64")
    c = np.asarray(odds_c, dtype="float64")
    valid = (
        np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        & (a > 1.0) & (b > 1.0) & (c > 1.0)
    )

    safe_a = np.where(valid, a, 1.0)
    safe_b = np.where(valid, b, 1.0)
    safe_c = np.where(valid, c, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_a = np.where(valid, 1.0 / safe_a, np.nan)
        raw_b = np.where(valid, 1.0 / safe_b, np.nan)
        raw_c = np.where(valid, 1.0 / safe_c, np.nan)
        total = raw_a + raw_b + raw_c
        return raw_a / total, raw_b / total, raw_c / total, total - 1.0


def fair_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Remove the overround from a two-way market.

    Returns ``(fair_a, fair_b)`` summing to 1.0. For AH the two sides are the
    favourite and the underdog; for O/U they are over and under. Invalid odds
    still raise, so callers can keep relying on that.
    """
    _safe_inverse(odds_a)
    _safe_inverse(odds_b)
    fair_a, fair_b, _ = fair_two_way_array((odds_a,), (odds_b,))
    return float(fair_a[0]), float(fair_b[0])


def fair_two_way_shin(odds_a: float, odds_b: float, tol: float = 1e-7, max_iter: int = 100) -> tuple[float, float, float]:
    """Remove overround using Shin's method (accounting for insider trading / favorite-longshot bias).

    Returns ``(p_a, p_b, z)`` where p_a + p_b == 1.0 and z is the estimated insider proportion.
    """
    _safe_inverse(odds_a)
    _safe_inverse(odds_b)
    q_a = 1.0 / float(odds_a)
    q_b = 1.0 / float(odds_b)
    q_sum = q_a + q_b

    if q_sum <= 1.0:
        p_a, p_b = fair_two_way(odds_a, odds_b)
        return p_a, p_b, 0.0

    z_low = 0.0
    z_high = 0.4
    z_mid = 0.0
    p_a = q_a / q_sum
    p_b = q_b / q_sum

    for _ in range(max_iter):
        z_mid = (z_low + z_high) / 2.0
        term_a = math.sqrt(z_mid**2 + 4.0 * (1.0 - z_mid) * (q_a**2) / q_sum)
        term_b = math.sqrt(z_mid**2 + 4.0 * (1.0 - z_mid) * (q_b**2) / q_sum)
        denom = 2.0 * (1.0 - z_mid)
        if denom == 0:
            break
        p_a = (term_a - z_mid) / denom
        p_b = (term_b - z_mid) / denom
        sum_p = p_a + p_b
        if abs(sum_p - 1.0) < tol:
            break
        if sum_p > 1.0:
            z_low = z_mid
        else:
            z_high = z_mid

    s = p_a + p_b
    return float(p_a / s), float(p_b / s), float(z_mid)


def fair_three_way(odds_home: float, odds_draw: float, odds_away: float) -> tuple[float, float, float]:
    """Remove the overround from a 1X2 market. Delegates to the array version."""
    _safe_inverse(odds_home)
    _safe_inverse(odds_draw)
    _safe_inverse(odds_away)
    fair_a, fair_b, fair_c, _ = fair_three_way_array((odds_home,), (odds_draw,), (odds_away,))
    return float(fair_a[0]), float(fair_b[0]), float(fair_c[0])


def overround(odds_a: float, odds_b: float) -> float:
    """Bookmaker margin of a two-way market, e.g. ``0.05`` for 5%."""
    _safe_inverse(odds_a)
    _safe_inverse(odds_b)
    _, _, margin = fair_two_way_array((odds_a,), (odds_b,))
    return float(margin[0])


# --------------------------------------------------------------------------- #
# settle one recorded pick
# --------------------------------------------------------------------------- #
def settle_pick(
    *,
    market: Market,
    goal_diff_fav: float,
    total_goals: float,
    ah_open: float,
    ou_line_open: float,
    ah_fav_price: float | None,
    ah_dog_price: float | None,
    ou_over_price: float | None,
    ou_under_price: float | None,
    stake: float = 1.0,
) -> tuple[float, float]:
    """Return ``(outcome_unit, profit)`` for a recorded pick."""
    if market == "AH_fav":
        unit = ah_unit(goal_diff_fav, ah_open)
        odds = ah_fav_price
    elif market == "AH_dog":
        unit = ah_unit_dog(goal_diff_fav, ah_open)
        odds = ah_dog_price
    elif market == "OU_over":
        unit = ou_unit(total_goals, ou_line_open, "over")
        odds = ou_over_price
    elif market == "OU_under":
        unit = ou_unit(total_goals, ou_line_open, "under")
        odds = ou_under_price
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown market {market!r}")

    return unit, unit_to_profit(unit, odds, stake)


# --------------------------------------------------------------------------- #
# outcome labeling & 5-tier settlement summary
# --------------------------------------------------------------------------- #
def outcome_name(unit: float) -> str:
    """Return descriptive name for the 5 quarter-line outcomes."""
    if math.isclose(unit, 1.0, abs_tol=1e-5):
        return "ชนะเต็ม (+1.0)"
    if math.isclose(unit, 0.5, abs_tol=1e-5):
        return "ชนะครึ่ง (+0.5)"
    if math.isclose(unit, 0.0, abs_tol=1e-5):
        return "เสมอยก (0.0)"
    if math.isclose(unit, -0.5, abs_tol=1e-5):
        return "เสียครึ่ง (-0.5)"
    if math.isclose(unit, -1.0, abs_tol=1e-5):
        return "เสียเต็ม (-1.0)"
    return f"อื่นๆ ({unit:+.2f})"


def summarize_outcomes(
    units: object, profits: object | None = None
) -> dict[str, Any]:
    """Calculate 5-tier settlement breakdown, win rates, and net financial P&L."""
    u = np.asarray(units, dtype="float64")
    full_wins = int(np.sum(np.isclose(u, 1.0, atol=1e-5)))
    half_wins = int(np.sum(np.isclose(u, 0.5, atol=1e-5)))
    pushes = int(np.sum(np.isclose(u, 0.0, atol=1e-5)))
    half_losses = int(np.sum(np.isclose(u, -0.5, atol=1e-5)))
    full_losses = int(np.sum(np.isclose(u, -1.0, atol=1e-5)))
    total = len(u)
    decided = full_wins + half_wins + half_losses + full_losses

    nominal_win_rate = (full_wins + half_wins) / decided if decided > 0 else 0.0
    effective_win_rate = (full_wins + 0.5 * half_wins) / decided if decided > 0 else 0.0

    p = np.asarray(profits, dtype="float64") if profits is not None else u
    total_profit = float(np.sum(p))
    roi = (total_profit / total) * 100.0 if total > 0 else 0.0

    return {
        "total": total,
        "full_wins": full_wins,
        "half_wins": half_wins,
        "pushes": pushes,
        "half_losses": half_losses,
        "full_losses": full_losses,
        "decided": decided,
        "nominal_win_rate": nominal_win_rate,
        "effective_win_rate": effective_win_rate,
        "total_profit": total_profit,
        "roi": roi,
    }

