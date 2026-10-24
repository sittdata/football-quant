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
from typing import Literal

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
