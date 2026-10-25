"""Outcome maths: the formulas from README section 4, plus the payout helper."""

from __future__ import annotations

import math

import pytest

from fbpredict import labels


# --------------------------------------------------------------------------- #
# AH unit outcomes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "goal_diff,ah,expected",
    [
        (1, -0.75, 0.5),   # won by one on -0.75 -> half win
        (2, -0.75, 1.0),   # won by two -> full win
        (0, -0.75, -1.0),  # draw -> full loss
        (-1, -0.75, -1.0),
        (1, -0.5, 1.0),    # whole line, won by one -> full win
        (0, -0.5, -1.0),
        (1, -1.0, 0.0),    # whole line landed exactly -> push
        (0, -0.25, -0.5),  # quarter line, draw -> half loss
        (1, -1.25, -0.5),  # won by one on -1.25 -> half loss
        (2, -1.25, 1.0),   # won by two -> full win (both halves land)
        (0, 0.0, 0.0),     # level line, draw -> push
        (1, 0.0, 1.0),
    ],
)
def test_ah_unit_matches_the_brief(goal_diff: int, ah: float, expected: float) -> None:
    assert labels.ah_unit(goal_diff, ah) == expected


def test_ah_unit_dog_is_the_mirror() -> None:
    for diff, ah in [(1, -0.75), (0, -0.5), (3, -1.25), (-2, -0.25)]:
        fav = labels.ah_unit(diff, ah)
        dog = labels.ah_unit_dog(diff, ah)
        # a quarter line can be +0.5/-0.5, a whole line 0/0, otherwise mirror
        assert math.isclose(fav + dog, 0.0, abs_tol=1e-12) or (fav == 1.0 and dog == -1.0)


def test_ah_unit_rejects_non_quarter_lines() -> None:
    with pytest.raises(ValueError):
        labels.ah_unit(1, -0.6)


# --------------------------------------------------------------------------- #
# O/U unit outcomes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "total,line,side,expected",
    [
        (3, 2.5, "over", 1.0),
        (2, 2.5, "over", -1.0),
        (2, 2.0, "over", 0.0),    # exact push
        (3, 2.25, "over", 1.0),   # both halves of 2.25 land
        (2, 2.25, "over", -0.5),  # half loss
        (3, 2.75, "over", 0.5),   # half win: 2.5 lands, 3.0 pushes
        (1, 2.5, "under", 1.0),
        (2, 2.0, "under", 0.0),
        (3, 3.0, "over", 0.0),
    ],
)
def test_ou_unit(total: int, line: float, side: str, expected: float) -> None:
    assert labels.ou_unit(total, line, side) == expected


# --------------------------------------------------------------------------- #
# profit with real odds
# --------------------------------------------------------------------------- #
def test_unit_to_profit_applies_the_price() -> None:
    assert math.isclose(labels.unit_to_profit(1.0, 2.0, 1.0), 1.0)
    assert math.isclose(labels.unit_to_profit(0.5, 2.0, 1.0), 0.5)
    assert math.isclose(labels.unit_to_profit(-1.0, 2.0, 1.0), -1.0)
    assert math.isclose(labels.unit_to_profit(-0.5, 2.0, 1.0), -0.5)
    assert labels.unit_to_profit(0.0, 2.0, 1.0) == 0.0
    # 1.90 odds is the usual -110 style price
    assert math.isclose(labels.unit_to_profit(1.0, 1.90, 10.0), 9.0)


def test_unit_to_profit_rejects_impossible_odds() -> None:
    for bad in (1.0, 0.9, float("nan"), None):
        with pytest.raises(ValueError):
            labels.unit_to_profit(1.0, bad)


# --------------------------------------------------------------------------- #
# binary labels drop only exact pushes
# --------------------------------------------------------------------------- #
def test_ah_binary_drops_only_exact_push() -> None:
    assert labels.ah_binary(1, -0.75) == 1    # half win counts as a cover
    assert labels.ah_binary(0, -0.25) == 0    # half loss counts as a miss
    assert labels.ah_binary(1, -1.0) is None  # exact push


def test_ou_binary_drops_only_exact_push() -> None:
    assert labels.ou_binary(3, 2.25, "over") == 1
    assert labels.ou_binary(2, 2.25, "over") == 0
    assert labels.ou_binary(2, 2.0, "over") is None


# --------------------------------------------------------------------------- #
# vig removal
# --------------------------------------------------------------------------- #
def test_fair_two_way_sums_to_one_and_removes_the_margin() -> None:
    fair_a, fair_b = labels.fair_two_way(1.90, 1.90)
    assert math.isclose(fair_a + fair_b, 1.0)
    assert math.isclose(fair_a, 0.5)
    assert math.isclose(labels.overround(1.90, 1.90), 1 / 1.90 * 2 - 1)


def test_fair_two_way_respects_an_asymmetric_market() -> None:
    fair_fav, fair_dog = labels.fair_two_way(1.40, 3.20)
    assert fair_fav > fair_dog
    assert math.isclose(fair_fav + fair_dog, 1.0)


def test_fair_three_way_sums_to_one() -> None:
    parts = labels.fair_three_way(1.80, 3.60, 4.50)
    assert math.isclose(sum(parts), 1.0)
    assert parts[0] > parts[1] > parts[2]


# --------------------------------------------------------------------------- #
# settle_pick wires it together
# --------------------------------------------------------------------------- #
def test_settle_pick_each_market() -> None:
    common = dict(
        goal_diff_fav=1,
        total_goals=3,
        ah_open=-0.75,
        ou_line_open=2.5,
        ah_fav_price=1.95,
        ah_dog_price=1.95,
        ou_over_price=2.00,
        ou_under_price=1.90,
        stake=1.0,
    )
    unit, profit = labels.settle_pick(market="AH_fav", **common)
    assert unit == 0.5 and math.isclose(profit, 0.5 * 0.95)

    unit, profit = labels.settle_pick(market="AH_dog", **common)
    assert unit == -0.5 and math.isclose(profit, -0.5)

    unit, profit = labels.settle_pick(market="OU_over", **common)
    assert unit == 1.0 and math.isclose(profit, 1.0)

    unit, profit = labels.settle_pick(market="OU_under", **common)
    assert unit == -1.0 and math.isclose(profit, -1.0)


# --------------------------------------------------------------------------- #
# non-finite input must fail loudly, not crash with a bare TypeError
# --------------------------------------------------------------------------- #
def test_non_finite_inputs_raise_a_clear_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        labels.ah_unit(float("nan"), -0.5)
    with pytest.raises(ValueError, match="finite"):
        labels.ou_unit(float("inf"), 2.5, "over")
    with pytest.raises(ValueError, match="finite"):
        labels.ah_unit(1, float("nan"))


# --------------------------------------------------------------------------- #
# one implementation of the margin rule
# --------------------------------------------------------------------------- #
def test_fair_two_way_array_agrees_with_the_scalar_version() -> None:
    fair_a, fair_b, margin = labels.fair_two_way_array([1.90, 2.10], [1.90, 1.75])
    scalar_a, scalar_b = labels.fair_two_way(1.90, 1.90)

    assert math.isclose(float(fair_a[0]), scalar_a)
    assert math.isclose(float(fair_b[0]), scalar_b)
    assert math.isclose(float(fair_a[0] + fair_b[0]), 1.0)
    assert math.isclose(float(fair_a[1] + fair_b[1]), 1.0)
    assert float(margin[0]) > 0.0


def test_fair_two_way_array_returns_nan_for_unusable_odds() -> None:
    fair_a, fair_b, margin = labels.fair_two_way_array([1.0, float("nan")], [2.0, 2.0])
    assert math.isnan(float(fair_a[0])) and math.isnan(float(fair_b[0]))
    assert math.isnan(float(fair_a[1])) and math.isnan(float(fair_b[1]))
    assert math.isnan(float(margin[0]))


def test_scalar_fair_two_way_still_rejects_bad_odds() -> None:
    with pytest.raises(ValueError):
        labels.fair_two_way(1.0, 1.9)
    with pytest.raises(ValueError):
        labels.overround(1.9, 0.5)


def test_fair_two_way_shin() -> None:
    p_a, p_b, z = labels.fair_two_way_shin(1.90, 1.90)
    assert math.isclose(p_a, 0.5, abs_tol=1e-5)
    assert math.isclose(p_b, 0.5, abs_tol=1e-5)
    assert math.isclose(p_a + p_b, 1.0, abs_tol=1e-7)

    # Asymmetric odds: 1.50 vs 2.60
    p_fav, p_dog, z_val = labels.fair_two_way_shin(1.50, 2.60)
    assert p_fav > p_dog
    assert math.isclose(p_fav + p_dog, 1.0, abs_tol=1e-7)
    assert z_val >= 0.0

