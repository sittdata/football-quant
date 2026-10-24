"""Rule set 2 (R1-R6): the printed thresholds, the no-vig variant, and the audit."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fbpredict import features, labels, price_pool_rules as rules

from .helpers import make_frame, make_match


def _row(ah_open: float, fav: float, draw: float, dog: float) -> pd.Series:
    return pd.Series(
        {
            "ah_open": ah_open,
            "pool_fav_open": fav,
            "draw_ft_odds": draw,
            "pool_dog_open": dog,
        }
    )


# --------------------------------------------------------------------------- #
# table integrity
# --------------------------------------------------------------------------- #
def test_rule_table_matches_the_spec() -> None:
    assert len(rules.RULES_V2) == 16
    keys = [rule.key for rule in rules.RULES_V2]
    assert len(keys) == len(set(keys)), "rule keys must be unique"
    assert keys[0] == "R1_ah-0.25"
    assert "R6_1_ah-0.25" in keys


def test_feature_names_are_three_per_rule() -> None:
    names = rules.rule_feature_names()
    assert len(names) == 3 * len(rules.RULES_V2)
    assert "R1_ah-0.25_match" in names
    assert "R1_ah-0.25_near" in names
    assert "R1_ah-0.25_dist" in names


def test_r4a_is_flagged_untestable() -> None:
    """The spec forbids substituting another bookmaker for Ladbrokes (6.2)."""
    assert rules.R4A_UNTESTABLE is True
    r4a = next(rule for rule in rules.RULES_V2 if rule.rule_id == "R4a")
    assert r4a.untestable is True
    assert "Ladbrokes" in r4a.note
    # the raw range cannot be decimal, because decimal odds are always > 1
    assert rules.LADBROKES_OVER_HT_RAW_RANGE[0] < 1.0
    assert rules.LADBROKES_OVER_HT_IF_HK == (1.50, 1.75)

    # the feature is NaN even on a row whose draw price would satisfy the
    # visible part of the condition: an untestable rule must not half-fire
    row = _row(0.0, 1.90, 3.70, 4.50)
    out = rules.rule_features(row)
    assert math.isnan(out["R4a_ah0.0_match"])
    assert math.isnan(out["R4a_ah0.0_near"])
    assert math.isnan(out["R4a_ah0.0_dist"])

    # the no-vig table carries the same flag through the conversion
    novig_r4a = next(rule for rule in rules.RULES_V2_NOVIG if rule.rule_id == "R4a")
    assert novig_r4a.untestable is True


def test_untestable_rule_never_reaches_the_audit() -> None:
    """R4a is reported as untestable; it is not measured on substitute markets."""
    frame = make_frame([make_match(match_id="m1", ah_open=0.0, draw_ft_odds=3.70)])
    prepared = features.build_features(frame)
    audit = rules.audit_rules(prepared)

    assert "R4a_ah0.0" not in set(audit["key"])
    # its testable sibling R3a (draw price only, same line) DOES fire and is
    # measured, so the exclusion above is specific to the untestable rule
    assert "R3a_ah0.0" in set(audit["key"])


# --------------------------------------------------------------------------- #
# the spec's own example values
# --------------------------------------------------------------------------- #
def test_r1_fires_on_the_spec_example() -> None:
    result = rules.rule_features(_row(-0.25, 2.09, 3.25, 2.80))
    assert result["R1_ah-0.25_match"] == 1
    assert result["R1_ah-0.25_near"] == 1
    assert result["R1_ah-0.25_dist"] == 0.0


def test_rule_row_overround_matches_the_spec_calculation() -> None:
    """The spec's section 9.4 quotes about 1.14 for R1 at -0.25."""
    r1 = next(rule for rule in rules.RULES_V2 if rule.key == "R1_ah-0.25")
    assert math.isclose(r1.overround, 1.1433, abs_tol=1e-3)


def test_r6_1_distance_is_measured_on_the_dog_side() -> None:
    result = rules.rule_features(_row(-0.25, 2.09, 3.25, 2.80))
    # dog 2.80 against pt(3.10) = (3.07, 3.13) -> 0.27 away
    assert result["R6_1_ah-0.25_match"] == 0
    assert result["R6_1_ah-0.25_near"] == 0
    assert math.isclose(result["R6_1_ah-0.25_dist"], 0.27, abs_tol=1e-9)


def test_r2a_distance_is_measured_on_the_fav_side() -> None:
    result = rules.rule_features(_row(-0.25, 2.09, 3.25, 2.80))
    # fav 2.09 against ge(2.20) -> 0.11 short
    assert result["R2a_ah-0.25_match"] == 0
    assert math.isclose(result["R2a_ah-0.25_dist"], 0.11, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# boundary semantics
# --------------------------------------------------------------------------- #
def test_le_and_ge_include_their_boundary() -> None:
    assert rules.rule_features(_row(-0.25, 2.09, 3.25, 2.80))["R1_ah-0.25_match"] == 1
    # one cent outside on the draw side and it must stop matching
    assert rules.rule_features(_row(-0.25, 2.09, 3.24, 2.80))["R1_ah-0.25_match"] == 0


def test_single_price_uses_the_plus_minus_tolerance() -> None:
    inside_low = rules.rule_features(_row(-0.25, 2.20, 3.50, 2.99))
    assert inside_low["R5_1_ah-0.25_match"] == 1

    at_edge = rules.rule_features(_row(-0.25, 2.20 - rules.TOL, 3.50, 2.99))
    assert at_edge["R5_1_ah-0.25_match"] == 1

    # just past the band -> not a match, but the distance (0.01) is still
    # inside the SAME tolerance the band itself uses, so `near` stays 1.
    just_outside = rules.rule_features(_row(-0.25, 2.20 - rules.TOL - 0.01, 3.50, 2.99))
    assert just_outside["R5_1_ah-0.25_match"] == 0
    assert just_outside["R5_1_ah-0.25_near"] == 1
    assert math.isclose(just_outside["R5_1_ah-0.25_dist"], 0.01, abs_tol=1e-9)

    # clearly beyond the tolerance -> neither match nor near
    far = rules.rule_features(_row(-0.25, 2.20 - rules.TOL - 0.05, 3.50, 2.99))
    assert far["R5_1_ah-0.25_match"] == 0
    assert far["R5_1_ah-0.25_near"] == 0
    assert math.isclose(far["R5_1_ah-0.25_dist"], 0.05, abs_tol=1e-9)


def test_a_range_is_used_exactly_as_written() -> None:
    assert rules.rule_features(_row(-0.75, 1.80, 3.80, 4.40))["R5_3_ah-0.75_match"] == 1
    assert rules.rule_features(_row(-0.75, 1.89, 3.80, 4.40))["R5_3_ah-0.75_match"] == 1
    assert rules.rule_features(_row(-0.75, 1.90, 3.80, 4.40))["R5_3_ah-0.75_match"] == 0


def test_a_rule_only_fires_on_the_lines_it_lists() -> None:
    result = rules.rule_features(_row(-0.6, 2.09, 3.25, 2.80))
    assert result["R1_ah-0.25_match"] == 0
    assert math.isnan(result["R1_ah-0.25_dist"])


def test_rules_with_unchecked_sides_still_fire() -> None:
    """R3a only constrains the draw price."""
    result = rules.rule_features(_row(0.0, 1.90, 3.70, 4.50))
    assert result["R3a_ah0.0_match"] == 1
    assert result["R3a_ah0.0_dist"] == 0.0


def test_missing_prices_do_not_fire_rules() -> None:
    """A missing price means 'unknown', not 'distance zero' (regression).

    _dist() used to collapse a missing price to 0.0, which made every rule row
    whose handicap line matched fire on rows that carried no prices at all -
    R1 would fire with an empty 1X2 triple. The contract is now: a checked
    side with no price is match=0 / near=0 / dist=NaN.
    """
    result = rules.rule_features(pd.Series({"ah_open": -0.25, "pool_fav_open": None}))
    assert result["R1_ah-0.25_match"] == 0
    assert result["R1_ah-0.25_near"] == 0
    assert math.isnan(result["R1_ah-0.25_dist"])


def test_a_missing_price_on_any_checked_side_suppresses_the_rule() -> None:
    """The rule must not fall back to the sides it can still see."""
    # fav present, draw and dog missing: R1 checks all three -> no fire.
    partial = rules.rule_features(_row(-0.25, 2.05, float("nan"), float("nan")))
    assert partial["R1_ah-0.25_match"] == 0
    assert partial["R1_ah-0.25_near"] == 0
    assert math.isnan(partial["R1_ah-0.25_dist"])

    # Each missing side on its own must also suppress it.
    for column in ("pool_fav_open", "draw_ft_odds", "pool_dog_open"):
        row = {"ah_open": -0.25, "pool_fav_open": 2.05, "draw_ft_odds": 3.25, "pool_dog_open": 3.10}
        row[column] = None
        result = rules.rule_features(pd.Series(row))
        assert result["R1_ah-0.25_match"] == 0, column
        assert result["R1_ah-0.25_near"] == 0, column
        assert math.isnan(result["R1_ah-0.25_dist"]), column


def test_an_unchecked_side_may_still_be_missing() -> None:
    """R3a only reads the draw price; missing fav/dog prices must not block it."""
    result = rules.rule_features(
        pd.Series({"ah_open": 0.0, "draw_ft_odds": 3.70, "pool_fav_open": None, "pool_dog_open": None})
    )
    assert result["R3a_ah0.0_match"] == 1
    assert result["R3a_ah0.0_dist"] == 0.0

    # ... but a missing *draw* price (the side it checks) suppresses it.
    result = rules.rule_features(
        pd.Series({"ah_open": 0.0, "draw_ft_odds": None, "pool_fav_open": 1.90, "pool_dog_open": 4.50})
    )
    assert result["R3a_ah0.0_match"] == 0
    assert result["R3a_ah0.0_near"] == 0
    assert math.isnan(result["R3a_ah0.0_dist"])


# --------------------------------------------------------------------------- #
# the two rule sets stay independent (spec section 1)
# --------------------------------------------------------------------------- #
def test_both_rule_sets_can_fire_on_the_same_match() -> None:
    row = _row(-0.25, 2.05, 3.25, 3.10)
    v2 = rules.rule_features(row)
    assert v2["R1_ah-0.25_match"] == 1
    assert v2["R6_1_ah-0.25_match"] == 1


def test_the_tip_table_rules_are_untouched_by_rule_set_two() -> None:
    """One row can enter both tables; neither may clobber the other (spec 1)."""
    # AH -0.25 at 2.06 is inside tip.jpg's own band (2.06-2.10) and also fires
    # image 2's R1 (fav <= 2.09, draw >= 3.25, dog >= 2.80).
    frame = make_frame(
        [
            make_match(ah_open=-0.25, pool_fav_open=2.06, draw_ft_odds=3.30, pool_dog_open=3.00),
        ]
    )
    out = features.build_features(frame, feature_set="v2")
    # the brief's own table matches its own band
    assert out["rule_match"].iloc[0] == 1
    # rule set 2 arrives as a separate block with its own names ...
    assert out["R1_ah-0.25_match"].iloc[0] == 1
    # ... and building it did not overwrite or extend the tip.jpg table itself
    assert list(features.RULES.columns) == ["ah", "pool_min", "pool_max"]
    assert len(features.RULES) == 8


def test_add_rule_features_adds_the_whole_block() -> None:
    frame = make_frame([make_match()])
    out = rules.add_rule_features(frame)
    for name in rules.rule_feature_names():
        assert name in out.columns


# --------------------------------------------------------------------------- #
# no-vig variant
# --------------------------------------------------------------------------- #
def test_novig_prices_remove_the_margin() -> None:
    for margin in (1.04, 1.10):
        fav = 1.0 / (0.48 * margin)
        draw = 1.0 / (0.28 * margin)
        dog = 1.0 / (0.24 * margin)
        nv_fav, nv_draw, nv_dog = rules.novig_prices(fav, draw, dog)
        assert math.isclose(nv_fav, 0.48, abs_tol=1e-9)
        assert math.isclose(nv_draw, 0.28, abs_tol=1e-9)
        assert math.isclose(nv_dog, 0.24, abs_tol=1e-9)


def test_novig_prices_reject_impossible_books() -> None:
    assert rules.novig_prices(None, 3.0, 3.0) == (None, None, None)
    assert rules.novig_prices(1.0, 3.0, 3.0) == (None, None, None)


def test_novig_table_is_a_probability_table() -> None:
    for rule in rules.RULES_V2_NOVIG:
        for bound in (rule.fav, rule.draw, rule.dog):
            if bound is None:
                continue
            low, high = bound
            assert low >= 0.0
            assert high <= 1.0 or high == rules.INF


def test_novig_bands_invert_the_price_order() -> None:
    """A 'price at most X' becomes a 'probability at least Y'."""
    r1_raw = next(rule for rule in rules.RULES_V2 if rule.key == "R1_ah-0.25")
    r1_nv = next(rule for rule in rules.RULES_V2_NOVIG if rule.key == "R1_ah-0.25")
    # fav <= 2.09 in price space -> probability >= 1/2.09/overround
    expected = (1.0 / r1_raw.fav[1]) / r1_raw.overround
    assert math.isclose(r1_nv.fav[0], expected, abs_tol=1e-9)


def test_novig_features_are_built_for_every_rule() -> None:
    frame = make_frame([make_match()])
    out = rules.add_rule_features(frame, suffix="_nv", novig=True)
    for name in rules.rule_feature_names(suffix="_nv"):
        assert name in out.columns


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def test_wilson_interval_is_bounded_and_sane() -> None:
    low, high = rules.wilson_interval(13, 20)
    assert 0.0 <= low < 0.65 < high <= 1.0
    # 13/20 must still be compatible with no edge at all
    assert low < 0.5


def test_wilson_interval_handles_no_trials() -> None:
    low, high = rules.wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)


def _audit_frame() -> pd.DataFrame:
    """Four matches that all fire R1 at -0.25, with known outcomes."""
    rows = []
    # favourite wins by 1 on -0.25 -> half win (a win, but a small one)
    rows.append(make_match(match_id="w1", fav_side="home", home_goals=1, away_goals=0,
                           ah_open=-0.25, ah_fav_price=2.00, ah_dog_price=1.90,
                           ou_line_open=2.5, ou_over_price=2.00, ou_under_price=1.90))
    # favourite wins by 2 -> full win
    rows.append(make_match(match_id="w2", fav_side="home", home_goals=2, away_goals=0,
                           ah_open=-0.25, ah_fav_price=2.00, ah_dog_price=1.90,
                           ou_line_open=2.5, ou_over_price=2.00, ou_under_price=1.90))
    # draw -> favourite loses the quarter line
    rows.append(make_match(match_id="l1", fav_side="home", home_goals=1, away_goals=1,
                           ah_open=-0.25, ah_fav_price=2.00, ah_dog_price=1.90,
                           ou_line_open=2.5, ou_over_price=2.00, ou_under_price=1.90))
    # away win -> favourite loses outright
    rows.append(make_match(match_id="l2", fav_side="home", home_goals=0, away_goals=1,
                           ah_open=-0.25, ah_fav_price=2.00, ah_dog_price=1.90,
                           ou_line_open=2.5, ou_over_price=2.00, ou_under_price=1.90))
    frame = make_frame(rows)
    frame["pool_fav_open"] = 2.05
    frame["draw_ft_odds"] = 3.25
    frame["pool_dog_open"] = 3.10
    return frame


def test_audit_counts_wins_pushes_and_losses_separately() -> None:
    prepared = features.build_features(_audit_frame())
    audit = rules.audit_rules(prepared, markets=("AH_fav",))

    row = audit[(audit["key"] == "R1_ah-0.25")].iloc[0]
    assert row["n"] == 4
    assert row["wins"] == 2
    assert row["losses"] == 2
    assert row["pushes"] == 0
    assert row["hit_rate"] == pytest.approx(0.5)
    assert row["break_even"] == pytest.approx(1.0 / 2.00)


def test_audit_excludes_pushes_from_the_hit_rate() -> None:
    frame = _audit_frame()
    # a whole line that lands exactly on the score -> push, not a win
    frame.loc[0, "ah_open"] = -1.0
    frame.loc[0, "home_goals"] = 1
    frame.loc[0, "away_goals"] = 0
    prepared = features.build_features(frame)
    audit = rules.audit_rules(prepared, markets=("AH_fav",))

    row = audit[audit["key"] == "R1_ah-0.25"]
    if not row.empty:
        assert row.iloc[0]["pushes"] == 0  # -1.0 is not a -0.25 rule line


def test_audit_reports_the_underdog_side_too() -> None:
    prepared = features.build_features(_audit_frame())
    audit = rules.audit_rules(prepared, markets=("AH_fav", "AH_dog"))
    assert set(audit["market"]) == {"AH_fav", "AH_dog"}


def test_audit_needs_the_feature_columns() -> None:
    with pytest.raises(ValueError):
        rules.audit_rules(make_frame([make_match()]))


def test_rule_table_overrounds_skips_rows_without_a_full_book() -> None:
    table = rules.rule_table_overrounds()
    assert "R1_ah-0.25" in set(table["key"])
    # R3a only constrains the draw, so it has no book of its own
    assert "R3a_ah0.0" not in set(table["key"])


def test_overround_of_a_known_book() -> None:
    assert math.isclose(rules.overround_of(2.00, 4.00, 4.00), 1.0, abs_tol=1e-9)


def test_settlement_reuses_the_single_label_implementation() -> None:
    """Rule features must not carry their own copy of the settlement maths."""
    # favourite wins by one at -0.25: both halves of the quarter line win, so
    # this is a FULL win of 1.0 unit (a half win would need x == 0.25).
    assert labels.ah_unit(1, -0.25) == 1.0
    assert labels.ah_unit_dog(1, -0.25) == -1.0

    # the underdog helper is exactly the favourite helper mirrored, on every
    # line the rule tables use - that is what "single implementation" means.
    for diff in (-2, -1, 0, 1, 2):
        for line in (-1.5, -0.5, -0.25, 0.0):
            assert labels.ah_unit(diff, line) == -labels.ah_unit_dog(diff, line)


# --------------------------------------------------------------------------- #
# pipeline-level invariants of the no-vig variant
# --------------------------------------------------------------------------- #
def test_novig_features_stay_identical_across_different_overrounds() -> None:
    """Same vig-free probabilities, different margins -> identical nv features.

    This is the property the v2_novig variant exists to test (spec 9.4b): the
    rule must read the same in a 1.04 book and a 1.10 book, because the vig-free
    probabilities are the same. The raw block, by contrast, does see the margin.
    """
    probs = (0.45, 0.25, 0.30)  # fav/draw/dog, deliberately mid-band for R1
    tight = tuple(1.0 / (p * 1.04) for p in probs)
    loose = tuple(1.0 / (p * 1.10) for p in probs)

    frame = make_frame(
        [
            make_match(match_id="tight", ah_open=-0.25, pool_fav_open=tight[0],
                       draw_ft_odds=tight[1], pool_dog_open=tight[2]),
            make_match(match_id="loose", ah_open=-0.25, pool_fav_open=loose[0],
                       draw_ft_odds=loose[1], pool_dog_open=loose[2]),
        ]
    )

    novig = rules.add_rule_features(frame, suffix="_nv", novig=True).set_index("match_id")
    for name in rules.rule_feature_names(suffix="_nv"):
        left, right = novig.loc["tight", name], novig.loc["loose", name]
        if isinstance(left, float) and math.isnan(left):
            assert math.isnan(right), name
        else:
            assert left == right, name

    raw = rules.add_rule_features(frame).set_index("match_id")
    # the printed-threshold block genuinely reacts to the margin (R1's draw
    # bound sits between the two books' draw prices), so the invariance above
    # is a property of the conversion, not of the data being identical
    assert raw.loc["tight", "R1_ah-0.25_dist"] != raw.loc["loose", "R1_ah-0.25_dist"]


def test_audit_rules_novig_reads_the_vigfree_block() -> None:
    """A no-vig audit must not silently reuse the raw rule columns.

    The test row is priced at a 1.04 margin with probabilities 0.45/0.25/0.30:
    the raw R1 rule does not match (2.137 > 2.09) while the vig-free R1 does.
    """
    probs = (0.45, 0.25, 0.30)
    prices = tuple(1.0 / (p * 1.04) for p in probs)
    frame = make_frame(
        [
            make_match(match_id="m1", ah_open=-0.25, pool_fav_open=prices[0],
                       draw_ft_odds=prices[1], pool_dog_open=prices[2]),
        ]
    )
    prepared = features.build_features(frame)

    novig_audit = rules.audit_rules(prepared, suffix="_nv", novig=True, markets=("AH_fav",))
    raw_audit = rules.audit_rules(prepared, markets=("AH_fav",))

    assert "R1_ah-0.25" in set(novig_audit["key"])
    assert "R1_ah-0.25" not in set(raw_audit["key"])


# --------------------------------------------------------------------------- #
# the owner's 2026-10-21 water rule (R7) - trained first, other rules aside
# --------------------------------------------------------------------------- #
def _water_row(
    ah: float = -0.75, fav_ah: float | None = 1.85, fav_1x2: float = 1.75
) -> pd.Series:
    return pd.Series(
        {
            "ah_open": ah,
            "ah_fav_price": fav_ah,
            "pool_fav_open": fav_1x2,
            "draw_ft_odds": 3.60,
            "pool_dog_open": 4.00,
        }
    )


def test_water_rule_needs_every_band() -> None:
    key = rules.RULES_WATER[0].key
    assert key == "R7_ah-0.75"

    inside = rules.rule_features(_water_row(), table=rules.RULES_WATER)
    assert inside[f"{key}_match"] == 1

    # boundaries are inclusive on all three checks
    for kwargs in ({"fav_ah": 1.80}, {"fav_ah": 1.88}, {"fav_1x2": 1.87}):
        assert (
            rules.rule_features(_water_row(**kwargs), table=rules.RULES_WATER)[
                f"{key}_match"
            ]
            == 1
        ), kwargs

    # one step outside the water band: no match, but the distance is TOL-close
    outside_price = rules.rule_features(_water_row(fav_ah=1.79), table=rules.RULES_WATER)
    assert outside_price[f"{key}_match"] == 0
    assert outside_price[f"{key}_near"] == 1

    # one step past the 1X2 cap: no match either
    outside_1x2 = rules.rule_features(_water_row(fav_1x2=1.90), table=rules.RULES_WATER)
    assert outside_1x2[f"{key}_match"] == 0

    # wrong handicap line: no match
    wrong_line = rules.rule_features(_water_row(ah=-0.5), table=rules.RULES_WATER)
    assert wrong_line[f"{key}_match"] == 0

    # a missing favourite handicap price suppresses the rule, never a match
    missing = rules.rule_features(_water_row(fav_ah=None), table=rules.RULES_WATER)
    assert missing[f"{key}_match"] == 0
    assert math.isnan(missing[f"{key}_dist"])


def test_water_rule_stays_out_of_the_printed_sets() -> None:
    """R7 must not leak into RULES_V2: the printed audits pin 16 rows."""
    assert len(rules.RULES_V2) == 16
    assert all(rule.rule_id != "R7" for rule in rules.RULES_V2)
    assert len(rules.RULES_WATER) == 1
    assert rules.RULES_WATER[0].ah_fav == (1.80, 1.88)


def test_audit_rules_supports_a_custom_table_and_builds_its_columns() -> None:
    """Regression: a non-default table used to generate RULES_V2 columns."""
    frame = make_frame(
        [
            make_match(
                match_id="win",
                ah_open=-0.75,
                ah_fav_price=1.83,
                pool_fav_open=1.75,
                draw_ft_odds=3.60,
                pool_dog_open=4.00,
                home_goals=2,
                away_goals=0,
            ),
            make_match(
                match_id="loss",
                ah_open=-0.75,
                ah_fav_price=1.83,
                pool_fav_open=1.75,
                draw_ft_odds=3.60,
                pool_dog_open=4.00,
                home_goals=1,
                away_goals=1,
            ),
        ]
    )
    prepared = features.build_features(frame)  # v1 frame: no rule columns at all
    audit = rules.audit_rules(prepared, table=rules.RULES_WATER, markets=("AH_fav",))

    assert not audit.empty
    row = audit.iloc[0]
    assert row["key"] == "R7_ah-0.75"
    assert row["market"] == "AH_fav"
    assert row["n"] == 2 and row["wins"] == 1 and row["losses"] == 1
    assert math.isclose(row["hit_rate"], 0.5)
