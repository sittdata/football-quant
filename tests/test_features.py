"""Feature engineering: the rule table, and the leakage guards."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fbpredict import features

from .helpers import make_frame, make_match


# --------------------------------------------------------------------------- #
# the tip.jpg rule table
# --------------------------------------------------------------------------- #
def test_rule_features_exact_hit() -> None:
    result = features.rule_features(-0.75, 1.62)
    assert result["rule_match"] == 1
    assert result["rule_near"] == 1
    assert result["rule_dist"] == 0.0


def test_rule_features_boundaries_are_inclusive() -> None:
    assert features.rule_features(-0.5, 1.79)["rule_match"] == 1
    assert features.rule_features(-0.5, 1.83)["rule_match"] == 1


def test_rule_features_near_but_outside() -> None:
    result = features.rule_features(-1.0, 1.52)  # band is 1.47-1.50
    assert result["rule_match"] == 0
    assert result["rule_near"] == 1
    assert math.isclose(result["rule_dist"], 0.02, abs_tol=1e-9)


def test_rule_features_far_outside() -> None:
    result = features.rule_features(-1.0, 2.50)
    assert result["rule_match"] == 0
    assert result["rule_near"] == 0
    assert result["rule_dist"] > 0.9


def test_rule_features_unknown_handicap_returns_nan_distance() -> None:
    result = features.rule_features(-3.0, 1.10)
    assert result["rule_match"] == 0
    assert result["rule_near"] == 0
    assert math.isnan(result["rule_dist"])


def test_rule_features_tolerates_missing_input() -> None:
    assert features.rule_features(None, 1.62)["rule_match"] == 0
    assert features.rule_features(-0.75, None)["rule_match"] == 0


# --------------------------------------------------------------------------- #
# the goal-based rules
# --------------------------------------------------------------------------- #
def test_draw_thresholds_come_from_the_image() -> None:
    frame = make_frame(
        [
            make_match(match_id="a", draw_ht_odds=2.30),
            make_match(match_id="b", draw_ht_odds=2.35),
            make_match(match_id="c", draw_ht_odds=2.45),
            make_match(match_id="d", draw_ht_odds=2.50),
        ]
    )
    out = features.build_features(frame)
    assert list(out["draw_ht_high"]) == [0, 1, 1, 1]
    assert list(out["draw_ht_vhigh"]) == [0, 0, 1, 1]


def test_draw_ft_band_is_4_30_plus_minus_0_05() -> None:
    frame = make_frame(
        [
            make_match(match_id="a", draw_ft_odds=4.20),
            make_match(match_id="b", draw_ft_odds=4.25),
            make_match(match_id="c", draw_ft_odds=4.30),
            make_match(match_id="d", draw_ft_odds=4.35),
            make_match(match_id="e", draw_ft_odds=4.40),
        ]
    )
    out = features.build_features(frame)
    assert list(out["draw_ft_match"]) == [0, 1, 1, 1, 0]


def test_interaction_feature() -> None:
    frame = make_frame([make_match(ah_open=-0.75, ou_line_open=2.5)])
    out = features.build_features(frame)
    assert math.isclose(out["ah_abs"].iloc[0], 0.75)
    assert math.isclose(out["total_x_ah"].iloc[0], 2.5 * 0.75)


# --------------------------------------------------------------------------- #
# favourite orientation
# --------------------------------------------------------------------------- #
def test_goal_diff_uses_fav_side_home() -> None:
    frame = make_frame([make_match(fav_side="home", home_goals=3, away_goals=1)])
    out = features.build_features(frame)
    assert out["goal_diff_fav"].iloc[0] == 2


def test_goal_diff_uses_fav_side_away() -> None:
    frame = make_frame([make_match(fav_side="away", home_goals=1, away_goals=3)])
    out = features.build_features(frame)
    assert out["goal_diff_fav"].iloc[0] == 2  # the away side is the favourite


def test_fav_side_falls_back_to_the_lower_price() -> None:
    frame = make_frame(
        [
            make_match(fav_side=None, pool_fav_open=1.30, pool_dog_open=6.0),
        ]
    )
    frame["pool_home_open"] = [1.30]
    frame["pool_away_open"] = [6.0]
    assert features.favourite_is_home(frame).iloc[0] == "home"

    frame["pool_home_open"] = [6.0]
    frame["pool_away_open"] = [1.30]
    assert features.favourite_is_home(frame).iloc[0] == "away"


# --------------------------------------------------------------------------- #
# the important one: no leakage from rolling features
# --------------------------------------------------------------------------- #
def test_rolling_features_never_see_the_current_match() -> None:
    frame = make_frame(
        [
            make_match(
                match_id="m1",
                kickoff_at="2025-01-01T12:00:00+00:00",
                home_team="Alpha",
                away_team="Beta",
                home_goals=2,
                away_goals=1,
                ou_line_open=2.5,
            ),
            make_match(
                match_id="m2",
                kickoff_at="2025-01-08T12:00:00+00:00",
                home_team="Alpha",
                away_team="Gamma",
                home_goals=5,
                away_goals=0,
                ou_line_open=2.5,
            ),
        ]
    )
    out = features.build_features(frame).set_index("match_id")

    # first appearance: nothing to look back on
    assert pd.isna(out.loc["m1", "home_gf_avg"])

    # second match sees ONLY the first one (2 goals), not its own 5
    assert math.isclose(out.loc["m2", "home_gf_avg"], 2.0)
    assert math.isclose(out.loc["m2", "home_ga_avg"], 1.0)
    assert math.isclose(out.loc["m2", "home_total_avg"], 3.0)
    assert math.isclose(out.loc["m2", "home_over_rate"], 1.0)  # 3 goals > 2.5 line


def test_rolling_is_per_team() -> None:
    frame = make_frame(
        [
            make_match(match_id="m1", kickoff_at="2025-01-01T12:00:00+00:00",
                       home_team="Alpha", away_team="Beta", home_goals=4, away_goals=0),
            make_match(match_id="m2", kickoff_at="2025-01-08T12:00:00+00:00",
                       home_team="Gamma", away_team="Delta", home_goals=1, away_goals=0),
            make_match(match_id="m3", kickoff_at="2025-01-15T12:00:00+00:00",
                       home_team="Alpha", away_team="Delta", home_goals=0, away_goals=0),
        ]
    )
    out = features.build_features(frame).set_index("match_id")
    # Alpha's second match averages only Alpha's own history
    assert math.isclose(out.loc["m3", "home_gf_avg"], 4.0)
    # Delta played m2 (0 goals scored, 1 conceded)
    assert math.isclose(out.loc["m3", "away_gf_avg"], 0.0)
    assert math.isclose(out.loc["m3", "away_ga_avg"], 1.0)


# --------------------------------------------------------------------------- #
# shapes and odds
# --------------------------------------------------------------------------- #
def test_every_feature_column_exists_after_build() -> None:
    out = features.build_features(make_frame([make_match()]))
    for column in features.FEATURE_COLUMNS:
        assert column in out.columns, column


def test_fair_probabilities_are_vig_free() -> None:
    frame = make_frame([make_match(ah_fav_price=1.90, ah_dog_price=1.90,
                                   ou_over_price=2.10, ou_under_price=1.75)])
    out = features.build_features(frame)
    assert math.isclose(out["fair_fav_ah"].iloc[0] + out["fair_dog_ah"].iloc[0], 1.0)
    assert math.isclose(out["fair_over"].iloc[0] + out["fair_under"].iloc[0], 1.0)
    assert out["ah_overround"].iloc[0] > 0


def test_missing_feature_report_counts() -> None:
    frame = make_frame([make_match(), make_match(match_id="m2", draw_ht_odds=None)])
    out = features.build_features(frame)
    report = features.missing_feature_report(out).set_index("feature")
    assert report.loc["draw_ht_odds", "missing"] == 1
    assert report.loc["draw_ht_odds", "present"] == 1


# --------------------------------------------------------------------------- #
# the red row from tip.jpg must actually be applied
# --------------------------------------------------------------------------- #
def test_red_row_special_rule_is_applied() -> None:
    frame = make_frame(
        [
            make_match(match_id="hit", ah_open=-0.5, pool_fav_open=1.585),
            make_match(match_id="price_outside", ah_open=-0.5, pool_fav_open=1.80),
            make_match(match_id="line_moved", ah_open=-0.75, pool_fav_open=1.585),
        ]
    )
    out = features.build_features(frame).set_index("match_id")
    assert out.loc["hit", "special_ah_not_drifted"] == 1
    assert out.loc["price_outside", "special_ah_not_drifted"] == 0
    assert out.loc["line_moved", "special_ah_not_drifted"] == 0


def test_special_rule_handles_missing_input() -> None:
    assert features.special_rule_features(None, 1.585) == 0
    assert features.special_rule_features(-0.5, None) == 0
    assert features.special_rule_features(float("nan"), 1.585) == 0


def test_rolling_window_has_a_single_default() -> None:
    import inspect

    assert features.DEFAULT_ROLLING_WINDOW == 5
    assert (
        inspect.signature(features.build_features).parameters["rolling_window"].default
        == features.DEFAULT_ROLLING_WINDOW
    )
    assert (
        inspect.signature(features.add_team_rolling).parameters["window"].default
        == features.DEFAULT_ROLLING_WINDOW
    )


def test_fair_probs_still_remove_the_margin() -> None:
    """The deduplicated helper must keep the same behaviour as before."""
    frame = make_frame([make_match(ah_fav_price=1.80, ah_dog_price=2.05,
                                   ou_over_price=1.95, ou_under_price=1.95)])
    out = features.build_features(frame)
    assert math.isclose(out["fair_fav_ah"].iloc[0] + out["fair_dog_ah"].iloc[0], 1.0)
    assert math.isclose(out["fair_over"].iloc[0] + out["fair_under"].iloc[0], 1.0)
    assert out["fair_fav_ah"].iloc[0] > out["fair_dog_ah"].iloc[0]


def test_fair_probs_are_nan_when_odds_are_missing() -> None:
    frame = make_frame([make_match(ah_fav_price=None, ah_dog_price=None,
                                   ou_over_price=None, ou_under_price=None)])
    out = features.build_features(frame)
    assert pd.isna(out["fair_fav_ah"].iloc[0])
    assert pd.isna(out["fair_over"].iloc[0])


# --------------------------------------------------------------------------- #
# opening -> closing drift (the red row of tip.jpg, for real)
# --------------------------------------------------------------------------- #
def test_closing_line_drift_features() -> None:
    frame = make_frame(
        [
            make_match(match_id="held", ah_open=-0.5, ah_close=-0.5,
                       ou_line_open=2.5, ou_line_close=2.5),
            make_match(match_id="moved", ah_open=-0.5, ah_close=-0.75,
                       ou_line_open=2.5, ou_line_close=3.0),
            make_match(match_id="no_closing", ah_open=-0.5, ah_close=None,
                       ou_line_open=2.5, ou_line_close=None),
        ]
    )
    out = features.build_features(frame).set_index("match_id")

    assert out.loc["held", "ah_drift"] == 0.0
    assert out.loc["held", "ah_not_drifted_to_075"] == 1

    assert out.loc["moved", "ah_drift"] == pytest.approx(-0.25)
    assert out.loc["moved", "ah_not_drifted_to_075"] == 0
    assert out.loc["moved", "ou_drift"] == pytest.approx(0.5)

    # no closing line -> NaN drift, and the flag must not guess
    assert pd.isna(out.loc["no_closing", "ah_drift"])
    assert out.loc["no_closing", "ah_not_drifted_to_075"] == 0


def test_drift_features_are_absent_without_closing_columns() -> None:
    """A provider that supplies no closing line must not break the build."""
    out = features.build_features(make_frame([make_match()]))
    assert pd.isna(out["ah_drift"].iloc[0])
    assert out["ah_not_drifted_to_075"].iloc[0] == 0


# --------------------------------------------------------------------------- #
# feature sets v1 / v2 / v2_novig
# --------------------------------------------------------------------------- #
def test_feature_set_columns_and_unknown_sets() -> None:
    assert features.feature_set_columns("v1") == features.FEATURE_COLUMNS
    v2 = features.feature_set_columns("v2")
    v2_novig = features.feature_set_columns("v2_novig")
    assert len(features.RULE_SET_V2_FEATURES) == 48
    assert "R1_ah-0.25_match" in v2
    assert "fav_no_drop" in v2
    # the no-vig set carries the nv block INSTEAD of the raw one (never both:
    # running the two variants together was ruled out in the plan)
    assert "R1_ah-0.25_nv_match" in v2_novig
    assert "R1_ah-0.25_match" not in v2_novig
    assert "fav_no_drop_nv" in v2_novig
    with pytest.raises(ValueError):
        features.feature_set_columns("v9")


def test_v2_build_adds_the_rule_block_and_the_novig_build_uses_nv_names() -> None:
    frame = make_frame([make_match()])
    v2 = features.build_features(frame, feature_set="v2")
    for name in features.RULE_SET_V2_FEATURES:
        assert name in v2.columns, name
    nv = features.build_features(frame, feature_set="v2_novig")
    for name in features.RULE_SET_V2_NOVIG_FEATURES:
        assert name in nv.columns, name
    assert "R1_ah-0.25_match" not in nv.columns
    # v1 stays exactly what it was: no rule-set-2 columns at all
    v1 = features.build_features(frame)
    assert "R1_ah-0.25_match" not in v1.columns
    assert "fav_no_drop" not in v1.columns


def test_closing_flags_are_nan_in_normal_mode_even_with_closing_prices() -> None:
    """The leakage guard: the normal 24h mode must never read closing prices."""
    frame = make_frame(
        [make_match(pool_fav_close=1.75, pool_dog_close=4.40, draw_ft_close=3.50)]
    )
    out = features.build_features(frame, feature_set="v2")
    assert pd.isna(out["fav_no_drop"].iloc[0])
    assert pd.isna(out["dog_no_drop"].iloc[0])


def test_closing_flags_fill_only_in_late_mode() -> None:
    frame = make_frame(
        [
            # favourite price held up, underdog price fell
            make_match(match_id="mixed", pool_fav_open=1.80, pool_fav_close=1.85,
                       pool_dog_open=4.20, pool_dog_close=4.00),
            # favourite price fell, underdog price rose
            make_match(match_id="other", pool_fav_open=1.80, pool_fav_close=1.75,
                       pool_dog_open=4.20, pool_dog_close=4.40),
            # no closing prices at all -> the flag must stay unknown
            make_match(match_id="unknown", pool_fav_open=1.80, pool_fav_close=None,
                       pool_dog_open=4.20, pool_dog_close=None),
        ]
    )
    out = features.build_features(
        frame, feature_set="v2", use_closing_features=True
    ).set_index("match_id")

    assert out.loc["mixed", "fav_no_drop"] == 1.0   # 1.85 >= 1.80
    assert out.loc["mixed", "dog_no_drop"] == 0.0   # 4.00 < 4.20
    assert out.loc["other", "fav_no_drop"] == 0.0
    assert out.loc["other", "dog_no_drop"] == 1.0
    assert pd.isna(out.loc["unknown", "fav_no_drop"])
    assert pd.isna(out.loc["unknown", "dog_no_drop"])


def test_novig_closing_flags_compare_probabilities() -> None:
    frame = make_frame(
        [
            make_match(
                pool_fav_open=2.00, draw_ft_odds=3.50, pool_dog_open=3.50,
                pool_fav_close=1.90, draw_ft_close=3.50, pool_dog_close=3.70,
            )
        ]
    )
    out = features.build_features(
        frame, feature_set="v2_novig", use_closing_features=True
    )
    # the favourite's price fell, so its vig-free probability rose: no drop
    assert out["fav_no_drop_nv"].iloc[0] == 1.0
    assert out["dog_no_drop_nv"].iloc[0] == 0.0


def test_missing_feature_report_can_measure_a_feature_set() -> None:
    out = features.build_features(make_frame([make_match()]), feature_set="v2")
    report = features.missing_feature_report(
        out, features.feature_set_columns("v2")
    ).set_index("feature")
    assert report.loc["R1_ah-0.25_match", "present"] == 1
    assert report.loc["fav_no_drop", "missing"] == 1


# --------------------------------------------------------------------------- #
# the owner's 2026-10-21 water rule, trained first with every other rule aside
# --------------------------------------------------------------------------- #
def test_water_feature_set_contains_only_the_owner_rule() -> None:
    assert features.feature_set_columns("water") == features.RULE_WATER_FEATURES
    assert len(features.RULE_WATER_FEATURES) == 3
    assert "R7_ah-0.75_match" in features.feature_set_columns("water")

    out = features.build_features(make_frame([make_match()]), feature_set="water")
    assert "R7_ah-0.75_match" in out.columns

    # and the printed sets stay untouched by the new rule
    for name in ("v1", "v2", "v2_novig"):
        assert "R7_ah-0.75_match" not in features.feature_set_columns(name)
