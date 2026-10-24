"""Backtest: the split, the blind view, and the look-ahead guarantee."""

from __future__ import annotations

import pandas as pd
import pytest

from fbpredict import backtest, demo, features, model, report
from fbpredict.config import load_config

from .helpers import make_frame, make_match


# --------------------------------------------------------------------------- #
# blind_view, as described in the brief
# --------------------------------------------------------------------------- #
def test_blind_view_separates_past_from_today() -> None:
    frame = make_frame(
        [
            make_match(match_id="past", kickoff_at="2025-01-01T12:00:00+00:00"),
            make_match(match_id="today-a", kickoff_at="2025-01-02T12:00:00+00:00"),
            make_match(match_id="today-b", kickoff_at="2025-01-02T20:00:00+00:00"),
            make_match(match_id="future", kickoff_at="2025-01-05T12:00:00+00:00"),
        ]
    )
    known, todo = backtest.blind_view(frame, pd.Timestamp("2025-01-02T00:00:00+00:00"))

    assert list(known["match_id"]) == ["past"]
    assert sorted(todo["match_id"]) == ["today-a", "today-b"]


def test_blind_view_erases_the_results() -> None:
    frame = make_frame([make_match(match_id="today", kickoff_at="2025-01-02T12:00:00+00:00")])
    _, todo = backtest.blind_view(frame, pd.Timestamp("2025-01-02T00:00:00+00:00"))
    for column in ("home_goals", "away_goals", "ht_home", "ht_away"):
        assert todo[column].isna().all()


# --------------------------------------------------------------------------- #
# the split
# --------------------------------------------------------------------------- #
def test_time_split_is_ordered_and_never_shuffled() -> None:
    frame = make_frame(
        [
            make_match(match_id=f"m{index:02d}", kickoff_at=f"2025-01-{index + 1:02d}T12:00:00+00:00")
            for index in range(10)
        ]
    )
    train, test = backtest.time_split(frame, 0.8)

    assert len(train) == 8
    assert len(test) == 2
    assert train["kickoff_at"].max() <= test["kickoff_at"].min()
    assert list(test["match_id"]) == ["m08", "m09"]


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic_frame() -> pd.DataFrame:
    return demo.generate(seed=31, n_matches=120)


def test_every_prediction_happens_before_kickoff(synthetic_frame: pd.DataFrame) -> None:
    cfg = load_config()
    cfg.min_train = 30
    result = backtest.walk_forward(synthetic_frame, cfg)

    assert not result.predictions.empty
    assert (result.predictions["predicted_at"] < result.predictions["kickoff_at"]).all()


def test_the_model_only_sees_earlier_matches(synthetic_frame: pd.DataFrame) -> None:
    """The first test row must be predicted from the train window alone."""
    cfg = load_config()
    cfg.min_train = 20
    result = backtest.walk_forward(synthetic_frame, cfg)
    assert result.train_size == int(len(synthetic_frame) * cfg.train_ratio)
    assert result.n_models_fitted >= 1


def test_skips_everything_when_history_is_too_short(synthetic_frame: pd.DataFrame) -> None:
    cfg = load_config()
    cfg.min_train = 10_000
    result = backtest.walk_forward(synthetic_frame, cfg)

    assert result.taken == 0
    assert result.skipped == result.test_size
    assert result.predictions["market"].isna().all()


def test_skipped_rows_still_carry_probabilities(synthetic_frame: pd.DataFrame) -> None:
    """The report needs both heads scored even for matches we did not bet."""
    cfg = load_config()
    cfg.min_train = 20
    cfg.edge_threshold = 0.99  # force everything to be skipped
    result = backtest.walk_forward(synthetic_frame, cfg)

    assert result.taken == 0
    assert result.predictions["p_fav_cover"].notna().any()
    assert result.predictions["p_over"].notna().any()


def test_threshold_controls_how_many_matches_are_bet(synthetic_frame: pd.DataFrame) -> None:
    cfg = load_config()
    cfg.min_train = 20

    loose = backtest.walk_forward(synthetic_frame, cfg, threshold=0.0)
    strict = backtest.walk_forward(synthetic_frame, cfg, threshold=0.50)

    assert loose.taken >= strict.taken


def test_profit_is_zero_for_skipped_matches(synthetic_frame: pd.DataFrame) -> None:
    cfg = load_config()
    cfg.min_train = 20
    cfg.edge_threshold = 0.99
    result = backtest.walk_forward(synthetic_frame, cfg)
    assert (result.predictions["profit"].fillna(0.0) == 0.0).all()


def test_forward_predict_handles_upcoming_matches(synthetic_frame: pd.DataFrame) -> None:
    from fbpredict import model

    bundle, _ = model.train_model(synthetic_frame)
    cfg = load_config()
    upcoming = synthetic_frame.head(5).copy()
    upcoming["home_goals"] = None
    upcoming["away_goals"] = None

    picks = backtest.forward_predict(upcoming, bundle, cfg)
    assert len(picks) == 5
    assert {"match_id", "market", "edge", "predicted_at"} <= set(picks.columns)


def test_feature_coverage_describes_the_feature_frame(synthetic_frame: pd.DataFrame) -> None:
    """Regression: coverage used to be measured on the predictions frame.

    That frame only carries a handful of columns, so every feature was reported
    as absent even though the model had trained on them - which made the
    report's coverage table meaningless.
    """
    cfg = load_config()
    cfg.min_train = 30
    result = backtest.walk_forward(synthetic_frame, cfg)

    assert result.feature_coverage is not None
    coverage = result.feature_coverage.set_index("feature")

    # columns the model actually consumes (ah_open/pool_fav_open are raw inputs,
    # not model features, so they are deliberately absent from this table)
    for feature in ("ou_line_open", "pool_fav_open", "home_gf_avg", "ah_abs"):
        assert coverage.loc[feature, "present"] > 0, feature
    # the demo emits closing lines now, so the drift features are exercised
    assert coverage.loc["ah_drift", "present"] > 0

    metrics = report.evaluate(result, cfg)
    by_name = {row["feature"]: row for row in metrics["feature_coverage"]}
    assert by_name["ah_abs"]["present"] > 0
    assert by_name["ah_abs"]["missing"] < len(result.predictions)


# --------------------------------------------------------------------------- #
# the prediction clock and the closing-price guard
# --------------------------------------------------------------------------- #
def test_closing_features_are_refused_outside_the_closing_window() -> None:
    """Refuse, not warn: asking for closing prices at the 24h clock is look-ahead."""
    frame = demo.generate(seed=61, n_matches=120)
    cfg = load_config()
    cfg.min_train = 30

    with pytest.raises(ValueError, match="refusing"):
        backtest.walk_forward(frame, cfg, feature_set="v2", use_closing_features=True)

    with pytest.raises(ValueError, match="refusing"):
        backtest.walk_forward(
            frame,
            cfg,
            feature_set="v2",
            predicted_before_minutes=1440,
            use_closing_features=True,
        )

    # v1 has no closing features at all, so the request itself is invalid
    with pytest.raises(ValueError, match="refusing"):
        backtest.walk_forward(
            frame,
            cfg,
            feature_set="v1",
            predicted_before_minutes=5,
            use_closing_features=True,
        )


def test_late_mode_fills_closing_features_inside_the_window() -> None:
    frame = demo.generate(seed=67, n_matches=120)
    cfg = load_config()
    cfg.min_train = 30

    result = backtest.walk_forward(
        frame,
        cfg,
        feature_set="v2",
        predicted_before_minutes=5,
        use_closing_features=True,
    )

    coverage = result.feature_coverage.set_index("feature")
    assert coverage.loc["fav_no_drop", "present"] > 0
    assert "fav_no_drop" in result.active_features
    assert "closing-price features were filled" in " ".join(result.notes)

    # the prediction clock really moved: 5 minutes before kick-off
    deltas = (
        result.predictions["kickoff_at"] - result.predictions["predicted_at"]
    ).dt.total_seconds()
    assert float((deltas - 300.0).abs().max()) < 0.01
    assert result.predicted_before_minutes == 5


def test_normal_mode_reports_the_closing_features_as_dropped() -> None:
    frame = demo.generate(seed=71, n_matches=120)
    cfg = load_config()
    cfg.min_train = 30

    result = backtest.walk_forward(frame, cfg, feature_set="v2")

    assert result.feature_set == "v2"
    assert "fav_no_drop" not in result.active_features
    assert "fav_no_drop" in result.dropped_features
    assert "dog_no_drop" in result.dropped_features


def test_unknown_feature_set_is_refused_before_any_work() -> None:
    frame = demo.generate(seed=71, n_matches=60)
    cfg = load_config()
    with pytest.raises(ValueError):
        backtest.walk_forward(frame, cfg, feature_set="v9")


def test_forward_predict_refuses_early_serving_with_closing_features() -> None:
    frame = demo.generate(seed=73, n_matches=140)
    bundle, _ = model.train_model(frame, feature_set="v2", use_closing_features=True)

    upcoming = frame.tail(3).copy()
    upcoming["home_goals"] = None
    upcoming["away_goals"] = None
    cfg = load_config()

    earliest_kickoff = pd.Timestamp(upcoming["kickoff_at"].min())
    with pytest.raises(ValueError, match="closing-price"):
        backtest.forward_predict(
            upcoming, bundle, cfg, predicted_at=earliest_kickoff - pd.Timedelta(hours=24)
        )

    # inside the window it serves normally - and a single stamp can only serve
    # matches whose closing price already exists, i.e. kicking off soon
    one_match = upcoming.tail(1)
    picks = backtest.forward_predict(
        one_match,
        bundle,
        cfg,
        predicted_at=pd.Timestamp(one_match["kickoff_at"].iloc[0]) - pd.Timedelta(minutes=5),
    )
    assert len(picks) == 1


# --------------------------------------------------------------------------- #
# the report carries the spec sections
# --------------------------------------------------------------------------- #
def test_report_carries_the_spec_sections(synthetic_frame: pd.DataFrame) -> None:
    cfg = load_config()
    cfg.min_train = 30
    frame = synthetic_frame.copy()
    # make a handful of matches sit exactly on an image-2 rule (R5_1 at -0.25:
    # fav 2.20 / draw 3.50 / dog 2.99) so the per-rule table is non-empty; the
    # synthetic prices almost never hit a narrow band by chance
    for index in (10, 40, 70, 100):
        frame.loc[index, "ah_open"] = -0.25
        frame.loc[index, "pool_fav_open"] = 2.20
        frame.loc[index, "draw_ft_odds"] = 3.50
        frame.loc[index, "pool_dog_open"] = 2.99

    result = backtest.walk_forward(frame, cfg, feature_set="v2")
    metrics = report.evaluate(result, cfg)

    assert metrics["features"]["feature_set"] == "v2"
    assert metrics["rule_performance"], "per-rule performance table is missing"
    assert any(row["key"] == "R5_1_ah-0.25" for row in metrics["rule_performance"])
    assert metrics["overround_comparison"]["measured"]["n"] > 0
    assert metrics["overround_comparison"]["table"]
    assert [row["rule_id"] for row in metrics["untestable_rules"]] == ["R4a"]
    assert metrics["odds_format"]["prices_checked"] > 0

    text = report.render_markdown(metrics)
    for header in (
        "Rule set 2 - per-rule performance",
        "Overround comparison",
        "Odds format record",
        "Rules that stay untestable",
    ):
        assert header in text, header


def test_report_json_is_serialisable_with_the_new_sections(
    synthetic_frame: pd.DataFrame, tmp_path
) -> None:
    import json

    cfg = load_config()
    cfg.min_train = 30
    cfg.reports_dir = tmp_path
    result = backtest.walk_forward(synthetic_frame, cfg, feature_set="v2_novig")
    metrics = report.evaluate(result, cfg)

    md_path, json_path = report.save_report(metrics, cfg, stem="tmp_report_test")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["features"]["feature_set"] == "v2_novig"
    assert payload["untestable_rules"][0]["rule_id"] == "R4a"
    assert md_path.read_text(encoding="utf-8").startswith("# Backtest report")
