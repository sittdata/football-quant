"""Model training and the market-selection rule."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fbpredict import demo, features, model

from .helpers import make_match


def _flat_row() -> pd.Series:
    """A row whose fair prices are exactly 0.5/0.5 on both markets."""
    return pd.Series(
        make_match(ah_fav_price=2.0, ah_dog_price=2.0, ou_over_price=2.0, ou_under_price=2.0)
    )


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def test_skips_when_no_edge_clears_the_threshold() -> None:
    pick = model.select_market(_flat_row(), 0.50, 0.50, threshold=0.03)
    assert pick.market is None
    assert "below threshold" in pick.reason
    assert pick.edge == pytest.approx(0.0)


def test_picks_the_favourite_when_it_has_the_edge() -> None:
    pick = model.select_market(_flat_row(), 0.60, 0.50, threshold=0.03)
    assert pick.market == "AH_fav"
    assert pick.edge == pytest.approx(0.10)
    assert pick.side_odds == pytest.approx(2.0)
    assert pick.p_model == pytest.approx(0.60)


def test_can_pick_the_underdog_side() -> None:
    pick = model.select_market(_flat_row(), 0.40, 0.50, threshold=0.03)
    assert pick.market == "AH_dog"
    assert pick.p_model == pytest.approx(0.60)  # 1 - 0.40


def test_can_pick_the_over() -> None:
    pick = model.select_market(_flat_row(), 0.50, 0.62, threshold=0.03)
    assert pick.market == "OU_over"
    assert pick.p_model == pytest.approx(0.62)


def test_can_pick_the_under() -> None:
    pick = model.select_market(_flat_row(), 0.50, 0.35, threshold=0.03)
    assert pick.market == "OU_under"
    assert pick.p_model == pytest.approx(0.65)


def test_chooses_the_largest_edge_across_both_markets() -> None:
    row = pd.Series(
        make_match(ah_fav_price=1.80, ah_dog_price=2.10, ou_over_price=2.00, ou_under_price=2.00)
    )
    pick = model.select_market(row, 0.62, 0.55, threshold=0.01)
    assert pick.market in pick.all_edges
    assert pick.edge == pytest.approx(max(pick.all_edges.values()))


def test_blacklist_excludes_blacklisted_market() -> None:
    # Normally OU_over would win here (0.62 vs fair ~0.50 -> edge ~0.12)
    pick = model.select_market(
        _flat_row(),
        0.50,
        0.62,
        threshold=0.03,
        blacklist_markets=["OU_over"],
    )
    assert pick.market != "OU_over"
    # edges still present in all_edges for transparency
    assert "OU_over" in pick.all_edges


def test_allowed_markets_filters_candidates() -> None:
    # Only allow dog and under
    pick = model.select_market(
        _flat_row(),
        0.65,  # fav cover would normally win
        0.65,  # over would normally have edge
        threshold=0.03,
        allowed_markets=["AH_dog", "OU_under"],
    )
    # fav and over are filtered out
    assert pick.market in (None, "AH_dog", "OU_under")



def test_refuses_when_only_one_side_is_priced() -> None:
    row = pd.Series(make_match(ah_fav_price=1.9, ah_dog_price=None,
                               ou_over_price=None, ou_under_price=None))
    pick = model.select_market(row, 0.9, 0.9)
    assert pick.market is None
    assert "no usable prices" in pick.reason


def test_ignores_impossible_prices() -> None:
    row = pd.Series(make_match(ah_fav_price=1.0, ah_dog_price=1.0,
                               ou_over_price=0.5, ou_under_price=2.0))
    pick = model.select_market(row, 0.9, 0.9)
    assert pick.market is None


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def test_train_model_reports_what_it_used() -> None:
    frame = demo.generate(seed=11, n_matches=140)
    bundle, report = model.train_model(frame, kind="logreg", seed=1)

    assert report.n_rows_used > 0
    assert report.n_rows_used <= report.n_rows_in
    assert 0.0 < report.class_balance["ah_cover_rate"] < 1.0
    assert 0.0 < report.class_balance["ou_over_rate"] < 1.0
    assert set(bundle.feature_columns) == set(features.FEATURE_COLUMNS)


def test_probabilities_are_bounded_and_ordered() -> None:
    frame = demo.generate(seed=13, n_matches=120)
    bundle, _ = model.train_model(frame)
    probabilities = bundle.predict_proba_df(frame)

    for column in ("p_fav_cover", "p_over"):
        values = probabilities[column]
        assert values.notna().all()
        assert ((values >= 0.0) & (values <= 1.0)).all()


def test_train_model_refuses_a_tiny_sample() -> None:
    frame = demo.generate(seed=17, n_matches=8)
    with pytest.raises(ValueError):
        model.train_model(frame)


def test_ensure_feature_columns_fills_gaps() -> None:
    frame = pd.DataFrame([{"kickoff_at": "2025-01-01T12:00:00+00:00"}])
    prepared = model.ensure_feature_columns(frame)
    for column in features.FEATURE_COLUMNS:
        assert column in prepared.columns


# --------------------------------------------------------------------------- #
# freezing the model
# --------------------------------------------------------------------------- #
def test_save_then_verify_round_trip(tmp_path: Path) -> None:
    bundle, _ = model.train_model(demo.generate(seed=19, n_matches=90))
    meta = bundle.save(tmp_path / "model_v1.pkl", extra={"note": "frozen before forward test"})

    assert Path(meta["path"]).exists()
    ok, info = model.ModelBundle.verify(tmp_path / "model_v1.pkl")
    assert ok is True
    assert info["metadata"]["note"] == "frozen before forward test"

    reloaded = model.ModelBundle.load(tmp_path / "model_v1.pkl")
    assert reloaded.feature_columns == bundle.feature_columns


def test_verify_detects_a_changed_file(tmp_path: Path) -> None:
    bundle, _ = model.train_model(demo.generate(seed=23, n_matches=90))
    bundle.save(tmp_path / "model_v1.pkl")
    with open(tmp_path / "model_v1.pkl", "ab") as handle:
        handle.write(b"tampered")
    ok, _ = model.ModelBundle.verify(tmp_path / "model_v1.pkl")
    assert ok is False


def test_verify_reports_missing_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "nope.pkl"
    path.write_bytes(b"x")
    ok, info = model.ModelBundle.verify(path)
    assert ok is False
    assert "sidecar" in info["reason"]


# --------------------------------------------------------------------------- #
# a market whose own line is unknown cannot be selected
# --------------------------------------------------------------------------- #
def test_refuses_a_market_whose_line_is_missing() -> None:
    row = pd.Series(
        make_match(
            ah_open=None,
            ou_line_open=None,
            ah_fav_price=2.0,
            ah_dog_price=2.0,
            ou_over_price=2.0,
            ou_under_price=2.0,
        )
    )
    pick = model.select_market(row, 0.90, 0.90, threshold=0.01)
    assert pick.market is None
    assert "line is missing" in pick.reason


def test_still_selects_when_only_the_other_line_is_missing() -> None:
    """A missing OU line must not block a perfectly settleable AH bet."""
    row = pd.Series(
        make_match(
            ah_open=-0.5,
            ou_line_open=None,
            ah_fav_price=2.0,
            ah_dog_price=2.0,
            ou_over_price=2.0,
            ou_under_price=2.0,
        )
    )
    pick = model.select_market(row, 0.70, 0.50, threshold=0.01)
    assert pick.market == "AH_fav"


# --------------------------------------------------------------------------- #
# the frozen bundle cannot be loaded unverified
# --------------------------------------------------------------------------- #
def test_load_refuses_a_tampered_model(tmp_path: Path) -> None:
    bundle, _ = model.train_model(demo.generate(seed=29, n_matches=90))
    bundle.save(tmp_path / "model_v1.pkl")
    with open(tmp_path / "model_v1.pkl", "ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="refusing to load"):
        model.ModelBundle.load(tmp_path / "model_v1.pkl")


def test_load_can_still_be_forced_when_explicitly_asked(tmp_path: Path) -> None:
    bundle, _ = model.train_model(demo.generate(seed=31, n_matches=90))
    bundle.save(tmp_path / "model_v1.pkl")
    loaded = model.ModelBundle.load(tmp_path / "model_v1.pkl", require_verified=False)
    assert loaded.feature_columns == bundle.feature_columns


# --------------------------------------------------------------------------- #
# the rolling window travels with the model
# --------------------------------------------------------------------------- #
def test_rolling_window_is_recorded_and_used() -> None:
    frame = demo.generate(seed=37, n_matches=90)
    bundle, _ = model.train_model(frame, rolling_window=3)

    assert bundle.rolling_window == 3
    assert bundle.metadata["rolling_window"] == 3

    probabilities = bundle.predict_proba_df(frame)
    assert probabilities["p_fav_cover"].notna().all()
    assert probabilities["p_over"].notna().all()


def test_rolling_window_falls_back_to_the_default() -> None:
    frame = demo.generate(seed=41, n_matches=90)
    bundle, _ = model.train_model(frame)
    assert bundle.rolling_window == features.DEFAULT_ROLLING_WINDOW

    bundle.metadata["rolling_window"] = "not-a-number"
    assert bundle.rolling_window == features.DEFAULT_ROLLING_WINDOW


# --------------------------------------------------------------------------- #
# feature sets travel with the model
# --------------------------------------------------------------------------- #
def test_train_model_records_the_feature_set() -> None:
    frame = demo.generate(seed=43, n_matches=140)
    bundle, report = model.train_model(frame, feature_set="v2")

    assert bundle.feature_set == "v2"
    assert bundle.metadata["feature_set"] == "v2"
    assert report.feature_set == "v2"
    # the rule-set-2 block was actually consumed by the model
    assert "R1_ah-0.25_match" in bundle.feature_columns
    assert bundle.use_closing_features is False


def test_default_feature_set_is_still_v1() -> None:
    bundle, _ = model.train_model(demo.generate(seed=47, n_matches=90))
    assert bundle.feature_set == "v1"
    assert set(bundle.feature_columns) == set(features.FEATURE_COLUMNS)


def test_normal_mode_drops_the_closing_features_entirely() -> None:
    """Leakage guard: with closing features off they never reach the model.

    The demo frame carries closing 1X2 prices, so a build bug (silently filling
    them) would show up here as the flag surviving into feature_columns.
    """
    frame = demo.generate(seed=53, n_matches=140)
    bundle, report = model.train_model(
        frame, feature_set="v2", use_closing_features=False
    )

    assert "fav_no_drop" not in bundle.feature_columns
    assert "dog_no_drop" not in bundle.feature_columns
    assert "fav_no_drop" in report.dropped_empty_features
    assert "dog_no_drop" in report.dropped_empty_features


def test_late_mode_keeps_the_closing_features() -> None:
    frame = demo.generate(seed=59, n_matches=140)
    bundle, report = model.train_model(
        frame, feature_set="v2", use_closing_features=True
    )

    assert bundle.use_closing_features is True
    assert bundle.metadata["use_closing_features"] is True
    assert "fav_no_drop" in bundle.feature_columns
    assert "fav_no_drop" not in report.dropped_empty_features
