"""Backtest: walk-forward, time-ordered, no shuffling.

Two things the brief is emphatic about, and both are enforced here:

* the split is 80/20 **by time, oldest to newest, never shuffled**
  (README section 7.A),
* ``blind_view`` hides the result of the day being predicted, and the model is
  only ever fitted on matches strictly before the one it is predicting
  (README section 7.A + section 8.1).

``predicted_at`` is set to a fixed offset *before* kick-off so the row also
satisfies the database's look-ahead constraint.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from . import features, labels, model
from .config import Config
from .features import DEFAULT_ROLLING_WINDOW
from .model import ModelKind, Pick

RESULT_COLUMNS = ("home_goals", "away_goals", "ht_home", "ht_away")


@dataclass
class BacktestResult:
    run_id: str
    predictions: pd.DataFrame
    train_size: int
    test_size: int
    n_models_fitted: int
    skipped: int
    taken: int
    predicted_before_hours: float
    notes: list[str] = field(default_factory=list)
    #: How much of each feature the test window actually carried. Measured on
    #: the feature frame, not on the predictions frame (which only holds a
    #: handful of columns and would report every feature as absent).
    feature_coverage: pd.DataFrame | None = None
    #: Which feature set the run trained on (v1 / v2 / v2_novig).
    feature_set: str = "v1"
    #: The explicit prediction clock in minutes, when the caller gave one.
    predicted_before_minutes: float | None = None
    #: Features actually used by the last fitted model, and the ones dropped
    #: because the training window never observed a value.
    active_features: list[str] = field(default_factory=list)
    dropped_features: list[str] = field(default_factory=list)
    #: The built feature frame the run used (all rows, in-sample). The report
    #: reads it for the per-rule performance, overround and odds-format
    #: sections; it is deliberately not serialised into the metrics JSON.
    data_frame: pd.DataFrame | None = None


def closing_feature_guard(
    feature_set: str, use_closing_features: bool, lead_minutes: float
) -> None:
    """Refuse (never merely warn) a run that would use closing prices too early.

    The closing price does not exist until near kick-off. A run that asks for
    the closing-dependent features while predicting earlier than
    :data:`fbpredict.features.CLOSING_PRICE_LEAD_MINUTES` would be using a
    price from the future, so it is refused before any work happens.
    """
    if not use_closing_features:
        return
    if feature_set == "v1":
        raise ValueError(
            f"refusing: feature set {feature_set!r} contains no closing-price "
            "features, so use_closing_features cannot be honoured"
        )
    if lead_minutes > features.CLOSING_PRICE_LEAD_MINUTES:
        raise ValueError(
            f"refusing: closing-price features requested but the prediction lead is "
            f"{lead_minutes:.0f} min, earlier than the closing-price window "
            f"({features.CLOSING_PRICE_LEAD_MINUTES} min before kick-off). Predict "
            "inside the window (for example --predicted-before-minutes 5) or drop "
            "use_closing_features."
        )


# --------------------------------------------------------------------------- #
# the brief's own helper, kept verbatim in spirit
# --------------------------------------------------------------------------- #
def blind_view(matches: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into ``(known, todo)`` where ``todo`` has its results erased.

    ``known``  - every match strictly before ``as_of`` (results visible)
    ``todo``   - matches on ``as_of``'s day, results removed
    """
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")

    frame = matches.copy()
    if not pd.api.types.is_datetime64_any_dtype(frame["kickoff_at"]):
        frame["kickoff_at"] = pd.to_datetime(frame["kickoff_at"], utc=True, format="ISO8601")

    known = frame[frame["kickoff_at"] < as_of].copy()
    day_end = as_of + pd.Timedelta(days=1)
    todo = frame[(frame["kickoff_at"] >= as_of) & (frame["kickoff_at"] < day_end)].copy()
    for column in RESULT_COLUMNS:
        if column in todo.columns:
            todo[column] = None
    return known, todo


# --------------------------------------------------------------------------- #
# time-ordered split
# --------------------------------------------------------------------------- #
def time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Oldest ``train_ratio`` for training, the rest for testing. Never shuffled."""
    ordered = df.sort_values("kickoff_at", kind="mergesort").reset_index(drop=True)
    cut = int(len(ordered) * train_ratio)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #
def walk_forward(
    df: pd.DataFrame,
    cfg: Config,
    *,
    kind: ModelKind = "logreg",
    threshold: float | None = None,
    refit_every: int | None = None,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    predicted_before_hours: float = 24.0,
    predicted_before_minutes: float | None = None,
    feature_set: str = "v1",
    use_closing_features: bool = False,
    blacklist_markets: Sequence[str] | None = None,
    allowed_markets: Sequence[str] | None = None,
    run_id: str | None = None,
) -> BacktestResult:
    """Predict every match in the test window using only earlier matches.

    ``predicted_before_minutes`` is the explicit prediction clock the spec
    requires whenever closing prices are in play; it overrides
    ``predicted_before_hours``. Asking for the closing-price features outside
    the closing window is refused (see :func:`closing_feature_guard`).
    """
    threshold = cfg.edge_threshold if threshold is None else threshold
    refit_every = max(1, cfg.refit_every if refit_every is None else refit_every)
    active_blacklist = cfg.blacklist_markets if blacklist_markets is None else tuple(blacklist_markets)
    active_allowed = cfg.allowed_markets if allowed_markets is None else tuple(allowed_markets)
    run_id = run_id or f"bt-{uuid.uuid4().hex[:10]}"

    if predicted_before_minutes is not None:
        if predicted_before_minutes < 0:
            raise ValueError("predicted_before_minutes must not be negative")
        predicted_before_hours = predicted_before_minutes / 60.0

    closing_feature_guard(
        feature_set, use_closing_features, predicted_before_hours * 60.0
    )
    features.feature_set_columns(feature_set)  # refuse an unknown set early

    if df.empty:
        raise ValueError("no matches to backtest")

    data = features.build_features(
        df,
        rolling_window=rolling_window,
        feature_set=feature_set,
        use_closing_features=use_closing_features,
    )
    data = data.sort_values("kickoff_at", kind="mergesort").reset_index(drop=True)

    n_total = len(data)
    cut = int(n_total * cfg.train_ratio)
    notes: list[str] = []

    if n_total - cut < 1:
        raise ValueError(
            f"only {n_total} matches: a {cfg.train_ratio:.0%} split leaves no test rows"
        )
    if cut < cfg.min_train:
        notes.append(
            f"training window has {cut} matches but FB_MIN_TRAIN is {cfg.min_train}; "
            "predictions will start late or not at all"
        )

    records: list[dict] = []
    bundle: model.ModelBundle | None = None
    models_fitted = 0
    skipped = 0
    taken = 0

    for index in range(cut, n_total):
        history = data.iloc[:index]
        row = data.iloc[index]
        record = _base_record(row, predicted_before_hours)

        if len(history) < cfg.min_train:
            record["skip_reason"] = "not enough training history"
            records.append(record)
            skipped += 1
            continue

        if bundle is None or (index - cut) % refit_every == 0:
            try:
                bundle, _ = model.train_model(
                    history,
                    kind=kind,
                    seed=cfg.seed,
                    rolling_window=rolling_window,
                    feature_set=feature_set,
                    use_closing_features=use_closing_features,
                )
                models_fitted += 1
            except ValueError as exc:
                record["skip_reason"] = f"training failed: {exc}"
                records.append(record)
                skipped += 1
                continue

        probabilities = bundle.predict_proba_df(data.iloc[[index]])
        p_fav_cover = float(probabilities["p_fav_cover"].iloc[0])
        p_over = float(probabilities["p_over"].iloc[0])
        record["p_fav_cover"] = p_fav_cover
        record["p_over"] = p_over

        pick = model.select_market(
            row,
            p_fav_cover,
            p_over,
            threshold=threshold,
            blacklist_markets=active_blacklist,
            allowed_markets=active_allowed,
        )
        record["edge"] = pick.edge

        if pick.market is None:
            record["skip_reason"] = pick.reason
            records.append(record)
            skipped += 1
            continue

        unit, profit = _settle(row, pick, cfg.stake)
        record.update(
            {
                "market": pick.market,
                "p_model": pick.p_model,
                "p_implied": pick.p_implied,
                "odds_used": pick.side_odds,
                "stake": cfg.stake,
                "outcome_unit": unit,
                "profit": profit,
                "skip_reason": "",
            }
        )
        records.append(record)
        taken += 1

    predictions = pd.DataFrame(records)
    if not predictions.empty:
        predictions["kickoff_at"] = pd.to_datetime(predictions["kickoff_at"], utc=True)
        predictions["predicted_at"] = pd.to_datetime(predictions["predicted_at"], utc=True)

    notes.append(
        f"features were built once over the whole frame; every rolling statistic is "
        f"shift(1) so a match never sees its own result"
    )
    notes.append(
        f"predicted_at is fixed at kickoff - {predicted_before_hours:g}h to satisfy the "
        f"database look-ahead constraint"
    )
    if feature_set != "v1":
        notes.append(
            f"feature set {feature_set}: the rule features are policy inputs from the "
            f"owner's own rule work (same person, same period), so this run can compare "
            f"their usefulness but cannot establish it"
        )
    if use_closing_features:
        notes.append(
            f"closing-price features were filled (late mode: predictions sit inside "
            f"the {features.CLOSING_PRICE_LEAD_MINUTES}-minute closing window)"
        )

    active_features: list[str] = list(bundle.feature_columns) if bundle is not None else []
    dropped_features: list[str] = []
    if bundle is not None:
        dropped_features = [str(name) for name in bundle.metadata.get("dropped_empty_features", [])]

    return BacktestResult(
        run_id=run_id,
        predictions=predictions,
        train_size=cut,
        test_size=n_total - cut,
        n_models_fitted=models_fitted,
        skipped=skipped,
        taken=taken,
        predicted_before_hours=predicted_before_hours,
        notes=notes,
        feature_coverage=features.missing_feature_report(
            data.iloc[cut:], features.feature_set_columns(feature_set)
        ),
        feature_set=feature_set,
        predicted_before_minutes=predicted_before_minutes,
        active_features=active_features,
        dropped_features=dropped_features,
        data_frame=data,
    )


def _base_record(row: pd.Series, predicted_before_hours: float) -> dict:
    """One row per test match, carrying the model probabilities for BOTH heads.

    Recording ``p_fav_cover`` / ``p_over`` even for skipped matches is what lets
    the report score the model honestly instead of only scoring the matches it
    chose to bet on.
    """
    return {
        "match_id": row.get("match_id"),
        "kickoff_at": row.get("kickoff_at"),
        "predicted_at": row.get("kickoff_at") - timedelta(hours=predicted_before_hours),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "fav_side": row.get("fav_side"),
        "ah_open": row.get("ah_open"),
        "ou_line_open": row.get("ou_line_open"),
        "goal_diff_fav": row.get("goal_diff_fav"),
        "total_goals": row.get("total_goals"),
        "ah_label": row.get("ah_label"),
        "ou_label": row.get("ou_label"),
        "fair_fav_ah": row.get("fair_fav_ah"),
        "fair_over": row.get("fair_over"),
        "p_fav_cover": np.nan,
        "p_over": np.nan,
        "market": None,
        "p_model": np.nan,
        "p_implied": np.nan,
        "edge": np.nan,
        "odds_used": np.nan,
        "stake": 0.0,
        "outcome_unit": np.nan,
        "profit": 0.0,
        "skip_reason": "",
        "is_synthetic": int(row.get("is_synthetic") or 0),
    }


def _settle(row: pd.Series, pick: Pick, stake: float) -> tuple[float, float]:
    return labels.settle_pick(
        market=pick.market,  # type: ignore[arg-type]
        goal_diff_fav=float(row["goal_diff_fav"]),
        total_goals=float(row["total_goals"]),
        ah_open=float(row["ah_open"]),
        ou_line_open=float(row["ou_line_open"]),
        ah_fav_price=row.get("ah_fav_price"),
        ah_dog_price=row.get("ah_dog_price"),
        ou_over_price=row.get("ou_over_price"),
        ou_under_price=row.get("ou_under_price"),
        stake=stake,
    )


# --------------------------------------------------------------------------- #
# plan B: forward predictions against a frozen model
# --------------------------------------------------------------------------- #
def forward_predict(
    upcoming: pd.DataFrame,
    bundle: model.ModelBundle,
    cfg: Config,
    *,
    threshold: float | None = None,
    blacklist_markets: Sequence[str] | None = None,
    allowed_markets: Sequence[str] | None = None,
    predicted_at: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Predict matches that have no result yet, for the live forward test.

    Rows are *not* written to the database here; the caller does that through
    :func:`fbpredict.db.insert_prediction`, which enforces ``predicted_at <
    kickoff_at``.
    """
    threshold = cfg.edge_threshold if threshold is None else threshold
    active_blacklist = cfg.blacklist_markets if blacklist_markets is None else tuple(blacklist_markets)
    active_allowed = cfg.allowed_markets if allowed_markets is None else tuple(allowed_markets)
    if upcoming.empty:
        return pd.DataFrame()

    # Use the feature set and window the model was trained with, so serving
    # cannot silently differ from training.
    prepared = features.build_features(
        upcoming,
        rolling_window=bundle.rolling_window,
        feature_set=bundle.feature_set,
        use_closing_features=bundle.use_closing_features,
    )
    probabilities = bundle.predict_proba_df(prepared)

    stamp = pd.Timestamp(predicted_at) if predicted_at is not None else pd.Timestamp.utcnow()
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")

    # The same closing-window guard as the backtest. A frozen late-mode model
    # may only be asked about matches whose closing price already exists by
    # the prediction time; anything earlier would be look-ahead.
    if bundle.use_closing_features:
        kickoffs = pd.to_datetime(prepared["kickoff_at"], utc=True)
        earliest = kickoffs - pd.Timedelta(minutes=features.CLOSING_PRICE_LEAD_MINUTES)
        too_early = prepared.loc[stamp < earliest, "match_id"]
        if len(too_early):
            raise ValueError(
                "refusing: this model reads closing-price features, but the prediction "
                f"timestamp {stamp} is earlier than the closing-price window for "
                f"{len(too_early)} of {len(prepared)} match(es) "
                f"(first: {too_early.iloc[0]})"
            )

    rows: list[dict] = []
    for position, (_, row) in enumerate(prepared.iterrows()):
        pick = model.select_market(
            row,
            float(probabilities["p_fav_cover"].iloc[position]),
            float(probabilities["p_over"].iloc[position]),
            threshold=threshold,
            blacklist_markets=active_blacklist,
            allowed_markets=active_allowed,
        )
        rows.append(
            {
                "match_id": row["match_id"],
                "predicted_at": stamp,
                "kickoff_at": row["kickoff_at"],
                "market": pick.market,
                "p_model": pick.p_model,
                "p_implied": pick.p_implied,
                "edge": pick.edge,
                "odds_used": pick.side_odds,
                "stake": cfg.stake if pick.market else 0.0,
                "skip_reason": pick.reason,
            }
        )
    return pd.DataFrame(rows)
