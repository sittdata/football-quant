"""The model: one multi-output classifier plus a market-selection rule.

Follows README section 5:

* two heads, ``p_fav_cover`` and ``p_over``, trained together,
* start with logistic regression, keep XGBoost as an optional upgrade,
* pick the market with the highest edge against the vig-free implied
  probability, and **skip the match** when nothing clears the threshold.

The bundle is saved with a SHA-256 and the git commit so a forward test can
prove the model was frozen before it started (README 7.B).
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import features
from .features import DEFAULT_ROLLING_WINDOW, FEATURE_COLUMNS
from .labels import fair_two_way

ModelKind = Literal["logreg", "xgboost"]

MARKETS = ("AH_fav", "AH_dog", "OU_over", "OU_under")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def ensure_feature_columns(
    df: pd.DataFrame, columns: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Guarantee every feature column exists (missing ones become NaN)."""
    out = df.copy()
    for column in (FEATURE_COLUMNS if columns is None else columns):
        if column not in out.columns:
            out[column] = np.nan
    return out


def prepare_frame(
    df: pd.DataFrame,
    *,
    need_labels: bool = False,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    feature_set: str = "v1",
    use_closing_features: bool = False,
) -> pd.DataFrame:
    """Return a frame that definitely has the feature set's columns (and labels).

    Callers may hand in a raw match table straight from the database; building
    the features here means a caller can never accidentally train or predict on
    an all-NaN feature matrix. The closing-price flags are only filled when the
    caller explicitly asks (late-prediction mode) - the normal mode leaves them
    NaN so the model cannot learn from a price that does not exist yet.
    """
    expected = features.feature_set_columns(feature_set)
    has_features = all(column in df.columns for column in expected)
    has_labels = "ah_label" in df.columns and "ou_label" in df.columns
    if not has_features or (need_labels and not has_labels):
        df = features.build_features(
            df,
            rolling_window=rolling_window,
            feature_set=feature_set,
            use_closing_features=use_closing_features,
        )
    return ensure_feature_columns(df, expected)


def _base_estimator(kind: ModelKind, seed: int):
    if kind == "logreg":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=4000,
                        C=1.0,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if kind == "xgboost":  # pragma: no cover - optional dependency
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError(
                "xgboost is not installed. Run: python -m pip install xgboost"
            ) from exc
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        random_state=seed,
                        eval_metric="logloss",
                        n_jobs=4,
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown model kind {kind!r}")


def _positive_probability(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """P(class == 1) even when one output saw only a single class."""
    proba = np.atleast_2d(proba)
    if proba.shape[1] == 1:
        only = int(classes[0])
        return np.full(proba.shape[0], 1.0 if only == 1 else 0.0)
    index = int(np.where(classes == 1)[0][0]) if (classes == 1).any() else 1
    return proba[:, index]


# --------------------------------------------------------------------------- #
# the bundle
# --------------------------------------------------------------------------- #
@dataclass
class ModelBundle:
    model: MultiOutputClassifier
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    kind: ModelKind = "logreg"
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def rolling_window(self) -> int:
        """The window this model was trained with.

        Recorded at fit time, so serving can never silently use a different
        window from training.
        """
        try:
            return int(self.metadata.get("rolling_window", DEFAULT_ROLLING_WINDOW))
        except (TypeError, ValueError):
            return DEFAULT_ROLLING_WINDOW

    @property
    def feature_set(self) -> str:
        """Which feature set this model was trained on (``v1`` / ``v2`` / ``v2_novig``)."""
        return str(self.metadata.get("feature_set", "v1"))

    @property
    def use_closing_features(self) -> bool:
        """Whether this model may read closing-price features (late mode only)."""
        return bool(self.metadata.get("use_closing_features", False))

    def predict_proba_df(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = prepare_frame(
            df,
            rolling_window=self.rolling_window,
            feature_set=self.feature_set,
            use_closing_features=self.use_closing_features,
        )
        matrix = prepared[self.feature_columns].to_numpy(dtype="float64")
        probabilities = self.model.predict_proba(matrix)
        classes = self.model.classes_

        out = pd.DataFrame(index=df.index)
        out["p_fav_cover"] = _positive_probability(probabilities[0], np.asarray(classes[0]))
        out["p_over"] = _positive_probability(probabilities[1], np.asarray(classes[1]))
        return out

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar = {
            "path": str(path),
            "sha256": digest,
            "kind": self.kind,
            "feature_columns": self.feature_columns,
            "metadata": {**self.metadata, **(extra or {})},
        }
        path.with_suffix(".meta.json").write_text(
            json.dumps(sidecar, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
        )
        return sidecar

    @staticmethod
    def load(path: str | Path, *, require_verified: bool = True) -> "ModelBundle":
        """Load a bundle, refusing to unpickle an unverified file.

        Unpickling executes code, so the SHA-256 sidecar must be checked
        *before* the bytes are deserialized; checking afterwards would give
        false assurance.
        """
        if require_verified:
            ok, meta = ModelBundle.verify(path)
            if not ok:
                raise ValueError(
                    f"refusing to load {path}: "
                    f"{meta.get('reason', 'hash does not match its sidecar')}"
                )
        with open(path, "rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def verify(path: str | Path) -> tuple[bool, dict[str, Any]]:
        """Check the pickled bundle still matches its recorded hash."""
        path = Path(path)
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            return False, {"reason": "no .meta.json sidecar found"}
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        return actual == meta.get("sha256"), {**meta, "actual_sha256": actual}


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
@dataclass
class TrainingReport:
    n_rows_in: int
    n_rows_used: int
    dropped_missing_label: int
    dropped_push: int
    class_balance: dict[str, float]
    feature_medians: dict[str, float]
    #: features dropped because the training window never observed a value
    dropped_empty_features: list[str] = field(default_factory=list)
    #: which feature set was trained (v1 / v2 / v2_novig)
    feature_set: str = "v1"


def train_model(
    df: pd.DataFrame,
    *,
    kind: ModelKind = "logreg",
    seed: int = 42,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    feature_set: str = "v1",
    use_closing_features: bool = False,
) -> tuple[ModelBundle, TrainingReport]:
    """Fit the two-headed model on rows that have both labels."""
    prepared = prepare_frame(
        df,
        need_labels=True,
        rolling_window=rolling_window,
        feature_set=feature_set,
        use_closing_features=use_closing_features,
    )

    ah_label = pd.to_numeric(prepared["ah_label"], errors="coerce")
    ou_label = pd.to_numeric(prepared["ou_label"], errors="coerce")
    usable = ah_label.notna() & ou_label.notna()

    dropped_missing = int((ah_label.isna() | ou_label.isna()).sum())
    pushes = int(
        (
            (ah_label.isna() & prepared["goal_diff_fav"].notna())
            | (ou_label.isna() & prepared["total_goals"].notna())
        ).sum()
    )

    if usable.sum() < 10:
        raise ValueError(
            f"only {int(usable.sum())} labelled rows available - not enough to train. "
            "Fetch more matches or check that results were joined onto the odds."
        )

    train = prepared[usable]

    # A feature with no observed value anywhere in the training window cannot
    # inform the model, and handing it to the imputer only produces a warning
    # per fit (40+ of them in a walk-forward run). draw_ht_odds is the usual
    # case: 5DollarFootballAPI exposes no half-time 1X2 market. Drop it and
    # record what was dropped, rather than hiding the symptom. This is also
    # what removes the closing-price flags in the normal (24h) mode: they are
    # NaN for every training row, so they can never leak into the model.
    set_columns = features.feature_set_columns(feature_set)
    active_features = [
        column
        for column in set_columns
        if pd.to_numeric(train[column], errors="coerce").notna().any()
    ]
    dropped_features = [c for c in set_columns if c not in active_features]
    if not active_features:
        raise ValueError("no feature has any observed value in the training window")

    x = train[active_features].to_numpy(dtype="float64")
    y = np.column_stack([ah_label[usable].to_numpy(), ou_label[usable].to_numpy()]).astype(int)

    model = MultiOutputClassifier(_base_estimator(kind, seed))
    model.fit(x, y)

    report = TrainingReport(
        n_rows_in=len(prepared),
        n_rows_used=int(usable.sum()),
        dropped_missing_label=dropped_missing,
        dropped_push=pushes,
        class_balance={
            "ah_cover_rate": float(ah_label[usable].mean()),
            "ou_over_rate": float(ou_label[usable].mean()),
        },
        feature_medians={
            column: float(pd.to_numeric(train[column], errors="coerce").median())
            for column in set_columns
            if pd.to_numeric(train[column], errors="coerce").notna().any()
        },
        dropped_empty_features=dropped_features,
        feature_set=feature_set,
    )
    bundle = ModelBundle(
        model=model,
        feature_columns=list(active_features),
        kind=kind,
        metadata={
            "trained_rows": int(usable.sum()),
            "trained_from": str(train["kickoff_at"].min()) if "kickoff_at" in train else None,
            "trained_to": str(train["kickoff_at"].max()) if "kickoff_at" in train else None,
            "class_balance": report.class_balance,
            "rolling_window": int(rolling_window),
            "feature_set": feature_set,
            "use_closing_features": bool(use_closing_features),
            "dropped_empty_features": dropped_features,
        },
    )
    return bundle, report


# --------------------------------------------------------------------------- #
# market selection
# --------------------------------------------------------------------------- #
@dataclass
class Pick:
    market: str | None
    side_odds: float | None
    p_model: float | None
    p_implied: float | None
    edge: float | None
    all_edges: dict[str, float] = field(default_factory=dict)
    reason: str = ""


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def select_market(
    row: pd.Series,
    p_fav_cover: float,
    p_over: float,
    *,
    threshold: float = 0.03,
    blacklist_markets: Any = None,
    allowed_markets: Any = None,
) -> Pick:
    """Highest-edge market for one match, or an explicit skip.

    Supports blacklisting specific markets (e.g. ['OU_over']) or restricting
    selection to an explicit allowlist (e.g. ['AH_dog', 'OU_under']).
    """
    ah_open = _number(row.get("ah_open"))
    ou_line = _number(row.get("ou_line_open"))
    ah_fav = _number(row.get("ah_fav_price"))
    ah_dog = _number(row.get("ah_dog_price"))
    ou_over = _number(row.get("ou_over_price"))
    ou_under = _number(row.get("ou_under_price"))

    edges: dict[str, float] = {}
    odds_by_market: dict[str, float | None] = {
        "AH_fav": ah_fav,
        "AH_dog": ah_dog,
        "OU_over": ou_over,
        "OU_under": ou_under,
    }
    implied: dict[str, float] = {}

    # A market is only selectable when its own line is known: without the
    # handicap/total the bet cannot be settled at all.
    if ah_open is not None and ah_fav and ah_dog and ah_fav > 1.0 and ah_dog > 1.0:
        fair_fav, fair_dog = fair_two_way(ah_fav, ah_dog)
        edges["AH_fav"] = p_fav_cover - fair_fav
        edges["AH_dog"] = (1.0 - p_fav_cover) - fair_dog
        implied["AH_fav"] = fair_fav
        implied["AH_dog"] = fair_dog

    if ou_line is not None and ou_over and ou_under and ou_over > 1.0 and ou_under > 1.0:
        fair_over, fair_under = fair_two_way(ou_over, ou_under)
        edges["OU_over"] = p_over - fair_over
        edges["OU_under"] = (1.0 - p_over) - fair_under
        implied["OU_over"] = fair_over
        implied["OU_under"] = fair_under

    if not edges:
        return Pick(
            None,
            None,
            None,
            None,
            None,
            {},
            "no usable prices on both sides, or the handicap/total line is missing",
        )

    blacklist_set = {str(m).strip() for m in blacklist_markets} if blacklist_markets else set()
    allowed_set = {str(m).strip() for m in allowed_markets} if allowed_markets else set(odds_by_market.keys())

    candidate_edges = {
        m: e for m, e in edges.items()
        if m in allowed_set and m not in blacklist_set
    }

    if not candidate_edges:
        return Pick(
            None,
            None,
            None,
            None,
            None,
            edges,
            "all candidate markets were excluded by blacklist/allowlist",
        )

    best = max(candidate_edges, key=candidate_edges.get)
    best_edge = float(candidate_edges[best])
    if best_edge < threshold:
        return Pick(
            None,
            None,
            None,
            None,
            best_edge,
            edges,
            f"best edge {best_edge:.4f} below threshold {threshold:.4f}",
        )

    p_model = p_fav_cover if best.startswith("AH") else p_over
    if best.endswith("_dog") or best.endswith("_under"):
        p_model = 1.0 - p_model

    return Pick(
        market=best,
        side_odds=odds_by_market[best],
        p_model=float(p_model),
        p_implied=float(implied.get(best, np.nan)),
        edge=best_edge,
        all_edges=edges,
        reason="selected",
    )
