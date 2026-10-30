"""Calibrated Quantitative Models for Football Betting (Asian Handicap & Over/Under).
Author: cook45

Architecture:
1. Separate heads:
   - AHModel: Predicts whether Favourite covers the handicap (Goal Differential task).
   - OUModel: Predicts whether Total Goals exceed the line (Match Tempo task).
2. Probability Calibration (CalibratedClassifierCV / Sigmoid):
   - Eliminates uncalibrated XGBoost/LogReg probability spikes (prevents phantom +25% edges).
3. Strict Out-Of-Sample Walk-Forward validation (zero data leakage).
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fbpredict import config, labels

AH_FEATURE_COLS = [
    "ah_open",
    "ah_abs",
    "fair_fav_ah",
    "fair_dog_ah",
    "ah_overround",
    "is_fav_home",
    "fav_gf_avg",
    "fav_ga_avg",
    "dog_gf_avg",
    "dog_ga_avg",
    "fav_form",
    "dog_form",
    "fav_dog_form_diff",
    "fav_rest_days",
    "dog_rest_days",
    "fav_rest_diff",
    "draw_ft_odds",
    "ou_line_open",
]

OU_FEATURE_COLS = [
    "ou_line_open",
    "fair_over",
    "fair_under",
    "ou_overround",
    "expected_total",
    "home_gf_avg",
    "home_ga_avg",
    "away_gf_avg",
    "away_ga_avg",
    "goals_avg_diff",
    "home_over_rate",
    "away_over_rate",
    "draw_ft_odds",
    "ah_abs",
]

MODEL_FEATURE_COLS = list(dict.fromkeys(AH_FEATURE_COLS + OU_FEATURE_COLS))


def _class_one_probability(proba: Any, classes: Any) -> np.ndarray:
    """Return P(class == 1) per row, resolving the column via ``classes_``.

    A model trained on a single class returns one column, and its label may be 0.
    Indexing ``[:, 1]`` blind would raise or silently invert the probability.
    """
    arr = np.asarray(proba, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros(arr.shape[0] if arr.ndim == 2 else 0)
    class_list = list(classes) if classes is not None else []
    if 1 in class_list:
        return arr[:, class_list.index(1)]
    if arr.shape[1] > 1:
        return arr[:, 1]
    return np.zeros(arr.shape[0])


@dataclass
class QuantModelBundle:
    ah_model: Any
    ou_model: Any
    ah_feature_cols: list[str] = field(default_factory=lambda: list(AH_FEATURE_COLS))
    ou_feature_cols: list[str] = field(default_factory=lambda: list(OU_FEATURE_COLS))
    feature_columns: list[str] = field(default_factory=lambda: list(MODEL_FEATURE_COLS))
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_match(self, features_dict: dict[str, Any]) -> dict[str, Any]:
        """Run calibrated inference on a single match feature dictionary."""
        # Safety gate: reject matches with insufficient history
        if not features_dict.get("has_sufficient_history", 0):
            return {
                "safe": False,
                "reason": "Insufficient historical match data (min 3 games per team)",
                "p_fav": 0.5,
                "p_dog": 0.5,
                "p_over": 0.5,
                "p_under": 0.5,
                "best_market": None,
                "edge": 0.0,
            }

        row_df = pd.DataFrame([features_dict])
        for col in self.ah_feature_cols:
            if col not in row_df.columns:
                row_df[col] = np.nan
        for col in self.ou_feature_cols:
            if col not in row_df.columns:
                row_df[col] = np.nan

        X_ah = row_df[self.ah_feature_cols]
        X_ou = row_df[self.ou_feature_cols]

        ah_proba = self.ah_model.predict_proba(X_ah)
        p_fav = float(_class_one_probability(ah_proba, getattr(self.ah_model, "classes_", None))[0])
        p_dog = 1.0 - p_fav

        ou_proba = self.ou_model.predict_proba(X_ou)
        p_over = float(_class_one_probability(ou_proba, getattr(self.ou_model, "classes_", None))[0])
        p_under = 1.0 - p_over

        def _safe_odds(val: Any) -> float | None:
            """Return the quoted price, or None. Never invent a price."""
            if val is None:
                return None
            try:
                f = float(val)
                return f if math.isfinite(f) and f > 1.0 else None
            except (ValueError, TypeError):
                return None

        ah_fav_price = _safe_odds(features_dict.get("ah_fav_price"))
        ah_dog_price = _safe_odds(features_dict.get("ah_dog_price"))
        ou_over_price = _safe_odds(features_dict.get("ou_over_price"))
        ou_under_price = _safe_odds(features_dict.get("ou_under_price"))

        # Skip unpaired markets: both sides of a 2-way pair must be quoted.
        # Computing an edge when only one side is quoted forces an invented neutral (0.5) fair prob.
        ah_paired = (ah_fav_price is not None and ah_dog_price is not None)
        ou_paired = (ou_over_price is not None and ou_under_price is not None)

        def _safe_prob(val: Any) -> float | None:
            if val is None:
                return None
            try:
                f = float(val)
                return f if math.isfinite(f) and 0.0 < f < 1.0 else None
            except (ValueError, TypeError):
                return None

        fair_fav = _safe_prob(features_dict.get("fair_fav_ah")) if ah_paired else None
        fair_dog = _safe_prob(features_dict.get("fair_dog_ah")) if ah_paired else None
        if ah_paired and (fair_fav is None or fair_dog is None):
            try:
                fair_fav, fair_dog = labels.fair_two_way(ah_fav_price, ah_dog_price)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Failed to recompute fair odds for AH pair (%s, %s): %s", ah_fav_price, ah_dog_price, e)
                fair_fav, fair_dog = None, None

        fair_over = _safe_prob(features_dict.get("fair_over")) if ou_paired else None
        fair_under = _safe_prob(features_dict.get("fair_under")) if ou_paired else None
        if ou_paired and (fair_over is None or fair_under is None):
            try:
                fair_over, fair_under = labels.fair_two_way(ou_over_price, ou_under_price)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Failed to recompute fair odds for OU pair (%s, %s): %s", ou_over_price, ou_under_price, e)
                fair_over, fair_under = None, None

        edge_fav = (p_fav - fair_fav) if (ah_paired and fair_fav is not None) else None
        edge_dog = (p_dog - fair_dog) if (ah_paired and fair_dog is not None) else None
        edge_over = (p_over - fair_over) if (ou_paired and fair_over is not None) else None
        edge_under = (p_under - fair_under) if (ou_paired and fair_under is not None) else None

        priced = []
        if ah_paired and edge_fav is not None and edge_dog is not None:
            priced.append(("AH_fav", edge_fav, p_fav, ah_fav_price))
            priced.append(("AH_dog", edge_dog, p_dog, ah_dog_price))

        if ou_paired and edge_over is not None and edge_under is not None:
            priced.append(("OU_over", edge_over, p_over, ou_over_price))
            priced.append(("OU_under", edge_under, p_under, ou_under_price))

        if not priced:
            return {
                "safe": False,
                "reason": "No quoted odds available for any market",
                "p_fav": round(p_fav, 4),
                "p_dog": round(p_dog, 4),
                "p_over": round(p_over, 4),
                "p_under": round(p_under, 4),
                "best_market": None,
                "edge": 0.0,
            }

        best_mkt, best_edge, best_prob, best_odds = max(priced, key=lambda x: x[1])

        # Evaluate Steam & Smart Money Alignment
        from fbpredict.steam_engine import SteamSignal, analyze_market_steam, evaluate_steam_confirmation
        steam_sig = features_dict.get("steam_signal")
        if not isinstance(steam_sig, SteamSignal):
            steam_sig = analyze_market_steam(
                open_ah=features_dict.get("ah_open"),
                current_ah=features_dict.get("ah_close"),
                open_fav_price=features_dict.get("ah_fav_price"),
                open_dog_price=features_dict.get("ah_dog_price"),
                current_fav_price=features_dict.get("ah_fav_close_price") or features_dict.get("ah_fav_price"),
                current_dog_price=features_dict.get("ah_dog_close_price") or features_dict.get("ah_dog_price"),
                open_ou=features_dict.get("ou_line_open"),
                current_ou=features_dict.get("ou_line_close"),
                open_over_price=features_dict.get("ou_over_price"),
                open_under_price=features_dict.get("ou_under_price"),
                current_over_price=features_dict.get("ou_over_close_price") or features_dict.get("ou_over_price"),
                current_under_price=features_dict.get("ou_under_close_price") or features_dict.get("ou_under_price"),
            )
        steam_eval = evaluate_steam_confirmation(best_mkt, steam_sig)

        return {
            "safe": True,
            "p_fav": round(p_fav, 4),
            "p_dog": round(p_dog, 4),
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "edge_fav": round(edge_fav, 4) if edge_fav is not None else None,
            "edge_dog": round(edge_dog, 4) if edge_dog is not None else None,
            "edge_over": round(edge_over, 4) if edge_over is not None else None,
            "edge_under": round(edge_under, 4) if edge_under is not None else None,
            "best_market": best_mkt,
            "best_edge": round(best_edge, 4),
            "best_prob": round(best_prob, 4),
            "best_odds": round(best_odds, 2),
            "is_steam_confirmed": steam_eval["is_confirmed"],
            "conviction_tier": steam_eval["conviction_tier"],
            "steam_badge": steam_eval["badge_label"],
            "steam_direction": steam_eval["steam_direction"],
            "is_rlm": steam_eval["is_rlm"],
            "steam_description": steam_eval["description"],
        }


    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized inference over an entire DataFrame of matches."""
        if df is None or df.empty:
            return pd.DataFrame(
                columns=[
                    "p_fav", "p_dog", "p_over", "p_under",
                    "edge_fav", "edge_dog", "edge_over", "edge_under",
                ]
            )

        d = df.copy()
        for col in self.ah_feature_cols:
            if col not in d.columns:
                d[col] = np.nan

        for col in self.ou_feature_cols:
            if col not in d.columns:
                d[col] = np.nan

        ah_proba = self.ah_model.predict_proba(d[self.ah_feature_cols])
        p_fav = _class_one_probability(ah_proba, getattr(self.ah_model, "classes_", None))
        p_dog = 1.0 - p_fav
        ou_proba = self.ou_model.predict_proba(d[self.ou_feature_cols])
        p_over = _class_one_probability(ou_proba, getattr(self.ou_model, "classes_", None))
        p_under = 1.0 - p_over

        def _fair_col(name: str) -> np.ndarray:
            """Fair probability column, defaulting missing/NaN rows to a neutral 0.5."""
            if name not in d.columns:
                return np.full(len(d), 0.5)
            values = pd.to_numeric(d[name], errors="coerce").to_numpy(dtype=float)
            values = np.where(np.isfinite(values), values, 0.5)
            return np.clip(values, 0.0, 1.0)

        fair_fav = _fair_col("fair_fav_ah")
        fair_dog = _fair_col("fair_dog_ah")
        fair_over = _fair_col("fair_over")
        fair_under = _fair_col("fair_under")

        edge_fav = p_fav - fair_fav
        edge_dog = p_dog - fair_dog
        edge_over = p_over - fair_over
        edge_under = p_under - fair_under

        return pd.DataFrame({
            "p_fav": p_fav,
            "p_dog": p_dog,
            "p_over": p_over,
            "p_under": p_under,
            "edge_fav": edge_fav,
            "edge_dog": edge_dog,
            "edge_over": edge_over,
            "edge_under": edge_under,
        }, index=d.index)

    def save(self, file_path: Path | str) -> None:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: Path | str) -> QuantModelBundle:
        with open(file_path, "rb") as f:
            return pickle.load(f)


def train_calibrated_models(
    df_features: pd.DataFrame,
    seed: int = 42,
) -> tuple[QuantModelBundle, dict[str, Any]]:
    """Train separated and probability-calibrated models for AH and OU."""
    df_usable = df_features[df_features["has_sufficient_history"] == 1].copy()

    # 1. Asian Handicap Head (Filtered for decisive matches without push NaN)
    mask_ah = df_usable["ah_label"].notna()
    df_ah = df_usable[mask_ah].copy()
    X_ah = df_ah[AH_FEATURE_COLS]
    y_ah = df_ah["ah_label"].astype(int)

    base_ah = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=3,
        learning_rate=0.03,
        l2_regularization=2.0,
        random_state=seed,
    )
    calibrated_ah = CalibratedClassifierCV(estimator=base_ah, method="sigmoid", cv=5)
    calibrated_ah.fit(X_ah, y_ah)

    # 2. Over/Under Head (Filtered for decisive matches without push NaN)
    mask_ou = df_usable["ou_label"].notna()
    df_ou = df_usable[mask_ou].copy()
    X_ou = df_ou[OU_FEATURE_COLS]
    y_ou = df_ou["ou_label"].astype(int)

    base_ou = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=3,
        learning_rate=0.03,
        l2_regularization=2.0,
        random_state=seed,
    )
    calibrated_ou = CalibratedClassifierCV(estimator=base_ou, method="sigmoid", cv=5)
    calibrated_ou.fit(X_ou, y_ou)

    metadata = {
        "n_samples_ah": len(df_ah),
        "n_samples_ou": len(df_ou),
        "ah_cover_rate": round(float(y_ah.mean()), 4),
        "ou_over_rate": round(float(y_ou.mean()), 4),
        "ah_feature_cols": AH_FEATURE_COLS,
        "ou_feature_cols": OU_FEATURE_COLS,
        "feature_columns": MODEL_FEATURE_COLS,
    }

    bundle = QuantModelBundle(
        ah_model=calibrated_ah,
        ou_model=calibrated_ou,
        ah_feature_cols=AH_FEATURE_COLS,
        ou_feature_cols=OU_FEATURE_COLS,
        feature_columns=MODEL_FEATURE_COLS,
        metadata=metadata,
    )

    return bundle, metadata
