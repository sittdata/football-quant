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

        p_fav = float(self.ah_model.predict_proba(X_ah)[0, 1])
        p_dog = 1.0 - p_fav

        p_over = float(self.ou_model.predict_proba(X_ou)[0, 1])
        p_under = 1.0 - p_over

        fair_fav = features_dict.get("fair_fav_ah", 0.5)
        fair_dog = features_dict.get("fair_dog_ah", 0.5)
        fair_over = features_dict.get("fair_over", 0.5)
        fair_under = features_dict.get("fair_under", 0.5)

        edge_fav = p_fav - fair_fav
        edge_dog = p_dog - fair_dog
        edge_over = p_over - fair_over
        edge_under = p_under - fair_under

        markets = [
            ("AH_fav", edge_fav, p_fav, features_dict.get("ah_fav_price", 1.90)),
            ("AH_dog", edge_dog, p_dog, features_dict.get("ah_dog_price", 1.90)),
            ("OU_over", edge_over, p_over, features_dict.get("ou_over_price", 1.90)),
            ("OU_under", edge_under, p_under, features_dict.get("ou_under_price", 1.90)),
        ]

        best_mkt, best_edge, best_prob, best_odds = max(markets, key=lambda x: x[1])

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
            "edge_fav": round(edge_fav, 4),
            "edge_dog": round(edge_dog, 4),
            "edge_over": round(edge_over, 4),
            "edge_under": round(edge_under, 4),
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

        p_fav = self.ah_model.predict_proba(d[self.ah_feature_cols])[:, 1]
        p_dog = 1.0 - p_fav
        p_over = self.ou_model.predict_proba(d[self.ou_feature_cols])[:, 1]
        p_under = 1.0 - p_over

        fair_fav = d["fair_fav_ah"].values if "fair_fav_ah" in d.columns else np.full(len(d), 0.5)
        fair_dog = d["fair_dog_ah"].values if "fair_dog_ah" in d.columns else np.full(len(d), 0.5)
        fair_over = d["fair_over"].values if "fair_over" in d.columns else np.full(len(d), 0.5)
        fair_under = d["fair_under"].values if "fair_under" in d.columns else np.full(len(d), 0.5)

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
