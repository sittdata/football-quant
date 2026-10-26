"""Train and benchmark multileague_v3 using Understat xG, xGA & PPDA feature enrichment.

Compares:
- Baseline (v2): XGBoost on Thscore features (corners, HT dynamics)
- Upgraded (v3): XGBoost with Understat xG, xGA, npxGD, PPDA clash, and Luck metrics
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

# Fix console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fbpredict import config, db, features, model
from fbpredict.db import connect
from fbpredict.model import ModelBundle, train_model


def load_dataset() -> pd.DataFrame:
    con = connect(ROOT_DIR / "data" / "football.db")
    df = pd.read_sql_query(
        """
        SELECT * FROM matches
        WHERE kickoff_at IS NOT NULL
          AND ah_open IS NOT NULL
          AND ou_line_open IS NOT NULL
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
        ORDER BY kickoff_at ASC
        """,
        con,
    )
    con.close()
    print(f"[Dataset] Loaded {len(df)} labelled matches with opening market odds.")
    return df


def benchmark_and_train():
    cfg = config.Config.from_env()
    df = load_dataset()

    # Time-based Train / Test Split (Strictly Walk-forward: 80% train, 20% test)
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx].reset_index(drop=True).copy()
    test_df = df.iloc[split_idx:].reset_index(drop=True).copy()

    print(f"[Benchmark] Time-split: {len(train_df)} train matches, {len(test_df)} test matches.")
    print(f"  - Train date range: {train_df['kickoff_at'].min()[:10]} -> {train_df['kickoff_at'].max()[:10]}")
    print(f"  - Test date range : {test_df['kickoff_at'].min()[:10]} -> {test_df['kickoff_at'].max()[:10]}")

    results = []

    # 1. Baseline Model: multileague_v2 (thscore_v2)
    print("\n--- 1. Evaluating multileague_v2 (XGBoost on Thscore Features) ---")
    bundle_v2, _ = train_model(
        train_df,
        kind="xgboost",
        feature_set="thscore_v2",
        rolling_window=5,
        seed=42,
    )
    preds_v2 = bundle_v2.predict_proba_df(test_df)

    prep_test = model.prepare_frame(test_df, need_labels=True, feature_set="thscore_v2")
    y_test_ah = pd.to_numeric(prep_test["ah_label"], errors="coerce")
    y_test_ou = pd.to_numeric(prep_test["ou_label"], errors="coerce")
    valid_mask = y_test_ah.notna() & y_test_ou.notna()

    y_ah = y_test_ah[valid_mask].astype(int)
    y_ou = y_test_ou[valid_mask].astype(int)
    p_ah_v2 = preds_v2.loc[valid_mask, "p_fav_cover"]
    p_ou_v2 = preds_v2.loc[valid_mask, "p_over"]

    auc_ah_v2 = roc_auc_score(y_ah, p_ah_v2)
    auc_ou_v2 = roc_auc_score(y_ou, p_ou_v2)
    brier_ah_v2 = brier_score_loss(y_ah, p_ah_v2)
    brier_ou_v2 = brier_score_loss(y_ou, p_ou_v2)

    results.append({
        "Model": "v2 (Thscore XGBoost)",
        "Features": "thscore_v2 (Corners + HT)",
        "AH ROC-AUC": round(auc_ah_v2, 4),
        "OU ROC-AUC": round(auc_ou_v2, 4),
        "Avg ROC-AUC": round((auc_ah_v2 + auc_ou_v2) / 2, 4),
        "AH Brier": round(brier_ah_v2, 4),
        "OU Brier": round(brier_ou_v2, 4),
    })

    # 2. Upgraded Model: multileague_v3 (xg_v3)
    print("\n--- 2. Evaluating multileague_v3 (XGBoost + Understat xG & PPDA) ---")
    bundle_v3, _ = train_model(
        train_df,
        kind="xgboost",
        feature_set="xg_v3",
        rolling_window=5,
        seed=42,
    )
    preds_v3 = bundle_v3.predict_proba_df(test_df)
    p_ah_v3 = preds_v3.loc[valid_mask, "p_fav_cover"]
    p_ou_v3 = preds_v3.loc[valid_mask, "p_over"]

    auc_ah_v3 = roc_auc_score(y_ah, p_ah_v3)
    auc_ou_v3 = roc_auc_score(y_ou, p_ou_v3)
    brier_ah_v3 = brier_score_loss(y_ah, p_ah_v3)
    brier_ou_v3 = brier_score_loss(y_ou, p_ou_v3)

    results.append({
        "Model": "v3 (Understat xG + PPDA)",
        "Features": "xg_v3 (+xG, npxGD, PPDA, Luck)",
        "AH ROC-AUC": round(auc_ah_v3, 4),
        "OU ROC-AUC": round(auc_ou_v3, 4),
        "Avg ROC-AUC": round((auc_ah_v3 + auc_ou_v3) / 2, 4),
        "AH Brier": round(brier_ah_v3, 4),
        "OU Brier": round(brier_ou_v3, 4),
    })

    res_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("🏆 OUT-OF-SAMPLE BENCHMARK: MULTILEAGUE V2 vs V3 (xG ENRICHMENT)")
    print("=" * 80)
    print(res_df.to_string(index=False))

    # Train Final multileague_v3 on full dataset and freeze
    print("\n[Production] Training final multileague_v3 on full dataset...")
    final_bundle, report = train_model(
        df,
        kind="xgboost",
        feature_set="xg_v3",
        rolling_window=5,
        seed=42,
    )

    out_path = ROOT_DIR / "models" / "multileague_v3.pkl"
    sidecar = final_bundle.save(out_path)
    print(f"\n[OK] Model multileague_v3.pkl successfully frozen & saved!")
    print(f"  - Model Path      : {out_path}")
    print(f"  - SHA-256 Sidecar : {sidecar['sha256']}")
    print(f"  - Total Features  : {len(final_bundle.feature_columns)}")
    print(f"  - Active Rows Used: {report.n_rows_used}")


if __name__ == "__main__":
    benchmark_and_train()
