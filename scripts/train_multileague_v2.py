"""Train and benchmark multileague_v2 using Thscore feature enrichment.

Compares:
- Baseline (v1): Logistic Regression on original odds & rolling goals
- Upgraded (v2): XGBoost / Regularized MultiOutput with Thscore HT dynamics & Corner features
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

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

    # 1. Baseline Model: multileague_v1 (logreg on filtered_novig)
    print("\n--- 1. Evaluating Baseline (v1: LogReg on filtered_novig) ---")
    bundle_v1, _ = train_model(
        train_df,
        kind="logreg",
        feature_set="filtered_novig",
        rolling_window=5,
        seed=42,
    )
    preds_v1 = bundle_v1.predict_proba_df(test_df)
    
    # Ground truth labels for test set
    prep_test = model.prepare_frame(test_df, need_labels=True, feature_set="filtered_novig")
    y_test_ah = pd.to_numeric(prep_test["ah_label"], errors="coerce")
    y_test_ou = pd.to_numeric(prep_test["ou_label"], errors="coerce")
    valid_mask = y_test_ah.notna() & y_test_ou.notna()

    y_ah = y_test_ah[valid_mask].astype(int)
    y_ou = y_test_ou[valid_mask].astype(int)
    p_ah_v1 = preds_v1.loc[valid_mask, "p_fav_cover"]
    p_ou_v1 = preds_v1.loc[valid_mask, "p_over"]

    auc_ah_v1 = roc_auc_score(y_ah, p_ah_v1)
    auc_ou_v1 = roc_auc_score(y_ou, p_ou_v1)
    brier_ah_v1 = brier_score_loss(y_ah, p_ah_v1)
    brier_ou_v1 = brier_score_loss(y_ou, p_ou_v1)

    results.append({
        "Model": "v1 (Baseline LogReg)",
        "Features": "filtered_novig (old)",
        "AH ROC-AUC": round(auc_ah_v1, 4),
        "OU ROC-AUC": round(auc_ou_v1, 4),
        "Avg ROC-AUC": round((auc_ah_v1 + auc_ou_v1) / 2, 4),
        "AH Brier": round(brier_ah_v1, 4),
        "OU Brier": round(brier_ou_v1, 4),
    })

    # 2. Upgraded Model: v2 with Thscore Features (thscore_v2) - LogReg
    print("\n--- 2. Evaluating v2 LogReg (with Thscore HT & Tempo features) ---")
    bundle_v2_lr, _ = train_model(
        train_df,
        kind="logreg",
        feature_set="thscore_v2",
        rolling_window=5,
        seed=42,
    )
    preds_v2_lr = bundle_v2_lr.predict_proba_df(test_df)
    p_ah_v2_lr = preds_v2_lr.loc[valid_mask, "p_fav_cover"]
    p_ou_v2_lr = preds_v2_lr.loc[valid_mask, "p_over"]

    auc_ah_v2_lr = roc_auc_score(y_ah, p_ah_v2_lr)
    auc_ou_v2_lr = roc_auc_score(y_ou, p_ou_v2_lr)
    brier_ah_v2_lr = brier_score_loss(y_ah, p_ah_v2_lr)
    brier_ou_v2_lr = brier_score_loss(y_ou, p_ou_v2_lr)

    results.append({
        "Model": "v2 (Thscore LogReg)",
        "Features": "thscore_v2 (+HT Dynamics)",
        "AH ROC-AUC": round(auc_ah_v2_lr, 4),
        "OU ROC-AUC": round(auc_ou_v2_lr, 4),
        "Avg ROC-AUC": round((auc_ah_v2_lr + auc_ou_v2_lr) / 2, 4),
        "AH Brier": round(brier_ah_v2_lr, 4),
        "OU Brier": round(brier_ou_v2_lr, 4),
    })

    # 3. Upgraded Model: v2 with XGBoost (thscore_v2)
    print("\n--- 3. Evaluating v2 XGBoost (Non-linear Thscore Interactions) ---")
    bundle_v2_xgb, _ = train_model(
        train_df,
        kind="xgboost",
        feature_set="thscore_v2",
        rolling_window=5,
        seed=42,
    )
    preds_v2_xgb = bundle_v2_xgb.predict_proba_df(test_df)
    p_ah_v2_xgb = preds_v2_xgb.loc[valid_mask, "p_fav_cover"]
    p_ou_v2_xgb = preds_v2_xgb.loc[valid_mask, "p_over"]

    auc_ah_v2_xgb = roc_auc_score(y_ah, p_ah_v2_xgb)
    auc_ou_v2_xgb = roc_auc_score(y_ou, p_ou_v2_xgb)
    brier_ah_v2_xgb = brier_score_loss(y_ah, p_ah_v2_xgb)
    brier_ou_v2_xgb = brier_score_loss(y_ou, p_ou_v2_xgb)

    results.append({
        "Model": "v2 (Thscore XGBoost)",
        "Features": "thscore_v2 (+Non-linear)",
        "AH ROC-AUC": round(auc_ah_v2_xgb, 4),
        "OU ROC-AUC": round(auc_ou_v2_xgb, 4),
        "Avg ROC-AUC": round((auc_ah_v2_xgb + auc_ou_v2_xgb) / 2, 4),
        "AH Brier": round(brier_ah_v2_xgb, 4),
        "OU Brier": round(brier_ou_v2_xgb, 4),
    })

    # Comparison Table
    res_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("🏆 OUT-OF-SAMPLE BENCHMARK: MULTILEAGUE V1 vs V2")
    print("=" * 80)
    print(res_df.to_string(index=False))

    # Train Final multileague_v2 on FULL dataset and freeze
    print("\n[Production] Training final multileague_v2 on full dataset (2,400+ matches)...")
    final_bundle, report = train_model(
        df,
        kind="xgboost",
        feature_set="thscore_v2",
        rolling_window=5,
        seed=42,
    )

    out_path = ROOT_DIR / "models" / "multileague_v2.pkl"
    sidecar = final_bundle.save(out_path)
    print(f"\n[OK] Model multileague_v2.pkl successfully frozen & saved!")
    print(f"  - Model Path      : {out_path}")
    print(f"  - SHA-256 Sidecar : {sidecar['sha256']}")
    print(f"  - Total Features  : {len(final_bundle.feature_columns)}")
    print(f"  - Active Rows Used: {report.n_rows_used}")


if __name__ == "__main__":
    benchmark_and_train()
