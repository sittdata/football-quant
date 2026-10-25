"""Industrial Training and Strict Walk-Forward Benchmark for Football Quant Models.
Author: cook45

Enforces zero data leakage:
- 80% Train (Chronological past matches only)
- 20% Test (Chronological future out-of-sample matches only)
- Calibrated probabilities
- Focuses strictly on top European leagues with high liquidity
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fbpredict import config, labels
from fbpredict.config import is_whitelisted_league
from fbpredict.feature_engine import HistoricalFeatureStore
from fbpredict.quant_model import (
    MODEL_FEATURE_COLS,
    QuantModelBundle,
    train_calibrated_models,
)

DB_PATH = ROOT_DIR / "data" / "football.db"


def build_training_dataset():
    print("⏳ [1/4] Loading Top-Tier European matches from database...")
    con = sqlite3.connect(DB_PATH)
    query = """
    SELECT match_id, league, season, kickoff_at, home_team, away_team, fav_side,
           ah_open, ah_fav_price, ah_dog_price,
           ou_line_open, ou_over_price, ou_under_price, draw_ft_odds,
           home_goals, away_goals
    FROM matches
    WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
      AND ah_open IS NOT NULL AND ou_line_open IS NOT NULL
    ORDER BY kickoff_at ASC
    """
    df = pd.read_sql_query(query, con)
    con.close()

    # Filter strictly for whitelisted leagues
    df = df[df["league"].apply(is_whitelisted_league)].reset_index(drop=True)
    print(f"✅ Filtered {len(df)} matches from top European leagues.")

    print("⏳ [2/4] Enriching matches with strictly historical rolling features...")
    store = HistoricalFeatureStore(DB_PATH)

    rows = []
    for _, r in df.iterrows():
        feat = store.enrich_match(
            home_team=r["home_team"],
            away_team=r["away_team"],
            kickoff_iso=r["kickoff_at"],
            ah_open=r["ah_open"],
            ou_line_open=r["ou_line_open"],
            ah_fav_price=r["ah_fav_price"] or 1.95,
            ah_dog_price=r["ah_dog_price"] or 1.95,
            ou_over_price=r["ou_over_price"] or 1.95,
            ou_under_price=r["ou_under_price"] or 1.95,
            draw_ft_odds=r["draw_ft_odds"] or 3.30,
            fav_side=r["fav_side"],
        )

        h_g = int(r["home_goals"])
        a_g = int(r["away_goals"])
        tot_g = h_g + a_g
        fav_diff = (h_g - a_g) if r["fav_side"] == "home" else (a_g - h_g)

        ah_unit = labels.ah_unit(fav_diff, r["ah_open"])
        ou_unit = labels.ou_unit(tot_g, r["ou_line_open"], "over")
        ah_bin = labels.ah_binary(fav_diff, r["ah_open"])
        ou_bin = labels.ou_binary(tot_g, r["ou_line_open"], "over")

        feat["match_id"] = r["match_id"]
        feat["league"] = r["league"]
        feat["kickoff_at"] = r["kickoff_at"]
        feat["home_team"] = r["home_team"]
        feat["away_team"] = r["away_team"]
        feat["fav_side"] = r["fav_side"]
        feat["home_goals"] = h_g
        feat["away_goals"] = a_g
        feat["fav_diff"] = fav_diff
        feat["tot_goals"] = tot_g
        feat["ah_unit"] = ah_unit
        feat["ou_unit"] = ou_unit
        feat["ah_label"] = ah_bin
        feat["ou_label"] = ou_bin
        rows.append(feat)

    feat_df = pd.DataFrame(rows)
    # Only keep matches where both teams have sufficient history
    usable_df = feat_df[feat_df["has_sufficient_history"] == 1].reset_index(drop=True)
    print(f"✅ Usable matches with sufficient history: {len(usable_df)} matches.")
    return usable_df


def main():
    df = build_training_dataset()
    if len(df) < 200:
        print("⚠️ Too few usable matches to perform robust train/test split.")
        return

    # Chronological Walk-Forward Split (80% Train, 20% Test)
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print("\n" + "=" * 65)
    print("📊 STRICT WALK-FORWARD SPLIT (ZERO DATA LEAKAGE)")
    print("=" * 65)
    print(f"Train matches: {len(train_df)} ({train_df['kickoff_at'].min()[:10]} -> {train_df['kickoff_at'].max()[:10]})")
    print(f"Test matches : {len(test_df)} ({test_df['kickoff_at'].min()[:10]} -> {test_df['kickoff_at'].max()[:10]})")

    print("\n⏳ [3/4] Training calibrated models on Train set...")
    bundle, meta = train_calibrated_models(train_df, seed=42)

    print("⏳ [4/4] Evaluating Out-Of-Sample performance on unseen Test set...")
    mask_test_ah = test_df["ah_label"].notna()
    mask_test_ou = test_df["ou_label"].notna()

    X_test_ah = test_df.loc[mask_test_ah, bundle.ah_feature_cols]
    y_test_ah = test_df.loc[mask_test_ah, "ah_label"].astype(int)
    p_ah = bundle.ah_model.predict_proba(X_test_ah)[:, 1]

    X_test_ou = test_df.loc[mask_test_ou, bundle.ou_feature_cols]
    y_test_ou = test_df.loc[mask_test_ou, "ou_label"].astype(int)
    p_ou = bundle.ou_model.predict_proba(X_test_ou)[:, 1]

    brier_ah = brier_score_loss(y_test_ah, p_ah)
    brier_ou = brier_score_loss(y_test_ou, p_ou)
    auc_ah = roc_auc_score(y_test_ah, p_ah)
    auc_ou = roc_auc_score(y_test_ou, p_ou)

    print("\n--- Out-Of-Sample Calibration Metrics (Real Unseen Matches) ---")
    print(f"AH (Handicap)   : ROC-AUC = {auc_ah:.4f} | Brier Loss = {brier_ah:.4f}")
    print(f"OU (Over/Under) : ROC-AUC = {auc_ou:.4f} | Brier Loss = {brier_ou:.4f}")

    # Simulated Betting Benchmark on Unseen Test Set
    min_edge = 0.04  # 4% Edge threshold
    bets = []
    tot_staked = 0.0
    tot_profit = 0.0

    for idx, r in test_df.iterrows():
        pred = bundle.predict_match(r.to_dict())
        if not pred.get("safe") or pred.get("best_edge", 0.0) < min_edge:
            continue

        mkt = pred["best_market"]
        odds = pred["best_odds"]
        edge = pred["best_edge"]

        # Settlement
        unit_res = 0.0
        if mkt == "AH_fav":
            unit_res = r["ah_unit"]
        elif mkt == "AH_dog":
            unit_res = -r["ah_unit"]
        elif mkt == "OU_over":
            unit_res = r["ou_unit"]
        elif mkt == "OU_under":
            unit_res = -r["ou_unit"]

        profit = 0.0
        if unit_res > 0:
            profit = (odds - 1.0) * unit_res
        elif unit_res < 0:
            profit = -abs(unit_res)

        tot_staked += 1.0
        tot_profit += profit
        bets.append({
            "match": f"{r['home_team']} vs {r['away_team']}",
            "market": mkt,
            "odds": odds,
            "edge": edge,
            "unit_res": unit_res,
            "profit": profit,
        })

    print("\n" + "=" * 65)
    print("📈 OUT-OF-SAMPLE BETTING BENCHMARK (REALISTIC EDGE)")
    print("=" * 65)
    print(f"Total Test Matches  : {len(test_df)}")
    print(f"Bets Taken (Edge≥4%): {len(bets)}")
    if bets:
        wins = sum(1 for b in bets if b["unit_res"] > 0)
        draws = sum(1 for b in bets if b["unit_res"] == 0)
        losses = sum(1 for b in bets if b["unit_res"] < 0)
        win_rate = (wins / len(bets)) * 100
        roi = (tot_profit / tot_staked) * 100
        print(f"Results Breakdown   : {wins} Wins | {draws} Draws | {losses} Losses")
        print(f"Out-of-Sample Win%  : {win_rate:.2f}%")
        print(f"Total Net Units     : {tot_profit:+.2f} Units")
        print(f"Realistic Net ROI   : {roi:+.2f}%")
    else:
        print("No bets met the edge threshold in the test set.")

    # Train final bundle on full clean dataset and save
    print("\n💾 Freezing final production bundle...")
    final_bundle, _ = train_calibrated_models(df, seed=42)
    save_path = ROOT_DIR / "models" / "quant_calibrated_v1.pkl"
    final_bundle.save(save_path)
    print(f"✅ Production Model saved to {save_path}")


if __name__ == "__main__":
    main()
