"""
End-to-End Walk-Forward Train/Test Suite for Syndicate Quant Models.
Strict Out-of-Sample Validation (80% Train / 20% Unseen Future Test).

Steps:
1. Load 12,718 clean historical matches with Asian Handicap & Over/Under lines.
2. Temporal split (no shuffling, oldest 80% to train, newest 20% to test).
3. Train HistGradientBoosting + CalibratedClassifierCV (Sigmoid) on Train Set.
4. Predict calibrated probabilities on Out-Of-Sample Test Set.
5. Apply Shin Fair Odds & Steam Engine filter for +EV Edge selection.
6. Simulate Flat Betting vs Fractional Kelly Criterion portfolio growth.
7. Output institutional quant metrics (Brier Score, ROC-AUC, Win Rate, ROI %, Drawdown).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

from fbpredict.quant_model import train_calibrated_models

def run_train_test_pipeline():
    print("=" * 85)
    print(">>> [SYNDICATE QUANT ALPHA: WALK-FORWARD TRAIN & TEST PIPELINE] <<<")
    print("     (Strict Time-Ordered Out-Of-Sample Validation - Zero Lookahead)")
    print("=" * 85)

    # 1. Load Data
    data_path = ROOT_DIR / "data" / "clean_quant_dataset.pkl"
    print(f"\n[*] 1. Loading clean historical dataset from {data_path}...")
    with open(data_path, "rb") as f:
        df = pickle.load(f)

    # Ensure valid history flag
    df_usable = df[df["has_sufficient_history"] == 1].copy().reset_index(drop=True)
    n_total = len(df_usable)
    print(f"[+] Total Verified Matches with Sufficient History: {n_total:,} matches")

    # 2. Strict Temporal Split (80% Train, 20% Test)
    split_idx = int(n_total * 0.80)
    train_df = df_usable.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_usable.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"\n[*] 2. Splitting Chronologically (Time-Ordered):")
    print(f"    - Training Set (In-Sample Past)       : {len(train_df):,} matches (80%)")
    print(f"    - Test Set (Out-Of-Sample Future)     : {len(test_df):,} matches (20%)")

    # 3. Model Training on Train Set ONLY
    print(f"\n[*] 3. Training Probability-Calibrated Quant Models on Train Set...")
    bundle, meta = train_calibrated_models(train_df, seed=42)
    print(f"    [+] AH Model fitted on {meta['n_samples_ah']:,} Asian Handicap matches (Cover Rate: {meta['ah_cover_rate']:.2%})")
    print(f"    [+] OU Model fitted on {meta['n_samples_ou']:,} Over/Under matches (Over Rate: {meta['ou_over_rate']:.2%})")

    # 4. Out-Of-Sample Inference on Test Set
    print(f"\n[*] 4. Running Vectorized Inference on Unseen Test Set...")
    preds = bundle.predict_batch(test_df)
    test_merged = pd.concat([test_df, preds], axis=1)

    # Evaluate Calibration Metrics on Over/Under
    valid_ou = test_merged[test_merged["ou_label"].notna()].copy()
    y_ou = valid_ou["ou_label"].astype(int)
    p_ou = valid_ou["p_over"]
    brier_ou = brier_score_loss(y_ou, p_ou)
    auc_ou = roc_auc_score(y_ou, p_ou)

    # Evaluate Calibration Metrics on Asian Handicap
    valid_ah = test_merged[test_merged["ah_label"].notna()].copy()
    y_ah = valid_ah["ah_label"].astype(int)
    p_ah = valid_ah["p_fav"]
    brier_ah = brier_score_loss(y_ah, p_ah)
    auc_ah = roc_auc_score(y_ah, p_ah)

    print(f"    [+] Over/Under Model  -> Brier Score: {brier_ou:.4f} | ROC-AUC: {auc_ou:.4f}")
    print(f"    [+] Asian Handicap    -> Brier Score: {brier_ah:.4f} | ROC-AUC: {auc_ah:.4f}")

    # 5. Trading Simulation on Out-Of-Sample Test Set
    print(f"\n[*] 5. Executing Syndicate +EV Trading Simulation on Test Set...")
    initial_bankroll = 10000.0
    flat_stake = 100.0
    bankroll_flat = initial_bankroll
    bankroll_kelly = initial_bankroll

    min_edge_threshold = 0.03  # Minimum 3.0% edge over fair odds to trade
    trades = []

    for idx, row in test_merged.iterrows():
        # Scan 4 primary markets
        markets = [
            ("AH_fav", row.get("edge_fav", 0), row.get("p_fav", 0), row.get("ah_fav_price", 1.90), row.get("ah_label") == 1, row.get("ah_label") == 0, row.get("ah_label")),
            ("AH_dog", row.get("edge_dog", 0), row.get("p_dog", 0), row.get("ah_dog_price", 1.90), row.get("ah_label") == 0, row.get("ah_label") == 1, row.get("ah_label")),
            ("OU_over", row.get("edge_over", 0), row.get("p_over", 0), row.get("ou_over_price", 1.90), row.get("ou_label") == 1, row.get("ou_label") == 0, row.get("ou_label")),
            ("OU_under", row.get("edge_under", 0), row.get("p_under", 0), row.get("ou_under_price", 1.90), row.get("ou_label") == 0, row.get("ou_label") == 1, row.get("ou_label")),
        ]

        # Select market with maximum positive edge
        best = max(markets, key=lambda x: x[1])
        mkt_name, edge, prob, odds, is_win, is_loss, label_val = best

        if edge >= min_edge_threshold and pd.notna(odds) and odds > 1.20 and pd.notna(label_val):
            is_push = (not is_win) and (not is_loss)

            # Flat Bet P&L
            if is_push:
                pnl_flat = 0.0
            elif is_win:
                pnl_flat = flat_stake * (odds - 1.0)
            else:
                pnl_flat = -flat_stake
            bankroll_flat += pnl_flat

            # Quarter-Kelly Staking
            kelly_f = max(0.0, edge / (odds - 1.0)) * 0.25
            kelly_f = min(0.04, kelly_f)  # Risk cap at 4%
            stake_k = bankroll_kelly * kelly_f
            
            if is_push:
                pnl_k = 0.0
            elif is_win:
                pnl_k = stake_k * (odds - 1.0)
            else:
                pnl_k = -stake_k
            bankroll_kelly += pnl_k

            trades.append({
                "market": mkt_name,
                "edge": edge,
                "prob": prob,
                "odds": odds,
                "win": is_win,
                "loss": is_loss,
                "push": is_push,
                "pnl_flat": pnl_flat,
                "pnl_kelly": pnl_k,
                "bankroll_flat": bankroll_flat,
                "bankroll_kelly": bankroll_kelly
            })

    trades_df = pd.DataFrame(trades)
    n_trades = len(trades_df)
    n_wins = len(trades_df[trades_df["win"] == True])
    n_losses = len(trades_df[trades_df["loss"] == True])
    n_pushes = len(trades_df[trades_df["push"] == True])

    win_rate = (n_wins / (n_wins + n_losses)) * 100.0 if (n_wins + n_losses) > 0 else 0.0
    flat_profit = bankroll_flat - initial_bankroll
    flat_roi = (flat_profit / (n_trades * flat_stake)) * 100.0 if n_trades > 0 else 0.0
    
    kelly_profit = bankroll_kelly - initial_bankroll
    kelly_gain = (kelly_profit / initial_bankroll) * 100.0

    # Max Drawdown
    equity_flat = trades_df["bankroll_flat"].values
    peaks = np.maximum.accumulate(equity_flat)
    dd_flat = (equity_flat - peaks) / peaks * 100.0
    max_dd = abs(min(dd_flat)) if len(dd_flat) > 0 else 0.0

    print("=" * 85)
    print(">>> [OUT-OF-SAMPLE TEST SET PERFORMANCE SUMMARY] <<<")
    print("=" * 85)
    print(f"  Test Evaluation Window        : {len(test_df):,} unseen historical matches")
    print(f"  Trades Executed (+EV >= 3%)   : {n_trades:,} trades")
    print(f"  Win / Loss / Push Record      : {n_wins}W - {n_losses}L - {n_pushes}P")
    print(f"  Realized Win Rate             : {win_rate:.2f}%")
    print("-" * 85)
    print(f"  [1. FLAT STAKING ($100 per bet)]:")
    print(f"    - Net Realized Profit       : ${flat_profit:+,.2f}")
    print(f"    - Yield / Return on Turnover: {flat_roi:+.2f}%")
    print(f"    - Maximum Equity Drawdown   : {max_dd:.2f}%")
    print("-" * 85)
    print(f"  [2. FRACTIONAL KELLY STAKING (Portfolio Compounding)]:")
    print(f"    - Starting Bankroll         : ${initial_bankroll:,.2f}")
    print(f"    - Final Portfolio Bankroll  : ${bankroll_kelly:,.2f}")
    print(f"    - Total Net Gain            : {kelly_gain:+.2f}%")
    print("=" * 85)

if __name__ == "__main__":
    run_train_test_pipeline()
