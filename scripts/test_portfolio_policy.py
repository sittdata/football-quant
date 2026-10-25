import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fbpredict.quant_model import QuantModelBundle, train_calibrated_models

df = pd.read_pickle(ROOT_DIR / "data" / "clean_quant_dataset.pkl")
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].reset_index(drop=True)
test = df.iloc[split_idx:].reset_index(drop=True)

bundle, _ = train_calibrated_models(train, seed=42)
preds = bundle.predict_batch(test)

stake = 1.0

for min_edge in [0.03, 0.04, 0.05, 0.06]:
    bets = []
    equity = [0.0]
    tot_profit = 0.0

    for idx, r in test.iterrows():
        p = preds.loc[idx]
        lg = r["league"]
        is_champ = "Championship" in lg
        
        # League-specific Market Routing Strategy:
        if is_champ:
            # Lower league (Championship): OU Over + Under + AH Favourite
            candidates = [
                ("OU_over", p["edge_over"], p["p_over"], r["ou_over_price"]),
                ("OU_under", p["edge_under"], p["p_under"], r["ou_under_price"]),
                ("AH_fav", p["edge_fav"], p["p_fav"], r["ah_fav_price"]),
            ]
        else:
            # Top 5 European leagues: Value in AH Underdog + OU
            candidates = [
                ("AH_dog", p["edge_dog"], p["p_dog"], r["ah_dog_price"]),
                ("OU_over", p["edge_over"], p["p_over"], r["ou_over_price"]),
                ("OU_under", p["edge_under"], p["p_under"], r["ou_under_price"]),
            ]
            
        best_mkt, edge, prob, odds = max(candidates, key=lambda x: x[1])
        if edge < min_edge:
            continue
            
        odds = odds or 1.90
        unit_res = 0.0
        if best_mkt == "AH_fav":
            unit_res = r["ah_unit"]
        elif best_mkt == "AH_dog":
            unit_res = -r["ah_unit"]
        elif best_mkt == "OU_over":
            unit_res = r["ou_unit"]
        elif best_mkt == "OU_under":
            unit_res = -r["ou_unit"]
            
        profit = 0.0
        if unit_res > 0:
            profit = (odds - 1.0) * stake * unit_res
        elif unit_res < 0:
            profit = -stake * abs(unit_res)
            
        tot_profit += profit
        equity.append(tot_profit)

        bets.append({
            "kickoff": r["kickoff_at"],
            "league": lg,
            "match": f"{r['home_team']} vs {r['away_team']}",
            "market": best_mkt,
            "odds": odds,
            "prob": prob,
            "edge": edge,
            "unit_res": unit_res,
            "profit": profit,
        })

    b_df = pd.DataFrame(bets)
    if b_df.empty:
        continue

    wins = len(b_df[b_df["unit_res"] > 0])
    draws = len(b_df[b_df["unit_res"] == 0])
    losses = len(b_df[b_df["unit_res"] < 0])
    pnl = b_df["profit"].sum()
    roi = (pnl / len(b_df)) * 100

    eq = np.array(equity)
    peaks = np.maximum.accumulate(eq)
    drawdowns = peaks - eq
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    print("=" * 70)
    print(f"🎯 QUANT STRATEGY BACKTEST (MIN EDGE >= {min_edge*100:.1f}%)")
    print(f"Total Bets: {len(b_df)} | Record: 🟢 {wins}W | ⚪ {draws}D | 🔴 {losses}L")
    print(f"Realized Hit Rate: {wins/len(b_df)*100:.2f}% | Expected Prob: {b_df['prob'].mean()*100:.2f}%")
    print(f"Net PnL: {pnl:+.2f} Units | Net ROI: {roi:+.2f}% | Max Drawdown: {max_dd:.2f} Units")
    print("-" * 70)
    print("🏆 BY LEAGUE:")
    for lg, grp in b_df.groupby("league"):
        w = (grp["unit_res"] > 0).sum()
        d = (grp["unit_res"] == 0).sum()
        l = (grp["unit_res"] < 0).sum()
        lpnl = grp["profit"].sum()
        print(f"  {lg[:28]:<28} | Bets: {len(grp):<3} | {w}/{d}/{l} | PnL: {lpnl:+6.2f} U | ROI: {lpnl/len(grp)*100:+5.1f}%")
    print("📊 BY MARKET:")
    for mkt, grp in b_df.groupby("market"):
        w = (grp["unit_res"] > 0).sum()
        d = (grp["unit_res"] == 0).sum()
        l = (grp["unit_res"] < 0).sum()
        mpnl = grp["profit"].sum()
        print(f"  {mkt:<15} | Bets: {len(grp):<3} | {w}/{d}/{l} | PnL: {mpnl:+6.2f} U | ROI: {mpnl/len(grp)*100:+5.1f}%")
