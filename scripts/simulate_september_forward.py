"""Simulate realistic betting performance from 2026-09-01 to 2026-10-24.

Starting capital: 1,000 THB.
Uses frozen multileague_v2 model.
Tests:
1. Flat stake 100 THB per bet
2. Flat stake 50 THB per bet (recommended 5% bankroll)
3. Staking with different edge thresholds (5%, 8%, 12%, 15%)
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fbpredict import config, db, features, labels, model
from fbpredict.db import connect
from fbpredict.model import ModelBundle


def settle_ah(fav_scored: float, dog_scored: float, ah_line: float, bet_on_fav: bool, price: float, stake: float) -> float:
    """Return net profit/loss for Asian Handicap bet."""
    diff = fav_scored - dog_scored
    # ah_unit returns +1, +0.5, 0, -0.5, -1
    unit = labels.ah_unit(diff, ah_line)
    if not bet_on_fav:
        unit = -unit

    if unit > 0:
        return unit * (price - 1.0) * stake
    else:
        return unit * stake  # unit is negative or zero


def settle_ou(total_goals: float, ou_line: float, bet_on_over: bool, price: float, stake: float) -> float:
    """Return net profit/loss for Over/Under bet."""
    unit = labels.ou_unit(total_goals, ou_line)
    if not bet_on_over:
        unit = -unit

    if unit > 0:
        return unit * (price - 1.0) * stake
    else:
        return unit * stake


def run_simulation():
    con = connect(ROOT_DIR / "data" / "football.db")
    query = """
    SELECT match_id, league, season, kickoff_at, home_team, away_team, fav_side,
           ah_open, ah_fav_price, ah_dog_price, pool_fav_open, pool_dog_open,
           draw_ft_odds, draw_ht_odds, ou_line_open, ou_over_price, ou_under_price,
           home_goals, away_goals, ht_home, ht_away
    FROM matches
    WHERE kickoff_at >= '2026-09-01'
      AND kickoff_at <= '2026-10-24'
      AND home_goals IS NOT NULL
      AND away_goals IS NOT NULL
      AND ah_open IS NOT NULL
      AND ou_line_open IS NOT NULL
    ORDER BY kickoff_at ASC
    """
    df = pd.read_sql_query(query, con)
    con.close()

    print(f"[Simulation] Loaded {len(df)} matches from 2026-09-01 to 2026-10-24.")

    # Load frozen multileague_v2 model
    bundle = ModelBundle.load(ROOT_DIR / "models" / "multileague_v2.pkl", require_verified=True)
    preds = bundle.predict_proba_df(df)

    candidates = []
    for idx, row in df.iterrows():
        p_fav_cover = float(preds.loc[idx, "p_fav_cover"])
        p_dog_cover = 1.0 - p_fav_cover
        p_over = float(preds.loc[idx, "p_over"])
        p_under = 1.0 - p_over

        fav_p = float(row.get("ah_fav_price") or 1.90)
        dog_p = float(row.get("ah_dog_price") or 1.90)
        over_p = float(row.get("ou_over_price") or 1.90)
        under_p = float(row.get("ou_under_price") or 1.90)

        fair_fav, fair_dog = labels.fair_two_way(fav_p, dog_p)
        fair_over, fair_under = labels.fair_two_way(over_p, under_p)

        edges = {
            "AH_fav": (p_fav_cover - fair_fav, p_fav_cover, fav_p),
            "AH_dog": (p_dog_cover - fair_dog, p_dog_cover, dog_p),
            "OU_over": (p_over - fair_over, p_over, over_p),
            "OU_under": (p_under - fair_under, p_under, under_p),
        }

        # Pick market with highest edge
        best_market = max(edges, key=lambda k: edges[k][0])
        best_edge, best_prob, best_price = edges[best_market]

        candidates.append({
            "match_id": row["match_id"],
            "league": row["league"],
            "kickoff_at": row["kickoff_at"],
            "home": row["home_team"],
            "away": row["away_team"],
            "fav_side": row["fav_side"],
            "ah_open": float(row["ah_open"]),
            "ou_line_open": float(row["ou_line_open"]),
            "home_goals": float(row["home_goals"]),
            "away_goals": float(row["away_goals"]),
            "best_market": best_market,
            "best_edge": best_edge,
            "best_prob": best_prob,
            "best_price": best_price,
        })

    cand_df = pd.DataFrame(candidates)

    print("\n" + "=" * 80)
    print("📊 SIMULATION: STARTING BANKROLL = 1,000 THB (2026-09-01 -> 2026-10-24)")
    print("=" * 80)

    # Test different edge thresholds & stake sizes
    thresholds = [0.08, 0.10, 0.12, 0.15]
    stakes = [100, 50]

    for stake in stakes:
        print(f"\n>>> PLAN: ลงคู่ละ {stake} บาท (ทุนเริ่มต้น 1,000 บาท) <<<")
        print(f"{'Min Edge':<10} | {'Total Bets':<10} | {'Win Rate':<10} | {'Profit (THB)':<14} | {'End Bankroll':<14} | {'ROI':<8}")
        print("-" * 75)

        for edge_th in thresholds:
            bets = cand_df[cand_df["best_edge"] >= edge_th].copy()
            if bets.empty:
                continue

            bankroll = 1000.0
            total_profit = 0.0
            wins = 0
            pushes = 0
            losses = 0

            for _, b in bets.iterrows():
                h_goals = b["home_goals"]
                a_goals = b["away_goals"]
                tot_goals = h_goals + a_goals
                fav_scored = h_goals if b["fav_side"] == "home" else a_goals
                dog_scored = a_goals if b["fav_side"] == "home" else h_goals

                m = b["best_market"]
                pnl = 0.0
                if m == "AH_fav":
                    pnl = settle_ah(fav_scored, dog_scored, b["ah_open"], True, b["best_price"], stake)
                elif m == "AH_dog":
                    pnl = settle_ah(fav_scored, dog_scored, b["ah_open"], False, b["best_price"], stake)
                elif m == "OU_over":
                    pnl = settle_ou(tot_goals, b["ou_line_open"], True, b["best_price"], stake)
                elif m == "OU_under":
                    pnl = settle_ou(tot_goals, b["ou_line_open"], False, b["best_price"], stake)

                total_profit += pnl
                bankroll += pnl

                if pnl > 0:
                    wins += 1
                elif pnl == 0:
                    pushes += 1
                else:
                    losses += 1

            decided = wins + losses
            wr = (wins / decided * 100) if decided > 0 else 0
            turnover = len(bets) * stake
            roi = (total_profit / turnover * 100) if turnover > 0 else 0

            sign = "+" if total_profit > 0 else ""
            print(f"{edge_th*100:.0f}%{'':<7} | {len(bets):<10} | {wr:5.1f}%{'':<4} | {sign}{total_profit:8.1f} บาท   | {bankroll:8.1f} บาท   | {sign}{roi:5.1f}%")


if __name__ == "__main__":
    run_simulation()
