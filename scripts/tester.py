"""Ultimate Football Quant Backtester & Performance Auditor.
Author: cook45

Key Capabilities:
1. Strict Walk-Forward (Zero Data Leakage).
2. Probability Calibration & True Vig-Free Edge Testing.
3. Multi-Dimension Breakdown (By League, Market, Edge Sensitivity).
4. Risk & Drawdown Analytics (Max Drawdown, Win/Loss Streaks, Profit Factor).
5. Visual ASCII Equity Curve in Terminal.

Usage:
    python scripts/tester.py
    python scripts/tester.py --edge 0.05
    python scripts/tester.py --league "EPL" --edge 0.04
    python scripts/tester.py --market AH
    python scripts/tester.py --sensitivity
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fbpredict import labels
from fbpredict.config import is_safe_match, is_whitelisted_league
from fbpredict.feature_engine import HistoricalFeatureStore
from fbpredict.quant_model import (
    MODEL_FEATURE_COLS,
    QuantModelBundle,
    train_calibrated_models,
)

DB_PATH = ROOT_DIR / "data" / "football.db"


def load_and_enrich_dataset(cache_file: Path = ROOT_DIR / "data" / "clean_quant_dataset.pkl") -> pd.DataFrame:
    """Load, filter, and enrich historical matches with rolling stats."""
    if cache_file.exists():
        try:
            df = pd.read_pickle(cache_file)
            if not df.empty and "has_sufficient_history" in df.columns:
                return df
        except Exception:
            pass

    print("⏳ Loading & enriching top European league fixtures from database...")
    con = sqlite3.connect(DB_PATH)
    query = """
    SELECT match_id, league, season, kickoff_at, home_team, away_team, fav_side,
           ah_open, ah_fav_price, ah_dog_price,
           ou_line_open, ou_over_price, ou_under_price, draw_ft_odds,
           ah_close, ou_line_close, pool_fav_close, pool_dog_close,
           home_goals, away_goals
    FROM matches
    WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
      AND ah_open IS NOT NULL AND ou_line_open IS NOT NULL
    ORDER BY kickoff_at ASC
    """
    df = pd.read_sql_query(query, con)
    con.close()

    df = df[df["league"].apply(is_whitelisted_league)].reset_index(drop=True)
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
            ah_close=r.get("ah_close"),
            ou_line_close=r.get("ou_line_close"),
            ah_fav_close_price=r.get("pool_fav_close"),
            ah_dog_close_price=r.get("pool_dog_close"),
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

    enriched_df = pd.DataFrame(rows)
    enriched_df = enriched_df[enriched_df["has_sufficient_history"] == 1].reset_index(drop=True)
    enriched_df.to_pickle(cache_file)
    return enriched_df


def render_ascii_chart(values: list[float], height: int = 10, width: int = 60) -> str:
    """Generate a high-res ASCII equity curve for terminal visualization."""
    if not values or len(values) < 2:
        return ""
    
    # Resample values to fit terminal width
    indices = np.linspace(0, len(values) - 1, width, dtype=int)
    sampled = [values[i] for i in indices]

    min_val = min(sampled)
    max_val = max(sampled)
    val_range = max(1e-6, max_val - min_val)

    grid = [[" " for _ in range(width)] for _ in range(height)]

    for col, v in enumerate(sampled):
        norm = (v - min_val) / val_range
        row = int(norm * (height - 1))
        row = max(0, min(height - 1, row))
        grid[height - 1 - row][col] = "█"

    lines = []
    for r in range(height):
        # Y-axis label
        current_val = max_val - (r / (height - 1)) * val_range
        prefix = f"{current_val:+6.1f} | "
        lines.append(prefix + "".join(grid[r]))
    lines.append(" " * 8 + "+" + "-" * width)
    return "\n".join(lines)


def run_backtest(
    df: pd.DataFrame,
    min_edge: float = 0.04,
    league_filter: str | None = None,
    market_filter: str | None = None,
    train_ratio: float = 0.80,
    stake_unit: float = 1.0,
    use_kelly: bool = False,
    steam_only: bool = False,
    show_chart: bool = True,
    bundle: QuantModelBundle | None = None,
) -> dict[str, Any]:
    """Execute strict walk-forward out-of-sample backtest."""
    from fbpredict.staking import calculate_kelly_fraction

    data = df.copy()
    if league_filter:
        data = data[data["league"].str.contains(league_filter, case=False, na=False)].reset_index(drop=True)

    if len(data) < 50:
        print(f"⚠️ Not enough matches to test (found only {len(data)} matches).")
        return {}

    # Strict Walk-Forward Split
    split_idx = int(len(data) * train_ratio)
    train_df = data.iloc[:split_idx].reset_index(drop=True)
    test_df = data.iloc[split_idx:].reset_index(drop=True)

    # Train calibrated model strictly on historical train set if not provided
    if bundle is None:
        bundle, _ = train_calibrated_models(train_df, seed=42)

    bets = []
    tot_staked = 0.0
    tot_profit = 0.0
    equity_curve = [0.0]

    batch_preds = bundle.predict_batch(test_df)

    for idx, r in test_df.iterrows():
        p_row = batch_preds.loc[idx]
        edge_fav = p_row["edge_fav"]
        edge_dog = p_row["edge_dog"]
        edge_over = p_row["edge_over"]
        edge_under = p_row["edge_under"]

        markets = [
            ("AH_fav", edge_fav, p_row["p_fav"], r.get("ah_fav_price", 1.90) or 1.90),
            ("AH_dog", edge_dog, p_row["p_dog"], r.get("ah_dog_price", 1.90) or 1.90),
            ("OU_over", edge_over, p_row["p_over"], r.get("ou_over_price", 1.90) or 1.90),
            ("OU_under", edge_under, p_row["p_under"], r.get("ou_under_price", 1.90) or 1.90),
        ]

        if market_filter:
            if market_filter.upper() == "AH":
                markets = [m for m in markets if m[0].startswith("AH")]
            elif market_filter.upper() == "OU":
                markets = [m for m in markets if m[0].startswith("OU")]

        best_mkt, edge, prob, odds = max(markets, key=lambda x: x[1])

        if edge < min_edge:
            continue

        # Check Steam & Smart Money Alignment
        s_dir = r.get("steam_direction", "NEUTRAL")
        is_steam_confirmed = (
            (best_mkt == "AH_dog" and s_dir == "STEAM_DOG")
            or (best_mkt == "AH_fav" and s_dir == "STEAM_FAV")
            or (best_mkt == "OU_over" and s_dir == "STEAM_OVER")
            or (best_mkt == "OU_under" and s_dir == "STEAM_UNDER")
        )

        if steam_only and not is_steam_confirmed:
            continue

        actual_stake = (
            calculate_kelly_fraction(prob=prob, decimal_odds=odds, fraction=0.25)
            if use_kelly
            else stake_unit
        )

        # Settle bet
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
            profit = (odds - 1.0) * actual_stake * unit_res
        elif unit_res < 0:
            profit = -actual_stake * abs(unit_res)

        tot_staked += actual_stake
        tot_profit += profit
        equity_curve.append(tot_profit)


        bets.append({
            "kickoff": r["kickoff_at"],
            "league": r["league"],
            "match": f"{r['home_team']} vs {r['away_team']}",
            "market": best_mkt,
            "odds": odds,
            "prob": prob,
            "edge": edge,
            "unit_res": unit_res,
            "profit": profit,
        })

    if not bets:
        if show_chart:
            print(f"⚠️ No bets met Edge threshold ≥ {min_edge*100:.1f}%.")
        return {"bundle": bundle, "bets": 0, "win_rate": 0.0, "pnl": 0.0, "roi": 0.0, "max_dd": 0.0, "profit_factor": 0.0}

    b_df = pd.DataFrame(bets)
    wins = len(b_df[b_df["unit_res"] > 0])
    draws = len(b_df[b_df["unit_res"] == 0])
    losses = len(b_df[b_df["unit_res"] < 0])
    win_rate = (wins / len(b_df)) * 100
    roi = (tot_profit / tot_staked) * 100

    # Drawdown & Streaks
    eq = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq)
    drawdowns = peaks - eq
    max_dd = float(np.max(drawdowns))

    # Profit Factor
    gross_win = float(b_df[b_df["profit"] > 0]["profit"].sum())
    gross_loss = abs(float(b_df[b_df["profit"] < 0]["profit"].sum()))
    pf = (gross_win / gross_loss) if gross_loss > 0 else 999.0

    # Streaks
    max_win_streak = 0
    max_loss_streak = 0
    cur_win = 0
    cur_loss = 0
    for res in b_df["unit_res"]:
        if res > 0:
            cur_win += 1
            cur_loss = 0
        elif res < 0:
            cur_loss += 1
            cur_win = 0
        else:
            pass
        max_win_streak = max(max_win_streak, cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    # Output Presentation
    print("\n" + "=" * 70)
    print("🎯 QUANT AUDIT REPORT: STRICT OUT-OF-SAMPLE TESTER")
    print("=" * 70)
    print(f"Holdout Test Window : {test_df['kickoff_at'].min()[:10]} -> {test_df['kickoff_at'].max()[:10]}")
    print(f"Total Test Matches  : {len(test_df)}")
    print(f"Bets Qualified      : {len(b_df)} (Edge Threshold: ≥ {min_edge*100:.1f}%)")
    print("-" * 70)
    print(f"Outcome Record      : 🟢 {wins} Wins | ⚪ {draws} Draws | 🔴 {losses} Losses")
    print(f"Realized Hit Rate   : {win_rate:.2f}% (Expected Model Prob: {b_df['prob'].mean()*100:.2f}%)")
    print(f"Total Units Staked  : {tot_staked:.1f} Units")
    print(f"Net Profit/Loss     : {tot_profit:+.2f} Units")
    print(f"Realized Net ROI    : {roi:+.2f}%")
    print(f"Profit Factor       : {pf:.2f}")
    print(f"Max Drawdown        : {max_dd:.2f} Units")
    print(f"Longest Win Streak  : {max_win_streak} matches")
    print(f"Longest Loss Streak : {max_loss_streak} matches")
    print("=" * 70)

    # 1. Performance by League
    print("\n🏆 PERFORMANCE BY LEAGUE:")
    print("-" * 70)
    print(f"{'League':<30} | {'Bets':<5} | {'W/D/L':<10} | {'Hit Rate':<9} | {'Net PnL':<9} | {'ROI'}")
    print("-" * 70)
    for lg, grp in b_df.groupby("league"):
        w = len(grp[grp["unit_res"] > 0])
        d = len(grp[grp["unit_res"] == 0])
        l = len(grp[grp["unit_res"] < 0])
        hr = (w / len(grp)) * 100
        pnl = grp["profit"].sum()
        sub_roi = (pnl / len(grp)) * 100
        wdl = f"{w}/{d}/{l}"
        print(f"{lg[:30]:<30} | {len(grp):<5} | {wdl:<10} | {hr:6.1f}%   | {pnl:+7.2f} U | {sub_roi:+6.1f}%")

    # 2. Performance by Market (AH vs OU)
    print("\n📊 PERFORMANCE BY MARKET:")
    print("-" * 70)
    print(f"{'Market':<15} | {'Bets':<5} | {'W/D/L':<10} | {'Hit Rate':<9} | {'Net PnL':<9} | {'ROI'}")
    print("-" * 70)
    for mkt, grp in b_df.groupby("market"):
        w = len(grp[grp["unit_res"] > 0])
        d = len(grp[grp["unit_res"] == 0])
        l = len(grp[grp["unit_res"] < 0])
        hr = (w / len(grp)) * 100
        pnl = grp["profit"].sum()
        sub_roi = (pnl / len(grp)) * 100
        wdl = f"{w}/{d}/{l}"
        print(f"{mkt:<15} | {len(grp):<5} | {wdl:<10} | {hr:6.1f}%   | {pnl:+7.2f} U | {sub_roi:+6.1f}%")

    # 3. Visual Equity Curve
    if show_chart and len(equity_curve) > 5:
        print("\n📈 OUT-OF-SAMPLE BANKROLL TRAJECTORY (EQUITY CURVE):")
        print(render_ascii_chart(equity_curve))

    return {
        "bundle": bundle,
        "bets": len(b_df),
        "win_rate": win_rate,
        "pnl": tot_profit,
        "roi": roi,
        "max_dd": max_dd,
        "profit_factor": pf,
    }


def run_sensitivity_analysis(
    df: pd.DataFrame,
    league_filter: str | None = None,
    market_filter: str | None = None,
):
    """Evaluate how Edge Threshold affects sample size, win rate, and ROI."""
    target_name = f"League='{league_filter}'" if league_filter else "All Whitelisted Leagues"
    if market_filter:
        target_name += f" | Market='{market_filter}'"
    print("\n" + "=" * 70)
    print(f"🔬 EDGE SENSITIVITY SCAN ({target_name})")
    print("=" * 70)
    print(f"{'Min Edge':<10} | {'Bets':<8} | {'Win Rate':<12} | {'Net PnL':<12} | {'ROI':<10} | {'Max DD'}")
    print("-" * 70)

    # Train bundle once for the dataset split and reuse across thresholds
    cached_bundle = None
    for edge in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        res = run_backtest(
            df,
            min_edge=edge,
            league_filter=league_filter,
            market_filter=market_filter,
            show_chart=False,
            bundle=cached_bundle,
        )
        if res and res.get("bundle"):
            cached_bundle = res["bundle"]
        if res and res.get("bets", 0) > 0:
            print(
                f"≥ {edge*100:4.1f}%     | {res['bets']:<8} | {res['win_rate']:6.2f}%     | {res['pnl']:+8.2f} U   | {res['roi']:+6.2f}%   | {res['max_dd']:5.2f} U"
            )


def main():
    parser = argparse.ArgumentParser(description="Cook45 Quantitative Model Tester")
    parser.add_argument("--edge", type=float, default=0.04, help="Minimum edge threshold (default: 0.04)")
    parser.add_argument("--league", type=str, default=None, help="Filter by league (e.g. EPL, La Liga, Serie A)")
    parser.add_argument("--market", type=str, default=None, choices=["AH", "OU"], help="Filter by market (AH or OU)")
    parser.add_argument("--kelly", action="store_true", help="Use Fractional Kelly Staking (default: Flat 1.0U)")
    parser.add_argument("--steam-only", action="store_true", help="Only take bets confirmed by Sharp Money / Steam")
    parser.add_argument("--sensitivity", action="store_true", help="Run multi-threshold sensitivity scan")
    parser.add_argument("--recache", action="store_true", help="Re-compute feature dataset from database")
    args = parser.parse_args()

    cache_p = ROOT_DIR / "data" / "clean_quant_dataset.pkl"
    if args.recache and cache_p.exists():
        cache_p.unlink()

    df = load_and_enrich_dataset(cache_p)
    print(f"✅ Loaded {len(df)} enriched top-tier matches with historical features.")

    if args.sensitivity:
        run_sensitivity_analysis(df, league_filter=args.league, market_filter=args.market)
    else:
        run_backtest(
            df,
            min_edge=args.edge,
            league_filter=args.league,
            market_filter=args.market,
            use_kelly=args.kelly,
            steam_only=args.steam_only,
            show_chart=True,
        )



if __name__ == "__main__":
    main()
