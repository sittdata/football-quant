"""Thscore Full-Stack Real-Time Predictor & Odds Pipeline.

Integrates:
1. Real-time Asian Handicap & Over/Under lines & water prices directly from Thscore (SBOBET/Bet365/Crown)
2. In-game stats (Half-Time scores, corners, cards, tempo)
3. Inference via trained multileague_v2.pkl (MultiOutput XGBoost)
4. Vig-free Fair Odds and Positive Edge (+Edge %) calculation

Usage:
    python scripts/thscore_fullstack_predictor.py --company 31 --top 10
    python scripts/thscore_fullstack_predictor.py --company 8 --save-db
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

# Fix console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fbpredict import db, features, labels
from fbpredict.db import connect
from fbpredict.model import ModelBundle
from fbpredict.providers.thscore_feed import get_thscore_live_data, ThscoreLiveMatch


def run_fullstack_pipeline(
    company_id: int = 31,
    top_n: int = 10,
    min_edge: float = 0.05,
    capital: float = 0.0,
    save_db: bool = False,
):
    print("=" * 85)
    print(f"🚀 THSCORE FULL-STACK PIPELINE (Company ID: {company_id} - SBOBET/Bet365/Crown)")
    if capital > 0:
        print(f"💰 พอร์ตเป้าหมายวันนี้: {capital:,.2f} บาท")
    print("=" * 85)

    print("\n[1/4] Fetching live match feeds & Asian bookmaker odds directly from Thscore...")
    live_matches = get_thscore_live_data(company_id=company_id)
    print(f"[OK] Fetched {len(live_matches)} matches.")

    # Filter matches with valid AH and OU lines
    valid_matches = [m for m in live_matches if m.ah_line is not None and m.ou_line is not None]
    print(f"[OK] {len(valid_matches)} matches have active Asian Handicap and Over/Under market lines.")

    if not valid_matches:
        print("[Notice] No matches currently open with complete AH/OU odds.")
        return

    # Convert to DataFrame matching model input format
    rows = []
    for m in valid_matches:
        # Convert Asian/Hong Kong price format to Decimal odds (+ 1.0) if price < 1.5
        def to_decimal(p, default=1.90):
            if p is None:
                return default
            return (p + 1.0) if p <= 1.5 else p

        h_price = to_decimal(m.ah_home_price, 1.90)
        a_price = to_decimal(m.ah_away_price, 1.90)
        o_price = to_decimal(m.ou_over_price, 1.90)
        u_price = to_decimal(m.ou_under_price, 1.90)

        # In Asian Handicap: negative or 0 line means home is fav, positive means away is fav
        fav_side = "home" if m.ah_line <= 0 else "away"
        fav_price = h_price if fav_side == "home" else a_price
        dog_price = a_price if fav_side == "home" else h_price

        rows.append({
            "match_id": f"thscore:{m.match_id}",
            "provider": "thscore_feed",
            "league": m.league,
            "season": "2026",
            "kickoff_at": m.kickoff_at,
            "home_team": m.home_team,
            "away_team": m.away_team,
            "fav_side": fav_side,
            "ah_open": float(m.ah_line),
            "ah_fav_price": float(fav_price),
            "ah_dog_price": float(dog_price),
            "ou_line_open": float(m.ou_line),
            "ou_over_price": float(o_price),
            "ou_under_price": float(u_price),
            "pool_fav_open": m.pool_home if fav_side == "home" else m.pool_away,
            "pool_dog_open": m.pool_away if fav_side == "home" else m.pool_home,
            "draw_ft_odds": m.pool_draw or 3.20,
            "draw_ht_odds": 2.10,
            "home_goals": m.home_goals,
            "away_goals": m.away_goals,
            "ht_home": m.ht_home,
            "ht_away": m.ht_away,
            "home_corner": m.home_corner,
            "away_corner": m.away_corner,
            "home_red": m.home_red,
            "away_red": m.away_red,
            "home_yellow": m.home_yellow,
            "away_yellow": m.away_yellow,
        })

    df_feed = pd.DataFrame(rows)

    # Optional: Save into SQLite
    if save_db:
        print("\n[2/4] Archiving live Thscore matches into local SQLite (data/football.db)...")
        con = connect(ROOT_DIR / "data" / "football.db")
        cur = con.cursor()
        saved = 0
        for _, r in df_feed.iterrows():
            try:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO matches (
                        match_id, provider, league, season, kickoff_at, home_team, away_team,
                        fav_side, ah_open, ah_fav_price, ah_dog_price, pool_fav_open, pool_dog_open,
                        draw_ft_odds, ou_line_open, ou_over_price, ou_under_price,
                        ht_home, ht_away, home_goals, away_goals
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r["match_id"], r["provider"], r["league"], r["season"], r["kickoff_at"],
                        r["home_team"], r["away_team"], r["fav_side"], r["ah_open"],
                        r["ah_fav_price"], r["ah_dog_price"], r["pool_fav_open"], r["pool_dog_open"],
                        r["draw_ft_odds"], r["ou_line_open"], r["ou_over_price"], r["ou_under_price"],
                        r["ht_home"], r["ht_away"], r["home_goals"], r["away_goals"]
                    )
                )
                saved += 1
            except Exception:
                continue
        con.commit()
        con.close()
        print(f"[OK] Synced {saved} matches to SQLite.")

    # [3/4] Run multileague_v2 inference
    model_path = ROOT_DIR / "models" / "multileague_v2.pkl"
    print(f"\n[3/4] Loading trained MultiOutput model from: {model_path}...")
    bundle = ModelBundle.load(model_path, require_verified=True)

    print("[3/4] Generating predictions and calculating Vig-free Market Edges...")
    preds = bundle.predict_proba_df(df_feed)

    output_rows = []
    for idx, row in df_feed.iterrows():
        p_fav_cover = float(preds.loc[idx, "p_fav_cover"])
        p_dog_cover = 1.0 - p_fav_cover
        p_over = float(preds.loc[idx, "p_over"])
        p_under = 1.0 - p_over

        fav_team = row["home_team"] if row["fav_side"] == "home" else row["away_team"]
        dog_team = row["away_team"] if row["fav_side"] == "home" else row["home_team"]

        # Vig-free fair implied probabilities
        fair_fav, fair_dog = labels.fair_two_way(row["ah_fav_price"], row["ah_dog_price"])
        fair_over, fair_under = labels.fair_two_way(row["ou_over_price"], row["ou_under_price"])

        edge_fav = p_fav_cover - fair_fav
        edge_dog = p_dog_cover - fair_dog
        edge_over = p_over - fair_over
        edge_under = p_under - fair_under

        markets = {
            f"ต่อ {fav_team} (AH {row['ah_open']})": (edge_fav, p_fav_cover, row["ah_fav_price"]),
            f"รอง {dog_team} (AH {abs(float(row['ah_open'])) if float(row['ah_open'])<0 else row['ah_open']})": (edge_dog, p_dog_cover, row["ah_dog_price"]),
            f"สูง (Over {row['ou_line_open']})": (edge_over, p_over, row["ou_over_price"]),
            f"ต่ำ (Under {row['ou_line_open']})": (edge_under, p_under, row["ou_under_price"]),
        }

        # Pick best market
        best_name = max(markets, key=lambda k: markets[k][0])
        best_edge, best_p, best_price = markets[best_name]

        output_rows.append({
            "league": row["league"],
            "home": row["home_team"],
            "away": row["away_team"],
            "kickoff": row["kickoff_at"],
            "fav_team": fav_team,
            "dog_team": dog_team,
            "ah_line": row["ah_open"],
            "p_fav": p_fav_cover,
            "p_dog": p_dog_cover,
            "ou_line": row["ou_line_open"],
            "p_over": p_over,
            "p_under": p_under,
            "ht_home": row["ht_home"],
            "ht_away": row["ht_away"],
            "corners": f"{row['home_corner']}-{row['away_corner']}" if row['home_corner'] is not None else "N/A",
            "best_market": best_name,
            "best_edge": best_edge,
            "best_p": best_p,
            "best_price": best_price,
        })

    # Filter & Sort
    df_results = pd.DataFrame(output_rows).sort_values("best_edge", ascending=False)
    filtered = df_results[df_results["best_edge"] >= min_edge].reset_index(drop=True)

    print("\n" + "=" * 85)
    print(f"🎯 TOP PICKS WITH HIGHEST POSITIVE EDGE (Thscore Asian Odds + In-Game Dynamics)")
    print("=" * 85)

    display_set = filtered.head(top_n) if not filtered.empty else df_results.head(top_n)
    stake_per_bet = (capital / len(display_set)) if (capital > 0 and len(display_set) > 0) else 0.0

    if stake_per_bet > 0:
        print(f"📊 แผนเดิมพันวันนี้: ทุนรวม {capital:,.1f} บาท | คัด {len(display_set)} คู่ | ลงคู่ละ >> {stake_per_bet:,.1f} บาทเท่ากัน <<\n")

    for i, r in display_set.reset_index(drop=True).iterrows():
        print(f"\n{i+1}. ⚽ [{r['league']}] {r['home']} vs {r['away']}")
        print(f"   ⏰ เวลาเตะ: {r['kickoff']}")
        if r["ht_home"] is not None:
            print(f"   📊 สถานะเกมสด: HT {r['ht_home']}-{r['ht_away']} | เตะมุม {r['corners']}")
        print(f"   📊 ตลาดแฮนดิแคป (AH {r['ah_line']}): ต่อ {r['fav_team']} ({r['p_fav']*100:.1f}%) | รอง {r['dog_team']} ({r['p_dog']*100:.1f}%)")
        print(f"   📊 ตลาดสูงต่ำ (OU {r['ou_line']} ลูก): สูง ({r['p_over']*100:.1f}%) | ต่ำ ({r['p_under']*100:.1f}%)")
        edge_sign = "+" if r['best_edge'] > 0 else ""
        print(f"   ⭐ โมเดลชี้เป้าตลาดที่ดีที่สุด (Best Edge): >> {r['best_market']} << (Edge: {edge_sign}{r['best_edge']*100:.2f}%)")
        if stake_per_bet > 0:
            potential_payout = stake_per_bet * r['best_price']
            print(f"   💵 แนะนำเดิมพัน: ลง {stake_per_bet:,.1f} บาท (ค่าน้ำ {r['best_price']:.2f}) -> ถ้าชนะรับกลับ {potential_payout:,.1f} บาท")


def main():
    parser = argparse.ArgumentParser(description="Run Full-Stack Thscore Live Odds & Predictor")
    parser.add_argument("--company", type=int, default=31, help="Bookmaker company ID (31: SBOBET, 8: Bet365, 3: Crown)")
    parser.add_argument("--top", type=int, default=8, help="Number of top picks")
    parser.add_argument("--min-edge", type=float, default=0.08, help="Minimum positive edge threshold")
    parser.add_argument("--capital", type=float, default=0.0, help="Total betting capital in THB (e.g. 500)")
    parser.add_argument("--save-db", action="store_true", help="Sync matches to SQLite database")
    args = parser.parse_args()

    run_fullstack_pipeline(
        company_id=args.company,
        top_n=args.top,
        min_edge=args.min_edge,
        capital=args.capital,
        save_db=args.save_db,
    )


if __name__ == "__main__":
    main()
