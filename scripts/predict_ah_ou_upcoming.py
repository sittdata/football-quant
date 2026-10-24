"""Run Asian Handicap (AH) and Over/Under (OU) predictions on upcoming matches.

Uses the frozen MultiOutput model (models/multileague_v1.pkl or models/multileague_xgb.pkl)
to predict:
1. AH Fav Cover vs AH Dog Cover
2. OU Over vs OU Under
3. Vig-free fair implied odds and edge calculations
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Fix console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fbpredict import features, labels
from fbpredict.db import connect
from fbpredict.model import ModelBundle


def run_ah_ou_predictions(model_name: str = "multileague_v1.pkl", top_n: int = 10):
    model_path = ROOT_DIR / "models" / model_name
    print(f"\n[AH/OU Predictor] Loading frozen model from: {model_path}...")
    bundle = ModelBundle.load(model_path, require_verified=True)

    con = connect(ROOT_DIR / "data" / "football.db")
    # Fetch upcoming matches with opening odds
    query = """
    SELECT match_id, league, season, kickoff_at, home_team, away_team, fav_side,
           ah_open, ah_fav_price, ah_dog_price, pool_fav_open, pool_dog_open,
           draw_ft_odds, draw_ht_odds, ou_line_open, ou_over_price, ou_under_price,
           ah_close, ou_line_close, pool_fav_close, pool_dog_close, draw_ft_close,
           home_goals, away_goals
    FROM matches
    WHERE kickoff_at >= '2026-10-24'
      AND ah_open IS NOT NULL
      AND ou_line_open IS NOT NULL
    ORDER BY kickoff_at ASC
    LIMIT 50
    """
    df_raw = pd.read_sql_query(query, con)
    con.close()

    if df_raw.empty:
        print("[AH/OU Predictor] No upcoming matches found with valid odds.")
        return

    print(f"[AH/OU Predictor] Found {len(df_raw)} upcoming matches with market odds.")
    
    # Feature preparation matching the model's feature set
    preds = bundle.predict_proba_df(df_raw)

    output_rows = []
    for idx, row in df_raw.iterrows():
        p_fav_cover = float(preds.loc[idx, "p_fav_cover"])
        p_dog_cover = 1.0 - p_fav_cover
        p_over = float(preds.loc[idx, "p_over"])
        p_under = 1.0 - p_over

        fav_side = str(row.get("fav_side") or "home")
        fav_team = row["home_team"] if fav_side == "home" else row["away_team"]
        dog_team = row["away_team"] if fav_side == "home" else row["home_team"]

        # Calculate Edge vs Bookmaker Vig-free Implied Probabilities
        ah_fav_p = float(row.get("ah_fav_price") or 1.90)
        ah_dog_p = float(row.get("ah_dog_price") or 1.90)
        fair_fav, fair_dog = labels.fair_two_way(ah_fav_p, ah_dog_p)

        ou_over_p = float(row.get("ou_over_price") or 1.90)
        ou_under_p = float(row.get("ou_under_price") or 1.90)
        fair_over, fair_under = labels.fair_two_way(ou_over_p, ou_under_p)

        edge_fav = p_fav_cover - fair_fav
        edge_dog = p_dog_cover - fair_dog
        edge_over = p_over - fair_over
        edge_under = p_under - fair_under

        # Pick best market with highest positive edge
        edges = {
            f"ต่อ {fav_team} (AH {row['ah_open']})": edge_fav,
            f"รอง {dog_team} (AH {abs(float(row['ah_open'])) if float(row['ah_open'])<0 else row['ah_open']})": edge_dog,
            f"สูง (Over {row['ou_line_open']})": edge_over,
            f"ต่ำ (Under {row['ou_line_open']})": edge_under
        }
        best_market = max(edges, key=edges.get)
        best_edge = edges[best_market]

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
            "best_market": best_market,
            "best_edge": best_edge
        })

    # Sort by Edge strength
    df_results = pd.DataFrame(output_rows).sort_values("best_edge", ascending=False)

    print("\n" + "=" * 85)
    print("🎯 UPCOMING ASIAN HANDICAP (AH) & OVER/UNDER (OU) PICKS CURATED BY AI")
    print("=" * 85)

    for i, r in df_results.head(top_n).reset_index(drop=True).iterrows():
        print(f"\n{i+1}. ⚽ [{r['league']}] {r['home']} vs {r['away']}")
        print(f"   ⏰ คิกออฟ: {r['kickoff']}")
        print(f"   📊 ตลาดแฮนดิแคป (AH {r['ah_line']}):")
        print(f"      - ต่อ {r['fav_team']}: ความน่าจะเป็น {r['p_fav']*100:.1f}%")
        print(f"      - รอง {r['dog_team']}: ความน่าจะเป็น {r['p_dog']*100:.1f}%")
        print(f"   📊 ตลาดสูงต่ำ (OU {r['ou_line']} ลูก):")
        print(f"      - สูง (Over): ความน่าจะเป็น {r['p_over']*100:.1f}%")
        print(f"      - ต่ำ (Under): ความน่าจะเป็น {r['p_under']*100:.1f}%")
        edge_sign = "+" if r['best_edge'] > 0 else ""
        print(f"   ⭐ โมเดลชี้เป้าตลาดที่ดีที่สุด (Best Edge): >> {r['best_market']} << (Edge: {edge_sign}{r['best_edge']*100:.2f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run AH & OU predictions using frozen model")
    parser.add_argument("--model", default="multileague_v2.pkl", help="Model filename (default: multileague_v2.pkl)")
    parser.add_argument("--top", type=int, default=8, help="Number of top picks")
    args = parser.parse_args()

    run_ah_ou_predictions(model_name=args.model, top_n=args.top)
