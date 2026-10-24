"""Analyze and recommend upcoming matches using trained ML models on Thscore API data.

Usage:
    python scripts/recommend_upcoming_matches.py --mode livescores --top 10
    python scripts/recommend_upcoming_matches.py --mode schedule --date 2026-10-24 --top 15
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import requests

# Fix console encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "dsa_project" / "models"
DATA_DIR = ROOT_DIR / "dsa_project" / "data"
API_KEY = "B95794317BA548F2"


def load_all_models():
    models = {}
    
    # 1. Corner regressor
    corner_path = MODELS_DIR / "thscore_corner_regressor.pkl"
    if corner_path.exists():
        with open(corner_path, "rb") as f:
            models["corner"] = pickle.load(f)

    # 2. HT Goal classifier (Over 0.5 HT)
    ht_path = MODELS_DIR / "thscore_ht_goal_classifier.pkl"
    if ht_path.exists():
        with open(ht_path, "rb") as f:
            models["ht_goal"] = pickle.load(f)

    # 3. Total Goals regressor
    goals_path = MODELS_DIR / "best_total_goals_regressor.pkl"
    if goals_path.exists():
        with open(goals_path, "rb") as f:
            models["goals"] = pickle.load(f)

    # 4. BTTS classifier
    btts_path = MODELS_DIR / "best_classifier_target_btts.pkl"
    if btts_path.exists():
        with open(btts_path, "rb") as f:
            models["btts"] = pickle.load(f)

    return models


def load_league_stats() -> dict[str, dict]:
    """Compute empirical baseline statistics per league from historical data."""
    csv_path = DATA_DIR / "thscore_multi_league_matches.csv"
    if not csv_path.exists():
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        stats = {}
        global_ht = float(df["has_ht_goal"].mean()) if "has_ht_goal" in df else 0.72
        global_corners = float(df["total_corners"].dropna().mean()) if "total_corners" in df else 9.6
        global_goals = float(df["total_goals"].mean()) if "total_goals" in df else 2.65
        global_btts = float(df["btts"].mean()) if "btts" in df else 0.50

        for league, group in df.groupby("league"):
            stats[league] = {
                "ht_rate": float(group["has_ht_goal"].mean()) if len(group) >= 2 else global_ht,
                "corners_avg": float(group["total_corners"].dropna().mean()) if len(group.dropna(subset=["total_corners"])) >= 2 else global_corners,
                "goals_avg": float(group["total_goals"].mean()) if len(group) >= 2 else global_goals,
                "btts_rate": float(group["btts"].mean()) if len(group) >= 2 else global_btts,
            }
        stats["__global__"] = {
            "ht_rate": global_ht,
            "corners_avg": global_corners,
            "goals_avg": global_goals,
            "btts_rate": global_btts,
        }
        return stats
    except Exception as e:
        print(f"[Warning] Failed to load league stats: {e}")
        return {}


def get_fixtures(mode: str = "livescores", date_str: str = "2026-10-24") -> list[dict]:
    if mode == "livescores":
        url = f"http://api2.thscore.info/football_th/livescores.aspx?api_key={API_KEY}"
    else:
        url = f"http://api2.thscore.info/football_th/schedule.aspx?date={date_str}&api_key={API_KEY}"
    
    try:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception as e:
        print(f"Failed to fetch fixtures from Thscore API: {e}")
        return []

    # status 0 = upcoming (not started yet)
    upcoming = [m for m in data if m.get("status") == 0]
    return upcoming


def evaluate_upcoming(mode: str = "livescores", date_str: str = "2026-10-24", top_n: int = 10, filter_league: str | None = None):
    models = load_all_models()
    league_stats = load_league_stats()
    global_defaults = league_stats.get("__global__", {
        "ht_rate": 0.72,
        "corners_avg": 9.6,
        "goals_avg": 2.65,
        "btts_rate": 0.50,
    })

    print(f"\n[AI Scanner] Requesting Thscore fixtures (mode={mode}, date={date_str})...")
    fixtures = get_fixtures(mode=mode, date_str=date_str)
    if not fixtures:
        print("[AI Scanner] No upcoming matches found.")
        return

    print(f"[AI Scanner] Fetched {len(fixtures)} upcoming fixtures. Running inference...")

    recommendations = []
    major_keywords = [
        "พรีเมียร์", "ลาลีกา", "บุนเดส", "กัลโช่", "เซเรีย", "ลีกเอิง", "แชมเปี้ยนชิพ", 
        "เจลีก", "เนเธอร์แลนด์", "โปรตุเกส", "เบลเยียม", "สก็อต", "ยูฟ่า", "ไทย", "สเปน"
    ]

    for m in fixtures:
        league = m.get("leagueName") or m.get("leagueShortName") or "Unknown"
        home = m.get("homeName", "")
        away = m.get("awayName", "")
        kickoff = m.get("matchTime", "")

        if filter_league and filter_league.lower() not in league.lower():
            continue

        is_major = any(k in league.lower() for k in major_keywords)

        # Lookup stats
        l_stat = league_stats.get(league, global_defaults)
        league_ht_rate = l_stat["ht_rate"]
        league_goals_avg = l_stat["goals_avg"]
        league_corner_avg = l_stat["corners_avg"]
        league_btts_rate = l_stat["btts_rate"]

        # 1. HT Goal Model
        # features: ["league_ht_rate", "league_goals_avg", "home_red", "away_red"]
        p_ht_goal = league_ht_rate
        if "ht_goal" in models:
            scaler = models["ht_goal"]["scaler"]
            clf = models["ht_goal"]["model"]
            ht_feat = np.array([[league_ht_rate, league_goals_avg, 0, 0]])
            scaled_x = scaler.transform(ht_feat)
            p_ht_goal = float(clf.predict_proba(scaled_x)[0, 1])

        # 2. Corner Regressor Model
        # features: ["league_corner_avg", "league_goals_avg", "home_red", "away_red", "total_reds"]
        pred_corners = league_corner_avg
        if "corner" in models:
            scaler = models["corner"]["scaler"]
            reg = models["corner"]["model"]
            corner_feat = np.array([[league_corner_avg, league_goals_avg, 0, 0, 0]])
            scaled_x = scaler.transform(corner_feat)
            pred_corners = float(reg.predict(scaled_x)[0])

        recommendations.append({
            "matchId": m.get("matchId"),
            "league": league,
            "home": home,
            "away": away,
            "kickoff": kickoff,
            "p_ht_goal": p_ht_goal,
            "pred_corners": pred_corners,
            "p_btts": league_btts_rate,
            "is_major": 1 if is_major else 0,
        })

    if not recommendations:
        print("[AI Scanner] No matches matched filter criteria.")
        return

    # Sort: Major leagues first, then highest HT Goal probability
    df_rec = pd.DataFrame(recommendations).sort_values(
        by=["is_major", "p_ht_goal"], 
        ascending=[False, False]
    )

    print("\n" + "=" * 85)
    print(f"🎯 THSCORE AI MODEL PREDICTIONS: UPCOMING MATCHES ({mode.upper()})")
    print("=" * 85)

    top_picks = df_rec.head(top_n).reset_index(drop=True)
    for i, row in top_picks.iterrows():
        badge = "⭐ [Major League]" if row["is_major"] == 1 else "📌"
        print(f"\n{i+1}. {badge} ⚽ [{row['league']}] {row['home']} vs {row['away']}")
        print(f"   ⏰ เวลาเตะ: {row['kickoff']}")
        print(f"   🎯 ประตูครึ่งแรก (HT Over 0.5): โอกาสสูงถึง {row['p_ht_goal']*100:.1f}%")
        print(f"   🚩 คาดการณ์ลูกเตะมุมรวม (Expected Corners): ~{row['pred_corners']:.1f} ลูก")
        print(f"   🔥 โอกาสยิงทั้งสองฝั่ง (BTTS): {row['p_btts']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Run predictions on upcoming Thscore matches")
    parser.add_argument("--mode", choices=["livescores", "schedule"], default="livescores")
    parser.add_argument("--date", default="2026-10-24", help="Date in YYYY-MM-DD")
    parser.add_argument("--top", type=int, default=10, help="Top N picks")
    parser.add_argument("--league", default=None, help="League filter")
    args = parser.parse_args()

    evaluate_upcoming(mode=args.mode, date_str=args.date, top_n=args.top, filter_league=args.league)


if __name__ == "__main__":
    main()
