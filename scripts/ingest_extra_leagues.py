"""Ingest Additional High-Liquidity European Leagues into SQLite.
Author: cook45

Sources: football-data.co.uk (Bet365 Asian Handicap & Over/Under closing/opening lines)
Leagues:
- Germany 2. Bundesliga (D2)
- Netherlands Eredivisie (N1)
- Portugal Primeira Liga (P1)
- Belgium Pro League (B1)
- Spain Segunda Division (SP2)
- Italy Serie B (I2)
Seasons: 2023/24, 2024/25
"""

from __future__ import annotations

import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = ROOT_DIR / "data" / "football.db"


def safe_float(v: Any) -> float | None:
    if v is None or pd.isna(v):
        return None
    try:
        f = float(str(v).strip())
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None

LEAGUES = [
    # Top 5 European Leagues & Championship
    ("E0", "England Premier League"),
    ("E1", "England Championship"),
    ("SP1", "Spain La Liga"),
    ("I1", "Italy Serie A"),
    ("D1", "Germany Bundesliga"),
    ("F1", "France Ligue 1"),
    # Second Tier & Top Secondary Liquidity Leagues
    ("D2", "Germany 2. Bundesliga"),
    ("N1", "Netherlands Eredivisie"),
    ("P1", "Portugal Primeira Liga"),
    ("B1", "Belgium Pro League"),
    ("SP2", "Spain Segunda Division"),
    ("I2", "Italy Serie B"),
]

SEASONS = ["2223", "2324", "2425", "2526", "2627"]


def parse_kickoff(date_str: str, time_str: str | None = None) -> str:
    """Parse date DD/MM/YYYY or DD/MM/YY and time into ISO UTC."""
    if pd.isna(date_str) or not str(date_str).strip():
        return datetime.now(timezone.utc).isoformat()
    d_part = str(date_str).strip()
    t_part = str(time_str).strip() if pd.notna(time_str) and str(time_str).strip() else "15:00"

    for fmt in ["%d/%m/%Y %H:%M", "%d/%m/%y %H:%M", "%Y-%m-%d %H:%M"]:
        try:
            dt = datetime.strptime(f"{d_part} {t_part}", fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            pass
    return f"{d_part}T{t_part}:00+00:00"


def ingest_league(con: sqlite3.Connection, code: str, league_name: str, season_code: str) -> int:
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{code}.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return 0

    if df.empty or "HomeTeam" not in df.columns:
        return 0

    cur = con.cursor()
    inserted = 0

    for idx, r in df.iterrows():
        home = str(r["HomeTeam"]).strip()
        away = str(r["AwayTeam"]).strip()
        if not home or not away or pd.isna(r.get("FTHG")) or pd.isna(r.get("FTAG")):
            continue

        h_goals = int(r["FTHG"])
        a_goals = int(r["FTAG"])
        ht_h = int(r["HTHG"]) if pd.notna(r.get("HTHG")) else None
        ht_a = int(r["HTAG"]) if pd.notna(r.get("HTAG")) else None

        kickoff_iso = parse_kickoff(r.get("Date"), r.get("Time"))
        season_str = f"{season_code[:2]}/{season_code[2:]}"
        match_id = f"fduk:{code}_{season_code}_{idx}"

        # Odds extraction
        draw_odds = safe_float(r.get("B365D"))

        ou_over = safe_float(r.get("B365>2.5"))
        ou_under = safe_float(r.get("B365<2.5"))
        has_ou = ou_over is not None and ou_under is not None
        ou_line = 2.50 if has_ou else None
        ou_line_close = 2.50 if has_ou else None

        # Asian Handicap Line (AHh is Home line)
        ahh = safe_float(r.get("AHh"))
        b365_ahh = safe_float(r.get("B365AHH"))
        b365_aha = safe_float(r.get("B365AHA"))
        has_ah = ahh is not None and b365_ahh is not None and b365_aha is not None

        if has_ah:
            if ahh < 0:
                fav_side = "home"
                ah_open = ahh
                ah_fav_price = b365_ahh
                ah_dog_price = b365_aha
            elif ahh > 0:
                fav_side = "away"
                ah_open = -ahh  # Normalize favourite line to negative
                ah_fav_price = b365_aha
                ah_dog_price = b365_ahh
            else:
                # Level ball (0.0)
                if b365_ahh <= b365_aha:
                    fav_side = "home"
                    ah_open = 0.0
                    ah_fav_price = b365_ahh
                    ah_dog_price = b365_aha
                else:
                    fav_side = "away"
                    ah_open = 0.0
                    ah_fav_price = b365_aha
                    ah_dog_price = b365_ahh

            # Closing odds extraction: store None when missing; never fabricate no-move close from open
            ahch = safe_float(r.get("AHCh"))
            c_ahh = safe_float(r.get("B365CAHH")) or safe_float(r.get("PCAHH"))
            c_aha = safe_float(r.get("B365CAHA")) or safe_float(r.get("PCAHA"))

            if fav_side == "home":
                ah_close = ahch
                pool_fav_close = c_ahh
                pool_dog_close = c_aha
            else:
                ah_close = -ahch if ahch is not None else None
                pool_fav_close = c_aha
                pool_dog_close = c_ahh
        else:
            fav_side = None
            ah_open = None
            ah_fav_price = None
            ah_dog_price = None
            ah_close = None
            pool_fav_close = None
            pool_dog_close = None

        cur.execute(
            """
            INSERT OR REPLACE INTO matches (
                match_id, provider, league, season, kickoff_at,
                home_team, away_team, fav_side,
                ah_open, ah_fav_price, ah_dog_price,
                ou_line_open, ou_over_price, ou_under_price, draw_ft_odds,
                ah_close, ou_line_close, pool_fav_close, pool_dog_close,
                home_goals, away_goals, ht_home, ht_away,
                odds_source, result_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_id,
                "football-data.co.uk",
                league_name,
                season_str,
                kickoff_iso,
                home,
                away,
                fav_side,
                ah_open,
                ah_fav_price,
                ah_dog_price,
                ou_line,
                ou_over,
                ou_under,
                draw_odds,
                ah_close,
                ou_line_close,
                pool_fav_close,
                pool_dog_close,
                h_goals,
                a_goals,
                ht_h,
                ht_a,
                "football-data.co.uk/bet365",
                "football-data.co.uk",
            ),
        )
        inserted += 1

    con.commit()
    return inserted


def main():
    print("=" * 70)
    print("🚀 INGESTING ADDITIONAL HIGH-LIQUIDITY LEAGUES (football-data.co.uk)")
    print("=" * 70)

    con = sqlite3.connect(DB_PATH)
    total_new = 0

    for code, name in LEAGUES:
        for s in SEASONS:
            print(f"⏳ Ingesting {name} ({code}) Season {s}...")
            cnt = ingest_league(con, code, name, s)
            print(f"   ✅ Saved {cnt} matches.")
            total_new += cnt

    con.close()
    print("=" * 70)
    print(f"🎉 INGESTION COMPLETE: Added {total_new} matches to {DB_PATH.name}!")
    print("=" * 70)


if __name__ == "__main__":
    main()
