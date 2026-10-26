"""Understat xG & Advanced Metrics Ingestion Script.
Author: cook45

Usage:
    python scripts/fetch_understat_xg.py --league EPL --season 2024
    python scripts/fetch_understat_xg.py --all-leagues --season 2024
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fbpredict.providers.understat import (
    fetch_understat_league_data,
    parse_understat_league_payload,
    store_understat_matches,
)

SUPPORTED_LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]
DB_PATH = ROOT_DIR / "data" / "football.db"


def ingest_league(con: sqlite3.Connection, league: str, season: int) -> int:
    print(f"📥 Fetching Understat data: {league} (Season: {season}/{season+1})...")
    try:
        payload = fetch_understat_league_data(league, season)
        records = parse_understat_league_payload(payload, league, season)
        if not records:
            print(f"  ⚠️ No finished matches found for {league} {season}")
            return 0
        n = store_understat_matches(con, records)
        print(f"  ✅ Stored {n} matches with xG, xGA, PPDA & xPTS into football.db")
        return n
    except Exception as e:
        print(f"  ❌ Error fetching {league} {season}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Ingest Understat Advanced xG & Pressing Stats")
    parser.add_argument("--league", choices=SUPPORTED_LEAGUES, default="EPL", help="League name")
    parser.add_argument("--season", type=int, default=2024, help="Single season starting year")
    parser.add_argument("--seasons", type=int, nargs="+", default=None, help="List of seasons (e.g. 2021 2022 2023 2024)")
    parser.add_argument("--all-leagues", action="store_true", help="Fetch all 5 major European leagues")
    args = parser.parse_args()

    seasons_to_run = args.seasons if args.seasons else [args.season]
    leagues_to_run = SUPPORTED_LEAGUES if args.all_leagues else [args.league]

    con = sqlite3.connect(DB_PATH)
    try:
        total = 0
        for s in seasons_to_run:
            for lg in leagues_to_run:
                total += ingest_league(con, lg, s)
        print(f"\n🎉 Finished ingestion! Total matches processed & saved: {total}")
    finally:
        con.close()


if __name__ == "__main__":
    main()

