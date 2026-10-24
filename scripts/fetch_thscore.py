"""Utility script to fetch data via the official Thscore REST API.

Usage:
    python scripts/fetch_thscore.py --mode livescores
    python scripts/fetch_thscore.py --mode schedule --date 2026-10-24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Fix console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fbpredict.config import Config
from fbpredict.db import connect, init_db
from fbpredict.providers.thscore_api import ThscoreApiProvider


def main():
    parser = argparse.ArgumentParser(description="Fetch matches using Thscore API")
    parser.add_argument("--mode", choices=["livescores", "schedule"], default="livescores")
    parser.add_argument("--date", help="Date in YYYY-MM-DD format (for schedule mode)")
    args = parser.parse_args()

    cfg = Config.from_env()
    con = init_db(cfg.db_path)
    provider = ThscoreApiProvider(config=cfg, con=con)

    print(f"[Thscore API] Connected to {cfg.db_path}")
    if args.mode == "livescores":
        print("[Thscore API] Fetching live scores & today's fixtures...")
        matches = provider.get_livescores()
    else:
        date_str = args.date or "2026-10-24"
        print(f"[Thscore API] Fetching schedule & results for {date_str}...")
        matches = provider.get_schedule_by_date(date_str)

    print(f"\n[OK] Fetched {len(matches)} matches from Thscore API (Payload archived in SQLite raw_payload).")
    completed = [m for m in matches if m.status == -1]
    in_progress = [m for m in matches if m.status > 0]
    upcoming = [m for m in matches if m.status == 0]

    print(f"  - Completed: {len(completed)}")
    print(f"  - In-Progress (Live): {len(in_progress)}")
    print(f"  - Upcoming: {len(upcoming)}")

    print("\nSample Matches:")
    for m in matches[:10]:
        score_str = f"{m.home_goals}-{m.away_goals}" if m.home_goals is not None else "vs"
        ht_str = f"(HT: {m.ht_home}-{m.ht_away})" if m.ht_home is not None else ""
        corner_str = f"[Corners: {m.home_corner}-{m.away_corner}]" if m.home_corner is not None else ""
        print(f"  * [{m.league}] {m.home_team} {score_str} {m.away_team} {ht_str} {corner_str}")


if __name__ == "__main__":
    main()
