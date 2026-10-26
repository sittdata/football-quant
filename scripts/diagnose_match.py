#!/usr/bin/env python3
"""Diagnostic tool to inspect matches, paper trading bets, and settlement integrity."""

import argparse
import io
import sqlite3
import sys
from pathlib import Path

# Force UTF-8 stdout if possible on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football.db"

def inspect_match(query: str) -> None:
    if not DB_PATH.exists():
        print(f"[!] Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print(f"\n========================================================")
    print(f"[*] INSPECTING TEAM / MATCH QUERY: '{query}'")
    print(f"========================================================\n")

    # 1. Search in matches table
    sql = """
        SELECT match_id, league, kickoff_at, home_team, away_team, 
               home_goals, away_goals, ah_open, ah_fav_price, ah_dog_price,
               ou_line_open, ou_over_price, ou_under_price, provider
        FROM matches
        WHERE home_team LIKE ? OR away_team LIKE ?
        ORDER BY kickoff_at DESC
        LIMIT 10
    """
    param = f"%{query}%"
    cursor.execute(sql, (param, param))
    matches = cursor.fetchall()

    if not matches:
        print("[!] No matches found matching that team name in 'matches' table.")
    else:
        print(f"[+] Recent Matches in DB ({len(matches)} found):")
        for m in matches:
            score = f"{m['home_goals']}-{m['away_goals']}" if m['home_goals'] is not None else "N/A"
            print(f"  * ID: {m['match_id']} | [{m['league']}] {m['kickoff_at']}")
            print(f"    Teams : {m['home_team']} vs {m['away_team']}")
            print(f"    Score : {score} | Provider: {m['provider']}")
            print(f"    Lines : AH={m['ah_open']} (Fav:{m['ah_fav_price']}, Dog:{m['ah_dog_price']}) | O/U={m['ou_line_open']}")
            print("-" * 50)

    # 2. Search in paper_bets table
    print("\n[*] Paper Trading Bets on this match:")
    bet_sql = """
        SELECT bet_id, kickoff_at, league, home_team, away_team, market, 
               pick_desc, odds, edge, stake_units, status, actual_score, 
               unit_res, pnl_units, settled_at
        FROM paper_bets
        WHERE home_team LIKE ? OR away_team LIKE ?
        ORDER BY kickoff_at DESC
        LIMIT 10
    """
    cursor.execute(bet_sql, (param, param))
    bets = cursor.fetchall()

    if not bets:
        print("  (No paper bets placed on this team/match)")
    else:
        for b in bets:
            print(f"  * Bet ID: {b['bet_id']} | [{b['league']}] {b['home_team']} vs {b['away_team']}")
            print(f"    Kickoff: {b['kickoff_at']} | Pick: {b['pick_desc']} @ {b['odds']}")
            print(f"    Stake: {b['stake_units']} units | Edge: {b['edge']:.3%}")
            print(f"    Status: {b['status']} | Score: {b['actual_score']} | Res: {b['unit_res']} | PnL: {b['pnl_units']}")
            print(f"    Settled At: {b['settled_at']}")
            print("-" * 50)

    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose match and settlement status")
    parser.add_argument("query", help="Team name or keyword to inspect")
    args = parser.parse_args()
    inspect_match(args.query)
