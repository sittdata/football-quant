"""Automated Match Settlement & Result Sync Daemon.
Author: cook45

Syncs real finished scores from Thscore and database to settle open paper bets:
1. Queries pending 'OPEN' bets in paper_bets table.
2. Checks completed results in football.db and Thscore feeds (schedule.aspx & livescores).
3. Calculates exact unit outcomes and updates ledger with Kelly-scaled P&L.
4. Syncs paper_trading_ledger.csv.

Usage:
    python scripts/auto_settle_feed.py --now
    python scripts/auto_settle_feed.py --daemon --interval 300
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fbpredict import config, labels
from fbpredict.config import Config
from fbpredict.feature_engine import normalize_team_name
from fbpredict.providers.thscore_api import ThscoreApiProvider
from fbpredict.providers.thscore_feed import fetch_matches_feed

DB_PATH = ROOT_DIR / "data" / "football.db"
LEDGER_CSV = ROOT_DIR / "data" / "paper_trading_ledger.csv"


def sync_finished_scores_from_thscore(con: sqlite3.Connection) -> int:
    """Fetch recent completed matches from Thscore and update football.db."""
    updated = 0
    cur = con.cursor()

    # 1. Check live feed for completed matches today
    try:
        leagues_map, live_items = fetch_matches_feed()
        for m in live_items:
            if m["status"] == -1 and m["home_goals"] is not None and m["away_goals"] is not None:
                h_norm = normalize_team_name(m["home_team"]).lower()
                a_norm = normalize_team_name(m["away_team"]).lower()
                if not h_norm or not a_norm or len(h_norm) < 2 or len(a_norm) < 2:
                    continue

                h_g = int(m["home_goals"])
                a_g = int(m["away_goals"])

                # Update in matches table
                res = cur.execute(
                    """
                    UPDATE matches
                    SET home_goals = ?, away_goals = ?, ht_home = ?, ht_away = ?
                    WHERE home_goals IS NULL
                      AND (
                          (LOWER(home_team) LIKE ? OR ? LIKE '%' || LOWER(home_team) || '%')
                          AND (LOWER(away_team) LIKE ? OR ? LIKE '%' || LOWER(away_team) || '%')
                      )
                    """,
                    (h_g, a_g, m["ht_home"], m["ht_away"], f"%{h_norm}%", h_norm, f"%{a_norm}%", a_norm),
                )
                if res.rowcount > 0:
                    updated += res.rowcount
        con.commit()
    except Exception as e:
        print(f"⚠️ Live feed check warning: {e}")

    # 2. Check Thscore Schedule for open bets dates
    open_dates = [
        r[0][:10]
        for r in cur.execute("SELECT DISTINCT kickoff_at FROM paper_bets WHERE status = 'OPEN'").fetchall()
        if r[0]
    ]

    cfg = Config.from_env()
    provider = ThscoreApiProvider(config=cfg, con=con)

    for d_str in open_dates:
        try:
            matches = provider.get_schedule_by_date(d_str)
            for m in matches:
                if m.status == -1 and m.home_goals is not None and m.away_goals is not None:
                    h_norm = normalize_team_name(m.home_team).lower()
                    a_norm = normalize_team_name(m.away_team).lower()
                    if not h_norm or not a_norm or len(h_norm) < 2 or len(a_norm) < 2:
                        continue

                    h_g = int(m.home_goals)
                    a_g = int(m.away_goals)

                    res = cur.execute(
                        """
                        UPDATE matches
                        SET home_goals = ?, away_goals = ?, ht_home = ?, ht_away = ?
                        WHERE home_goals IS NULL
                          AND (
                              (LOWER(home_team) LIKE ? OR ? LIKE '%' || LOWER(home_team) || '%')
                              AND (LOWER(away_team) LIKE ? OR ? LIKE '%' || LOWER(away_team) || '%')
                          )
                        """,
                        (h_g, a_g, m.ht_home, m.ht_away, f"%{h_norm}%", h_norm, f"%{a_norm}%", a_norm),
                    )
                    if res.rowcount > 0:
                        updated += res.rowcount
            con.commit()
        except Exception as e:
            pass

    return updated


def settle_open_paper_bets(con: sqlite3.Connection) -> int:
    """Evaluate and settle all OPEN paper bets where match scores exist."""
    cur = con.cursor()
    open_bets = cur.execute(
        """
        SELECT bet_id, kickoff_at, home_team, away_team, market, odds, stake_units
        FROM paper_bets
        WHERE status = 'OPEN'
        """
    ).fetchall()

    if not open_bets:
        return 0

    min_date = min((b[1][:10] for b in open_bets if b[1]), default="")
    query = """
        SELECT kickoff_at, home_team, away_team, home_goals, away_goals, fav_side, ah_open, ou_line_open
        FROM matches
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
    """
    if min_date:
        query += " AND kickoff_at >= ?"
        finished_matches = cur.execute(query, (min_date,)).fetchall()
    else:
        finished_matches = cur.execute(query).fetchall()

    settled_cnt = 0
    now_str = datetime.now(timezone.utc).isoformat()

    for bet in open_bets:
        bet_id, kickoff, home, away, mkt, odds, stake_units = bet
        stake = float(stake_units) if stake_units else 1.0
        bet_date = kickoff[:10] if kickoff else ""
        h_norm = normalize_team_name(home).lower()
        a_norm = normalize_team_name(away).lower()

        matched_m = None
        for m in finished_matches:
            m_kickoff, m_home, m_away, m_hg, m_ag, m_fav, m_ah, m_ou = m
            m_date = m_kickoff[:10] if m_kickoff else ""
            if m_date != bet_date:
                continue

            mh_norm = normalize_team_name(m_home).lower()
            ma_norm = normalize_team_name(m_away).lower()

            home_match = (home == m_home) or (h_norm and h_norm == mh_norm) or (h_norm and len(h_norm) >= 3 and h_norm in mh_norm) or (mh_norm and len(mh_norm) >= 3 and mh_norm in h_norm)
            away_match = (away == m_away) or (a_norm and a_norm == ma_norm) or (a_norm and len(a_norm) >= 3 and a_norm in ma_norm) or (ma_norm and len(ma_norm) >= 3 and ma_norm in a_norm)

            if home_match and away_match:
                matched_m = m
                break

        if not matched_m:
            continue

        _, _, _, h_g, a_g, fav_side, ah_open, ou_line = matched_m
        h_g = int(h_g)
        a_g = int(a_g)
        score_str = f"{h_g}:{a_g}"
        tot_g = h_g + a_g
        fav_diff = (h_g - a_g) if fav_side == "home" else (a_g - h_g)

        ah_u = labels.ah_unit(fav_diff, ah_open)
        ou_u = labels.ou_unit(tot_g, ou_line, "over")

        unit_res = 0.0
        if mkt == "AH_fav":
            unit_res = ah_u
        elif mkt == "AH_dog":
            unit_res = -ah_u
        elif mkt == "OU_over":
            unit_res = ou_u
        elif mkt == "OU_under":
            unit_res = -ou_u

        if unit_res == 1.0:
            status = "WON"
            pnl = (odds - 1.0) * stake
        elif unit_res == 0.5:
            status = "HALF_WON"
            pnl = (odds - 1.0) * stake * 0.5
        elif unit_res == -1.0:
            status = "LOST"
            pnl = -stake
        elif unit_res == -0.5:
            status = "HALF_LOST"
            pnl = -stake * 0.5
        else:
            status = "PUSH"
            pnl = 0.0


        cur.execute(
            """
            UPDATE paper_bets
            SET status = ?,
                actual_score = ?,
                unit_res = ?,
                pnl_units = ?,
                settled_at = ?
            WHERE bet_id = ?
            """,
            (status, score_str, unit_res, round(pnl, 3), now_str, bet_id),
        )
        settled_cnt += 1
        print(f"🏁 [SETTLED] {home} vs {away} -> Score: {score_str} | {status} ({pnl:+.2f} U)")

    con.commit()
    return settled_cnt


def sync_csv_ledger():
    """Export SQLite paper_bets to CSV for audit trail."""
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM paper_bets ORDER BY bet_id DESC", con)
    con.close()
    if not df.empty:
        df.to_csv(LEDGER_CSV, index=False, encoding="utf-8")


def run_cycle() -> dict[str, int]:
    con = sqlite3.connect(DB_PATH, timeout=30.0)
    # Check open bets count
    open_cnt = con.execute("SELECT count(*) FROM paper_bets WHERE status = 'OPEN'").fetchone()[0]
    if open_cnt == 0:
        con.close()
        return {"matches_synced": 0, "bets_settled": 0}

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Checking results for {open_cnt} OPEN paper bets...")
    updated_scores = sync_finished_scores_from_thscore(con)
    settled = settle_open_paper_bets(con)
    con.close()

    if settled > 0:
        sync_csv_ledger()
        print(f"✅ Settle cycle complete. Settled {settled} bets.")

    return {"matches_synced": updated_scores, "bets_settled": settled}


def main():
    parser = argparse.ArgumentParser(description="Cook45 Auto Settle Daemon")
    parser.add_argument("--now", action="store_true", help="Run a single settlement pass now")
    parser.add_argument("--daemon", action="store_true", help="Run continuous background sync")
    parser.add_argument("--interval", type=int, default=300, help="Poll interval in seconds (default: 300)")
    args = parser.parse_args()

    if args.daemon:
        print(f"🚀 Auto-Settle Daemon started (Polling every {args.interval}s)...")
        while True:
            try:
                run_cycle()
            except Exception as e:
                print(f"⚠️ Error in sync cycle: {e}")
            time.sleep(args.interval)
    else:
        run_cycle()
        print("✅ Finished settlement pass.")


if __name__ == "__main__":
    main()
