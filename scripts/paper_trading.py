"""Quantitative Paper Trading Engine & Audit Ledger (100% Risk-Free Simulation).
Author: cook45

Manages paper trading ledger:
- Strict Top-Tier League Whitelist
- Historical Feature Enrichment (No NaNs)
- Calibrated Probability & True Edge Calculation
- Automatic Settlement & Performance Reporting

Commands:
    python scripts/paper_trading.py --scan     # Scan upcoming fixtures and log new paper bets
    python scripts/paper_trading.py --settle   # Settle finished matches and calculate P&L
    python scripts/paper_trading.py --report   # Display performance audit dashboard
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fbpredict import config, labels
from fbpredict.config import is_safe_match, is_whitelisted_league
from fbpredict.feature_engine import HistoricalFeatureStore, normalize_team_name
from fbpredict.quant_model import QuantModelBundle
from fbpredict.staking import calculate_kelly_fraction

DB_PATH = ROOT_DIR / "data" / "football.db"
MODEL_PATH = ROOT_DIR / "models" / "quant_calibrated_v1.pkl"
LEDGER_CSV = ROOT_DIR / "data" / "paper_trading_ledger.csv"


def init_db(con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_bets (
            bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            kickoff_at TEXT NOT NULL,
            league TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            market TEXT NOT NULL,
            pick_desc TEXT NOT NULL,
            odds REAL NOT NULL,
            model_prob REAL NOT NULL,
            edge REAL NOT NULL,
            stake_units REAL NOT NULL DEFAULT 1.0,
            status TEXT NOT NULL DEFAULT 'OPEN',
            actual_score TEXT,
            unit_res REAL,
            pnl_units REAL,
            settled_at TEXT,
            UNIQUE (kickoff_at, home_team, away_team, market)
        );
        """
    )
    con.commit()


def get_feature_store() -> HistoricalFeatureStore:
    return HistoricalFeatureStore(DB_PATH)


def get_model() -> QuantModelBundle:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Please run scripts/train_quant_models.py first."
        )
    return QuantModelBundle.load(MODEL_PATH)


def scan_upcoming(min_edge: float = 0.04) -> None:
    """Scan upcoming matches in database/feeds, evaluate with model, and record paper bets."""
    con = sqlite3.connect(DB_PATH)
    init_db(con)

    print("🔎 Syncing live fixtures from Thscore Feed...")
    now_utc = datetime.now(timezone.utc)
    try:
        from fbpredict.providers.thscore_feed import fetch_matches_feed
        _, live_items = fetch_matches_feed()
        cur_sync = con.cursor()
        for m in live_items:
            # Only unplayed upcoming matches with valid handicap and total lines
            if m.get("status") == 0 and m.get("ah_line") is not None and m.get("ou_line") is not None:
                h = m["home_team"]
                a = m["away_team"]
                lg = m["league"]
                k_iso = m["kickoff_at"]
                if is_safe_match(h, a, lg):
                    mid = f"thscore:{m['match_id']}"
                    cur_sync.execute(
                        """
                        INSERT OR REPLACE INTO matches (
                            match_id, provider, league, season, kickoff_at,
                            home_team, away_team, fav_side, ah_open, ah_fav_price, ah_dog_price,
                            ou_line_open, ou_over_price, ou_under_price, draw_ft_odds,
                            odds_source, result_source
                        ) VALUES (?, 'thscore_feed', ?, '2026', ?, ?, ?, 'home', ?, 1.95, 1.95, ?, 1.95, 1.95, 3.30, 'thscore', 'thscore')
                        """,
                        (mid, lg, k_iso, h, a, m["ah_line"], m["ou_line"]),
                    )
        con.commit()
    except Exception as e:
        print(f"⚠️ Live Thscore feed sync warning: {e}")

    print("🔎 Evaluating upcoming fixtures in Top European Leagues...")
    query = """
    SELECT match_id, league, kickoff_at, home_team, away_team, fav_side,
           ah_open, ah_fav_price, ah_dog_price,
           ou_line_open, ou_over_price, ou_under_price, draw_ft_odds
    FROM matches
    WHERE (home_goals IS NULL OR away_goals IS NULL)
      AND ah_open IS NOT NULL AND ou_line_open IS NOT NULL
    ORDER BY kickoff_at ASC
    """
    df = pd.read_sql_query(query, con)

    if df.empty:
        print("ℹ️ No upcoming unplayed matches found in database.")
        con.close()
        return

    store = get_feature_store()
    bundle = get_model()

    new_bets = 0
    cur = con.cursor()

    for _, r in df.iterrows():
        home = r["home_team"]
        away = r["away_team"]
        league = r["league"]
        kickoff = r["kickoff_at"]

        # CRITICAL: Match MUST be strictly in the future! Never recommend already kicked off matches!
        try:
            k_dt = pd.to_datetime(kickoff, utc=True)
            if k_dt <= now_utc:
                continue
        except Exception:
            continue

        # 1. League & Squad Safety Firewall
        if not is_safe_match(home, away, league):
            continue

        # 2. Enrich with strictly historical rolling statistics
        feat = store.enrich_match(
            home_team=home,
            away_team=away,
            kickoff_iso=kickoff,
            ah_open=r["ah_open"],
            ou_line_open=r["ou_line_open"],
            ah_fav_price=r["ah_fav_price"],
            ah_dog_price=r["ah_dog_price"],
            ou_over_price=r["ou_over_price"],
            ou_under_price=r["ou_under_price"],
            draw_ft_odds=r["draw_ft_odds"],
            fav_side=r["fav_side"],
        )

        # 3. Predict with Calibrated Quantitative Model
        pred = bundle.predict_match(feat)
        if not pred.get("safe"):
            continue

        def _price(key: str) -> float | None:
            """Quoted decimal price from the feature row, or None. Never invent one."""
            val = feat.get(key)
            try:
                f = float(val)
            except (TypeError, ValueError):
                return None
            return f if math.isfinite(f) and f > 1.0 else None

        side = str(r["fav_side"]).strip().lower() if pd.notna(r["fav_side"]) else ""
        ah_open = float(r["ah_open"]) if pd.notna(r["ah_open"]) else None
        ou_line = float(r["ou_line_open"]) if pd.notna(r["ou_line_open"]) else None

        candidates = []
        is_champ = "Championship" in league
        if is_champ:
            # Lower league (Championship): OU Over + Under + AH Favourite
            if (
                side in ("home", "away")
                and ah_open is not None
                and _price("ah_fav_price") is not None
                and pred.get("edge_fav") is not None
            ):
                candidates.append(("AH_fav", pred["edge_fav"], pred["p_fav"], _price("ah_fav_price")))
        else:
            # Top 5 European leagues: Value in AH Underdog + OU
            if (
                side in ("home", "away")
                and ah_open is not None
                and _price("ah_dog_price") is not None
                and pred.get("edge_dog") is not None
            ):
                candidates.append(("AH_dog", pred["edge_dog"], pred["p_dog"], _price("ah_dog_price")))
        if ou_line is not None and _price("ou_over_price") is not None and pred.get("edge_over") is not None:
            candidates.append(("OU_over", pred["edge_over"], pred["p_over"], _price("ou_over_price")))
        if ou_line is not None and _price("ou_under_price") is not None and pred.get("edge_under") is not None:
            candidates.append(("OU_under", pred["edge_under"], pred["p_under"], _price("ou_under_price")))

        if not candidates:
            continue

        best_mkt, best_edge, best_prob, best_odds = max(candidates, key=lambda x: x[1])

        if best_edge < min_edge:
            continue

        # Pick Description formatting
        ah_val = abs(ah_open) if ah_open is not None else None
        fav_team = home if side == "home" else away
        dog_team = away if side == "home" else home

        if best_mkt == "AH_fav" and ah_val is not None:
            pick_desc = f"ต่อ {fav_team} (-{ah_val:.2f})"
        elif best_mkt == "AH_dog" and ah_val is not None:
            pick_desc = f"รอง {dog_team} (+{ah_val:.2f})"
        elif best_mkt == "OU_over":
            pick_desc = f"สูง Over {ou_line:.2f}"
        else:
            pick_desc = f"ต่ำ Under {ou_line:.2f}"

        # Steam & RLM Alpha Flag
        if pred.get("is_steam_confirmed"):
            pick_desc = f"🔥 {pick_desc}"
        elif pred.get("is_rlm"):
            pick_desc = f"⚡ {pick_desc}"

        created_at = now_utc.isoformat()
        stake_units = calculate_kelly_fraction(prob=best_prob, decimal_odds=best_odds, fraction=0.25)

        try:
            cur.execute(
                """
                INSERT INTO paper_bets (
                    created_at, kickoff_at, league, home_team, away_team,
                    market, pick_desc, odds, model_prob, edge, stake_units, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                """,
                (
                    created_at,
                    kickoff,
                    league,
                    home,
                    away,
                    best_mkt,
                    pick_desc,
                    best_odds,
                    best_prob,
                    best_edge,
                    stake_units,
                ),
            )
            con.commit()
            new_bets += 1
            print(f"🎯 [PAPER BET LOGGED] {home} vs {away} [{league}]")
            print(f"   👉 {pick_desc} @ {best_odds:.2f} | Edge: +{best_edge*100:.1f}% | WinProb: {best_prob*100:.1f}% | Kelly Stake: {stake_units:.2f} U\n")
        except sqlite3.IntegrityError:
            pass  # Already logged

    con.close()
    print(f"✅ Scanning complete. Added {new_bets} new paper bets to ledger.")
    sync_csv_ledger()


def settle_open_bets() -> None:
    """Check scores of finished matches and settle open paper bets."""
    con = sqlite3.connect(DB_PATH)
    init_db(con)

    cur = con.cursor()
    open_bets = cur.execute(
        """
        SELECT bet_id, kickoff_at, home_team, away_team, market, odds, stake_units
        FROM paper_bets
        WHERE status = 'OPEN'
        """
    ).fetchall()

    if not open_bets:
        print("ℹ️ No open paper bets ready to settle (matches still pending or scores not updated).")
        con.close()
        return

    # Load all completed matches from database
    finished_matches = cur.execute(
        """
        SELECT kickoff_at, home_team, away_team, home_goals, away_goals, fav_side, ah_open, ou_line_open
        FROM matches
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
        """
    ).fetchall()

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
        # No stored line/price -> cannot settle this market. Leave the bet OPEN.
        if mkt in ("AH_fav", "AH_dog") and (ah_open is None or fav_side not in ("home", "away")):
            print(f"[WARN] Skipping bet {bet_id}: missing AH line for matched fixture")
            continue
        if mkt in ("OU_over", "OU_under") and ou_line is None:
            print(f"[WARN] Skipping bet {bet_id}: missing OU line for matched fixture")
            continue

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
        print(f"🏁 [{status}] {home} vs {away} -> Score: {score_str} | Result: {unit_res:+.1f} | PnL: {pnl:+.2f} units")


    con.commit()
    con.close()
    print(f"✅ Settled {settled_cnt} open bets.")
    sync_csv_ledger()


def sync_csv_ledger():
    """Export SQLite paper_bets to CSV for easy inspection."""
    con = sqlite3.connect(DB_PATH)
    init_db(con)
    df = pd.read_sql_query("SELECT * FROM paper_bets ORDER BY bet_id DESC", con)
    con.close()
    df.to_csv(LEDGER_CSV, index=False, encoding="utf-8-sig")


def print_report():
    """Display clean performance dashboard."""
    con = sqlite3.connect(DB_PATH)
    init_db(con)
    df = pd.read_sql_query("SELECT * FROM paper_bets", con)
    con.close()

    total_bets = len(df)
    settled = df[df["status"] != "OPEN"]
    open_bets = df[df["status"] == "OPEN"]

    print("\n" + "=" * 65)
    print("📊 QUANTITATIVE PAPER TRADING AUDIT DASHBOARD")
    print("=" * 65)
    print(f"Total Bets Logged   : {total_bets}")
    print(f"Open (In-Play) Bets : {len(open_bets)}")
    print(f"Settled Bets        : {len(settled)}")

    if settled.empty:
        print("\nℹ️ No settled paper bets yet. Run --settle once matches complete.")
        return

    wins = len(settled[settled["unit_res"] > 0])
    draws = len(settled[settled["unit_res"] == 0])
    losses = len(settled[settled["unit_res"] < 0])
    tot_staked = float(settled["stake_units"].sum())
    tot_pnl = float(settled["pnl_units"].sum())
    win_rate = (wins / len(settled)) * 100
    roi = (tot_pnl / tot_staked) * 100 if tot_staked > 0 else 0.0

    print("-" * 65)
    print(f"Performance Record  : 🟢 {wins} Wins | ⚪ {draws} Draws | 🔴 {losses} Losses")
    print(f"Win Rate (Hit Rate) : {win_rate:.2f}%")
    print(f"Total Units Staked  : {tot_staked:.1f} Units")
    print(f"Net Realized PnL    : {tot_pnl:+.2f} Units")
    print(f"Realized Paper ROI  : {roi:+.2f}%")
    print("=" * 65)

    print("\n📌 Recent 5 Settled Bets:")
    for _, r in settled.tail(5).iterrows():
        print(f"  • {r['home_team']} vs {r['away_team']} [{r['league']}]")
        print(f"    Pick: {r['pick_desc']} @ {r['odds']:.2f} | Score: {r['actual_score']} | PnL: {r['pnl_units']:+.2f} units\n")


def main():
    parser = argparse.ArgumentParser(description="Cook45 Quant Paper Trading Engine")
    parser.add_argument("--scan", action="store_true", help="Scan upcoming fixtures and log paper bets")
    parser.add_argument("--settle", action="store_true", help="Settle finished paper bets against real scores")
    parser.add_argument("--report", action="store_true", help="Print audit dashboard")
    parser.add_argument("--edge", type=float, default=0.04, help="Minimum edge threshold (default: 0.04)")
    args = parser.parse_args()

    if args.scan:
        scan_upcoming(min_edge=args.edge)
    elif args.settle:
        settle_open_bets()
    elif args.report:
        print_report()
    else:
        # Default behavior: run report, then scan
        print_report()


if __name__ == "__main__":
    main()
