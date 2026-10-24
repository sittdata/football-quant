"""Predict upcoming matches script.

Fetches scheduled upcoming fixtures from 5DollarFootballAPI for allowed leagues,
prices them using per-fixture odds, and runs forward inference using multileague_v1.pkl.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from fbpredict import db, features, labels, model
from fbpredict.config import load_config
from fbpredict.providers.five_dollar import FiveDollarFootballProvider, to_match_row
from fbpredict.providers.base import ProviderError


LEAGUE_MAP = {
    "England Premier League": 4160026622,
    "Spain La Liga": 4212821298,
    "France Ligue 1": 3614399544,
}


def fetch_upcoming_matches(provider, league_id: int, bookmaker: str = "bet365", limit: int = 10) -> list[dict]:
    # Use order='desc' to get the latest/upcoming fixtures
    data = provider.league_fixtures(league_id, per_page=35, order="desc", include=None)
    fixtures = data.get("data", [])
    
    upcoming_rows = []
    
    for f in fixtures:
        if f.get("status") not in ("scheduled", "NS", "timed", "fixture"):
            continue
            
        row = to_match_row(f, bookmaker=bookmaker, season="2025", fetched_at=db.utcnow_iso())
        if row is None:
            continue
            
        # Fetch odds for this fixture
        fix_id = f.get("id")
        try:
            odds_resp = provider.fixture_odds(fix_id)
            if odds_resp and "data" in odds_resp:
                bms = odds_resp["data"].get("bookmakers", [])
                target_bm = next((b for b in bms if b.get("slug") == bookmaker), None)
                if not target_bm and bms:
                    target_bm = bms[0]
                if target_bm:
                    odds = target_bm.get("odds", {})
                    # 1X2
                    o_1x2 = odds.get("1x2", {}).get("opening") or {}
                    row["draw_ft_odds"] = o_1x2.get("draw")
                    
                    # Determine fav/dog
                    h_price = o_1x2.get("home")
                    a_price = o_1x2.get("away")
                    
                    # AH
                    o_ah = odds.get("asian_handicap", {}).get("opening") or {}
                    ah_line = o_ah.get("line")
                    if ah_line is not None:
                        row["ah_open"] = float(ah_line)
                        # Home/Away handicap prices
                        h_ah = o_ah.get("home")
                        a_ah = o_ah.get("away")
                        if float(ah_line) <= 0: # Home is fav (gives goals or 0)
                            row["fav_side"] = "home"
                            row["ah_fav_price"] = h_ah
                            row["ah_dog_price"] = a_ah
                            row["pool_fav_open"] = h_price
                            row["pool_dog_open"] = a_price
                        else: # Away is fav
                            row["fav_side"] = "away"
                            row["ah_fav_price"] = a_ah
                            row["ah_dog_price"] = h_ah
                            row["pool_fav_open"] = a_price
                            row["pool_dog_open"] = h_price
                            row["ah_open"] = -float(ah_line) # standardize favourite handicap as negative
                    
                    # OU (goal_line in 5dollar API)
                    o_ou = odds.get("goal_line", {}).get("opening") or odds.get("over_under", {}).get("opening") or {}
                    if o_ou.get("line") is not None:
                        row["ou_line_open"] = float(o_ou.get("line"))
                        row["ou_over_price"] = o_ou.get("over")
                        row["ou_under_price"] = o_ou.get("under")
        except Exception as e:
            continue
            
        # Check if fully priced
        if (
            row.get("ah_fav_price") and row.get("ah_dog_price") and
            row.get("ou_over_price") and row.get("ou_under_price") and
            row.get("ah_open") is not None and row.get("ou_line_open") is not None
        ):
            upcoming_rows.append(row)
            if len(upcoming_rows) >= limit:
                break
                
    return upcoming_rows


def main():
    cfg = load_config()
    cfg.edge_threshold = 0.06
    cfg.blacklist_markets = ("OU_over",)
    
    print("=" * 95)
    print("FORWARD PREDICTION: Next Upcoming Matches across White-listed Leagues")
    print(f"Model: models/multileague_v1.pkl | Edge Threshold: {cfg.edge_threshold:.0%}")
    print("Allowed Leagues:", cfg.allowed_leagues)
    print("=" * 95)
    
    bundle = model.ModelBundle.load("models/multileague_v1.pkl")
    con = db.init_db(cfg.db_path)
    provider = FiveDollarFootballProvider(cfg, con=con)
    
    # Load historical matches for computing rolling team form
    history_matches = db.fetch_matches_df(con, include_synthetic=False, require_result=True)
    
    all_upcoming = []
    for lg_name in cfg.allowed_leagues:
        lg_id = LEAGUE_MAP.get(lg_name)
        if not lg_id:
            continue
        print(f"Fetching upcoming priced fixtures for {lg_name}...")
        rows = fetch_upcoming_matches(provider, lg_id, bookmaker=cfg.bookmaker_slug, limit=6)
        print(f"  -> Found {len(rows)} priced upcoming matches.")
        all_upcoming.extend(rows)
        
    con.close()
    
    if not all_upcoming:
        print("\nNo upcoming matches with active betting prices found at this moment.")
        return
        
    up_df = pd.DataFrame(all_upcoming).drop_duplicates(subset="match_id", keep="first")

    # Combine with history to calculate rolling team form. The database already
    # holds these same fixtures (they are fetched ahead of kickoff), so the
    # historical copy of any upcoming match_id is dropped: keeping both produced
    # two rows per fixture and, because the rolling stats differ between them,
    # two contradictory picks for the same match.
    hist_only = history_matches[~history_matches["match_id"].isin(set(up_df["match_id"]))]
    full_df = pd.concat([hist_only, up_df], ignore_index=True)
    full_df["kickoff_at"] = pd.to_datetime(full_df["kickoff_at"], utc=True)
    full_df = full_df.sort_values("kickoff_at", kind="mergesort").reset_index(drop=True)
    
    # Predict probabilities
    preds = bundle.predict_proba_df(full_df)
    full_df["p_fav_cover"] = preds["p_fav_cover"]
    full_df["p_over"] = preds["p_over"]
    
    # Filter back to upcoming
    upcoming_pred = full_df[full_df["match_id"].isin(up_df["match_id"])].copy()
    
    from fbpredict import formatter
    from telegram_notify import send_message, configured as tg_configured, default_chat_id

    card_items = []
    skipped_items = []

    for _, row in upcoming_pred.iterrows():
        pick = model.select_market(
            row=row,
            p_fav_cover=float(row["p_fav_cover"]),
            p_over=float(row["p_over"]),
            threshold=cfg.edge_threshold,
            blacklist_markets=cfg.blacklist_markets,
            allowed_markets=cfg.allowed_markets,
        )
        
        details = formatter.parse_pick_details(row, pick)
        
        if pick.market and pick.market != "skip":
            card_items.append(details)
        else:
            skipped_items.append(details)

    # Print Terminal Box Cards
    print("\n" + "=" * 72)
    print("🎯  UPCOMING MATCHES PREDICTION — VALUE BETTING CARDS")
    print(f"    Edge Threshold: {cfg.edge_threshold:.0%} | Allowed: {', '.join(cfg.allowed_markets or ['ALL'])}")
    print("=" * 72)

    if card_items:
        terminal_report = formatter.format_matches_report(card_items, fmt="terminal")
        print(terminal_report)
    else:
        print("\n  [i] ไม่มีคู่แข่งขันที่มี Edge เกินเกณฑ์ ณ เวลานี้ (ข้ามทุกคู่)")

    print("\n" + "=" * 72)
    print(f"สรุปผล: พบทีเด็ดที่เข้าเกณฑ์ {len(card_items)} คู่ จากการวิเคราะห์ทั้งหมด {len(upcoming_pred)} คู่")
    print("=" * 72 + "\n")

    # Save to reports/upcoming_picks.md
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "upcoming_picks.md"

    md_content = [
        "# ⚽ รายงานวิเคราะห์ทีเด็ดบอลล่วงหน้า (Match Prediction Cards)",
        f"**เวลาประมวลผล:** {formatter.format_thai_kickoff(datetime.now(timezone.utc))} | **เกณฑ์ Edge:** `{cfg.edge_threshold:.0%}`\n",
        f"**จำนวนคู่ที่แนะนำ:** `{len(card_items)}` คู่ (จากทั้งหมด `{len(upcoming_pred)}` คู่)\n",
        formatter.format_matches_report(card_items, fmt="markdown") if card_items else "_ไม่มีคู่ที่เข้าเกณฑ์_",
    ]
    report_file.write_text("\n".join(md_content), encoding="utf-8")
    print(f"บันทึกรายงานฉบับเต็มไว้ที่: {report_file}")

    # If --telegram argument is supplied, send message
    if "--telegram" in sys.argv:
        if tg_configured():
            chat_id = default_chat_id()
            if chat_id:
                header = (
                    f"🎯 <b>ทีเด็ดบอลล่วงหน้า (AI Value Picks)</b>\n"
                    f"วิเคราะห์พบ <b>{len(card_items)}</b> คู่ที่มีความคุ้มค่าสูง:\n\n"
                )
                tg_body = formatter.format_matches_report(card_items, fmt="telegram")
                send_message(header + tg_body, chat_id=chat_id)
                print("ส่งรายงานเข้า Telegram เรียบร้อยแล้ว")
        else:
            print("[!] Telegram ไม่ได้ตั้งค่า Token ใน .env")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()

