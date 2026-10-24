"""Daily Quant Betting Card Generator + Telegram Auto-Push.

Runs the full pipeline:
1. Fetch today's upcoming matches from Thscore CDN (free, no API key)
2. Run multileague_v2 model inference on all matches
3. Filter picks with Positive Edge >= 8%
4. Format a clean betting card grouped by time slots
5. Push the card to Telegram via Bot API

Usage (manual):
    python scripts/daily_telegram_card.py

Designed to be triggered by Windows Task Scheduler at 17:30 daily.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")

from fbpredict.providers.thscore_feed import get_thscore_live_data
from fbpredict.model import ModelBundle
from fbpredict import labels

# ── Telegram Config ─────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_ALLOWED_CHAT_ID", "")


def send_telegram(text: str, parse_mode: str = "HTML") -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] TELEGRAM_BOT_TOKEN or TELEGRAM_ALLOWED_CHAT_ID not set in .env")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # Telegram limit is 4096 chars per message
    chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": parse_mode,
        }, timeout=15)
        if resp.status_code != 200:
            print(f"[ERROR] Telegram API returned {resp.status_code}: {resp.text}")
            return False
    return True


def generate_picks(min_edge: float = 0.08, capital: float = 500.0) -> str:
    """Run full inference pipeline and return formatted betting card text."""

    print("[1/4] Fetching live matches from Thscore CDN...")
    matches = get_thscore_live_data(company_id=31)
    upcoming = [m for m in matches if m.status == 0
                and m.ah_line is not None and m.ou_line is not None]
    print(f"[OK] {len(upcoming)} upcoming matches with SBOBET odds.")

    # Build DataFrame for model
    rows = []
    for m in upcoming:
        def to_dec(p):
            if p is None:
                return 1.90
            return (p + 1.0) if p <= 1.5 else p

        h_p, a_p = to_dec(m.ah_home_price), to_dec(m.ah_away_price)
        o_p, u_p = to_dec(m.ou_over_price), to_dec(m.ou_under_price)

        if m.ah_line > 0:
            fav_side, ah_fav = "home", -float(m.ah_line)
            fav_price, dog_price = h_p, a_p
        elif m.ah_line < 0:
            fav_side, ah_fav = "away", float(m.ah_line)
            fav_price, dog_price = a_p, h_p
        else:
            fav_side, ah_fav = "home", 0.0
            fav_price, dog_price = h_p, a_p

        rows.append({
            "match_id": f"thscore:{m.match_id}", "provider": "thscore_feed",
            "league": m.league, "season": "2026", "kickoff_at": m.kickoff_at,
            "home_team": m.home_team, "away_team": m.away_team,
            "fav_side": fav_side, "ah_open": ah_fav,
            "ah_fav_price": fav_price, "ah_dog_price": dog_price,
            "ou_line_open": float(m.ou_line),
            "ou_over_price": o_p, "ou_under_price": u_p,
            "pool_fav_open": 1.90, "pool_dog_open": 3.60,
            "draw_ft_odds": 3.20, "draw_ht_odds": 2.10,
            "home_goals": None, "away_goals": None,
            "ht_home": None, "ht_away": None,
            "home_corner": None, "away_corner": None,
            "home_red": None, "away_red": None,
            "home_yellow": None, "away_yellow": None,
            "home_rank": m.home_rank, "away_rank": m.away_rank,
            "weather": m.weather, "referee": m.referee,
            "b365_ah_line": m.b365_ah_line,
            "b365_ah_home": m.b365_ah_home,
            "b365_ah_away": m.b365_ah_away,
            "b365_ou_line": m.b365_ou_line,
            "b365_ou_over": m.b365_ou_over,
            "b365_ou_under": m.b365_ou_under,
        })

    df = pd.DataFrame(rows)

    # Archive upcoming matches with odds to football.db for future model retraining
    try:
        from fbpredict.db import connect, upsert_matches
        (ROOT_DIR / "data").mkdir(parents=True, exist_ok=True)
        db_con = connect(ROOT_DIR / "data" / "football.db")
        upsert_matches(db_con, rows)
        db_con.close()
        print(f"[DB] Archived {len(rows)} upcoming matches to football.db for retraining pipeline.")
    except Exception as e:
        print(f"[DB Warning] Could not archive to DB: {e}")

    print("[2/4] Loading model and running inference...")
    bundle = ModelBundle.load(ROOT_DIR / "models" / "multileague_v2.pkl", require_verified=True)
    preds = bundle.predict_proba_df(df)

    print("[3/4] Calculating edges and filtering picks...")
    picks = []
    for idx, r in df.iterrows():
        p_fav = float(preds.loc[idx, "p_fav_cover"])
        p_dog = 1.0 - p_fav
        p_over = float(preds.loc[idx, "p_over"])
        p_under = 1.0 - p_over
        fair_fav, fair_dog = labels.fair_two_way(r["ah_fav_price"], r["ah_dog_price"])
        fair_over, fair_under = labels.fair_two_way(r["ou_over_price"], r["ou_under_price"])

        fav_team = r["home_team"] if r["fav_side"] == "home" else r["away_team"]
        dog_team = r["away_team"] if r["fav_side"] == "home" else r["home_team"]
        ah_val = abs(r["ah_open"])

        mkts = [
            (f"ต่อ {fav_team} (-{ah_val:.2f})", p_fav - fair_fav, p_fav, r["ah_fav_price"]),
            (f"รอง {dog_team} (+{ah_val:.2f})", p_dog - fair_dog, p_dog, r["ah_dog_price"]),
            (f"สูง Over {r['ou_line_open']}", p_over - fair_over, p_over, r["ou_over_price"]),
            (f"ต่ำ Under {r['ou_line_open']}", p_under - fair_under, p_under, r["ou_under_price"]),
        ]
        best = max(mkts, key=lambda x: x[1])

        utc_dt = datetime.strptime(r["kickoff_at"], "%Y-%m-%d %H:%M:%S")
        thai_dt = utc_dt + timedelta(hours=7)

        rank_str = ""
        if (pd.notna(r.get("home_rank")) and r.get("home_rank")) or (pd.notna(r.get("away_rank")) and r.get("away_rank")):
            hr = f"#{r['home_rank']}" if r.get("home_rank") else "-"
            ar = f"#{r['away_rank']}" if r.get("away_rank") else "-"
            rank_str = f" [{hr} vs {ar}]"

        sharp_note = ""
        if "ต่อ" in best[0] and pd.notna(r.get("b365_ah_home")):
            b_p = to_dec(r["b365_ah_home"])
            if b_p < r["ah_fav_price"] - 0.03:
                sharp_note = f" ⚡B365={b_p:.2f}"
        elif "รอง" in best[0] and pd.notna(r.get("b365_ah_away")):
            b_p = to_dec(r["b365_ah_away"])
            if b_p < r["ah_dog_price"] - 0.03:
                sharp_note = f" ⚡B365={b_p:.2f}"

        if best[1] >= min_edge:
            picks.append({
                "time": thai_dt.strftime("%H:%M"),
                "dt": thai_dt,
                "hour": thai_dt.hour,
                "league": r["league"],
                "home": r["home_team"],
                "away": r["away_team"],
                "rank_str": rank_str,
                "sharp_note": sharp_note,
                "pick": best[0],
                "prob": best[2],
                "edge": best[1],
                "odds": best[3],
            })

    picks_df = pd.DataFrame(picks).sort_values(by=["dt", "edge"], ascending=[True, False])
    total = len(picks_df)
    stake = capital / min(total, 20) if total > 0 else 0

    print(f"[OK] {total} picks with Edge >= {min_edge * 100:.0f}%")

    # ── Format Telegram Message ─────────────────────────────────────────
    today_str = datetime.now().strftime("%d/%m/%Y")
    lines = [
        f"🏆 <b>ใบโพย Quant AI — {today_str}</b>",
        f"📊 คัด {total} คู่ (Edge ≥ {min_edge * 100:.0f}%) | ทุน {capital:,.0f} บ. | ไม้ละ {stake:,.0f} บ.",
        "",
    ]

    # Group by time slot
    def slot_label(h):
        if 14 <= h < 18:
            return "🌅 บ่าย (14:00–18:00)"
        if 18 <= h < 21:
            return "🌆 หัวค่ำ (18:00–21:00)"
        if 21 <= h < 24:
            return "🌙 รอบดึก (21:00–00:00)"
        return "🌃 ดึกมาก (00:00–06:00)"

    current_slot = None
    for i, (_, r) in enumerate(picks_df.head(20).iterrows()):
        sl = slot_label(r["hour"])
        if sl != current_slot:
            current_slot = sl
            lines.append(f"\n<b>━━ {sl} ━━</b>")

        edge_bar = "🔥" if r["edge"] >= 0.12 else ("⭐" if r["edge"] >= 0.10 else "✅")
        lines.append(
            f"{edge_bar} <b>{r['time']}</b> [{r['league']}]\n"
            f"   {r['home']} vs {r['away']}{r['rank_str']}\n"
            f"   👉 <b>{r['pick']}</b>{r['sharp_note']}\n"
            f"   Win {r['prob'] * 100:.0f}% | Edge +{r['edge'] * 100:.1f}% | น้ำ {r['odds']:.2f}"
        )

    lines.append(f"\n💰 แผน: ลง {min(total, 20)} คู่ x {stake:,.0f} บ. = {capital:,.0f} บ.")
    lines.append(f"🎯 เป้าหมาย: ถูก 14/20 → กำไร ~{14 * stake * 0.90 - 6 * stake:,.0f} บ.")
    lines.append("\n⚙️ Powered by Multileague AI v2 + Thscore CDN")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("🤖 DAILY QUANT BETTING CARD GENERATOR + TELEGRAM PUSH")
    print("=" * 60)

    card_text = generate_picks(min_edge=0.08, capital=500.0)

    # Print to console
    print("\n[4/4] Sending to Telegram...")
    ok = send_telegram(card_text)

    if ok:
        print("[SUCCESS] ✅ ใบโพยถูกส่งไปที่ Telegram เรียบร้อยแล้ว!")
    else:
        print("[FALLBACK] Printing card to console instead:")
        # Strip HTML for console display
        import re
        clean = re.sub(r"<[^>]+>", "", card_text)
        print(clean)


if __name__ == "__main__":
    main()
