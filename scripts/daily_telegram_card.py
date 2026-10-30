"""Daily Quant Betting Card Generator + Telegram Auto-Push.

Runs the automated daily pipeline:
1. Scan active & upcoming global fixtures across Pinnacle Sharp Odds and SofaScore
2. Compute Shin Vig-Removal true probabilities, insider trading parameter z, and field momentum
3. Filter alpha opportunities with positive edge
4. Format daily betting card in HTML
5. Dispatch via Telegram Bot API or print to console if no Telegram token configured.

Usage:
    python scripts/daily_telegram_card.py
"""

from __future__ import annotations

import html
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")

from fbpredict.syndicate_engine import SyndicateComboEngine, SyndicateSignal

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_ALLOWED_CHAT_ID", "").strip()


def _redact(value: str) -> str:
    """Strip the bot token from any string before it reaches logs or stdout."""
    if TELEGRAM_BOT_TOKEN:
        return value.replace(TELEGRAM_BOT_TOKEN, "<redacted>")
    return value


def _safe_cut_index(s: str, max_pos: int) -> int:
    """Find a cut position <= max_pos that avoids bisecting an HTML tag."""
    if len(s) <= max_pos:
        return len(s)
    last_open = s.rfind("<", 0, max_pos)
    last_close = s.rfind(">", 0, max_pos)
    if last_open > last_close and last_open > 0:
        return last_open
    return max_pos


def send_telegram(text: str, parse_mode: str = "HTML") -> bool:
    """Send text message to Telegram Bot API with clean paragraph chunking. Returns True on success."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[INFO] Telegram credentials not configured. Printing card to console.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # Split cleanly by newlines to avoid cutting HTML tags in half
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_len = 0

    max_chunk = 3800
    for line in lines:
        # A single line longer than the limit must be split, respecting tag boundaries
        while len(line) > max_chunk:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            cut = _safe_cut_index(line, max_chunk)
            if cut <= 0:
                cut = max_chunk
            chunks.append(line[:cut])
            line = line[cut:]
        if current_len + len(line) + 1 > max_chunk and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_len = len(line)
        else:
            current_chunk.append(line)
            current_len += len(line) + 1
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    for chunk in chunks:
        try:
            payload: dict[str, Any] = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code != 200 and parse_mode:
                # If Telegram rejected the chunk (e.g. 400 tag mismatch), re-send plain text
                retry_payload = {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": chunk,
                }
                resp = requests.post(url, json=retry_payload, timeout=20)
            if resp.status_code != 200:
                print(f"[ERROR] Telegram API failed ({resp.status_code}): {_redact(resp.text)}")
                return False
        except requests.RequestException as e:
            # Never log str(e): the request URL carries the bot token.
            print(f"[ERROR] Telegram connection error: {type(e).__name__}")
            return False
    return True


def generate_daily_card(limit: int = 12) -> str:
    """Scan market opportunities and build HTML card."""
    engine = SyndicateComboEngine()
    signals: list[SyndicateSignal] = engine.scan_opportunities(max_fixtures=limit)

    bkk_time = datetime.now(timezone(timedelta(hours=7)))
    date_str = bkk_time.strftime("%d/%m/%Y %H:%M")

    lines = [
        f"🏆 <b>SYNDICATE QUANT BETTING CARD — {date_str} (UTC+7)</b>",
        f"📊 <i>Sharp Pinnacle Odds + Shin Vig Removal + SofaScore Momentum</i>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    alpha_signals = [s for s in signals if s.verdict != "NEUTRAL" and s.edge_percent > 0]
    alpha_ids = {id(s) for s in alpha_signals}
    neutral_signals = [s for s in signals if id(s) not in alpha_ids]

    if alpha_signals:
        lines.append(f"\n🔥 <b>ACTIONABLE ALPHA PICKS ({len(alpha_signals)} Matches)</b>\n")
        for s in alpha_signals:
            status = f"🔴 LIVE {s.current_minute}' [{html.escape(str(s.current_score))}]" if s.is_live else "⏳ UPCOMING"
            h_team = html.escape(s.home_team)
            a_team = html.escape(s.away_team)
            league = html.escape(s.league)
            rec_mkt = html.escape(s.recommended_market)
            rationale = html.escape(s.rationale)

            lines.append(f"<b>{h_team} vs {a_team}</b> ({league})")
            lines.append(f"  • สถานะ: {status}")
            lines.append(f"  • แนะนำ: <b>{rec_mkt}</b>")
            lines.append(
                f"  • Edge (+EV): <b>{s.edge_percent:+.2f}%</b> | Quarter-Kelly: <b>{s.quarter_kelly_stake_percent:.2f}%</b>"
            )
            lines.append(f"  • Shin Insider z: {s.shin_insider_z*100:.1f}% | Market Vig: {s.market_vig_percent:.1f}%")
            lines.append(f"  • เหตุผล: <i>{rationale}</i>")
            lines.append("")
    else:
        lines.append("\nℹ️ <i>ไม่พบ Actionable Alpha ที่มี Edge เหนือ Vig ในรอบสแกนนี้</i>\n")

    if neutral_signals:
        lines.append(f"📋 <b>MARKET PRICED FIXTURES ({len(neutral_signals)} Matches)</b>")
        for s in neutral_signals[:6]:
            odds_str = "/".join(f"{o:.2f}" for o in s.pinnacle_quoted_odds)
            h_team = html.escape(s.home_team)
            a_team = html.escape(s.away_team)
            lines.append(f"  • {h_team} vs {a_team} | ราคา 1X2: {odds_str} | Vig: {s.market_vig_percent:.1f}%")

    lines.append("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("⚙️ <i>Powered by Antigravity Syndicate Quant Engine</i>")

    return "\n".join(lines)


def main() -> None:
    print("=" * 70)
    print("🤖 DAILY QUANT BETTING CARD GENERATOR + TELEGRAM PUSH")
    print("=" * 70)

    try:
        card_text = generate_daily_card()
    except Exception as e:
        print(f"[ERROR] Failed to generate card: {e}")
        card_text = f"⚠️ <b>Daily Quant Card Alert</b>: Scan error: {html.escape(str(e))}"

    # Try telegram dispatch
    sent = send_telegram(card_text)
    if sent:
        print("[SUCCESS] ✅ Betting card dispatched to Telegram successfully!")
    else:
        # Strip HTML and print cleanly to terminal
        print("\n--- BETTING CARD TEXT (STDOUT FALLBACK) ---")
        clean_text = re.sub(r"<[^>]+>", "", card_text)
        print(clean_text)
        print("------------------------------------------")


if __name__ == "__main__":
    main()
