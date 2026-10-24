"""Cook45 Telegram Commander & Quant AI Bot.

Interactive 24/7 Telegram Assistant:
- Run on-demand 20-pick Quant cards (/card, /picks)
- Trigger model retraining from phone (/retrain)
- Check system, database, and schedule status (/status)
- Search any team name to get instant AI match analysis
- Chat naturally with cook45 persona directly from bed

Usage:
    python telegram_bot.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import html
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

# Console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from fbpredict import labels
from fbpredict.config import Config
from fbpredict.db import connect
from fbpredict.model import ModelBundle
from fbpredict.providers.thscore_feed import get_thscore_live_data
from scripts.daily_telegram_card import generate_picks
from telegram_notify import API_ROOT, bot_token, default_chat_id, send_message

LOGS_DIR = PROJECT_ROOT / "logs"
BOT_LOG_PATH = LOGS_DIR / "bot.log"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_running: set[str] = set()
_running_lock = threading.Lock()


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(BOT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def allowed_chat_ids() -> set[str]:
    raw = os.getenv("TELEGRAM_ALLOWED_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID") or ""
    return {part.strip() for part in raw.replace(";", ",").split(",") if part.strip()}


def api_call(method: str, params: dict[str, Any] | None = None, timeout: float = 40.0) -> dict[str, Any]:
    token = bot_token()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
    reply = requests.get(
        API_ROOT.format(token=token, method=method),
        params=params or {},
        timeout=timeout,
    )
    payload = reply.json()
    if not payload.get("ok"):
        raise RuntimeError(f"{method}: {payload.get('description') or reply.status_code}")
    return payload["result"]


def get_updates(offset: int | None, timeout: int = 25) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"timeout": timeout, "allowed_updates": json.dumps(["message"])}
    if offset is not None:
        params["offset"] = offset
    return api_call("getUpdates", params, timeout=timeout + 15)


def cmd_help() -> str:
    return (
        "🤖 <b>cook45 Quant Commander</b>\n"
        "<i>ระบบคุมโมเดล AI ผ่านมือถือ 24 ชม.</i>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "📌 <b>คำสั่งด่วน:</b>\n"
        "  • <code>/card</code> หรือ <code>/picks</code> — ดึงโพย 20 คู่เด็ด (Edge ≥ 8%) ทันที\n"
        "  • <code>/status</code> — เช็กสถานะโมเดล, ฐานข้อมูล, และตารางเวลา\n"
        "  • <code>/retrain</code> — สั่ง Retrain โมเดล XGBoost ใหม่ด้วยผลล่าสุด\n"
        "  • <code>/help</code> — ดูคำสั่งทั้งหมด\n\n"
        "💬 <b>หรือพิมพ์ชื่อทีมมาคุยได้เลย:</b>\n"
        "  เช่น พิมพ์ <code>เชลซี</code>, <code>แมนยู</code>, <code>มาดริด</code>\n"
        "  กูจะส่องราคาและวิเคราะห์ Edge ของคู่นั้นให้ดูสดๆ ทันที!\n"
        "━━━━━━━━━━━━━━━━━━━━"
    )


def cmd_status() -> str:
    # 1. DB count
    con = connect(PROJECT_ROOT / "data" / "football.db")
    cur = con.cursor()
    total_db = cur.execute("SELECT count(*) FROM matches").fetchone()[0]
    finished_db = cur.execute("SELECT count(*) FROM matches WHERE home_goals IS NOT NULL").fetchone()[0]
    pending_db = cur.execute("SELECT count(*) FROM matches WHERE home_goals IS NULL").fetchone()[0]
    con.close()

    # 2. Model meta
    meta_path = PROJECT_ROOT / "models" / "multileague_v2.meta.json"
    meta = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    trained_rows = meta.get("metadata", {}).get("trained_rows", "N/A")
    features_count = len(meta.get("feature_columns", []))
    sha_short = meta.get("sha256", "N/A")[:16] + "..." if meta.get("sha256") else "N/A"

    now_thai = (datetime.now(timezone.utc) + timedelta(hours=7)).strftime("%d/%m/%Y %H:%M น.")

    return (
        f"📊 <b>สถานะระบบ Quant AI (cook45)</b>\n"
        f"📅 เวลาปัจจุบัน: {now_thai}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 <b>โมเดล AI (Multileague v2):</b>\n"
        f"  • สถาปัตยกรรม : XGBoost MultiOutput\n"
        f"  • จำนวนคู่ที่เทรน : <b>{trained_rows:,}</b> คู่\n"
        f"  • ฟีเจอร์ที่ใช้ : {features_count} มิติ (HT + Corners + Standings)\n"
        f"  • Hash SHA-256 : <code>{sha_short}</code>\n\n"
        f"🗄 <b>ฐานข้อมูล SQLite (data/football.db):</b>\n"
        f"  • ข้อมูลทั้งหมด : <b>{total_db:,}</b> คู่\n"
        f"  • แข่งจบแล้ว (มีผล) : {finished_db:,} คู่\n"
        f"  • บันทึกรอตรวจผล : {pending_db} คู่\n\n"
        f"⏰ <b>ตารางทำงานอัตโนมัติ (Task Scheduler):</b>\n"
        f"  • ยิงโพยเข้ามือถือ : <b>17:30 น. ทุกวัน</b>\n"
        f"  • Auto-Retrain  : <b>05:00 น. ทุกวันจันทร์</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🟢 สถานะ: <b>Online & พร้อมรบ</b>"
    )


def run_card_generation(chat_id: str) -> None:
    try:
        send_message("🔍 กำลังดึงราคาสดจาก Thscore CDN และรันโมเดล Multileague v2 สักครู่...", chat_id)
        card_text = generate_picks(min_edge=0.08, capital=500.0)
        # Split in chunks if needed
        chunks = [card_text[i : i + 4000] for i in range(0, len(card_text), 4000)]
        for chunk in chunks:
            send_message(chunk, chat_id)
        log("card generation sent via telegram command")
    except Exception as exc:
        send_message(f"❌ เกิดข้อผิดพลาดในการดึงโพย: {html.escape(str(exc))}", chat_id)
        log(f"card gen exception: {exc}\n{traceback.format_exc()}")
    finally:
        with _running_lock:
            _running.discard("card")


def cmd_card(chat_id: str) -> None:
    with _running_lock:
        if "card" in _running:
            send_message("⏳ กำลังคำนวณโพยอยู่แล้วเพื่อน รอสักแป๊บเดียว...", chat_id)
            return
        _running.add("card")
    thread = threading.Thread(target=run_card_generation, args=(chat_id,), daemon=True)
    thread.start()


def run_retrain_task(chat_id: str) -> None:
    try:
        send_message("🚀 เริ่มต้นระบบ Auto-Retrain Pipeline (อัปเดตผลบอลล่าสุด + Walk-Forward Test)...", chat_id)
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "auto_retrain_pipeline.py"), "--force"]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120)
        if res.returncode == 0:
            send_message("✅ <b>Auto-Retrain สำเร็จเรียบร้อย!</b> รายงานผลถูกส่งเข้าห้องแชทแล้ว", chat_id)
        else:
            tail = (res.stderr or res.stdout or "").strip()[-1000:]
            send_message(f"⚠️ Retrain มีปัญหา (rc={res.returncode}):\n<pre>{html.escape(tail)}</pre>", chat_id)
    except Exception as e:
        send_message(f"❌ Retrain ล้มเหลว: {html.escape(str(e))}", chat_id)
    finally:
        with _running_lock:
            _running.discard("retrain")


def cmd_retrain(chat_id: str) -> None:
    with _running_lock:
        if "retrain" in _running:
            send_message("⏳ กำลังรัน Retrain อยู่แล้วเพื่อน รอแป๊บ...", chat_id)
            return
        _running.add("retrain")
    thread = threading.Thread(target=run_retrain_task, args=(chat_id,), daemon=True)
    thread.start()


def analyze_team_query(team_query: str) -> str:
    """Search upcoming matches for team name and run instant prediction."""
    query = team_query.strip().lower()
    matches = get_thscore_live_data(company_id=31)
    upcoming = [
        m for m in matches
        if m.status == 0 and m.ah_line is not None and m.ou_line is not None
        and (query in m.home_team.lower() or query in m.away_team.lower() or query in m.league.lower())
    ]

    if not upcoming:
        return (
            f"🤔 ค้นหาคำว่า <b>'{html.escape(team_query)}'</b> ไม่พบคู่ที่กำลังจะเตะในวันนี้สัส!\n\n"
            f"💡 อาจเป็นเพราะ:\n"
            f"  1. วันนี้ไม่มีเตะ หรือยังไม่เปิดราคา\n"
            f"  2. ลองพิมพ์ชื่อสั้นลง เช่น <code>เชลซี</code>, <code>แมนยู</code>, <code>มาดริด</code>\n"
            f"  3. หรือกด <code>/card</code> เพื่อดู 20 คู่ที่มี Edge สูงสุดคืนนี้ไปเลย!"
        )

    # Build DF and run inference
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
            "match_id": f"thscore:{m.match_id}",
            "provider": "thscore_feed",
            "league": m.league,
            "season": "2026",
            "kickoff_at": m.kickoff_at,
            "home_team": m.home_team,
            "away_team": m.away_team,
            "fav_side": fav_side,
            "ah_open": ah_fav,
            "ah_fav_price": fav_price,
            "ah_dog_price": dog_price,
            "ou_line_open": float(m.ou_line),
            "ou_over_price": o_p,
            "ou_under_price": u_p,
            "pool_fav_open": 1.90,
            "pool_dog_open": 3.60,
            "draw_ft_odds": 3.20,
            "draw_ht_odds": 2.10,
            "home_goals": None,
            "away_goals": None,
            "ht_home": None,
            "ht_away": None,
            "home_corner": None,
            "away_corner": None,
            "home_red": None,
            "away_red": None,
            "home_yellow": None,
            "away_yellow": None,
            "home_rank": m.home_rank,
            "away_rank": m.away_rank,
            "weather": m.weather,
            "b365_ah_home": m.b365_ah_home,
            "b365_ah_away": m.b365_ah_away,
        })

    df = pd.DataFrame(rows)
    bundle = ModelBundle.load(PROJECT_ROOT / "models" / "multileague_v2.pkl", require_verified=True)
    preds = bundle.predict_proba_df(df)

    res_lines = [f"🎯 <b>วิเคราะห์คู่บอลตรงใจ: '{html.escape(team_query)}'</b>\n━━━━━━━━━━━━━━━━━━━━"]
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
        thai_dt = (utc_dt + timedelta(hours=7)).strftime("%H:%M น.")

        hr = f"#{r['home_rank']}" if pd.notna(r.get("home_rank")) and r.get("home_rank") else "-"
        ar = f"#{r['away_rank']}" if pd.notna(r.get("away_rank")) and r.get("away_rank") else "-"
        rank_str = f" [{hr} vs {ar}]" if hr != "-" or ar != "-" else ""

        edge_pct = best[1] * 100
        verdict = (
            "🔥 <b>Edge สูงมาก! โต๊ะเปิดช่อง น่ากดตามน้ำ</b>"
            if edge_pct >= 10
            else ("✅ <b>มี Edge บวก เล่นได้ตามสูตร</b>" if edge_pct >= 5 else "⚠️ <b>Edge น้อย/ติดลบ โต๊ะตั้งราคาแน่น ไม่แนะนำ</b>")
        )

        res_lines.append(
            f"⚽ <b>{r['home_team']} vs {r['away_team']}</b>{rank_str}\n"
            f"🏆 ลีก: {r['league']} | ⏰ เวลา: {thai_dt}\n"
            f"👉 <b>แนะนำ: {best[0]}</b>\n"
            f"📈 Win Prob: <b>{best[2]*100:.1f}%</b> | Edge: <b>{edge_pct:+.1f}%</b>\n"
            f"💧 ราคาค่าน้ำ: <b>{best[3]:.2f}</b>\n"
            f"💡 ความเห็น cook45: {verdict}\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )

    return "\n\n".join(res_lines)


def handle_update(update: dict[str, Any]) -> None:
    message = update.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = str(chat.get("id", ""))
    text = (message.get("text") or "").strip()
    if not chat_id or not text:
        return

    # Check permission
    if allowed_chat_ids() and chat_id not in allowed_chat_ids():
        log(f"ignored message from unauthorized chat {chat_id}")
        send_message(f"⛔ ไม่ได้รับอนุญาต Chat ID: <code>{html.escape(chat_id)}</code>", chat_id)
        return

    log(f"message from {chat_id}: {text!r}")

    # Command Handling
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        command = parts[0].split("@", 1)[0].lower()
        if command in ("/start", "/help"):
            send_message(cmd_help(), chat_id)
        elif command in ("/status", "/สถานะ"):
            send_message(cmd_status(), chat_id)
        elif command in ("/card", "/picks", "/โพย"):
            cmd_card(chat_id)
        elif command in ("/retrain", "/เทรน"):
            cmd_retrain(chat_id)
        else:
            send_message(f"ไม่รู้จักคำสั่ง {html.escape(command)} สัส! พิมพ์ /help ดูคำสั่งดิ๊", chat_id)
        return

    # Natural Language / Team Name search
    lower = text.lower()
    if lower in ("หวัดดี", "ดี", "hello", "hi", "hey", "มึง", "cook45", "กูเอง"):
        send_message(
            "เออว่าไง clack! นอนอยู่บนเตียงดิสัส 55555\n\n"
            "มีอะไรให้กูช่วย? พิมพ์ **ชื่อทีม** มาสิเดี๋ยวกูวิเคราะห์ให้ หรือพิมพ์ <code>/card</code> เอาโพย 20 คู่เด็ดไปกด!",
            chat_id,
        )
        return

    # Analyze as match/team search
    send_message(f"🔍 กำลังค้นหาและคำนวณคู่นี้ให้สักครู่...", chat_id)
    try:
        reply = analyze_team_query(text)
        send_message(reply, chat_id)
    except Exception as e:
        send_message(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์: {html.escape(str(e))}", chat_id)


def main() -> int:
    token = bot_token()
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set in .env", file=sys.stderr)
        return 1

    try:
        api_call("deleteWebhook")
    except Exception as exc:
        log(f"deleteWebhook skipped: {exc}")

    log(f"cook45 Telegram Commander online. Allowed chats: {sorted(allowed_chat_ids())}")
    print(f"[OK] cook45 Telegram Commander listening for messages...")

    offset: int | None = None
    while True:
        try:
            updates = get_updates(offset)
        except KeyboardInterrupt:
            log("stopped")
            return 0
        except Exception as exc:
            log(f"poll error: {exc}")
            time.sleep(5)
            continue

        for update in updates:
            offset = int(update["update_id"]) + 1
            try:
                handle_update(update)
            except Exception:
                log("handler exception\n" + traceback.format_exc())
                time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
