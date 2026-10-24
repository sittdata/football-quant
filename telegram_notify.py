"""Send fbpredict results to Telegram.

Configuration (put these in ``.env``, which is git-ignored):

    TELEGRAM_BOT_TOKEN=123456789:AA...        # from @BotFather
    TELEGRAM_CHAT_ID=123456789                # your chat id

The module is import-safe: when no token is configured every call is a no-op
that returns ``None``, so a pipeline can call :func:`notify_report` without
guarding it first.

CLI examples::

    python telegram_notify.py --test
    python telegram_notify.py --stem backtest_combined_all
    python telegram_notify.py --text "fetch finished"
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests

try:  # python-dotenv is already a project dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - optional at import time
    pass

API_ROOT = "https://api.telegram.org/bot{token}/{method}"
MESSAGE_LIMIT = 4096
CAPTION_LIMIT = 1024


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def bot_token() -> str | None:
    value = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    return value or None


def default_chat_id() -> str | None:
    value = (
        os.getenv("TELEGRAM_CHAT_ID")
        or os.getenv("TELEGRAM_ALLOWED_CHAT_ID")
        or ""
    ).strip()
    return value or None


def configured() -> bool:
    return bool(bot_token())


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #
def _call(
    method: str,
    data: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    token = bot_token()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    reply = requests.post(
        API_ROOT.format(token=token, method=method),
        data=data or {},
        files=files,
        timeout=timeout,
    )
    try:
        payload = reply.json()
    except ValueError as exc:  # non-JSON error page
        raise RuntimeError(f"telegram {method}: HTTP {reply.status_code}") from exc
    if not payload.get("ok"):
        raise RuntimeError(
            f"telegram {method} failed: {payload.get('description') or reply.status_code}"
        )
    return payload["result"]


def send_message(
    text: str,
    chat_id: str | None = None,
    *,
    parse_mode: str | None = "HTML",
    disable_preview: bool = True,
) -> dict[str, Any] | None:
    """Send a text message; returns ``None`` when Telegram is not configured."""
    if not bot_token():
        return None
    target = str(chat_id or default_chat_id() or "").strip()
    if not target:
        raise RuntimeError("no Telegram chat id: set TELEGRAM_CHAT_ID")
    data: dict[str, Any] = {
        "chat_id": target,
        "text": text[:MESSAGE_LIMIT],
        "disable_web_page_preview": "true" if disable_preview else "false",
    }
    if parse_mode:
        data["parse_mode"] = parse_mode
    try:
        return _call("sendMessage", data)
    except RuntimeError as exc:
        # A truncated HTML message can split a tag. Retry once as plain text
        # rather than dropping the notification.
        if "parse" not in str(exc).lower():
            raise
        data.pop("parse_mode", None)
        data["text"] = re.sub(r"<[^>]+>", "", data["text"])[:MESSAGE_LIMIT]
        return _call("sendMessage", data)


def send_document(
    path: str | Path,
    caption: str | None = None,
    chat_id: str | None = None,
) -> dict[str, Any] | None:
    """Upload a file; returns ``None`` when Telegram is not configured."""
    if not bot_token():
        return None
    target = str(chat_id or default_chat_id() or "").strip()
    if not target:
        raise RuntimeError("no Telegram chat id: set TELEGRAM_CHAT_ID")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        return _call(
            "sendDocument",
            {"chat_id": target, "caption": (caption or "")[:CAPTION_LIMIT]},
            files={"document": (path.name, handle)},
        )


# --------------------------------------------------------------------------- #
# report summary
# --------------------------------------------------------------------------- #
def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:+.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def summarize_stem(stem: str = "backtest_combined", reports_dir: str | Path = "reports") -> str:
    """A compact HTML summary of ``reports/<stem>.json``."""
    path = Path(reports_dir) / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    config = data.get("config") or {}
    sizes = data.get("sizes") or {}
    features = data.get("features") or {}
    betting = data.get("betting") or {}
    heads = data.get("heads") or {}

    lines = [f"<b>📊 {html.escape(stem)}</b>"]
    meta = []
    if features.get("feature_set"):
        meta.append(f"features={features['feature_set']}")
    if config.get("edge_threshold") is not None:
        meta.append(f"threshold={config['edge_threshold']}")
    if config.get("train_ratio") is not None:
        meta.append(f"split={config['train_ratio']:.0%}")
    if meta:
        lines.append(html.escape(" | ".join(meta)))

    lines.append(
        f"train/test: <b>{sizes.get('train', '?')}</b>/<b>{sizes.get('test', '?')}</b>"
        f" | bets <b>{betting.get('n_bets', '?')}</b>"
    )
    if betting.get("roi") is not None:
        lines.append(
            f"ROI: <b>{_pct(betting.get('roi'))}</b>"
            f" | profit {float(betting.get('profit') or 0):+.2f}u"
        )

    for name, block in heads.items():
        model = block.get("model") or {}
        base = block.get("market_baseline") or {}
        verdict = "✅" if block.get("model_beats_market_logloss") else "❌"
        lines.append(
            f"{html.escape(str(name))}: LL <b>{_num(model.get('log_loss'))}</b>"
            f" vs market {_num(base.get('log_loss'))} {verdict}"
        )

    by_market = betting.get("by_market") or {}
    if by_market:
        parts = [
            f"{html.escape(str(m))} {item.get('n', 0)}b {_pct(item.get('roi'))}"
            for m, item in by_market.items()
        ]
        lines.append("markets: " + " · ".join(parts))

    return "\n".join(lines)


def notify_report(
    stem: str = "backtest_combined",
    reports_dir: str | Path = "reports",
    *,
    chat_id: str | None = None,
    send_file: bool = True,
) -> bool:
    """Send the summary and (optionally) the Markdown report. No-op if unset."""
    if not bot_token():
        return False
    reports_dir = Path(reports_dir)
    send_message(summarize_stem(stem, reports_dir), chat_id)
    markdown = reports_dir / f"{stem}.md"
    if send_file and markdown.exists():
        send_document(markdown, caption=f"{stem} report", chat_id=chat_id)
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stem", default=None, help="report stem in reports/ (e.g. backtest_combined_all)")
    parser.add_argument("--text", default=None, help="plain-text message to send")
    parser.add_argument("--file", default=None, help="file to send as a document")
    parser.add_argument("--chat-id", default=None, help="override TELEGRAM_CHAT_ID")
    parser.add_argument("--test", action="store_true", help="send a connectivity test message")
    parser.add_argument("--discover", action="store_true", help="print chat ids that recently messaged the bot")
    parser.add_argument("--no-file", action="store_true", help="do not attach the .md report")
    args = parser.parse_args(argv)

    if not configured():
        print("TELEGRAM_BOT_TOKEN is not set; nothing sent. Add it to .env to enable.")
        return 0
    if args.discover:
        updates = _call("getUpdates", {"timeout": 0, "allowed_updates": json.dumps(["message"])})
        found: dict[str, str] = {}
        for update in updates:
            chat = (update.get("message") or {}).get("chat") or {}
            if chat.get("id"):
                found[str(chat["id"])] = (
                    chat.get("username") or chat.get("title") or chat.get("first_name") or ""
                )
        print(json.dumps(found, indent=2, ensure_ascii=False) if found else
              "no messages found; send /start to the bot first, then run this again")
        return 0
    if not (default_chat_id() or args.chat_id):
        print("TELEGRAM_CHAT_ID is not set; nothing sent. Add it to .env to enable.")
        return 0

    try:
        if args.test:
            send_message("✅ fbpredict Telegram notifications are working.", args.chat_id)
            print("test message sent")
        if args.text:
            send_message(html.escape(args.text), args.chat_id)
            print("message sent")
        if args.file:
            send_document(args.file, args.chat_id)
            print(f"file sent: {args.file}")
        if args.stem:
            notify_report(args.stem, chat_id=args.chat_id, send_file=not args.no_file)
            print(f"report sent: {args.stem}")
    except Exception as exc:  # surface the reason but keep the shell exit code simple
        print(f"telegram error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
