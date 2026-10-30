"""Tests for the daily Telegram card dispatcher."""

import importlib.util
from pathlib import Path

import requests


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "daily_telegram_card.py"
    spec = importlib.util.spec_from_file_location("daily_telegram_card", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_connection_error_does_not_leak_bot_token(monkeypatch, capsys):
    """A network failure must not echo the token-bearing request URL."""
    module = _load_module()
    token = "123456:SECRET-TOKEN"
    monkeypatch.setattr(module, "TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setattr(module, "TELEGRAM_CHAT_ID", "42")

    def boom(*args, **kwargs):
        raise requests.ConnectionError(
            f"failed POST https://api.telegram.org/bot{token}/sendMessage"
        )

    monkeypatch.setattr(module.requests, "post", boom)

    assert module.send_telegram("hello") is False
    out = capsys.readouterr().out
    assert token not in out
    assert "ConnectionError" in out


def test_single_long_line_is_hard_split(monkeypatch):
    """A line above the chunk limit must be split so Telegram does not reject it."""
    module = _load_module()
    monkeypatch.setattr(module, "TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setattr(module, "TELEGRAM_CHAT_ID", "42")

    sent = []

    class FakeResponse:
        status_code = 200
        text = "ok"

    def fake_post(url, json, timeout):
        sent.append(json["text"])
        return FakeResponse()

    monkeypatch.setattr(module.requests, "post", fake_post)

    long_line = "x" * 10000
    assert module.send_telegram(long_line) is True
    assert len(sent) > 1
    assert all(len(chunk) <= 3800 for chunk in sent)
    assert "".join(sent) == long_line


def test_api_error_response_is_redacted(monkeypatch, capsys):
    """A non-200 response body is redacted before printing."""
    module = _load_module()
    token = "123456:SECRET-TOKEN"
    monkeypatch.setattr(module, "TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setattr(module, "TELEGRAM_CHAT_ID", "42")

    class FakeResponse:
        status_code = 400
        text = f"bad request to /bot{token}/sendMessage"

    monkeypatch.setattr(module.requests, "post", lambda *a, **k: FakeResponse())

    assert module.send_telegram("hello") is False
    out = capsys.readouterr().out
    assert token not in out
    assert "<redacted>" in out


def test_html_400_retries_without_parse_mode(monkeypatch):
    """When Telegram rejects HTML formatting with 400, fallback retries without parse_mode."""
    module = _load_module()
    monkeypatch.setattr(module, "TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setattr(module, "TELEGRAM_CHAT_ID", "42")

    calls = []

    class FakeResponse:
        def __init__(self, code, text="ok"):
            self.status_code = code
            self.text = text

    def fake_post(url, json, timeout):
        calls.append(json)
        # First call with parse_mode fails with 400
        if "parse_mode" in json:
            return FakeResponse(400, "Bad Request: can't parse entities")
        return FakeResponse(200, "ok")

    monkeypatch.setattr(module.requests, "post", fake_post)

    assert module.send_telegram("<b>unclosed tag") is True
    assert len(calls) == 2
    assert "parse_mode" in calls[0]
    assert "parse_mode" not in calls[1]


def test_hard_split_avoids_bisecting_html_tags():
    """A cut inside an HTML tag <...> must split before the tag instead of through it."""
    module = _load_module()
    # Tag sits at index 3795..3815 crossing 3800: <span class='x'>
    line = ("a" * 3795) + "<span class='x'>bold text</span>" + ("b" * 100)
    cut = module._safe_cut_index(line, 3800)
    assert cut == 3795
    assert not line[:cut].endswith("<")
    assert line[cut:].startswith("<span")
