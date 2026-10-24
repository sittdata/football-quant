"""Provider plumbing: credential redaction and the raw-payload cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fbpredict import db
from fbpredict.config import Config
from fbpredict.providers.base import BaseProvider, _age_hours


def _cfg(**overrides) -> Config:
    cfg = Config()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class _ExplodingSession:
    """Stands in for requests.Session; records whether the network was used."""

    headers: dict = {}

    def __init__(self) -> None:
        self.calls = 0

    def get(self, *_args, **_kwargs):
        self.calls += 1
        raise RuntimeError("network must not be touched")


# --------------------------------------------------------------------------- #
# credential redaction
# --------------------------------------------------------------------------- #
def test_redaction_removes_a_query_string_api_key() -> None:
    provider = BaseProvider(name="oddspapi", cfg=_cfg(oddspapi_key="SECRETKEY123"))
    leaked = "https://api.oddspapi.io/v4/odds?apiKey=SECRETKEY123&fixtureId=42"

    cleaned = provider._redact(leaked)

    assert "SECRETKEY123" not in cleaned
    assert "fixtureId=42" in cleaned
    assert "<redacted>" in cleaned


def test_redaction_handles_a_requests_exception_message() -> None:
    """requests embeds the prepared URL - query string and all - in its errors."""
    provider = BaseProvider(name="oddspapi", cfg=_cfg(oddspapi_key="SECRETKEY123"))
    message = (
        "HTTPConnectionPool(host='api.oddspapi.io', port=443): Max retries exceeded "
        "with url: /v4/odds?apiKey=SECRETKEY123 (Caused by NewConnectionError)"
    )

    cleaned = provider._redact(message)

    assert "SECRETKEY123" not in cleaned
    assert "Max retries exceeded" in cleaned


def test_redaction_catches_other_secret_names() -> None:
    provider = BaseProvider(name="x", cfg=_cfg(api_football_key="OTHERKEY", football_data_org_key="THIRDKEY"))

    cleaned = provider._redact("GET /fixtures?token=OTHERKEY header X-Auth-Token: THIRDKEY")

    assert "OTHERKEY" not in cleaned
    assert "THIRDKEY" not in cleaned


def test_redaction_is_harmless_without_secrets() -> None:
    provider = BaseProvider(name="x", cfg=_cfg())
    assert provider._redact("plain text") == "plain text"


# --------------------------------------------------------------------------- #
# cache freshness
# --------------------------------------------------------------------------- #
def test_age_hours_treats_unparseable_timestamps_as_stale() -> None:
    assert _age_hours("not-a-date") == float("inf")
    assert _age_hours("") == float("inf")
    assert _age_hours(db.utcnow_iso()) < 0.01


def test_fresh_cache_is_reused_without_touching_the_network(tmp_path: Path) -> None:
    con = db.init_db(tmp_path / "t.db")
    cfg = _cfg(cache_max_age_hours=6.0, http_retries=1, request_delay=0.0)
    provider = BaseProvider(name="p", cfg=cfg, con=con)
    db.archive_raw(con, provider="p", endpoint="/e", payload={"v": 1}, request_key="k")

    session = _ExplodingSession()
    provider._session = session  # type: ignore[assignment]

    response = provider.request("https://example.test/e", request_key="k", endpoint="/e")

    assert response.from_cache is True
    assert response.data == {"v": 1}
    # the ORIGINAL fetch time is reported, not an empty string and not "now"
    assert response.fetched_at.endswith("+00:00")
    assert session.calls == 0


def test_stale_cache_entry_is_not_reused(tmp_path: Path) -> None:
    con = db.init_db(tmp_path / "t.db")
    cfg = _cfg(cache_max_age_hours=6.0, http_retries=1, request_delay=0.0)
    provider = BaseProvider(name="p", cfg=cfg, con=con)

    con.execute(
        "INSERT INTO raw_payload (provider, endpoint, request_key, fetched_at, payload) "
        "VALUES (?, ?, ?, ?, ?)",
        ("p", "/e", "k", "2020-01-01T00:00:00+00:00", json.dumps({"v": 1})),
    )
    con.commit()

    session = _ExplodingSession()
    provider._session = session  # type: ignore[assignment]

    with pytest.raises(RuntimeError):
        provider.request("https://example.test/e", request_key="k", endpoint="/e")

    assert session.calls == 1  # it went to the network instead of replaying 2020


def test_refresh_bypasses_a_fresh_cache(tmp_path: Path) -> None:
    con = db.init_db(tmp_path / "t.db")
    cfg = _cfg(cache_max_age_hours=6.0, http_retries=1, request_delay=0.0)
    provider = BaseProvider(name="p", cfg=cfg, con=con, refresh=True)
    db.archive_raw(con, provider="p", endpoint="/e", payload={"v": 1}, request_key="k")

    session = _ExplodingSession()
    provider._session = session  # type: ignore[assignment]

    with pytest.raises(RuntimeError):
        provider.request("https://example.test/e", request_key="k", endpoint="/e")

    assert session.calls == 1
