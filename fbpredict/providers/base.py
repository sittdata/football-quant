"""Shared HTTP plumbing: politeness, retries, and raw-payload archiving."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

from ..config import Config
from ..db import archive_raw, cached_payload

#: query parameters that must never survive into a log line or a traceback
_SECRET_PARAM = re.compile(
    r"(?i)\b([a-z0-9_]*(?:api[_-]?key|token|secret|password)[a-z0-9_]*)=([^&\s\"']+)"
)


def _age_hours(fetched_at: str) -> float:
    """Age of a stored timestamp in hours; unparseable values count as stale."""
    try:
        then = datetime.fromisoformat(str(fetched_at))
    except (TypeError, ValueError):
        return float("inf")
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds() / 3600.0


def _retry_after_seconds(reply: Any) -> float | None:
    """``Retry-After`` in seconds, when the server sends one."""
    raw = reply.headers.get("Retry-After") if hasattr(reply, "headers") else None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _rate_limit_snapshot(reply: Any) -> dict[str, str] | None:
    """The X-RateLimit-* headers, when the API sends them."""
    if not hasattr(reply, "headers"):
        return None
    headers = reply.headers
    snapshot = {
        "limit": headers.get("X-RateLimit-Limit"),
        "remaining": headers.get("X-RateLimit-Remaining"),
        "reset": headers.get("X-RateLimit-Reset"),
    }
    if all(value is None for value in snapshot.values()):
        return None
    return {key: value for key, value in snapshot.items() if value is not None}


class ProviderError(RuntimeError):
    """Raised when a provider cannot return usable data."""


class CacheMissError(ProviderError):
    """Offline mode was requested but nothing fresh is cached for this request.

    Callers that are walking a list can skip the item and carry on, instead of
    treating it as a failure.
    """


@dataclass
class Response:
    """A provider response plus the evidence of when it was obtained."""

    data: Any
    fetched_at: str
    from_cache: bool = False
    endpoint: str = ""


@dataclass
class BaseProvider:
    """Common behaviour for every data source."""

    name: str
    cfg: Config
    con: sqlite3.Connection | None = None
    use_cache: bool = True
    #: Skip the cache entirely and go to the network (a `--refresh` run).
    refresh: bool = False
    #: Never touch the network: serve only what is already cached.
    offline: bool = False
    #: Most recent X-RateLimit-* snapshot, so a caller can report quota left.
    last_rate_limit: dict[str, str] | None = None
    _session: requests.Session | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "User-Agent": (
                        "fbpredict/0.1 (student research project; "
                        "contact: set FB_CONTACT in .env)"
                    ),
                    "Accept": "application/json, text/html;q=0.9,*/*;q=0.8",
                }
            )
        return self._session

    def _pause(self) -> None:
        if self.cfg.request_delay > 0:
            time.sleep(self.cfg.request_delay)

    # ------------------------------------------------------------------ #
    def _secret_values(self) -> tuple[str, ...]:
        return tuple(
            value
            for value in (
                self.cfg.oddspapi_key,
                self.cfg.api_football_key,
                self.cfg.football_data_org_key,
                self.cfg.five_dollar_key,
            )
            if value
        )

    def _redact(self, text: object) -> str:
        """Strip credentials from anything that could reach a log or a screen.

        OddsPapi takes its key as a query parameter, so a raw URL - or the URL
        embedded in a ``requests`` exception message - can carry the key
        verbatim into stderr and into any pasted traceback.
        """
        cleaned = str(text)
        for secret in self._secret_values():
            cleaned = cleaned.replace(secret, "<redacted>")
        return _SECRET_PARAM.sub(lambda match: f"{match.group(1)}=<redacted>", cleaned)

    # ------------------------------------------------------------------ #
    def request(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        request_key: str | None = None,
        endpoint: str | None = None,
        as_json: bool = True,
    ) -> Response:
        """GET ``url``, archive the body, and return it.

        The cache key is ``(provider, endpoint, request_key)``; a cache hit
        avoids spending API quota again for the same logical request.
        """
        endpoint = endpoint or url
        key = request_key or url

        if self.use_cache and not self.refresh and self.con is not None:
            cached = cached_payload(
                self.con, provider=self.name, endpoint=endpoint, request_key=key
            )
            if cached is not None:
                payload, fetched_at = cached
                if _age_hours(fetched_at) <= self.cfg.cache_max_age_hours:
                    data = json.loads(payload) if as_json else payload
                    return Response(
                        data=data,
                        fetched_at=fetched_at,
                        from_cache=True,
                        endpoint=endpoint,
                    )

        if self.offline:
            raise CacheMissError(
                f"{self.name}: offline mode, and nothing fresh is cached for "
                f"{endpoint} ({key})"
            )

        last_error: Exception | None = None
        wait_seconds: float | None = None
        for attempt in range(1, max(1, self.cfg.http_retries) + 1):
            try:
                reply = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.cfg.http_timeout,
                )
            except requests.RequestException as exc:  # network hiccup
                last_error = exc
            else:
                snapshot = _rate_limit_snapshot(reply)
                if snapshot is not None:
                    self.last_rate_limit = snapshot

                if reply.status_code == 429 or 500 <= reply.status_code < 600:
                    wait_seconds = _retry_after_seconds(reply)
                    last_error = ProviderError(
                        f"{self.name}: HTTP {reply.status_code} from {self._redact(url)}"
                        + (
                            f" (server asked to wait {wait_seconds:g}s)"
                            if wait_seconds is not None
                            else ""
                        )
                    )
                elif reply.status_code >= 400:
                    raise ProviderError(
                        f"{self.name}: HTTP {reply.status_code} from {self._redact(url)} - "
                        f"{self._redact(reply.text[:300])}"
                    )
                else:
                    text = reply.text
                    fetched_at = self._archive(endpoint, key, text)
                    data = json.loads(text) if as_json else text
                    return Response(data=data, fetched_at=fetched_at, endpoint=endpoint)

            if wait_seconds is not None and wait_seconds > self.cfg.max_backoff_seconds:
                # Do not block an automated run for an hour waiting for the
                # quota window. Stop instead: the caller can resume later and
                # the raw-payload cache makes the resume cheap.
                raise ProviderError(
                    f"{self.name}: rate limited; the server asked to wait "
                    f"{wait_seconds:g}s, beyond FB_MAX_BACKOFF_SECONDS "
                    f"({self.cfg.max_backoff_seconds:g}). Re-run later to continue."
                )

            if attempt < max(1, self.cfg.http_retries):
                # Honour the server's own back-off when it sent one.
                delay = wait_seconds if wait_seconds is not None else min(
                    30.0, self.cfg.request_delay * (2 ** attempt)
                )
                time.sleep(max(0.0, delay))
            self._pause()

        raise ProviderError(
            f"{self.name}: giving up on {self._redact(url)}: "
            f"{type(last_error).__name__}: {self._redact(last_error)}"
        )

    def _archive(self, endpoint: str, key: str, text: str) -> str:
        if self.con is None:
            from ..db import utcnow_iso

            return utcnow_iso()
        return archive_raw(
            self.con,
            provider=self.name,
            endpoint=endpoint,
            payload=text,
            request_key=key,
            raw_dir=self.cfg.raw_dir,
            slug=key,
        )

    # ------------------------------------------------------------------ #
    def require_key(self, key: str | None, env_name: str) -> str:
        if not key:
            raise ProviderError(
                f"{self.name} needs {env_name}. Copy .env.example to .env and fill it in."
            )
        return key
