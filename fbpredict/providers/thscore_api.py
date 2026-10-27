"""Thscore Official JSON API Provider (api2.thscore.info).

Uses the official developer API feed:
- /football_th/livescores.aspx (Live matches, scores, HT, corners, cards)
- /football_th/schedule.aspx?date=YYYY-MM-DD (Daily schedule and full results)

Saves verbatim raw payloads to SQLite raw_payload table as coursework evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

import sqlite3
from ..config import Config
from ..db import archive_raw, connect, to_iso_utc
from .base import BaseProvider, ProviderError

API_BASE = "http://api2.thscore.info"


@dataclass
class ThscoreMatch:
    match_id: str
    league: str
    kickoff_at: str
    home_team: str
    away_team: str
    home_goals: int | None
    away_goals: int | None
    ht_home: int | None
    ht_away: int | None
    status: int
    home_corner: int | None = None
    away_corner: int | None = None
    home_red: int | None = None
    away_red: int | None = None
    home_yellow: int | None = None
    away_yellow: int | None = None
    weather: str | None = None
    temperature: str | None = None


class ThscoreApiProvider(BaseProvider):
    """Client for the official Thscore REST API."""

    name = "thscore_api"

    def __init__(self, config: Config | None = None, con: sqlite3.Connection | None = None):
        self.config = config or Config.from_env()
        self.con = con
        self.api_key = self.config.thscore_api_key
        if not self.api_key:
            raise ProviderError("THSCORE_API_KEY is not configured in .env or Config")

    def _fetch(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make an authenticated request and record the raw payload."""
        req_params = dict(params or {})
        req_params["api_key"] = self.api_key

        url = f"{API_BASE}{endpoint}"
        try:
            resp = requests.get(url, params=req_params, timeout=self.config.http_timeout)
            resp.raise_for_status()
        except Exception as e:
            raise ProviderError(f"Thscore API request failed for {endpoint}: {e}") from e

        data = resp.json()
        if "code" in data and data["code"] != 0 and "data" not in data:
            raise ProviderError(f"Thscore API error: {data.get('message', 'Unknown error')}")

        # Store verbatim payload for coursework evidence
        if self.con is not None:
            archive_raw(
                self.con,
                provider="thscore_api",
                endpoint=endpoint,
                request_key=json.dumps(params or {}, sort_keys=True),
                payload=resp.text,
                raw_dir=self.config.raw_dir,
            )

        return data

    def get_livescores(self) -> list[ThscoreMatch]:
        """Fetch today's live and scheduled football matches."""
        payload = self._fetch("/football_th/livescores.aspx")
        raw_list = payload.get("data", [])
        return [self._parse_item(m) for m in raw_list]

    def get_schedule_by_date(self, date_str: str) -> list[ThscoreMatch]:
        """Fetch matches and completed results for a specific date (YYYY-MM-DD)."""
        payload = self._fetch("/football_th/schedule.aspx", {"date": date_str})
        raw_list = payload.get("data", [])
        return [self._parse_item(m) for m in raw_list]

    def _parse_item(self, m: dict[str, Any]) -> ThscoreMatch:
        """Parse raw API dict into a standardized ThscoreMatch."""
        raw_time = m.get("matchTime", "")
        kickoff_iso = ""
        if raw_time:
            try:
                # format: DD-MM-YYYY HH:MM:SS
                dt = datetime.strptime(raw_time, "%d-%m-%Y %H:%M:%S").replace(tzinfo=timezone.utc)
                kickoff_iso = dt.isoformat()
            except Exception:
                kickoff_iso = raw_time

        def _safe_int(val: Any) -> int | None:
            if val is None or val == "":
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        return ThscoreMatch(
            match_id=f"thscore:{m.get('matchId')}",
            league=m.get("leagueName", "Unknown League"),
            kickoff_at=kickoff_iso,
            home_team=m.get("homeName", "Unknown Home"),
            away_team=m.get("awayName", "Unknown Away"),
            home_goals=_safe_int(m.get("homeScore")),
            away_goals=_safe_int(m.get("awayScore")),
            ht_home=_safe_int(m.get("homeHalfScore")),
            ht_away=_safe_int(m.get("awayHalfScore")),
            status=m.get("status", 0),
            home_corner=_safe_int(m.get("homeCorner")),
            away_corner=_safe_int(m.get("awayCorner")),
            home_red=_safe_int(m.get("homeRed")),
            away_red=_safe_int(m.get("awayRed")),
            home_yellow=_safe_int(m.get("homeYellow")),
            away_yellow=_safe_int(m.get("awayYellow")),
            weather=m.get("weather"),
            temperature=m.get("temperature"),
        )
