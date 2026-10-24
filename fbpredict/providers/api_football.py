"""Results / labels from API-Football (api-sports.io).

Free tier is 100 requests/day and one request returns a full league-season,
so the label side costs a handful of calls for the whole project.

Register: https://www.api-football.com/
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..db import to_iso_utc
from .base import BaseProvider, ProviderError, Response

BASE_URL = "https://v3.football.api-sports.io"

#: friendly name -> API-Football league id
LEAGUES = {
    "EPL": 39,
    "PREMIER_LEAGUE": 39,
    "LALIGA": 140,
    "BUNDESLIGA": 78,
    "SERIEA": 135,
    "LIGUE1": 61,
    "EREDIVISIE": 88,
    "PRIMEIRA": 94,
    "BRASILEIRAO": 71,
    "UCL": 2,
    "EUROPA_LEAGUE": 3,
    "WORLD_CUP": 1,
}


class ApiFootballProvider(BaseProvider):
    def __init__(
        self, cfg: Config, con=None, use_cache: bool = True, refresh: bool = False
    ) -> None:
        super().__init__(
            name="api-football", cfg=cfg, con=con, use_cache=use_cache, refresh=refresh
        )

    def _headers(self) -> dict[str, str]:
        key = self.require_key(self.cfg.api_football_key, "API_FOOTBALL_KEY")
        return {"x-apisports-key": key}

    # ------------------------------------------------------------------ #
    def fetch_season(self, league: str | int = "EPL", season: str | None = None) -> Response:
        """Return the raw :class:`Response` so the caller can use its real
        ``fetched_at`` - which may be a cached timestamp, not "now"."""
        league_id = LEAGUES.get(str(league).upper(), league)
        season = season or self.cfg.season
        url = f"{BASE_URL}/fixtures"
        params = {"league": league_id, "season": season}
        response = self.request(
            url,
            params=params,
            headers=self._headers(),
            request_key=f"{league_id}-{season}",
            endpoint="/fixtures",
        )
        data = response.data or {}
        errors = data.get("errors")
        if errors:
            raise ProviderError(f"api-football reported errors: {errors}")
        return response

    # ------------------------------------------------------------------ #
    @staticmethod
    def to_match_rows(payload: dict[str, Any], fetched_at: str, season: str | None = None) -> list[dict]:
        rows: list[dict] = []
        for item in payload.get("response", []) or []:
            fixture = item.get("fixture") or {}
            teams = item.get("teams") or {}
            goals = item.get("goals") or {}
            score = item.get("score") or {}
            half = (score.get("halftime") or {}) if isinstance(score, dict) else {}

            home = (teams.get("home") or {}).get("name")
            away = (teams.get("away") or {}).get("name")
            if not home or not away:
                continue

            rows.append(
                {
                    "match_id": f"apifootball:{fixture.get('id')}",
                    "provider": "api-football",
                    "league": ((item.get("league") or {}).get("name")),
                    "season": str((item.get("league") or {}).get("season", season or "")),
                    "kickoff_at": to_iso_utc(fixture.get("date")),
                    "home_team": home,
                    "away_team": away,
                    "home_goals": goals.get("home"),
                    "away_goals": goals.get("away"),
                    "ht_home": half.get("home"),
                    "ht_away": half.get("away"),
                    "result_source": "api-football",
                    "result_fetched_at": fetched_at,
                }
            )
        return rows
