"""Results / labels from football-data.org (free tier, top leagues).

One request returns a whole competition-season, which is the cheapest way to
get the label side of the dataset.

Register: https://www.football-data.org/client/register
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..db import to_iso_utc
from .base import BaseProvider, Response

BASE_URL = "https://api.football-data.org/v4"

#: friendly name -> football-data.org competition code
COMPETITIONS = {
    "EPL": "PL",
    "PREMIER_LEAGUE": "PL",
    "LALIGA": "PD",
    "BUNDESLIGA": "BL1",
    "SERIEA": "SA",
    "LIGUE1": "FL1",
    "EREDIVISIE": "DED",
    "PRIMEIRA": "PPL",
    "BRASILEIRAO": "BSA",
    "CHAMPIONSHIP": "ELC",
    "UCL": "CL",
    "WORLD_CUP": "WC",
    "EUROS": "EC",
}


class FootballDataOrgProvider(BaseProvider):
    def __init__(
        self, cfg: Config, con=None, use_cache: bool = True, refresh: bool = False
    ) -> None:
        super().__init__(
            name="football-data.org", cfg=cfg, con=con, use_cache=use_cache, refresh=refresh
        )

    def _headers(self) -> dict[str, str]:
        key = self.require_key(self.cfg.football_data_org_key, "FOOTBALL_DATA_ORG_KEY")
        return {"X-Auth-Token": key}

    # ------------------------------------------------------------------ #
    def fetch_season(self, competition: str = "PL", season: str | None = None) -> Response:
        """Return the raw :class:`Response` so the caller can use its real
        ``fetched_at`` - which may be a cached timestamp, not "now"."""
        code = COMPETITIONS.get(competition.upper(), competition.upper())
        season = season or self.cfg.season
        url = f"{BASE_URL}/competitions/{code}/matches"
        params = {"season": season}
        return self.request(
            url,
            params=params,
            headers=self._headers(),
            request_key=f"{code}-{season}",
            endpoint="/competitions/{code}/matches",
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def to_match_rows(payload: dict[str, Any], fetched_at: str, season: str | None = None) -> list[dict]:
        """Convert a competition payload into ``matches`` rows (results only)."""
        rows: list[dict] = []
        for match in payload.get("matches", []) or []:
            score = match.get("score") or {}
            full = score.get("fullTime") or {}
            half = score.get("halfTime") or {}

            home = (match.get("homeTeam") or {}).get("name")
            away = (match.get("awayTeam") or {}).get("name")
            if not home or not away:
                continue

            rows.append(
                {
                    "match_id": f"fdorg:{match.get('id')}",
                    "provider": "football-data.org",
                    "league": (match.get("competition") or {}).get("name"),
                    "season": str((match.get("season") or {}).get("startDate", season or "") or "")[:4]
                    or season,
                    "kickoff_at": to_iso_utc(match.get("utcDate")),
                    "home_team": home,
                    "away_team": away,
                    "home_goals": full.get("home"),
                    "away_goals": full.get("away"),
                    "ht_home": half.get("home"),
                    "ht_away": half.get("away"),
                    "result_source": "football-data.org",
                    "result_fetched_at": fetched_at,
                }
            )
        return rows
