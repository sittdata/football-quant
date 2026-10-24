"""Central configuration.

Everything is read from the environment (optionally via a ``.env`` file at the
project root) so that API keys never end up in source control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional; the project still works without it
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - trivial fallback

    def load_dotenv(*_args, **_kwargs) -> bool:  # type: ignore[misc]
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _get(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


@dataclass
class Config:
    """Runtime configuration for one run of the pipeline."""

    # --- paths ---------------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "football.db")
    raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")

    # --- domain --------------------------------------------------------------
    league: str = "EPL"
    allowed_leagues: tuple[str, ...] = ("England Premier League", "Spain La Liga", "France Ligue 1")
    bookmaker: str = "Pinnacle"
    #: bookmaker slug for 5DollarFootballAPI. bet365 ships with every plan;
    #: pinnacle and the other 18 need the Ultra plan.
    bookmaker_slug: str = "bet365"
    season: str = "2025"

    # --- model / staking -----------------------------------------------------
    edge_threshold: float = 0.06
    blacklist_markets: tuple[str, ...] = ("OU_over",)
    allowed_markets: tuple[str, ...] = ()
    stake: float = 1.0
    min_train: int = 40
    train_ratio: float = 0.8
    refit_every: int = 1
    seed: int = 42

    # --- credentials (may legitimately be None) ------------------------------
    football_data_org_key: str | None = None
    api_football_key: str | None = None
    oddspapi_key: str | None = None
    #: 5DollarFootballAPI: opening/closing odds AND results in one API.
    five_dollar_key: str | None = None
    thscore_api_key: str | None = None

    # --- network politeness --------------------------------------------------
    http_timeout: float = 30.0
    http_retries: int = 3
    request_delay: float = 2.0
    #: How long a stored raw payload may be reused before it is re-fetched.
    #: Without this, a season fetched once would be replayed forever.
    cache_max_age_hours: float = 6.0
    #: Never sleep longer than this waiting for a rate-limit window. A longer
    #: Retry-After stops the run instead, so it can be resumed later (the raw
    #: payload cache makes a resume cheap).
    max_backoff_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "Config":
        raw_blacklist = _get("FB_BLACKLIST_MARKETS", "OU_over")
        blacklist_markets = (
            tuple(m.strip() for m in raw_blacklist.split(",") if m.strip())
            if raw_blacklist and raw_blacklist.lower() != "none"
            else ()
        )
        raw_allowed = _get("FB_ALLOWED_MARKETS", "")
        allowed_markets = (
            tuple(m.strip() for m in raw_allowed.split(",") if m.strip())
            if raw_allowed and raw_allowed.lower() != "none"
            else ()
        )

        raw_allowed_leagues = _get("FB_ALLOWED_LEAGUES", "England Premier League,Spain La Liga,France Ligue 1")
        allowed_leagues = (
            tuple(lg.strip() for lg in raw_allowed_leagues.split(",") if lg.strip())
            if raw_allowed_leagues and raw_allowed_leagues.lower() != "all"
            else ()
        )

        cfg = cls(
            league=_get("FB_LEAGUE", "EPL") or "EPL",
            allowed_leagues=allowed_leagues,
            bookmaker=_get("FB_BOOKMAKER", "Pinnacle") or "Pinnacle",
            bookmaker_slug=_get("FB_BOOKMAKER_SLUG", "bet365") or "bet365",
            season=_get("FB_SEASON", "2025") or "2025",
            edge_threshold=_get_float("FB_EDGE_THRESHOLD", 0.06),
            blacklist_markets=blacklist_markets,
            allowed_markets=allowed_markets,
            stake=_get_float("FB_STAKE", 1.0),
            min_train=_get_int("FB_MIN_TRAIN", 40),
            train_ratio=_get_float("FB_TRAIN_RATIO", 0.8),
            refit_every=_get_int("FB_REFIT_EVERY", 1),
            seed=_get_int("FB_SEED", 42),
            football_data_org_key=_get("FOOTBALL_DATA_ORG_KEY"),
            api_football_key=_get("API_FOOTBALL_KEY"),
            oddspapi_key=_get("ODDSPAPI_KEY"),
            five_dollar_key=_get("FIVE_DOLLAR_FOOTBALL_API_KEY"),
            thscore_api_key=_get("THSCORE_API_KEY"),
            http_timeout=_get_float("FB_HTTP_TIMEOUT", 30.0),
            http_retries=_get_int("FB_HTTP_RETRIES", 3),
            request_delay=_get_float("FB_REQUEST_DELAY", 2.0),
            cache_max_age_hours=_get_float("FB_CACHE_MAX_AGE_HOURS", 6.0),
            max_backoff_seconds=_get_float("FB_MAX_BACKOFF_SECONDS", 60.0),
        )
        return cfg

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.raw_dir,
            self.models_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def missing_keys(self) -> dict[str, bool]:
        """Which providers are unusable right now (True == key missing)."""
        return {
            "football-data.org": self.football_data_org_key is None,
            "api-football": self.api_football_key is None,
            "oddspapi": self.oddspapi_key is None,
            "5dollarfootballapi": self.five_dollar_key is None,
        }


def load_config() -> Config:
    """Convenience helper used by the CLI and the tests."""
    cfg = Config.from_env()
    cfg.ensure_dirs()
    return cfg
