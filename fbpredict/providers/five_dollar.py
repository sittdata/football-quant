"""Odds **and** results from 5DollarFootballAPI.

This is the best fit for the project so far: one response carries the *opening*
line and both prices for the Asian handicap and the goal line, the opening 1X2
prices, and the half-time score. That is exactly the shape the brief needs
(README sections 3-4), from a single provider, with no odds/results join step.

* Docs: https://5dollarfootballapi.com/docs
* Base: ``https://api.5dollarfootballapi.com/v1``
* Auth: ``Authorization: Bearer fb_live_...``
* Attribution: required on the $0 plans (see :data:`ATTRIBUTION`)

Two response shapes matter
--------------------------
``/v1/leagues/{id}/fixtures?include=odds`` returns, per fixture, an ``odds``
block whose stages are the **line only**::

    "asian_handicap": {"opening": -0.25, "closing": -0.5, "inplay": -1.0}

``/v1/fixtures/{id}/odds`` returns, per bookmaker, stages carrying the **line
and both prices**::

    "asian_handicap": {"opening": {"line": -0.25, "home": 1.95, "away": 1.85}}

Both are parsed here; the price fields come back ``None`` when only the list
shape was fetched. ``include=odds`` on a list is a paid feature, so on the free
plan you fetch odds one fixture at a time (``--per-fixture-odds``).

What this provider cannot supply
--------------------------------
The per-fixture odds endpoint exposes no half-time 1X2 market, so
``draw_ht_odds`` (the brief's HT draw price) stays NULL. It is only reachable
through ``/v1/fixtures/{id}/odds/history?market=1x2_half``, which is Ultra-only.
The half-time *lines* (``goal_line_half``, ``asian_handicap_half``) and the
``btts`` prices are available and are stored, but they are not substitutes for
the Ladbrokes Over HT price the spec's rule R4a names - that rule stays
untestable and is reported as such.

All prices from this API are decimal (1.5-500, never below 1.0), which is what
``odds_format_raw`` / ``odds_format_stored`` record on each stored row.
"""

from __future__ import annotations

import math
from typing import Any, Iterator

from ..config import Config
from .base import BaseProvider, ProviderError

BASE_URL = "https://api.5dollarfootballapi.com/v1"

#: Required on Free / Community / Education plans when the data is shown publicly.
ATTRIBUTION = "Football data by [5DollarFootballAPI](https://5dollarfootballapi.com)"

MARKET_1X2 = "1x2"
MARKET_AH = "asian_handicap"
MARKET_GOAL = "goal_line"
MARKET_AH_HALF = "asian_handicap_half"
MARKET_GOAL_HALF = "goal_line_half"
MARKET_BTTS = "btts"

#: every price this API returns is decimal (1.5-500, never below 1.0). Both of
#: these are recorded per row so a future format change is visible, not
#: silent (spec sections 9.2/9.5).
ODDS_FORMAT = "decimal"

#: every field this module can fill on a ``matches`` row. The odds-format
#: columns are deliberately absent: they are set once per row by
#: :func:`to_match_row`, and ``map_markets`` returning them as ``None`` would
#: clobber that value when a caller merges the two (``_attach_per_fixture_odds``
#: does exactly that).
ODDS_FIELDS = (
    "fav_side",
    "ah_open",
    "ah_close",
    "ah_fav_price",
    "ah_dog_price",
    "pool_fav_open",
    "pool_dog_open",
    "draw_ft_odds",
    "draw_ht_odds",
    "pool_fav_close",
    "pool_dog_close",
    "draw_ft_close",
    "ou_line_open",
    "ou_line_close",
    "ou_over_price",
    "ou_under_price",
    "goal_line_half_open",
    "goal_line_half_close",
    "ah_half_open",
    "ah_half_close",
    "btts_open",
    "btts_close",
)


# --------------------------------------------------------------------------- #
# small parsers - the two response shapes are handled here and nowhere else
# --------------------------------------------------------------------------- #
def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int(value: Any) -> int | None:
    number = _num(value)
    return None if number is None else int(number)


def _stage(block: Any, stage: str = "opening") -> Any:
    if not isinstance(block, dict):
        return None
    return block.get(stage)


def _line_of(stage_value: Any) -> float | None:
    """The line from either response shape: a bare number, or ``{line, ...}``."""
    if isinstance(stage_value, dict):
        return _num(stage_value.get("line"))
    return _num(stage_value)


def _side_prices(stage_value: Any, first: str, second: str) -> tuple[float | None, float | None]:
    """``(price_first, price_second)``; both ``None`` when only a line is present."""
    if not isinstance(stage_value, dict):
        return None, None
    return _num(stage_value.get(first)), _num(stage_value.get(second))


def _favourite_side_from_pair(home: float | None, away: float | None) -> str | None:
    if home is None or away is None or home == away:
        return None
    return "home" if home < away else "away"


def _favourite_side_from_1x2(stage_value: Any) -> str | None:
    if not isinstance(stage_value, dict):
        return None
    return _favourite_side_from_pair(_num(stage_value.get("home")), _num(stage_value.get("away")))


def map_markets(markets: Any) -> dict[str, Any]:
    """Map one bookmaker's market block onto the project's ``matches`` columns.

    ``markets`` is the ``odds`` object of a fixture (list shape) or of a
    bookmaker entry (per-fixture shape).
    """
    out: dict[str, Any] = {field: None for field in ODDS_FIELDS}
    if not isinstance(markets, dict):
        return out

    one_x_two = markets.get(MARKET_1X2) or {}
    opening_1x2 = _stage(one_x_two, "opening")
    home = _num((opening_1x2 or {}).get("home")) if isinstance(opening_1x2, dict) else None
    away = _num((opening_1x2 or {}).get("away")) if isinstance(opening_1x2, dict) else None
    if home is not None and away is not None:
        out["pool_fav_open"] = min(home, away)
        out["pool_dog_open"] = max(home, away)
    if isinstance(opening_1x2, dict):
        out["draw_ft_odds"] = _num(opening_1x2.get("draw"))

    ah = markets.get(MARKET_AH) or {}
    ah_opening = _stage(ah, "opening")
    ah_line = _line_of(ah_opening)
    if ah_line is not None:
        # The API signs the line from the HOME team's point of view: -0.5 means
        # home gives half a goal. Our convention stores the FAVOURITE's
        # handicap, which is always the negative one.
        out["ah_open"] = -abs(ah_line)
        if ah_line == 0:
            # a level handicap carries no favourite, so the 1X2 decides
            out["fav_side"] = _favourite_side_from_1x2(opening_1x2) or "home"
        else:
            out["fav_side"] = "home" if ah_line < 0 else "away"

        fav_price, dog_price = _side_prices(ah_opening, "home", "away")
        if out["fav_side"] == "away":
            fav_price, dog_price = dog_price, fav_price
        out["ah_fav_price"] = fav_price
        out["ah_dog_price"] = dog_price

    # The closing line is independent information: keep it even when the
    # opening line is missing, so the drift feature is merely NaN rather than
    # the whole record being dropped.
    ah_closing = _line_of(_stage(ah, "closing"))
    if ah_closing is not None:
        out["ah_close"] = -abs(ah_closing)

    goal = markets.get(MARKET_GOAL) or {}
    goal_opening = _stage(goal, "opening")
    goal_line = _line_of(goal_opening)
    if goal_line is not None:
        out["ou_line_open"] = goal_line
        out["ou_over_price"], out["ou_under_price"] = _side_prices(
            goal_opening, "over", "under"
        )

    goal_closing = _line_of(_stage(goal, "closing"))
    if goal_closing is not None:
        out["ou_line_close"] = goal_closing

    # Closing 1X2 is independent of the opening one: keep whichever stage is
    # present. The favourite/underdog sides follow fav_side (set above from the
    # handicap line) so they stay oriented the same way as the opening prices;
    # with no handicap at all fall back to the lower closing price.
    closing_1x2 = _stage(one_x_two, "closing")
    if isinstance(closing_1x2, dict):
        home_close = _num(closing_1x2.get("home"))
        away_close = _num(closing_1x2.get("away"))
        out["draw_ft_close"] = _num(closing_1x2.get("draw"))
        if home_close is not None and away_close is not None:
            fav_side = out.get("fav_side") or _favourite_side_from_pair(home_close, away_close)
            if fav_side == "away":
                out["pool_fav_close"], out["pool_dog_close"] = away_close, home_close
            else:
                out["pool_fav_close"], out["pool_dog_close"] = home_close, away_close

    # Half-time lines. The API has no HT 1X2 market, so these lines are the
    # closest thing to the "Over HT" condition the spec asks for; they are
    # stored raw and their use is decided later. ah_half follows the same
    # always-negative-for-the-favourite convention as ah_open.
    ah_half = markets.get(MARKET_AH_HALF) or {}
    ah_half_open = _line_of(_stage(ah_half, "opening"))
    if ah_half_open is not None:
        out["ah_half_open"] = -abs(ah_half_open)
    ah_half_close = _line_of(_stage(ah_half, "closing"))
    if ah_half_close is not None:
        out["ah_half_close"] = -abs(ah_half_close)

    goal_half = markets.get(MARKET_GOAL_HALF) or {}
    goal_half_open = _line_of(_stage(goal_half, "opening"))
    if goal_half_open is not None:
        out["goal_line_half_open"] = goal_half_open
    goal_half_close = _line_of(_stage(goal_half, "closing"))
    if goal_half_close is not None:
        out["goal_line_half_close"] = goal_half_close

    btts = markets.get(MARKET_BTTS) or {}
    btts_open_stage = _stage(btts, "opening")
    if isinstance(btts_open_stage, dict):
        out["btts_open"] = _num(btts_open_stage.get("yes"))
    btts_close_stage = _stage(btts, "closing")
    if isinstance(btts_close_stage, dict):
        out["btts_close"] = _num(btts_close_stage.get("yes"))

    return out


def odds_block_for(payload: Any, bookmaker: str = "bet365") -> dict[str, Any] | None:
    """Find one bookmaker's market block in a ``/fixtures/{id}/odds`` response."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return None
    wanted = str(bookmaker).strip().lower()
    for entry in data.get("bookmakers") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("slug", "")).strip().lower() == wanted:
            block = entry.get("odds")
            if isinstance(block, dict):
                return block
    return None


def to_match_row(
    fixture: Any,
    *,
    bookmaker: str = "bet365",
    season: str | None = None,
    fetched_at: str | None = None,
) -> dict[str, Any] | None:
    """Convert one fixture (list shape, ideally with ``include=odds``) to a row."""
    if not isinstance(fixture, dict):
        return None

    teams = fixture.get("teams") or {}
    home = (teams.get("home") or {}).get("name")
    away = (teams.get("away") or {}).get("name")
    if not home or not away or fixture.get("id") is None:
        return None

    goals = fixture.get("goals") or {}
    row: dict[str, Any] = {
        "match_id": f"5dfa:{fixture.get('id')}",
        "provider": "5dollarfootballapi",
        "league": (fixture.get("league") or {}).get("name"),
        "season": season,
        "kickoff_at": fixture.get("kickoff_utc"),
        "home_team": home,
        "away_team": away,
        "home_goals": _int(goals.get("home")),
        "away_goals": _int(goals.get("away")),
        "ht_home": _int(goals.get("half_home")),
        "ht_away": _int(goals.get("half_away")),
        "odds_source": f"5dollarfootballapi/{bookmaker}",
        "result_source": "5dollarfootballapi",
        "odds_fetched_at": fetched_at,
        "result_fetched_at": fetched_at,
        "is_synthetic": 0,
    }

    odds = fixture.get("odds")
    if isinstance(odds, dict):
        # spec section 9.5: record the format this API speaks. It is decimal
        # both raw and as stored - there is no conversion anywhere on this path.
        row["odds_format_raw"] = ODDS_FORMAT
        row["odds_format_stored"] = ODDS_FORMAT
    row.update(map_markets(odds))
    return row


def coverage(rows: list[dict[str, Any]]) -> dict[str, int]:
    """How many rows actually carry each thing the model needs."""
    total = len(rows)
    finished = sum(1 for row in rows if row.get("home_goals") is not None)
    return {
        "rows": total,
        "finished": finished,
        "with_ah_line": sum(1 for row in rows if row.get("ah_open") is not None),
        "with_ah_prices": sum(1 for row in rows if row.get("ah_fav_price") is not None),
        "with_ou_line": sum(1 for row in rows if row.get("ou_line_open") is not None),
        "with_1x2": sum(1 for row in rows if row.get("pool_fav_open") is not None),
        "with_half_time_score": sum(1 for row in rows if row.get("ht_home") is not None),
        "with_closing_ah": sum(1 for row in rows if row.get("ah_close") is not None),
        "with_ht_draw_odds": sum(1 for row in rows if row.get("draw_ht_odds") is not None),
    }


# --------------------------------------------------------------------------- #
# provider
# --------------------------------------------------------------------------- #
class FiveDollarFootballProvider(BaseProvider):
    def __init__(
        self,
        cfg: Config,
        con=None,
        use_cache: bool = True,
        refresh: bool = False,
        offline: bool = False,
    ) -> None:
        super().__init__(
            name="5dollarfootballapi",
            cfg=cfg,
            con=con,
            use_cache=use_cache,
            refresh=refresh,
            offline=offline,
        )

    def _headers(self) -> dict[str, str]:
        key = self.require_key(
            self.cfg.five_dollar_key, "FIVE_DOLLAR_FOOTBALL_API_KEY"
        )
        return {"Authorization": f"Bearer {key}"}

    # ------------------------------------------------------------------ #
    # discovery
    # ------------------------------------------------------------------ #
    def account_status(self) -> dict[str, Any]:
        """Plan, limits and today's usage."""
        return self.request(
            f"{BASE_URL}/status",
            headers=self._headers(),
            request_key="status",
            endpoint="/status",
        ).data

    def list_leagues(
        self, *, popular: bool = True, search: str | None = None, per_page: int = 100
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"per_page": min(max(1, per_page), 100)}
        if popular:
            params["popular"] = 1
        if search:
            params["search"] = search
        return self.request(
            f"{BASE_URL}/leagues",
            params=params,
            headers=self._headers(),
            request_key=f"leagues-{int(popular)}-{search}-{params['per_page']}",
            endpoint="/leagues",
        ).data

    def get_league(self, league_id: str | int) -> dict[str, Any]:
        """One competition, with the valid ``?season=`` values."""
        return self.request(
            f"{BASE_URL}/leagues/{league_id}",
            headers=self._headers(),
            request_key=f"league-{league_id}",
            endpoint="/leagues/{id}",
        ).data

    def list_bookmakers(self) -> dict[str, Any]:
        return self.request(
            f"{BASE_URL}/bookmakers",
            headers=self._headers(),
            request_key="bookmakers",
            endpoint="/bookmakers",
        ).data

    # ------------------------------------------------------------------ #
    # fixtures
    # ------------------------------------------------------------------ #
    def league_fixtures(
        self,
        league_id: str | int,
        *,
        season: str | None = None,
        page: int = 1,
        per_page: int = 50,
        include: str | None = "odds",
        order: str = "asc",
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "page": max(1, page),
            "per_page": min(max(1, per_page), 50 if include else 100),
            "order": order,
        }
        if season:
            params["season"] = season
        if include:
            params["include"] = include
        if start_time is not None:
            params["start_time"] = int(start_time)
        if end_time is not None:
            params["end_time"] = int(end_time)

        return self.request(
            f"{BASE_URL}/leagues/{league_id}/fixtures",
            params=params,
            headers=self._headers(),
            request_key=(
                f"league-{league_id}-{season}-{start_time}-{end_time}-{include}-{order}-{page}"
            ),
            endpoint="/leagues/{id}/fixtures",
        ).data

    def fixture_odds(
        self,
        fixture_id: str | int,
        *,
        bookmakers: str | None = None,
        market: str | None = None,
    ) -> dict[str, Any]:
        bookmakers = bookmakers or self.cfg.bookmaker_slug
        params: dict[str, Any] = {"bookmakers": bookmakers}
        if market:
            params["market"] = market
        return self.request(
            f"{BASE_URL}/fixtures/{fixture_id}/odds",
            params=params,
            headers=self._headers(),
            request_key=f"odds-{fixture_id}-{bookmakers}-{market}",
            endpoint="/fixtures/{id}/odds",
        ).data

    def iter_league_fixtures(
        self,
        league_id: str | int,
        *,
        season: str | None = None,
        include: str | None = "odds",
        per_page: int = 50,
        order: str = "asc",
        max_pages: int = 100,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Walk every page of a league's fixtures, following ``has_more``."""
        for page in range(1, max_pages + 1):
            payload = self.league_fixtures(
                league_id,
                season=season,
                page=page,
                per_page=per_page,
                include=include,
                order=order,
                start_time=start_time,
                end_time=end_time,
            )
            for fixture in payload.get("data") or []:
                yield fixture

            pagination = payload.get("pagination") or {}
            if not pagination.get("has_more"):
                return

    # ------------------------------------------------------------------ #
    def probe(self, fixture_id: str | int | None = None) -> dict[str, Any]:
        """Report the plan, quota, leagues and one real odds block."""
        report: dict[str, Any] = {"notes": []}

        try:
            report["account"] = self.account_status().get("data")
        except ProviderError as exc:
            report["notes"].append(f"/status: {exc}")
        report["rate_limit_headers"] = self.last_rate_limit

        try:
            leagues = self.list_leagues(popular=True, per_page=20)
            report["popular_leagues"] = [
                {
                    "id": row.get("id"),
                    "name": row.get("name"),
                    "short_name": row.get("short_name"),
                }
                for row in (leagues.get("data") or [])
            ]
        except ProviderError as exc:
            report["notes"].append(f"/leagues: {exc}")

        try:
            report["bookmakers"] = [
                row.get("slug") for row in (self.list_bookmakers().get("data") or [])
            ]
        except ProviderError as exc:
            report["notes"].append(f"/bookmakers: {exc}")

        if fixture_id is not None:
            try:
                payload = self.fixture_odds(fixture_id, bookmakers=self.cfg.bookmaker_slug)
                block = odds_block_for(payload, self.cfg.bookmaker_slug)
                report["bookmaker"] = self.cfg.bookmaker_slug
                report["odds_block_found"] = block is not None
                report["markets"] = sorted(block.keys()) if block else []
                report["mapped_to_project_columns"] = map_markets(block or {})
            except ProviderError as exc:
                report["notes"].append(f"/fixtures/{fixture_id}/odds: {exc}")

        report["attribution"] = ATTRIBUTION
        return report
