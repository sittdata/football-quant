"""fetch-5dollar end to end, with the network stubbed out."""

from __future__ import annotations

import pandas as pd

from fbpredict import cli, db
from fbpredict.config import Config


def _fixture(
    fixture_id: int,
    home: str,
    away: str,
    *,
    goals: dict | None = None,
    odds: dict | None = None,
    kickoff: str = "2026-07-12T00:00:00+00:00",
) -> dict:
    payload = {
        "id": fixture_id,
        "league": {"id": 1, "name": "Test League"},
        "teams": {"home": {"id": 1, "name": home}, "away": {"id": 2, "name": away}},
        "kickoff_utc": kickoff,
        "status": "finished" if goals else "scheduled",
        "goals": goals
        or {"home": None, "away": None, "half_home": None, "half_away": None},
    }
    if odds is not None:
        payload["odds"] = odds
    return payload


_FINISHED_GOALS = {"home": 2, "away": 1, "half_home": 1, "half_away": 0}

_LIST_ODDS = {
    "1x2": {"opening": {"home": 1.80, "draw": 3.60, "away": 4.20}},
    "asian_handicap": {"opening": -0.5, "closing": -0.75},
    "goal_line": {"opening": 2.5, "closing": 2.75},
}

_PINNACLE_BLOCK = {
    "1x2": {"opening": {"home": 1.82, "draw": 3.55, "away": 4.30}},
    "asian_handicap": {"opening": {"line": -0.5, "home": 1.95, "away": 1.88}},
    "goal_line": {"opening": {"line": 2.5, "over": 1.92, "under": 1.88}},
}


class _StubProvider:
    """Stands in for FiveDollarFootballProvider and records what was asked for."""

    last_rate_limit = {"limit": "60", "remaining": "58", "reset": "1700000000"}

    def __init__(self, fixtures, odds_by_fixture=None):
        self._fixtures = fixtures
        self._odds = odds_by_fixture or {}
        self.calls: list[tuple] = []

    def iter_league_fixtures(
        self, league_id, *, season=None, include="odds", max_pages=100, order="asc", **_kwargs
    ):
        self.calls.append(("list", league_id, season, include))
        yield from self._fixtures

    def fixture_odds(self, fixture_id, *, bookmakers=None, market=None):
        self.calls.append(("odds", str(fixture_id), bookmakers))
        return {
            "data": {
                "bookmakers": [
                    {"slug": bookmakers, "odds": self._odds.get(str(fixture_id), {})}
                ]
            }
        }

    def list_leagues(self, **_kwargs):
        return {"data": [{"id": 3120672213, "name": "Premier League", "short_name": "EPL"}]}


def _configured(tmp_path, monkeypatch, stub) -> Config:
    cfg = Config()
    cfg.data_dir = tmp_path
    cfg.db_path = tmp_path / "t.db"
    cfg.bookmaker_slug = "bet365"
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "FiveDollarFootballProvider", lambda *a, **k: stub)
    return cfg


def test_fetch_writes_results_and_opening_odds_in_one_pass(tmp_path, monkeypatch) -> None:
    fixtures = [
        _fixture(1, "Arsenal", "Chelsea", goals=_FINISHED_GOALS, odds=_LIST_ODDS),
        _fixture(2, "Everton", "Fulham", odds=_LIST_ODDS),
    ]
    stub = _StubProvider(fixtures)
    cfg = _configured(tmp_path, monkeypatch, stub)

    assert cli.main(["fetch-5dollar", "--league", "1", "--season", "26/27"]) == 0

    con = db.connect(cfg.db_path)
    stored = db.fetch_matches_df(con, include_synthetic=False, require_result=False)
    con.close()

    assert len(stored) == 2
    played = stored[stored["match_id"] == "5dfa:1"].iloc[0]
    assert played["home_goals"] == 2 and played["away_goals"] == 1
    assert played["ht_home"] == 1 and played["ht_away"] == 0
    assert played["ah_open"] == -0.5
    assert played["ou_line_open"] == 2.5
    assert played["pool_fav_open"] == 1.80
    assert played["season"] == "26/27"

    # the list shape carries the line but not the prices
    assert pd.isna(played["ah_fav_price"])


def test_per_fixture_odds_attach_the_prices(tmp_path, monkeypatch) -> None:
    fixtures = [
        _fixture(1, "Arsenal", "Chelsea", goals=_FINISHED_GOALS, odds=_LIST_ODDS),
        _fixture(2, "Everton", "Fulham", goals=_FINISHED_GOALS, odds=_LIST_ODDS),
    ]
    stub = _StubProvider(fixtures, {"1": _PINNACLE_BLOCK, "2": _PINNACLE_BLOCK})
    cfg = _configured(tmp_path, monkeypatch, stub)

    assert cli.main(["fetch-5dollar", "--league", "1", "--per-fixture-odds"]) == 0

    con = db.connect(cfg.db_path)
    stored = db.fetch_matches_df(con, include_synthetic=False, require_result=False)
    con.close()

    row = stored[stored["match_id"] == "5dfa:1"].iloc[0]
    assert row["ah_fav_price"] == 1.95
    assert row["ah_dog_price"] == 1.88
    assert row["ou_over_price"] == 1.92
    assert row["ou_under_price"] == 1.88
    # one odds call per fixture, and the bookmaker was passed through
    assert [call[0] for call in stub.calls].count("odds") == 2
    assert all(call[2] == "bet365" for call in stub.calls if call[0] == "odds")


def test_no_odds_skips_the_include(tmp_path, monkeypatch) -> None:
    stub = _StubProvider([_fixture(1, "Arsenal", "Chelsea", goals=_FINISHED_GOALS)])
    cfg = _configured(tmp_path, monkeypatch, stub)

    assert cli.main(["fetch-5dollar", "--league", "1", "--no-odds"]) == 0

    assert stub.calls[0][3] is None  # include=None
    con = db.connect(cfg.db_path)
    stored = db.fetch_matches_df(con, include_synthetic=False, require_result=False)
    con.close()
    assert pd.isna(stored["ah_open"].iloc[0])
    assert stored["home_goals"].iloc[0] == 2


def test_missing_league_prints_the_ids_instead_of_failing_obscurely(
    tmp_path, monkeypatch, capsys
) -> None:
    stub = _StubProvider([])
    _configured(tmp_path, monkeypatch, stub)

    assert cli.main(["fetch-5dollar"]) == 1

    out = capsys.readouterr().out
    assert "3120672213" in out
    assert "Premier League" in out


def test_attribution_is_printed(tmp_path, monkeypatch, capsys) -> None:
    """Free and Community plans require attribution in public output."""
    stub = _StubProvider([_fixture(1, "Arsenal", "Chelsea", goals=_FINISHED_GOALS)])
    _configured(tmp_path, monkeypatch, stub)

    assert cli.main(["fetch-5dollar", "--league", "1"]) == 0
    assert "5DollarFootballAPI" in capsys.readouterr().out


def test_max_fixtures_stops_early(tmp_path, monkeypatch) -> None:
    fixtures = [
        _fixture(index, f"Home {index}", f"Away {index}", goals=_FINISHED_GOALS)
        for index in range(1, 11)
    ]
    stub = _StubProvider(fixtures)
    cfg = _configured(tmp_path, monkeypatch, stub)

    assert cli.main(["fetch-5dollar", "--league", "1", "--max-fixtures", "3"]) == 0

    con = db.connect(cfg.db_path)
    stored = db.fetch_matches_df(con, include_synthetic=False, require_result=False)
    con.close()
    assert len(stored) == 3
