"""5DollarFootballAPI mapping: both response shapes, and the sign convention."""

from __future__ import annotations

from fbpredict.providers import five_dollar as fd


# --------------------------------------------------------------------------- #
# list shape: /v1/leagues/{id}/fixtures?include=odds  (line only, no prices)
# --------------------------------------------------------------------------- #
def test_list_shape_gives_the_line_but_no_prices() -> None:
    markets = {
        "1x2": {"opening": {"home": 1.53, "draw": 3.75, "away": 5.0}, "closing": {}},
        "asian_handicap": {"opening": -0.25, "closing": -0.5, "inplay": -1.0},
        "goal_line": {"opening": 2.5, "closing": 2.75, "inplay": 3.0},
    }
    mapped = fd.map_markets(markets)

    assert mapped["ah_open"] == -0.25
    assert mapped["ah_close"] == -0.5
    assert mapped["ah_fav_price"] is None
    assert mapped["ah_dog_price"] is None
    assert mapped["fav_side"] == "home"
    assert mapped["ou_line_open"] == 2.5
    assert mapped["ou_line_close"] == 2.75
    assert mapped["ou_over_price"] is None
    assert mapped["pool_fav_open"] == 1.53
    assert mapped["pool_dog_open"] == 5.0
    assert mapped["draw_ft_odds"] == 3.75


# --------------------------------------------------------------------------- #
# per-fixture shape: /v1/fixtures/{id}/odds  (line + both prices)
# --------------------------------------------------------------------------- #
def test_per_fixture_shape_gives_line_and_prices() -> None:
    markets = {
        "1x2": {"opening": {"home": 1.53, "draw": 3.75, "away": 5.0}},
        "asian_handicap": {
            "opening": {"line": -0.25, "home": 1.95, "away": 1.85},
            "closing": {"line": -0.5, "home": 2.02, "away": 1.78},
            "inplay": {"line": -1.0, "home": 1.9, "away": 1.9},
        },
        "goal_line": {
            "opening": {"line": 2.5, "over": 1.9, "under": 1.9},
            "closing": {"line": 2.75, "over": 1.85, "under": 1.95},
        },
    }
    mapped = fd.map_markets(markets)

    assert mapped["ah_open"] == -0.25
    assert mapped["ah_close"] == -0.5          # opening line did drift
    assert mapped["fav_side"] == "home"
    assert mapped["ah_fav_price"] == 1.95
    assert mapped["ah_dog_price"] == 1.85
    assert mapped["ou_line_open"] == 2.5
    assert mapped["ou_line_close"] == 2.75
    assert mapped["ou_over_price"] == 1.9
    assert mapped["ou_under_price"] == 1.9


def test_away_favourite_swaps_the_prices() -> None:
    """A positive line means HOME receives, so the away side is the favourite."""
    markets = {
        "1x2": {"opening": {"home": 5.0, "draw": 3.75, "away": 1.53}},
        "asian_handicap": {"opening": {"line": 0.5, "home": 2.05, "away": 1.80}},
    }
    mapped = fd.map_markets(markets)

    assert mapped["fav_side"] == "away"
    # our convention stores the favourite's handicap, always negative
    assert mapped["ah_open"] == -0.5
    assert mapped["ah_fav_price"] == 1.80
    assert mapped["ah_dog_price"] == 2.05
    # the price pool follows the same side
    assert mapped["pool_fav_open"] == 1.53
    assert mapped["pool_dog_open"] == 5.0


def test_level_line_falls_back_to_the_1x2_for_the_favourite() -> None:
    markets = {
        "1x2": {"opening": {"home": 1.60, "draw": 4.0, "away": 5.5}},
        "asian_handicap": {"opening": {"line": 0, "home": 1.9, "away": 1.9}},
    }
    mapped = fd.map_markets(markets)

    assert mapped["ah_open"] == 0.0
    assert mapped["fav_side"] == "home"


def test_the_sign_convention_is_always_negative_for_the_favourite() -> None:
    for line in (-2.0, -1.25, -0.5, -0.25, 0.25, 0.5, 1.25, 2.0):
        mapped = fd.map_markets({"asian_handicap": {"opening": line}})
        assert mapped["ah_open"] == -abs(line), line
        assert mapped["ah_open"] <= 0.0


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #
def test_missing_markets_are_all_none() -> None:
    for empty in (None, {}, [], "nonsense"):
        mapped = fd.map_markets(empty)
        assert all(mapped[field] is None for field in fd.ODDS_FIELDS), empty


# --------------------------------------------------------------------------- #
# closing 1X2, the half-time lines and btts (the columns that used to be lost)
# --------------------------------------------------------------------------- #
def test_closing_1x2_is_oriented_like_the_opening_prices() -> None:
    markets = {
        "1x2": {
            "opening": {"home": 1.53, "draw": 3.75, "away": 5.0},
            "closing": {"home": 1.45, "draw": 4.50, "away": 7.0},
        },
    }
    mapped = fd.map_markets(markets)
    assert mapped["pool_fav_close"] == 1.45
    assert mapped["pool_dog_close"] == 7.0
    assert mapped["draw_ft_close"] == 4.50


def test_closing_prices_swap_for_an_away_favourite() -> None:
    """The away side is the favourite, so the closing prices swap too."""
    markets = {
        "1x2": {
            "opening": {"home": 5.0, "draw": 3.75, "away": 1.53},
            "closing": {"home": 7.0, "draw": 4.50, "away": 1.45},
        },
        "asian_handicap": {"opening": {"line": 0.5, "home": 2.05, "away": 1.80}},
    }
    mapped = fd.map_markets(markets)
    assert mapped["fav_side"] == "away"
    assert mapped["pool_fav_close"] == 1.45
    assert mapped["pool_dog_close"] == 7.0


def test_closing_only_book_still_gets_oriented() -> None:
    """No opening and no handicap: the lower closing price is the favourite."""
    markets = {"1x2": {"closing": {"home": 5.0, "draw": 3.75, "away": 1.53}}}
    mapped = fd.map_markets(markets)
    assert mapped["fav_side"] is None
    assert mapped["pool_fav_open"] is None
    assert mapped["pool_fav_close"] == 1.53
    assert mapped["pool_dog_close"] == 5.0


def test_half_time_lines_are_read_from_both_stages() -> None:
    markets = {
        "asian_handicap_half": {
            "opening": {"line": -0.5, "home": 1.975, "away": 1.825},
            "closing": {"line": -0.75, "home": 2.0, "away": 1.8},
        },
        "goal_line_half": {
            "opening": {"line": 1.25, "over": 2.0, "under": 1.8},
            "closing": {"line": 1.0, "over": 2.05, "under": 1.75},
        },
        "btts": {
            "opening": {"yes": 1.8, "no": 1.95},
            "closing": {"yes": 1.95, "no": 1.8},
        },
    }
    mapped = fd.map_markets(markets)

    # ah_half keeps the same always-negative-for-the-favourite convention
    assert mapped["ah_half_open"] == -0.5
    assert mapped["ah_half_close"] == -0.75
    assert mapped["goal_line_half_open"] == 1.25
    assert mapped["goal_line_half_close"] == 1.0
    assert mapped["btts_open"] == 1.8
    assert mapped["btts_close"] == 1.95


def test_half_time_markets_missing_stay_null() -> None:
    mapped = fd.map_markets({"goal_line": {"opening": 2.5}})
    for field in (
        "goal_line_half_open",
        "goal_line_half_close",
        "ah_half_open",
        "ah_half_close",
        "btts_open",
        "btts_close",
    ):
        assert mapped[field] is None, field


def test_odds_format_is_recorded_only_when_odds_were_present() -> None:
    with_odds = _fixture(
        odds={"1x2": {"opening": {"home": 1.53, "draw": 3.75, "away": 5.0}}}
    )
    row = fd.to_match_row(with_odds)
    assert row is not None
    assert row["odds_format_raw"] == "decimal"
    assert row["odds_format_stored"] == "decimal"

    row_without = fd.to_match_row(_fixture())
    assert row_without is not None
    assert row_without.get("odds_format_raw") is None


def test_partial_markets_do_not_invent_values() -> None:
    mapped = fd.map_markets({"asian_handicap": {"closing": -0.75}})
    assert mapped["ah_open"] is None       # no opening line -> nothing to settle on
    assert mapped["ah_close"] == -0.75
    assert mapped["fav_side"] is None
    assert mapped["ou_line_open"] is None


def test_nan_and_garbage_are_treated_as_missing() -> None:
    mapped = fd.map_markets(
        {
            "asian_handicap": {"opening": {"line": "not-a-number", "home": None, "away": "x"}},
            "goal_line": {"opening": {"line": float("nan"), "over": 1.9, "under": 1.9}},
        }
    )
    assert mapped["ah_open"] is None
    assert mapped["ou_line_open"] is None
    assert mapped["ou_over_price"] is None


# --------------------------------------------------------------------------- #
# odds_block_for
# --------------------------------------------------------------------------- #
def _odds_response(slug: str, block: dict) -> dict:
    return {
        "success": 1,
        "data": {
            "fixture_id": 1,
            "bookmakers": [
                {"name": "Pinnacle", "slug": "pinnacle", "odds": {"1x2": {}}},
                {"name": "Bet 365", "slug": slug, "odds": block},
            ],
        },
    }


def test_odds_block_for_picks_the_requested_bookmaker() -> None:
    block = {"asian_handicap": {"opening": {"line": -0.5, "home": 1.9, "away": 1.9}}}
    payload = _odds_response("bet365", block)

    assert fd.odds_block_for(payload, "bet365") == block
    assert fd.odds_block_for(payload, "BET365") == block   # case-insensitive
    assert fd.odds_block_for(payload, "williamhill") is None


def test_odds_block_for_handles_a_broken_payload() -> None:
    assert fd.odds_block_for(None) is None
    assert fd.odds_block_for({}) is None
    assert fd.odds_block_for({"data": None}) is None
    assert fd.odds_block_for({"data": {"bookmakers": None}}) is None
    assert fd.odds_block_for({"data": {"bookmakers": ["nonsense"]}}) is None


# --------------------------------------------------------------------------- #
# to_match_row
# --------------------------------------------------------------------------- #
def _fixture(**overrides) -> dict:
    base = {
        "id": 422841891,
        "league": {"id": 1, "name": "USA USL Cup"},
        "teams": {
            "home": {"id": 11, "name": "Birmingham Legion FC"},
            "away": {"id": 22, "name": "Tulsa"},
        },
        "kickoff_utc": "2026-07-12T00:00:00+00:00",
        "status": "finished",
        "goals": {"home": 3, "away": 1, "half_home": 1, "half_away": 1},
    }
    base.update(overrides)
    return base


def test_to_match_row_maps_results_and_half_time() -> None:
    row = fd.to_match_row(_fixture(), season="26/27", fetched_at="2026-07-13T00:00:00+00:00")

    assert row is not None
    assert row["match_id"] == "5dfa:422841891"
    assert row["provider"] == "5dollarfootballapi"
    assert row["home_team"] == "Birmingham Legion FC"
    assert row["home_goals"] == 3 and row["away_goals"] == 1
    # half-time score is what the brief needs for its HT rules
    assert row["ht_home"] == 1 and row["ht_away"] == 1
    assert row["season"] == "26/27"
    assert row["is_synthetic"] == 0
    assert row["result_fetched_at"] == "2026-07-13T00:00:00+00:00"


def test_to_match_row_with_odds_included() -> None:
    fixture = _fixture(
        odds={
            "1x2": {"opening": {"home": 1.53, "draw": 3.75, "away": 5.0}},
            "asian_handicap": {"opening": -0.25, "closing": -0.5},
            "goal_line": {"opening": 2.5, "closing": 2.75},
        }
    )
    row = fd.to_match_row(fixture)

    assert row is not None
    assert row["ah_open"] == -0.25
    assert row["ou_line_open"] == 2.5
    assert row["pool_fav_open"] == 1.53
    assert row["odds_source"] == "5dollarfootballapi/bet365"


def test_to_match_row_without_odds_leaves_them_null() -> None:
    row = fd.to_match_row(_fixture())
    assert row is not None
    assert row["ah_open"] is None
    assert row["pool_fav_open"] is None
    assert row["home_goals"] == 3          # the result still lands


def test_unplayed_fixture_has_no_goals() -> None:
    row = fd.to_match_row(
        _fixture(status="scheduled", goals={"home": None, "away": None,
                                            "half_home": None, "half_away": None})
    )
    assert row is not None
    assert row["home_goals"] is None
    assert row["ht_home"] is None


def test_to_match_row_rejects_unusable_rows() -> None:
    assert fd.to_match_row(None) is None
    assert fd.to_match_row({}) is None
    assert fd.to_match_row({"id": 1, "teams": {}}) is None
    assert fd.to_match_row({"id": None, "teams": {"home": {"name": "A"}, "away": {"name": "B"}}}) is None


# --------------------------------------------------------------------------- #
# coverage
# --------------------------------------------------------------------------- #
def test_coverage_counts_what_is_actually_present() -> None:
    rows = [
        fd.to_match_row(
            _fixture(
                id=1,
                odds={
                    "1x2": {"opening": {"home": 1.5, "draw": 3.7, "away": 5.0}},
                    "asian_handicap": {"opening": -0.25},
                    "goal_line": {"opening": 2.5},
                },
            )
        ),
        fd.to_match_row(_fixture(id=2)),  # result only
    ]
    report = fd.coverage([row for row in rows if row])

    assert report["rows"] == 2
    assert report["finished"] == 2
    assert report["with_ah_line"] == 1
    assert report["with_ah_prices"] == 0     # list shape carries no prices
    assert report["with_ou_line"] == 1
    assert report["with_1x2"] == 1
    assert report["with_half_time_score"] == 2
    assert report["with_closing_ah"] == 0
    # the brief's HT draw price is simply not available from this endpoint
    assert report["with_ht_draw_odds"] == 0
