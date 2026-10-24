"""Joining odds rows to result rows: ID bookkeeping and the tolerance default."""

from __future__ import annotations

import inspect

import pandas as pd

from fbpredict import linking


def _odds(match_id: str, home: str, away: str, kickoff: str) -> dict:
    return {
        "match_id": match_id,
        "home_team": home,
        "away_team": away,
        "kickoff_at": kickoff,
    }


def _result(
    match_id: str, home: str, away: str, kickoff: str, home_goals: int, away_goals: int
) -> dict:
    return {
        "match_id": match_id,
        "home_team": home,
        "away_team": away,
        "kickoff_at": kickoff,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "ht_home": 1,
        "ht_away": 0,
        "result_source": "test",
        "result_fetched_at": kickoff,
    }


def test_team_names_are_normalised() -> None:
    assert linking.normalise_team("Arsenal FC") == "arsenal"
    assert linking.normalise_team("AS Roma") == "roma"
    assert linking.normalise_team("Borussia Mönchengladbach") == "borussia monchengladbach"
    assert linking.normalise_team(None) == ""


def test_matching_rows_are_joined() -> None:
    odds = pd.DataFrame(
        [_odds("odds:1", "Arsenal", "Chelsea", "2025-01-01T15:00:00+00:00")]
    )
    results = pd.DataFrame(
        [_result("res:1", "Arsenal FC", "Chelsea", "2025-01-01T15:00:00+00:00", 2, 1)]
    )

    report = linking.link_results(odds, results)

    assert len(report.linked) == 1
    # the linked row keeps the ODDS identity, with the result filled in
    assert report.linked["match_id"].iloc[0] == "odds:1"
    assert report.linked["home_goals"].iloc[0] == 2
    assert report.linked["result_source"].iloc[0] == "test"


def test_unmatched_results_uses_the_result_id_namespace() -> None:
    """Regression: comparing odds ids against result ids reported everything.

    Linked rows carry the odds match_id, so the matched result ids have to be
    tracked separately. Before the fix, ``unmatched_results`` always equalled
    the whole result set, which made the join look broken when it was not.
    """
    odds = pd.DataFrame(
        [
            _odds("odds:1", "Arsenal", "Chelsea", "2025-01-01T15:00:00+00:00"),
            _odds("odds:2", "Everton", "Fulham", "2025-01-02T15:00:00+00:00"),
        ]
    )
    results = pd.DataFrame(
        [
            _result("res:1", "Arsenal", "Chelsea", "2025-01-01T15:00:00+00:00", 2, 1),
            _result("res:2", "Everton", "Fulham", "2025-01-02T15:00:00+00:00", 0, 0),
        ]
    )

    report = linking.link_results(odds, results)

    assert len(report.linked) == 2
    assert report.unmatched_results == []
    assert report.unmatched_odds == []
    assert report.ambiguous == []


def test_a_result_with_no_odds_counterpart_is_reported() -> None:
    odds = pd.DataFrame(
        [_odds("odds:1", "Arsenal", "Chelsea", "2025-01-01T15:00:00+00:00")]
    )
    results = pd.DataFrame(
        [
            _result("res:1", "Arsenal", "Chelsea", "2025-01-01T15:00:00+00:00", 2, 1),
            _result("res:2", "Everton", "Fulham", "2025-01-02T15:00:00+00:00", 0, 0),
        ]
    )

    report = linking.link_results(odds, results)

    assert report.unmatched_results == ["res:2"]
    assert report.unmatched_odds == []


def test_home_and_away_are_never_swapped() -> None:
    odds = pd.DataFrame(
        [_odds("odds:1", "Arsenal", "Chelsea", "2025-01-01T15:00:00+00:00")]
    )
    results = pd.DataFrame(
        [_result("res:1", "Chelsea", "Arsenal", "2025-01-01T15:00:00+00:00", 2, 1)]
    )

    report = linking.link_results(odds, results)

    assert report.linked.empty
    assert report.unmatched_odds == ["odds:1"]


def test_kickoff_outside_the_tolerance_is_not_matched() -> None:
    odds = pd.DataFrame(
        [_odds("odds:1", "Arsenal", "Chelsea", "2025-01-01T15:00:00+00:00")]
    )
    results = pd.DataFrame(
        [_result("res:1", "Arsenal", "Chelsea", "2025-01-05T15:00:00+00:00", 2, 1)]
    )

    assert linking.link_results(odds, results).linked.empty

    wider = linking.link_results(odds, results, tolerance_hours=24 * 7)
    assert len(wider.linked) == 1


def test_duplicate_candidates_are_reported_as_ambiguous() -> None:
    odds = pd.DataFrame(
        [_odds("odds:1", "Arsenal", "Chelsea", "2025-01-01T15:00:00+00:00")]
    )
    results = pd.DataFrame(
        [
            _result("res:1", "Arsenal", "Chelsea", "2025-01-01T15:00:00+00:00", 2, 1),
            _result("res:2", "Arsenal", "Chelsea", "2025-01-01T18:00:00+00:00", 0, 3),
        ]
    )

    report = linking.link_results(odds, results)

    assert report.ambiguous == ["odds:1"]
    # the closest kick-off wins, but the caller is told it was a choice
    assert report.linked["home_goals"].iloc[0] == 2


def test_empty_input_returns_a_usable_report() -> None:
    empty = pd.DataFrame()
    report = linking.link_results(empty, empty)
    assert report.linked.empty
    assert report.unmatched_odds == []
    assert report.unmatched_results == []


def test_tolerance_default_is_single_sourced() -> None:
    signature = inspect.signature(linking.link_results)
    assert (
        signature.parameters["tolerance_hours"].default
        == linking.DEFAULT_TOLERANCE_HOURS
    )
