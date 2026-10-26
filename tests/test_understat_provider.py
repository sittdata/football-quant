"""Unit tests for Understat xG / PPDA / xPTS Provider.
Author: cook45
"""

import pytest
from unittest.mock import patch, MagicMock
from fbpredict.providers.understat import (
    parse_understat_league_payload,
    UnderstatMatchRecord,
    calculate_ppda_ratio,
)


def test_calculate_ppda_ratio():
    # 260 passes / 23 defensive actions
    ratio = calculate_ppda_ratio({"att": 260, "def": 23})
    assert round(ratio, 2) == 11.30

    # Zero defensive actions fallback
    ratio_zero = calculate_ppda_ratio({"att": 100, "def": 0})
    assert ratio_zero == 100.0

    # None / Empty fallback
    assert calculate_ppda_ratio(None) == 0.0


def test_parse_understat_league_payload():
    mock_payload = {
        "dates": [
            {
                "id": "1001",
                "isResult": True,
                "h": {"id": "1", "title": "Arsenal", "short_title": "ARS"},
                "a": {"id": "2", "title": "Chelsea", "short_title": "CHE"},
                "goals": {"h": "3", "a": "1"},
                "xG": {"h": "2.4500", "a": "0.8500"},
                "datetime": "2024-09-15 15:00:00",
                "forecast": {"w": "0.75", "d": "0.18", "l": "0.07"},
            }
        ],
        "teams": {
            "1": {
                "title": "Arsenal",
                "history": [
                    {
                        "date": "2024-09-15 15:00:00",
                        "h_a": "h",
                        "xG": 2.45,
                        "xGA": 0.85,
                        "npxG": 2.45,
                        "npxGA": 0.85,
                        "ppda": {"att": 180, "def": 20},
                        "ppda_allowed": {"att": 210, "def": 15},
                        "deep": 9,
                        "deep_allowed": 3,
                        "xpts": 2.35,
                        "npxGD": 1.60,
                    }
                ],
            },
            "2": {
                "title": "Chelsea",
                "history": [
                    {
                        "date": "2024-09-15 15:00:00",
                        "h_a": "a",
                        "xG": 0.85,
                        "xGA": 2.45,
                        "npxG": 0.85,
                        "npxGA": 2.45,
                        "ppda": {"att": 210, "def": 15},
                        "ppda_allowed": {"att": 180, "def": 20},
                        "deep": 3,
                        "deep_allowed": 9,
                        "xpts": 0.45,
                        "npxGD": -1.60,
                    }
                ],
            },
        },
    }

    records = parse_understat_league_payload(mock_payload, league="EPL", season=2024)
    assert len(records) == 1
    r = records[0]
    assert isinstance(r, UnderstatMatchRecord)
    assert r.match_id == "1001"
    assert r.home_team == "Arsenal"
    assert r.away_team == "Chelsea"
    assert r.home_xg == 2.45
    assert r.away_xg == 0.85
    assert r.home_ppda == 9.0  # 180 / 20
    assert r.away_ppda == 14.0  # 210 / 15
    assert r.home_deep == 9
    assert r.away_deep == 3
    assert r.home_xpts == 2.35
