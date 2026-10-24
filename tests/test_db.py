"""Storage: the evidence archive and the look-ahead guard."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from fbpredict import db

from .helpers import make_match


def test_look_ahead_is_refused_in_python(tmp_path: Path) -> None:
    con = db.init_db(tmp_path / "t.db")
    with pytest.raises(db.LookAheadError):
        db.insert_prediction(
            con,
            match_id="m1",
            run_id="r1",
            predicted_at="2025-01-01T13:00:00+00:00",
            kickoff_at="2025-01-01T12:00:00+00:00",
        )


def test_prediction_at_kickoff_is_refused(tmp_path: Path) -> None:
    con = db.init_db(tmp_path / "t.db")
    with pytest.raises(db.LookAheadError):
        db.insert_prediction(
            con,
            match_id="m1",
            run_id="r1",
            predicted_at="2025-01-01T12:00:00+00:00",
            kickoff_at="2025-01-01T12:00:00+00:00",
        )


def test_the_schema_itself_blocks_look_ahead(tmp_path: Path) -> None:
    """Defence in depth: even a raw INSERT cannot slip past the CHECK."""
    con = db.init_db(tmp_path / "t.db")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO predictions (match_id, run_id, predicted_at, kickoff_at) "
            "VALUES (?, ?, ?, ?)",
            ("m1", "r1", "2025-01-01T13:00:00+00:00", "2025-01-01T12:00:00+00:00"),
        )


def test_valid_prediction_is_accepted(tmp_path: Path) -> None:
    con = db.init_db(tmp_path / "t.db")
    row_id = db.insert_prediction(
        con,
        match_id="m1",
        run_id="r1",
        predicted_at="2025-01-01T09:00:00+00:00",
        kickoff_at="2025-01-01T12:00:00+00:00",
        market="AH_fav",
        p_model=0.58,
        p_implied=0.51,
        edge=0.07,
        odds_used=1.95,
        stake=1.0,
    )
    assert row_id > 0
    stored = db.fetch_predictions_df(con, "r1")
    assert stored["market"].iloc[0] == "AH_fav"


def test_archive_raw_keeps_the_payload_and_writes_a_file(tmp_path: Path) -> None:
    con = db.init_db(tmp_path / "t.db")
    fetched_at = db.archive_raw(
        con,
        provider="demo",
        endpoint="/odds",
        payload={"hello": "world"},
        request_key="fixture-1",
        raw_dir=tmp_path / "raw",
        slug="fixture-1",
    )
    row = con.execute("SELECT * FROM raw_payload").fetchone()
    assert row["provider"] == "demo"
    assert row["fetched_at"] == fetched_at
    assert json.loads(row["payload"]) == {"hello": "world"}
    assert Path(row["file_path"]).exists()


def test_archive_raw_stores_html_verbatim(tmp_path: Path) -> None:
    con = db.init_db(tmp_path / "t.db")
    html = "<html><body>ตารางบอล</body></html>"
    db.archive_raw(
        con, provider="thscore", endpoint="/football/schedule/today", payload=html, request_key="today"
    )
    row = con.execute("SELECT payload FROM raw_payload").fetchone()
    assert row["payload"] == html


def test_cached_payload_round_trip(tmp_path: Path) -> None:
    con = db.init_db(tmp_path / "t.db")
    db.archive_raw(
        con, provider="p", endpoint="/e", payload={"v": 1}, request_key="k"
    )
    cached = db.cached_payload(con, provider="p", endpoint="/e", request_key="k")
    assert cached is not None
    payload, fetched_at = cached
    assert json.loads(payload) == {"v": 1}
    # the ORIGINAL fetch time comes back, so a caller can judge staleness
    assert fetched_at.endswith("+00:00")
    assert db.cached_payload(con, provider="p", endpoint="/e", request_key="other") is None


def test_upsert_merges_odds_and_result_rows(tmp_path: Path) -> None:
    """Odds arrive first, results later; neither may wipe the other."""
    con = db.init_db(tmp_path / "t.db")

    odds_row = {
        "match_id": "m1",
        "provider": "oddspapi",
        "kickoff_at": "2025-01-01T12:00:00+00:00",
        "home_team": "Alpha",
        "away_team": "Beta",
        "fav_side": "home",
        "ah_open": -0.5,
        "ah_fav_price": 1.90,
        "ah_dog_price": 1.90,
        "pool_fav_open": 1.80,
        "ou_line_open": 2.5,
        "odds_source": "Pinnacle",
    }
    result_row = {
        "match_id": "m1",
        "provider": "football-data.org",
        "kickoff_at": "2025-01-01T12:00:00+00:00",
        "home_team": "Alpha",
        "away_team": "Beta",
        "home_goals": 2,
        "away_goals": 1,
        "ht_home": 1,
        "ht_away": 0,
        "result_source": "football-data.org",
    }

    db.upsert_matches(con, [odds_row])
    db.upsert_matches(con, [result_row])

    frame = db.fetch_matches_df(con, require_result=True)
    assert len(frame) == 1
    assert frame["ah_open"].iloc[0] == -0.5      # odds survived
    assert frame["home_goals"].iloc[0] == 2      # result landed
    assert frame["odds_source"].iloc[0] == "Pinnacle"


def test_require_result_filters_unplayed_matches(tmp_path: Path) -> None:
    con = db.init_db(tmp_path / "t.db")
    db.upsert_matches(con, [make_match(match_id="played")])
    db.upsert_matches(con, [make_match(match_id="future", home_goals=None, away_goals=None)])
    assert len(db.fetch_matches_df(con, require_result=True)) == 1
    assert len(db.fetch_matches_df(con, require_result=False)) == 2


def test_record_run_stores_metadata(tmp_path: Path) -> None:
    con = db.init_db(tmp_path / "t.db")
    db.record_run(
        con,
        run_id="run-1",
        kind="backtest",
        config={"threshold": 0.03},
        model_sha256="deadbeef",
        metrics={"roi": 0.01},
    )
    row = con.execute("SELECT * FROM runs WHERE run_id = 'run-1'").fetchone()
    assert json.loads(row["config_json"])["threshold"] == 0.03
    assert row["model_sha256"] == "deadbeef"


def test_utc_timestamps_are_normalised() -> None:
    assert db.to_iso_utc("2025-01-01T12:00:00Z") == "2025-01-01T12:00:00+00:00"
    assert db.to_iso_utc("2025-01-01T14:00:00+02:00") == "2025-01-01T12:00:00+00:00"


# --------------------------------------------------------------------------- #
# the columns added for the closing-price / half-line / traceability work
# --------------------------------------------------------------------------- #
def test_new_price_columns_round_trip(tmp_path: Path) -> None:
    con = db.init_db(tmp_path / "t.db")
    db.upsert_matches(
        con,
        [
            make_match(
                match_id="m1",
                pool_fav_close=1.75,
                pool_dog_close=4.40,
                draw_ft_close=3.50,
                goal_line_half_open=1.25,
                goal_line_half_close=1.25,
                ah_half_open=-0.5,
                ah_half_close=-0.25,
                btts_open=1.80,
                btts_close=1.95,
                odds_format_raw="decimal",
                odds_format_stored="decimal",
            )
        ],
    )
    frame = db.fetch_matches_df(con, require_result=True)
    stored = frame.iloc[0]

    assert stored["pool_fav_close"] == 1.75
    assert stored["draw_ft_close"] == 3.50
    assert stored["goal_line_half_open"] == 1.25
    assert stored["ah_half_close"] == -0.25
    assert stored["btts_open"] == 1.80
    assert stored["btts_close"] == 1.95
    assert stored["odds_format_raw"] == "decimal"
    assert stored["odds_format_stored"] == "decimal"


def test_an_older_database_gains_the_new_columns(tmp_path: Path) -> None:
    """CREATE TABLE IF NOT EXISTS cannot add columns, so _migrate must."""
    path = tmp_path / "old.db"
    con = db.init_db(path)
    con.execute("ALTER TABLE matches DROP COLUMN pool_fav_close")
    con.execute("ALTER TABLE matches DROP COLUMN odds_format_raw")
    con.commit()
    con.close()

    con = db.init_db(path)  # reopening must migrate, not fail
    columns = {row["name"] for row in con.execute("PRAGMA table_info(matches)")}
    assert "pool_fav_close" in columns
    assert "odds_format_raw" in columns
