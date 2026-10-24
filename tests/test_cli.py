"""CLI wiring: the settle step that closes the plan-B loop."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fbpredict import cli, db, demo
from fbpredict.config import Config

from .helpers import make_match


def _configured(tmp_path, monkeypatch) -> Config:
    cfg = Config()
    cfg.data_dir = tmp_path
    cfg.db_path = tmp_path / "t.db"
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    return cfg


def test_settle_fills_in_outcome_and_profit(tmp_path, monkeypatch) -> None:
    cfg = _configured(tmp_path, monkeypatch)

    con = db.init_db(cfg.db_path)
    db.upsert_matches(con, [make_match(match_id="m1")])
    db.insert_prediction(
        con,
        match_id="m1",
        run_id="run-1",
        predicted_at="2025-01-01T09:00:00+00:00",
        kickoff_at="2025-01-01T12:00:00+00:00",
        market="AH_fav",
        p_model=0.60,
        p_implied=0.50,
        edge=0.10,
        odds_used=1.90,
        stake=1.0,
    )
    con.close()

    assert cli.main(["settle", "--run-id", "run-1"]) == 0

    con = db.connect(cfg.db_path)
    settled = db.fetch_predictions_df(con, "run-1")
    con.close()

    # make_match: home 2-1 away, favourite is home, line -0.5 -> full win
    assert settled["outcome_unit"].iloc[0] == 1.0
    assert settled["profit"].iloc[0] == pytest.approx(0.9)
    assert settled["settled_at"].notna().all()


def test_settle_refuses_an_unknown_run(tmp_path, monkeypatch) -> None:
    cfg = _configured(tmp_path, monkeypatch)
    db.init_db(cfg.db_path).close()

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["settle", "--run-id", "nope"])
    assert "no predictions recorded" in str(excinfo.value)


def test_settle_refuses_when_there_are_no_results(tmp_path, monkeypatch) -> None:
    cfg = _configured(tmp_path, monkeypatch)

    con = db.init_db(cfg.db_path)
    db.insert_prediction(
        con,
        match_id="m1",
        run_id="run-x",
        predicted_at="2025-01-01T09:00:00+00:00",
        kickoff_at="2025-01-01T12:00:00+00:00",
        market="AH_fav",
        stake=1.0,
    )
    con.close()

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["settle", "--run-id", "run-x"])
    assert "no matches with results" in str(excinfo.value)


def test_settle_leaves_unplayed_bets_unsettled(tmp_path, monkeypatch) -> None:
    """A bet whose match has no result yet must never be settled with a guess."""
    cfg = _configured(tmp_path, monkeypatch)

    con = db.init_db(cfg.db_path)
    db.upsert_matches(con, [make_match(match_id="some-other-match")])
    db.insert_prediction(
        con,
        match_id="still-to-be-played",
        run_id="run-2",
        predicted_at="2025-01-01T09:00:00+00:00",
        kickoff_at="2025-01-01T12:00:00+00:00",
        market="OU_over",
        stake=1.0,
    )
    con.close()

    assert cli.main(["settle", "--run-id", "run-2"]) == 0

    con = db.connect(cfg.db_path)
    row = db.fetch_predictions_df(con, "run-2").iloc[0]
    con.close()

    assert pd.isna(row["outcome_unit"])
    assert pd.isna(row["profit"])
    assert pd.isna(row["settled_at"])


def test_persisted_predictions_satisfy_the_look_ahead_constraint(tmp_path, monkeypatch) -> None:
    """_persist_predictions must not be able to write a look-ahead row."""
    cfg = _configured(tmp_path, monkeypatch)

    con = db.init_db(cfg.db_path)
    db.upsert_matches(con, [make_match(match_id="m1")])
    with pytest.raises(db.LookAheadError):
        db.insert_prediction(
            con,
            match_id="m1",
            run_id="run-3",
            predicted_at="2025-01-01T13:00:00+00:00",
            kickoff_at="2025-01-01T12:00:00+00:00",
            market="AH_fav",
        )
    con.close()


def test_backtest_save_predictions_writes_verifiable_rows(tmp_path, monkeypatch) -> None:
    """--save-predictions stores a run that can later be settled."""
    cfg = _configured(tmp_path, monkeypatch)
    csv_path = tmp_path / "matches.csv"
    demo.write_csv(demo.generate(seed=43, n_matches=120), csv_path)

    assert cli.main(["backtest", "--csv", str(csv_path), "--save-predictions"]) == 0

    con = db.connect(cfg.db_path)
    stored = pd.read_sql_query(
        "SELECT run_id, predicted_at, kickoff_at, market FROM predictions", con
    )
    con.close()

    assert not stored.empty
    assert stored["run_id"].nunique() == 1
    assert (stored["predicted_at"] < stored["kickoff_at"]).all()


# --------------------------------------------------------------------------- #
# feature-set flags and the closing-price guard
# --------------------------------------------------------------------------- #
def test_backtest_feature_set_flags_flow_through(tmp_path, monkeypatch) -> None:
    cfg = _configured(tmp_path, monkeypatch)
    cfg.reports_dir = tmp_path / "reports"
    csv_path = tmp_path / "matches.csv"
    demo.write_csv(demo.generate(seed=79, n_matches=130), csv_path)

    code = cli.main(
        [
            "backtest",
            "--csv",
            str(csv_path),
            "--feature-set",
            "v2_novig",
            "--predicted-before-minutes",
            "5",
            "--use-closing-features",
            "--stem",
            "run_v2nv",
        ]
    )

    assert code == 0
    text = (tmp_path / "reports" / "run_v2nv.md").read_text(encoding="utf-8")
    assert "- feature set: `v2_novig`" in text
    assert "Rule set 2 - per-rule performance" in text
    assert "Overround comparison" in text
    assert "Odds format record" in text
    assert "Rules that stay untestable" in text
    assert "prediction clock: **5 minutes**" in text


def test_backtest_refuses_closing_features_at_the_24h_clock(
    tmp_path, monkeypatch, capsys
) -> None:
    cfg = _configured(tmp_path, monkeypatch)
    cfg.reports_dir = tmp_path / "reports"
    csv_path = tmp_path / "matches.csv"
    demo.write_csv(demo.generate(seed=83, n_matches=130), csv_path)

    code = cli.main(
        ["backtest", "--csv", str(csv_path), "--feature-set", "v2", "--use-closing-features"]
    )

    assert code == 1
    assert "refusing" in capsys.readouterr().err


def test_per_fixture_attach_records_the_odds_format() -> None:
    """Regression: the per-fixture path used to skip odds_format_raw/stored."""
    class _StubProvider:
        def fixture_odds(self, fixture_id, *, bookmakers=None):
            return {
                "data": {
                    "bookmakers": [
                        {
                            "slug": "bet365",
                            "odds": {
                                "1x2": {"opening": {"home": 1.53, "draw": 3.75, "away": 5.0}}
                            },
                        }
                    ]
                }
            }

    rows = [{"match_id": "5dfa:99"}]
    attached, attempted, skipped, stopped = cli._attach_per_fixture_odds(
        _StubProvider(), rows, "bet365"
    )

    assert (attached, attempted, skipped) == (1, 1, 0)
    assert rows[0]["odds_format_raw"] == "decimal"
    assert rows[0]["odds_format_stored"] == "decimal"
    assert rows[0]["pool_fav_open"] == 1.53


def test_audit_rules_water_flag_writes_its_own_report(
    tmp_path, monkeypatch, capsys
) -> None:
    cfg = _configured(tmp_path, monkeypatch)
    cfg.reports_dir = tmp_path / "reports"
    frame = demo.generate(seed=97, n_matches=25)
    for index in (5, 8):
        frame.loc[index, "ah_open"] = -0.75
        frame.loc[index, "ah_fav_price"] = 1.85
        frame.loc[index, "pool_fav_open"] = 1.80
        frame.loc[index, "draw_ft_odds"] = 3.60
        frame.loc[index, "pool_dog_open"] = 4.10
    csv_path = tmp_path / "matches.csv"
    demo.write_csv(frame, csv_path)

    assert cli.main(["audit-rules", "--csv", str(csv_path), "--water"]) == 0

    out = capsys.readouterr().out
    assert "RULES_WATER" in out
    payload = json.loads(
        (tmp_path / "reports" / "water_rule_audit.json").read_text(encoding="utf-8")
    )
    assert payload["rows"][0]["key"] == "R7_ah-0.75"
    assert payload["rows"][0]["market"] == "AH_fav"


def test_audit_rules_water_conflicts_with_novig(tmp_path, monkeypatch) -> None:
    cfg = _configured(tmp_path, monkeypatch)
    csv_path = tmp_path / "matches.csv"
    demo.write_csv(demo.generate(seed=99, n_matches=12), csv_path)
    with pytest.raises(SystemExit):
        cli.main(["audit-rules", "--csv", str(csv_path), "--water", "--novig"])


def test_audit_rules_novig_cli_reads_the_novig_block(tmp_path, monkeypatch, capsys) -> None:
    """Regression: the --novig audit used to generate raw features internally."""
    cfg = _configured(tmp_path, monkeypatch)
    cfg.reports_dir = tmp_path / "reports"
    frame = demo.generate(seed=89, n_matches=30)
    # force one book to sit exactly on the no-vig R1 band: probabilities
    # 0.45/0.25/0.30 priced at a 1.04 margin; the raw R1 band (fav <= 2.09)
    # does NOT contain 2.137, so only the vig-free block can fire.
    frame.loc[5, "ah_open"] = -0.25
    frame.loc[5, "pool_fav_open"] = 1.0 / (0.45 * 1.04)
    frame.loc[5, "draw_ft_odds"] = 1.0 / (0.25 * 1.04)
    frame.loc[5, "pool_dog_open"] = 1.0 / (0.30 * 1.04)
    csv_path = tmp_path / "matches.csv"
    demo.write_csv(frame, csv_path)

    assert (
        cli.main(["audit-rules", "--csv", str(csv_path), "--novig", "--stem", "audit_nv"]) == 0
    )

    out = capsys.readouterr().out
    assert "RULES_V2_NOVIG" in out
    text = (tmp_path / "reports" / "audit_nv.md").read_text(encoding="utf-8")
    assert "| R1 |" in text
    payload = json.loads((tmp_path / "reports" / "audit_nv.json").read_text(encoding="utf-8"))
    keys = {row["key"] for row in payload["rows"]}
    assert "R1_ah-0.25" in keys
