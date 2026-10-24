"""SQLite storage layer.

Two things matter here and both come straight out of the brief:

* ``raw_payload`` keeps the verbatim API/scrape response plus ``fetched_at``.
  That is the "evidence that I fetched it myself" the assignment asks for, and
  it also stops us from spending API quota twice for the same request.
* ``predictions`` has a ``CHECK (predicted_at < kickoff_at)`` constraint, so a
  look-ahead row cannot be written even by accident.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS raw_payload (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    provider    TEXT NOT NULL,
    endpoint    TEXT NOT NULL,
    request_key TEXT,
    fetched_at  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    file_path   TEXT,
    UNIQUE (provider, endpoint, request_key, fetched_at)
);
CREATE INDEX IF NOT EXISTS idx_raw_provider ON raw_payload (provider, fetched_at);

CREATE TABLE IF NOT EXISTS matches (
    match_id        TEXT PRIMARY KEY,
    provider        TEXT NOT NULL,
    league          TEXT,
    season          TEXT,
    kickoff_at      TEXT NOT NULL,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    fav_side        TEXT,            -- 'home' | 'away': which team the handicap is on

    ah_open         REAL,
    ah_fav_price    REAL,
    ah_dog_price    REAL,
    pool_fav_open   REAL,
    pool_dog_open   REAL,
    draw_ft_odds    REAL,
    draw_ht_odds    REAL,
    ou_line_open    REAL,
    ou_over_price   REAL,
    ou_under_price  REAL,
    ah_close        REAL,            -- closing handicap on the favourite (drift check)
    ou_line_close   REAL,            -- closing total line

    -- 1X2 closing prices, oriented like pool_fav_open / pool_dog_open
    pool_fav_close  REAL,
    pool_dog_close  REAL,
    draw_ft_close   REAL,

    -- half-time lines used as proxies (the API exposes no HT 1X2 market):
    -- goal_line_half = HT over/under line, ah_half = HT asian handicap (always
    -- stored on the favourite, negative), btts_* = both-teams-to-score "yes"
    goal_line_half_open  REAL,
    goal_line_half_close REAL,
    ah_half_open         REAL,
    ah_half_close        REAL,
    btts_open            REAL,
    btts_close           REAL,

    ht_home         INTEGER,
    ht_away         INTEGER,
    home_goals      INTEGER,
    away_goals      INTEGER,

    odds_source     TEXT,
    -- odds format traceability (spec section 9.5). This schema is one row per
    -- match, not one row per price, so the spec's fields map as:
    --   bookmaker      ~ odds_source
    --   captured_at    ~ odds_fetched_at
    --   is_open_price  ~ the *_open column suffix
    odds_format_raw     TEXT,
    odds_format_stored  TEXT,
    result_source   TEXT,
    odds_fetched_at TEXT,
    result_fetched_at TEXT,
    is_synthetic    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_matches_kickoff ON matches (kickoff_at);

CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id     TEXT NOT NULL,
    run_id       TEXT NOT NULL,
    predicted_at TEXT NOT NULL,
    kickoff_at   TEXT NOT NULL,
    market       TEXT,
    p_model      REAL,
    p_implied    REAL,
    edge         REAL,
    odds_used    REAL,
    stake        REAL,
    outcome_unit REAL,
    profit       REAL,
    settled_at   TEXT,
    CHECK (predicted_at < kickoff_at)
);
CREATE INDEX IF NOT EXISTS idx_pred_run ON predictions (run_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    git_commit    TEXT,
    config_json   TEXT,
    model_path    TEXT,
    model_sha256  TEXT,
    metrics_json  TEXT,
    notes         TEXT
);
"""


class LookAheadError(ValueError):
    """Raised when a prediction would be recorded at or after kick-off."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def utcnow_iso() -> str:
    """Timestamp string that sorts correctly as text (always +00:00)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def to_iso_utc(value: Any) -> str:
    """Normalise anything date-like into ``YYYY-MM-DDTHH:MM:SS+00:00``."""
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return text
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, pd.Timestamp):
        parsed = value.to_pydatetime()
    else:
        return str(value)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc).replace(microsecond=0)
    return parsed.isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root: str | Path) -> str | None:
    """Best-effort short git commit; returns None when not a git repo."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


# --------------------------------------------------------------------------- #
# connection / schema
# --------------------------------------------------------------------------- #
def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db(db_path: str | Path) -> sqlite3.Connection:
    con = connect(db_path)
    con.executescript(SCHEMA)
    con.commit()
    _migrate(con)
    return con


#: Columns added after the first release. ``CREATE TABLE IF NOT EXISTS`` will
#: not add them to an existing database, so they are applied explicitly.
MATCH_MIGRATIONS: dict[str, str] = {
    "ah_close": "REAL",
    "ou_line_close": "REAL",
    "pool_fav_close": "REAL",
    "pool_dog_close": "REAL",
    "draw_ft_close": "REAL",
    "goal_line_half_open": "REAL",
    "goal_line_half_close": "REAL",
    "ah_half_open": "REAL",
    "ah_half_close": "REAL",
    "btts_open": "REAL",
    "btts_close": "REAL",
    "odds_format_raw": "TEXT",
    "odds_format_stored": "TEXT",
}


def _migrate(con: sqlite3.Connection) -> None:
    """Add any column that an older database is missing."""
    existing = {row["name"] for row in con.execute("PRAGMA table_info(matches)")}
    added: list[str] = []
    for column, column_type in MATCH_MIGRATIONS.items():
        if column in existing:
            continue
        # identifiers come from our own constant, but check anyway
        if not column.replace("_", "").isalnum():
            continue
        con.execute(f"ALTER TABLE matches ADD COLUMN {column} {column_type}")
        added.append(column)
    if added:
        con.commit()


# --------------------------------------------------------------------------- #
# raw payload archive
# --------------------------------------------------------------------------- #
def archive_raw(
    con: sqlite3.Connection,
    *,
    provider: str,
    endpoint: str,
    payload: Any,
    request_key: str | None = None,
    raw_dir: str | Path | None = None,
    slug: str | None = None,
) -> str:
    """Store a verbatim response and return its ``fetched_at`` timestamp.

    ``payload`` may be a dict/list (stored as JSON) or an already-serialised
    string such as raw HTML.
    """
    fetched_at = utcnow_iso()
    if isinstance(payload, (dict, list)):
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    else:
        text = str(payload)

    file_path: str | None = None
    if raw_dir is not None:
        day = fetched_at[:10]
        folder = Path(raw_dir) / provider / day
        folder.mkdir(parents=True, exist_ok=True)
        stamp = fetched_at.replace(":", "").replace("-", "")
        safe_slug = slug or (request_key or "payload")
        safe_slug = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in safe_slug)[:80]
        target = folder / f"{stamp}_{safe_slug}.json"
        target.write_text(text, encoding="utf-8")
        file_path = str(target)

    con.execute(
        """
        INSERT OR IGNORE INTO raw_payload
            (provider, endpoint, request_key, fetched_at, payload, file_path)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (provider, endpoint, request_key, fetched_at, text, file_path),
    )
    con.commit()
    return fetched_at


def cached_payload(
    con: sqlite3.Connection,
    *,
    provider: str,
    endpoint: str,
    request_key: str,
) -> tuple[str, str] | None:
    """Return ``(payload, fetched_at)`` for the newest matching request.

    The timestamp is returned alongside the payload so the caller can decide
    whether the cached bytes are still fresh, instead of silently replaying a
    stale response while stamping it with the current time.
    """
    row = con.execute(
        """
        SELECT payload, fetched_at FROM raw_payload
        WHERE provider = ? AND endpoint = ? AND request_key = ?
        ORDER BY fetched_at DESC LIMIT 1
        """,
        (provider, endpoint, request_key),
    ).fetchone()
    if row is None:
        return None
    return row["payload"], row["fetched_at"]


# --------------------------------------------------------------------------- #
# matches
# --------------------------------------------------------------------------- #
MATCH_COLUMNS: tuple[str, ...] = (
    "match_id",
    "provider",
    "league",
    "season",
    "kickoff_at",
    "home_team",
    "away_team",
    "fav_side",
    "ah_open",
    "ah_fav_price",
    "ah_dog_price",
    "pool_fav_open",
    "pool_dog_open",
    "draw_ft_odds",
    "draw_ht_odds",
    "ou_line_open",
    "ou_over_price",
    "ou_under_price",
    "ah_close",
    "ou_line_close",
    "pool_fav_close",
    "pool_dog_close",
    "draw_ft_close",
    "goal_line_half_open",
    "goal_line_half_close",
    "ah_half_open",
    "ah_half_close",
    "btts_open",
    "btts_close",
    "ht_home",
    "ht_away",
    "home_goals",
    "away_goals",
    "odds_source",
    "odds_format_raw",
    "odds_format_stored",
    "result_source",
    "odds_fetched_at",
    "result_fetched_at",
    "is_synthetic",
)


def upsert_matches(con: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    """Insert or update matches. Missing keys are stored as NULL."""
    payload = []
    for row in rows:
        record = {col: row.get(col) for col in MATCH_COLUMNS}
        if record["kickoff_at"] is not None:
            record["kickoff_at"] = to_iso_utc(record["kickoff_at"])
        for stamp in ("odds_fetched_at", "result_fetched_at"):
            if record[stamp] is not None:
                record[stamp] = to_iso_utc(record[stamp])
        record["is_synthetic"] = int(bool(record.get("is_synthetic") or 0))
        if not record["match_id"] or not record["home_team"] or not record["away_team"]:
            continue
        payload.append(tuple(record[col] for col in MATCH_COLUMNS))

    if not payload:
        return 0

    placeholders = ", ".join("?" for _ in MATCH_COLUMNS)
    columns = ", ".join(MATCH_COLUMNS)
    updates = ", ".join(
        f"{col} = COALESCE(excluded.{col}, matches.{col})"
        for col in MATCH_COLUMNS
        if col != "match_id"
    )
    con.executemany(
        f"""
        INSERT INTO matches ({columns}) VALUES ({placeholders})
        ON CONFLICT(match_id) DO UPDATE SET {updates}
        """,
        payload,
    )
    con.commit()
    return len(payload)


def fetch_matches_df(
    con: sqlite3.Connection,
    *,
    include_synthetic: bool = True,
    require_result: bool = False,
) -> pd.DataFrame:
    query = "SELECT * FROM matches"
    clauses: list[str] = []
    if not include_synthetic:
        clauses.append("is_synthetic = 0")
    if require_result:
        clauses.append("home_goals IS NOT NULL AND away_goals IS NOT NULL")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY kickoff_at ASC"

    df = pd.read_sql_query(query, con)
    if not df.empty:
        df["kickoff_at"] = pd.to_datetime(df["kickoff_at"], utc=True, format="ISO8601")
    return df


# --------------------------------------------------------------------------- #
# predictions
# --------------------------------------------------------------------------- #
def insert_prediction(
    con: sqlite3.Connection,
    *,
    match_id: str,
    run_id: str,
    predicted_at: str,
    kickoff_at: str,
    market: str | None = None,
    p_model: float | None = None,
    p_implied: float | None = None,
    edge: float | None = None,
    odds_used: float | None = None,
    stake: float | None = None,
) -> int:
    """Record one prediction. Raises :class:`LookAheadError` on look-ahead."""
    predicted_at = to_iso_utc(predicted_at)
    kickoff_at = to_iso_utc(kickoff_at)
    if not predicted_at < kickoff_at:
        raise LookAheadError(
            f"refusing prediction for {match_id}: predicted_at={predicted_at} "
            f"is not before kickoff_at={kickoff_at}"
        )
    cur = con.execute(
        """
        INSERT INTO predictions
            (match_id, run_id, predicted_at, kickoff_at, market,
             p_model, p_implied, edge, odds_used, stake)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            match_id,
            run_id,
            predicted_at,
            kickoff_at,
            market,
            p_model,
            p_implied,
            edge,
            odds_used,
            stake,
        ),
    )
    con.commit()
    return int(cur.lastrowid)


def settle_predictions(
    con: sqlite3.Connection,
    run_id: str,
    settlements: Sequence[dict[str, Any]],
) -> int:
    """Fill in ``outcome_unit`` / ``profit`` once results are known."""
    settled_at = utcnow_iso()
    con.executemany(
        """
        UPDATE predictions
        SET outcome_unit = ?, profit = ?, settled_at = ?
        WHERE run_id = ? AND match_id = ?
        """,
        [
            (row.get("outcome_unit"), row.get("profit"), settled_at, run_id, row["match_id"])
            for row in settlements
        ],
    )
    con.commit()
    return len(settlements)


def fetch_predictions_df(con: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM predictions WHERE run_id = ? ORDER BY kickoff_at ASC",
        con,
        params=(run_id,),
    )


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
def record_run(
    con: sqlite3.Connection,
    *,
    run_id: str,
    kind: str,
    config: dict[str, Any] | None = None,
    model_path: str | None = None,
    model_sha256: str | None = None,
    metrics: dict[str, Any] | None = None,
    notes: str | None = None,
    project_root: str | Path | None = None,
) -> None:
    con.execute(
        """
        INSERT INTO runs
            (run_id, kind, created_at, git_commit, config_json,
             model_path, model_sha256, metrics_json, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            metrics_json = COALESCE(excluded.metrics_json, runs.metrics_json),
            notes        = COALESCE(excluded.notes, runs.notes)
        """,
        (
            run_id,
            kind,
            utcnow_iso(),
            git_commit(project_root or Path.cwd()),
            json.dumps(config or {}, default=str, sort_keys=True),
            model_path,
            model_sha256,
            json.dumps(metrics or {}, default=str, sort_keys=True),
            notes,
        ),
    )
    con.commit()
