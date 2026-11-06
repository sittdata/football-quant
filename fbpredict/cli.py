"""Command line entry point:  python -m fbpredict <command> [...]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import backtest, db, demo, features, labels, linking, model, price_pool_rules, report
from .cli_1x2 import cmd_run_1x2
from .config import load_config
from .providers import five_dollar
from .providers.api_football import ApiFootballProvider
from .providers.base import CacheMissError, ProviderError
from .providers.five_dollar import FiveDollarFootballProvider
from .providers.football_data_org import FootballDataOrgProvider
from .providers.oddspapi import OddsPapiProvider, dumps_mapping_template
from .providers.thscore import ThscoreProvider, odds_span_report


# --------------------------------------------------------------------------- #
def _load_matches(args, cfg, *, include_synthetic: bool | None = None) -> pd.DataFrame:
    """Get a match frame from a CSV or from the local database.

    Synthetic demo rows are excluded unless explicitly asked for, so a real
    backtest can never silently train on generated data.
    """
    if getattr(args, "csv", None):
        df = pd.read_csv(args.csv)
        if "kickoff_at" in df.columns:
            df["kickoff_at"] = pd.to_datetime(df["kickoff_at"], utc=True, format="ISO8601")
        return df

    if include_synthetic is None:
        include_synthetic = bool(getattr(args, "include_synthetic", False))

    con = db.init_db(cfg.db_path)
    try:
        df = db.fetch_matches_df(
            con, include_synthetic=include_synthetic, require_result=True
        )
    finally:
        con.close()
    if df.empty:
        raise SystemExit(
            "No matches with results in the database yet.\n"
            "Try one of:\n"
            "  python -m fbpredict demo\n"
            "  python -m fbpredict fetch-results --provider football-data-org --competition PL"
        )
    return df


# --------------------------------------------------------------------------- #
def cmd_init_db(args, cfg) -> int:
    con = db.init_db(cfg.db_path)
    tables = [
        row["name"]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    con.close()
    print(f"database ready: {cfg.db_path}")
    print("tables:", ", ".join(tables))
    return 0


def cmd_show_config(args, cfg) -> int:
    missing = cfg.missing_keys()
    print("paths")
    print(f"  db       : {cfg.db_path}")
    print(f"  raw      : {cfg.raw_dir}")
    print(f"  models   : {cfg.models_dir}")
    print(f"  reports  : {cfg.reports_dir}")
    print("domain")
    print(f"  league   : {cfg.league}")
    print(f"  bookmaker: {cfg.bookmaker}")
    print(f"  season   : {cfg.season}")
    print("model")
    print(f"  threshold: {cfg.edge_threshold}")
    print(f"  blacklist: {', '.join(cfg.blacklist_markets) if cfg.blacklist_markets else 'none'}")
    print(f"  allowed  : {', '.join(cfg.allowed_markets) if cfg.allowed_markets else 'all'}")
    print(f"  min_train: {cfg.min_train}")
    print(f"  split    : {cfg.train_ratio:.0%} train")
    print("api keys")
    for name, absent in missing.items():
        print(f"  {name:<20} {'MISSING' if absent else 'set'}")
    return 0


def cmd_demo(args, cfg) -> int:
    print("generating SYNTHETIC data (pipeline demo only - not evidence)")
    frame = demo.generate(seed=args.seed, n_matches=args.matches)
    csv_path = cfg.data_dir / "synthetic_matches.csv"
    demo.write_csv(frame, csv_path)
    print(f"  wrote {csv_path}  ({len(frame)} matches)")

    # Demo rows go to their own database. The shared one is for real fetches,
    # and mixing the two would silently corrupt a real backtest.
    demo_db = cfg.data_dir / "demo.db"
    con = db.init_db(demo_db)
    try:
        count = db.upsert_matches(con, frame.to_dict(orient="records"))
        print(f"  stored {count} synthetic rows in {demo_db}")
        stored = db.fetch_matches_df(con, include_synthetic=True, require_result=True)
    finally:
        con.close()

    return _run_backtest(stored, cfg, args, stem="demo_backtest")


def _restrict_to_priced(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only matches that can actually be settled.

    A row without both-side prices can never produce a bet, but it still shifts
    the 80/20 index split. So a partial odds backfill would leave the test
    window completely unpriced and the backtest would bet nothing. Filter
    before splitting, not after.
    """
    needed = ("ah_open", "ah_fav_price", "ah_dog_price", "ou_over_price", "ou_under_price")
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        print(f"  ! cannot filter to priced matches: missing {missing}")
        return frame

    keep = pd.Series(True, index=frame.index)
    for column in needed:
        keep &= pd.to_numeric(frame[column], errors="coerce").notna()

    kept = frame[keep].reset_index(drop=True)
    print(f"  priced matches only      {len(kept)} of {len(frame)}")
    return kept


def cmd_backtest(args, cfg) -> int:
    frame = _load_matches(args, cfg)
    if getattr(args, "require_odds", False):
        frame = _restrict_to_priced(frame)
    return _run_backtest(frame, cfg, args, stem=args.stem)


def _maybe_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out  # NaN -> None


def _persist_predictions(con, result) -> int:
    """Store a backtest's predictions so the run can be settled later.

    Uses :func:`fbpredict.db.insert_prediction`, so the look-ahead constraint
    is enforced on the way in.
    """
    saved = 0
    for _, row in result.predictions.iterrows():
        try:
            db.insert_prediction(
                con,
                match_id=str(row["match_id"]),
                run_id=result.run_id,
                predicted_at=row["predicted_at"],
                kickoff_at=row["kickoff_at"],
                market=row["market"],
                p_model=_maybe_float(row["p_model"]),
                p_implied=_maybe_float(row["p_implied"]),
                edge=_maybe_float(row["edge"]),
                odds_used=_maybe_float(row["odds_used"]),
                stake=float(row["stake"] or 0.0),
            )
            saved += 1
        except db.LookAheadError:
            continue
    return saved


def _parse_market_list(val: str | None) -> tuple[str, ...] | None:
    if val is None:
        return None
    val = val.strip()
    if not val or val.lower() == "none":
        return ()
    return tuple(part.strip() for part in val.split(",") if part.strip())


def _run_backtest(frame: pd.DataFrame, cfg, args, *, stem: str) -> int:
    threshold = args.threshold if args.threshold is not None else cfg.edge_threshold
    kind = args.model
    feature_set = getattr(args, "feature_set", "v1")
    predicted_before_minutes = getattr(args, "predicted_before_minutes", None)
    use_closing_features = bool(getattr(args, "use_closing_features", False))
    blacklist_markets = _parse_market_list(getattr(args, "blacklist_markets", None))
    allowed_markets = _parse_market_list(getattr(args, "allowed_markets", None))

    print(
        f"backtesting {len(frame)} matches | model={kind} | "
        f"threshold={threshold} | split={cfg.train_ratio:.0%} | feature set={feature_set}"
    )
    result = backtest.walk_forward(
        frame,
        cfg,
        kind=kind,
        threshold=threshold,
        predicted_before_minutes=predicted_before_minutes,
        feature_set=feature_set,
        use_closing_features=use_closing_features,
        blacklist_markets=blacklist_markets,
        allowed_markets=allowed_markets,
    )

    metrics = report.evaluate(result, cfg)
    md_path, json_path = report.save_report(metrics, cfg, stem=stem)

    sizes = metrics["sizes"]
    betting = metrics["betting"]
    print("")
    print(f"  run id          : {result.run_id}")
    active_bl = cfg.blacklist_markets if blacklist_markets is None else blacklist_markets
    active_al = cfg.allowed_markets if allowed_markets is None else allowed_markets
    if active_bl:
        print(f"  blacklist       : {', '.join(active_bl)}")
    if active_al:
        print(f"  allowed         : {', '.join(active_al)}")
    print(f"  train/test      : {sizes['train']} / {sizes['test']}")
    print(f"  models fitted   : {sizes['models_fitted']}")

    features_block = metrics.get("features") or {}
    print(
        f"  feature set     : {features_block.get('feature_set', 'v1')} "
        f"({features_block.get('active', 0)} features used)"
    )
    dropped = features_block.get("dropped_empty") or []
    if dropped:
        preview = ", ".join(str(name) for name in dropped[:8])
        if len(dropped) > 8:
            preview += f", +{len(dropped) - 8} more"
        print(f"  dropped (no data): {len(dropped)} -> {preview}")
    if predicted_before_minutes is not None:
        print(f"  prediction clock: {predicted_before_minutes:g} min before kick-off")
    print(f"  bets / skipped  : {sizes['taken']} / {sizes['skipped']}")
    print(f"  profit          : {betting['profit']:.3f} units on {betting['staked']:.3f} staked")
    print(f"  ROI             : {betting['roi'] * 100:.2f}%" if betting["roi"] == betting["roi"] else "  ROI             : n/a")
    print("")
    for name, block in metrics["heads"].items():
        model_ll = block["model"]["log_loss"]
        base_ll = block["market_baseline"]["log_loss"]
        print(f"  {name:<24} model log loss {model_ll:.4f} vs market {base_ll:.4f}")

    if getattr(args, "save_predictions", False):
        con = db.init_db(cfg.db_path)
        try:
            saved = _persist_predictions(con, result)
        finally:
            con.close()
        print("")
        print(f"  saved {saved} predictions for run {result.run_id}")
        print(f"  settle them later with: settle --run-id {result.run_id}")

    print("")
    for note in metrics["notes"]:
        print(f"  ! {note}")
    print("")
    print(f"report: {md_path}")
    print(f"json  : {json_path}")
    return 0


def cmd_fetch_results(args, cfg) -> int:
    con = db.init_db(cfg.db_path)
    try:
        if args.provider == "football-data-org":
            provider = FootballDataOrgProvider(cfg, con=con, refresh=args.refresh)
            response = provider.fetch_season(args.competition, args.season)
        elif args.provider == "api-football":
            provider = ApiFootballProvider(cfg, con=con, refresh=args.refresh)
            response = provider.fetch_season(args.league, args.season)
        else:
            raise SystemExit(f"unknown provider {args.provider}")

        # Use the response's own timestamp. On a cache hit that is the ORIGINAL
        # fetch time, so the recorded evidence date is never a lie.
        rows = provider.to_match_rows(response.data, response.fetched_at, args.season)
        count = db.upsert_matches(con, rows)
    finally:
        con.close()

    source = "cache" if response.from_cache else "network"
    print(f"fetched {len(rows)} result rows from {source} at {response.fetched_at}, stored {count}")
    print(f"raw payloads archived under {cfg.raw_dir}")
    return 0


def cmd_probe_odds(args, cfg) -> int:
    con = db.init_db(cfg.db_path)
    try:
        provider = OddsPapiProvider(cfg, con=con)
        result = provider.probe(args.fixture_id)
    finally:
        con.close()

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print("")
    print("Starter mapping (edit to match what you just saw):")
    print(dumps_mapping_template())
    return 0


def cmd_probe_five_dollar(args, cfg) -> int:
    """Plan, quota, league ids, bookmakers, and one real odds block."""
    con = db.init_db(cfg.db_path)
    try:
        provider = FiveDollarFootballProvider(cfg, con=con)
        report = provider.probe(args.fixture_id)
    finally:
        con.close()

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


def _attach_per_fixture_odds(
    provider, rows: list[dict], bookmaker: str, max_odds: int | None = None
) -> tuple[int, int, int, str]:
    """Free/Community plan: the fixture list carries no odds, so fetch each one.

    One request per match. Returns
    ``(attached, attempted, skipped_uncached, stopped_because)``.

    A quota stop is not a failure: every response is cached as it arrives, so a
    later run resumes instead of starting over. In ``--cache-only`` mode an
    uncached fixture is skipped rather than treated as an error.
    """
    attached = 0
    attempted = 0
    skipped = 0
    stopped = ""
    for row in rows:
        if max_odds is not None and attempted >= max_odds:
            stopped = f"hit the --max-odds limit ({max_odds})"
            break

        fixture_id = str(row.get("match_id", "")).split(":", 1)[-1]
        attempted += 1
        try:
            payload = provider.fixture_odds(fixture_id, bookmakers=bookmaker)
        except CacheMissError:
            attempted -= 1
            skipped += 1
            continue
        except ProviderError as exc:
            attempted -= 1
            stopped = str(exc)
            break

        block = five_dollar.odds_block_for(payload, bookmaker)
        if block is None:
            continue
        row.update(five_dollar.map_markets(block))
        # record the format of the prices just attached, same as the
        # list-include path does (spec sections 9.2/9.5)
        row["odds_format_raw"] = five_dollar.ODDS_FORMAT
        row["odds_format_stored"] = five_dollar.ODDS_FORMAT
        row["odds_fetched_at"] = db.utcnow_iso()
        attached += 1
        if attempted % 50 == 0:
            print(f"  ... priced {attempted}/{len(rows)} fixtures")
    return attached, attempted, skipped, stopped


def cmd_fetch_five_dollar(args, cfg) -> int:
    """Pull a league's results and opening odds from 5DollarFootballAPI.

    Unlike the other sources this one needs no odds/results join: a single
    fixture row carries the result, the half-time score and the opening prices.
    """
    bookmaker = args.bookmaker or cfg.bookmaker_slug

    con = db.init_db(cfg.db_path)
    try:
        provider = FiveDollarFootballProvider(
            cfg, con=con, refresh=args.refresh, offline=args.cache_only
        )

        if args.league is None:
            print("--league is required. Popular league ids:")
            for row in (provider.list_leagues(popular=True, per_page=25).get("data") or []):
                print(f"  {row.get('id'):>12}  {row.get('name')} ({row.get('short_name')})")
            return 1

        # include=odds on a *list* is Pro-only, so per-fixture mode must not ask
        # for it - otherwise every run dies with 403 insufficient_plan.
        include = None if (args.no_odds or args.per_fixture_odds) else "odds"

        rows: list[dict] = []
        try:
            for fixture in provider.iter_league_fixtures(
                args.league,
                season=args.season,
                include=include,
                max_pages=args.max_pages,
                order="asc",
            ):
                row = five_dollar.to_match_row(
                    fixture,
                    bookmaker=bookmaker,
                    season=args.season,
                    fetched_at=db.utcnow_iso(),
                )
                if row is not None:
                    rows.append(row)
                if args.max_fixtures and len(rows) >= args.max_fixtures:
                    break
        except ProviderError as exc:
            if "insufficient_plan" in str(exc):
                print(
                    "This plan cannot batch odds on fixture lists "
                    "(include=odds starts with Pro).\n"
                    "Re-run with --per-fixture-odds to price one match at a time, "
                    "or --no-odds for results and lines only."
                )
                return 1
            raise

        attached = attempted = skipped = 0
        stopped = ""
        if args.per_fixture_odds and rows:
            attached, attempted, skipped, stopped = _attach_per_fixture_odds(
                provider, rows, bookmaker, max_odds=args.max_odds
            )

        count = db.upsert_matches(con, rows)
        coverage = five_dollar.coverage(rows)
        rate_limit = provider.last_rate_limit
    finally:
        con.close()

    print(f"league {args.league}: fetched {len(rows)} fixtures, stored {count}")
    if args.per_fixture_odds:
        print(f"  per-fixture odds          attached {attached} of {attempted} attempted")
        if skipped:
            print(f"  not cached (skipped)      {skipped}")
        if stopped:
            print(f"  pricing stopped           {stopped}")
            print("  re-run the same command later; already-fetched odds are cached")
    for key, value in coverage.items():
        print(f"  {key:<24} {value}")
    if rate_limit:
        print(f"  rate limit               {rate_limit}")
    print("")
    print(f"raw payloads archived under {cfg.raw_dir}")
    print(five_dollar.ATTRIBUTION)
    return 0


def cmd_scrape_thscore(args, cfg) -> int:
    con = db.init_db(cfg.db_path)
    try:
        provider = ThscoreProvider(cfg, con=con)
        rows = provider.scrape_day(
            args.day,
            confirm_robots=args.i_understand_robots,
            date_hint=args.date,
            headless=not args.headed,
        )
        print(odds_span_report(rows))
        print("")
        if args.save:
            match_rows = provider.to_match_rows(rows, date_hint=args.date)
            count = db.upsert_matches(con, match_rows)
            print(f"stored {count} result rows (scores only - odds left NULL on purpose)")
        else:
            print("(dry run: nothing written; pass --save to store the scores)")
    finally:
        con.close()
    return 0


def cmd_link(args, cfg) -> int:
    con = db.init_db(cfg.db_path)
    try:
        all_rows = db.fetch_matches_df(con, include_synthetic=False, require_result=False)
    finally:
        con.close()

    if all_rows.empty:
        raise SystemExit("no rows to link")

    with_odds = all_rows[all_rows["ah_open"].notna() & all_rows["pool_fav_open"].notna()]
    with_results = all_rows[all_rows["home_goals"].notna()]

    result = linking.link_results(with_odds, with_results, tolerance_hours=args.tolerance)
    print(result.summary)
    if result.ambiguous:
        print("ambiguous (closest kick-off kept):", ", ".join(result.ambiguous[:10]))
    if result.unmatched_odds:
        print("unmatched odds rows:", ", ".join(result.unmatched_odds[:10]))

    if not args.dry_run and not result.linked.empty:
        con = db.init_db(cfg.db_path)
        try:
            db.upsert_matches(con, result.linked.to_dict(orient="records"))
        finally:
            con.close()
        print(f"wrote {len(result.linked)} linked rows back into the database")
    return 0


def cmd_predict_forward(args, cfg) -> int:
    # load() verifies the SHA-256 sidecar BEFORE unpickling, so a tampered
    # artifact is refused rather than executed.
    bundle = model.ModelBundle.load(args.model)
    _, meta = model.ModelBundle.verify(args.model)
    print(f"model {args.model}")
    print("  hash matches sidecar: True")
    print(f"  trained rows        : {meta.get('metadata', {}).get('trained_rows')}")
    print(f"  rolling window      : {bundle.rolling_window}")
    print(f"  feature set         : {bundle.feature_set}")
    if bundle.use_closing_features:
        print(
            "  closing features    : ON - predictions are refused unless the match "
            f"is inside the {features.CLOSING_PRICE_LEAD_MINUTES}-minute closing window"
        )

    frame = _load_matches(args, cfg)
    upcoming = frame[frame["home_goals"].isna()]
    if upcoming.empty:
        print("no upcoming matches (all rows already have results)")
        return 0

    threshold = args.threshold if getattr(args, "threshold", None) is not None else cfg.edge_threshold
    blacklist_markets = _parse_market_list(getattr(args, "blacklist_markets", None))
    allowed_markets = _parse_market_list(getattr(args, "allowed_markets", None))

    picks = backtest.forward_predict(
        upcoming,
        bundle,
        cfg,
        threshold=threshold,
        blacklist_markets=blacklist_markets,
        allowed_markets=allowed_markets,
    )
    print("")
    for _, row in picks.iterrows():
        market = row["market"] or f"skip ({row['skip_reason']})"
        edge_info = f" [edge={row['edge']:.2%}]" if pd.notna(row.get("edge")) else ""
        print(f"  {row.get('home_team', 'Home')} vs {row.get('away_team', 'Away')}: {market}{edge_info}")
    print("")

    if getattr(args, "save", False):
        con = db.init_db(cfg.db_path)
        run_id = getattr(args, "run_id", None) or f"paper-{pd.Timestamp.utcnow().strftime('%Y%m%d-%H%M%S')}"
        saved = 0
        try:
            for _, row in picks.iterrows():
                if not row.get("market"):
                    continue
                try:
                    db.insert_prediction(
                        con,
                        match_id=str(row["match_id"]),
                        run_id=run_id,
                        predicted_at=row["predicted_at"],
                        kickoff_at=row["kickoff_at"],
                        market=row["market"],
                        p_model=_maybe_float(row["p_model"]),
                        p_implied=_maybe_float(row["p_implied"]),
                        edge=_maybe_float(row["edge"]),
                        odds_used=_maybe_float(row["odds_used"]),
                        stake=float(row["stake"] or 0.0),
                    )
                    saved += 1
                except db.LookAheadError:
                    continue
        finally:
            con.close()
        print(f"Paper-trading: saved {saved} predictions under run id: {run_id}")
        print(f"Settle them after matches complete with: .\\run.ps1 settle --run-id {run_id}")
    else:
        print("Predictions were NOT written to the database. Pass --save to store them for paper trading,")
        print("which can later be evaluated with: .\\run.ps1 settle --run-id <run_id>")
    return 0


def _num(value, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number:
        return "n/a"
    return f"{number:.{digits}f}{suffix}"


def cmd_audit_rules(args, cfg) -> int:
    """Measure rule set 2 on the matches we actually have.

    Descriptive and in-sample: it reports what happened when each rule fired,
    with a confidence interval and that rule's own break-even price, so a small
    lucky run is not mistaken for an edge.
    """
    frame = _load_matches(args, cfg)
    frame = _restrict_to_priced(frame)

    prepared = features.build_features(frame)
    novig = bool(getattr(args, "novig", False))
    water = bool(getattr(args, "water", False))
    filtered = bool(getattr(args, "filtered", False))
    if water and novig:
        raise SystemExit("--water audits raw decimal prices; do not combine it with --novig")
    if water and filtered:
        raise SystemExit("choose either --water or --filtered")

    if water:
        table = price_pool_rules.RULES_WATER
        markets = ("AH_fav",)
        suffix = ""
        rule_set_name = "RULES_WATER (R7)"
        stem = args.stem if args.stem != "rule_audit" else "water_rule_audit"
    elif filtered:
        table = price_pool_rules.RULES_FILTERED_NOVIG if novig else price_pool_rules.RULES_FILTERED
        markets = ("AH_fav", "AH_dog")
        suffix = "_nv" if novig else ""
        rule_set_name = "RULES_FILTERED_NOVIG" if novig else "RULES_FILTERED"
        stem = args.stem if args.stem != "rule_audit" else "filtered_rule_audit"
    else:
        table = price_pool_rules.RULES_V2_NOVIG if novig else price_pool_rules.RULES_V2
        markets = ("AH_fav", "AH_dog", "OU_over")
        # The no-vig block gets its own column suffix so it can never silently
        # reuse raw rule columns that share the same names.
        suffix = "_nv" if novig else ""
        rule_set_name = "RULES_V2_NOVIG" if novig else "RULES_V2"
        stem = args.stem

    audit = price_pool_rules.audit_rules(
        prepared, table=table, suffix=suffix, novig=novig, markets=markets
    )
    if audit.empty:
        print("no rule fired on any match that had usable prices")
        return 0

    context = {
        "generated_at": db.utcnow_iso(),
        "n_matches": int(len(prepared)),
        "rule_set": rule_set_name,
        "source": getattr(args, "csv", None) or f"local database ({cfg.league})",
    }
    md_path, _ = report.save_rule_audit(audit, cfg, stem=stem, context=context)

    print(f"audited {len(prepared)} matches with rule set {rule_set_name}")
    print("")
    print(
        f"  {'rule':<9} {'line':>6} {'market':<8} {'n':>4}  {'W/P/L':<9} "
        f"{'hit':>7}  {'95% CI':<13} {'odds':>5} {'B/E':>7} {'ROI':>8}"
    )
    for _, row in audit.iterrows():
        wp = f"{row['wins']}/{row['pushes']}/{row['losses']}"
        ci = f"{row['ci_low'] * 100:.0f}-{row['ci_high'] * 100:.0f}%"
        print(
            f"  {row['rule_id']:<9} {row['line']:>6} {row['market']:<8} {row['n']:>4}  "
            f"{wp:<9} {_num(row['hit_rate'] * 100, 1, '%'):>7}  {ci:<13} "
            f"{_num(row['avg_odds'], 2):>5} {_num(row['break_even'] * 100, 1, '%'):>7} "
            f"{_num(row['roi'] * 100, 1, '%'):>8}"
        )
    print("")
    resolvable = audit[audit["n"] >= report.MIN_MEANINGFUL_N]
    print(f"  rules with n >= {report.MIN_MEANINGFUL_N}: {len(resolvable)} of {len(audit)}")
    print("  ! in-sample description only - this is not evidence of an edge")
    print("  ! pushes are excluded from the hit rate (a stake returned is not a win)")
    print("  ! break-even is 1/odds, not 50%")
    print("")
    print(f"report: {md_path}")
    return 0


def cmd_settle(args, cfg) -> int:
    """Fill in outcome_unit / profit for predictions already recorded.

    This is the plan-B settlement step: predict before kick-off, then come back
    after the results land and settle the run.
    """
    con = db.init_db(cfg.db_path)
    try:
        predictions = db.fetch_predictions_df(con, args.run_id)
        if predictions.empty:
            raise SystemExit(f"no predictions recorded for run {args.run_id}")

        matches = db.fetch_matches_df(con, include_synthetic=True, require_result=True)
        if matches.empty:
            raise SystemExit("no matches with results in the database yet")
        matches = matches.set_index("match_id")

        settlements: list[dict] = []
        unsettled = 0
        for _, pick in predictions.iterrows():
            if not pick.get("market"):
                continue
            if pick["match_id"] not in matches.index:
                unsettled += 1
                continue

            match = matches.loc[pick["match_id"]]
            if isinstance(match, pd.DataFrame):  # duplicated id in the table
                match = match.iloc[0]

            diff = float(match["home_goals"]) - float(match["away_goals"])
            if str(match.get("fav_side")) != "home":
                diff = -diff
            total = float(match["home_goals"]) + float(match["away_goals"])

            unit, profit = labels.settle_pick(
                market=pick["market"],
                goal_diff_fav=diff,
                total_goals=total,
                ah_open=float(match["ah_open"]),
                ou_line_open=float(match["ou_line_open"]),
                ah_fav_price=match.get("ah_fav_price"),
                ah_dog_price=match.get("ah_dog_price"),
                ou_over_price=match.get("ou_over_price"),
                ou_under_price=match.get("ou_under_price"),
                stake=float(pick["stake"] or cfg.stake),
            )
            settlements.append(
                {"match_id": pick["match_id"], "outcome_unit": unit, "profit": profit}
            )

        settled = db.settle_predictions(con, args.run_id, settlements)
    finally:
        con.close()

    total_profit = sum(row["profit"] for row in settlements)
    print(f"settled {settled} predictions for run {args.run_id}")
    if unsettled:
        print(f"  {unsettled} bets still have no result recorded")
    print(f"profit: {total_profit:+.3f} units")
    return 0


def cmd_train_model(args, cfg) -> int:
    """Train and freeze a ModelBundle to disk with its SHA-256 sidecar."""
    frame = _load_matches(args, cfg)
    if getattr(args, "require_odds", False):
        frame = _restrict_to_priced(frame)

    out_path = Path(args.out) if args.out else cfg.models_dir / f"{args.model}_{args.feature_set}.pkl"
    print(f"training model on {len(frame)} matches | kind={args.model} | feature_set={args.feature_set}")
    bundle, report_obj = model.train_model(
        frame,
        kind=args.model,
        feature_set=args.feature_set,
        rolling_window=args.rolling_window,
        use_closing_features=bool(getattr(args, "use_closing_features", False)),
        seed=cfg.seed,
    )
    sidecar = bundle.save(out_path)
    print(f"model saved to {out_path}")
    print(f"  sha256         : {sidecar['sha256']}")
    print(f"  trained rows   : {report_obj.n_rows_used}")
    print(f"  active features: {len(bundle.feature_columns)}")
    return 0


def cmd_verify_model(args, cfg) -> int:
    ok, meta = model.ModelBundle.verify(args.model)
    print(json.dumps({"hash_matches": ok, **meta}, indent=2, ensure_ascii=False, default=str))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fbpredict",
        description="Football prediction research pipeline (AH + O/U).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create the SQLite database").set_defaults(func=cmd_init_db)
    sub.add_parser("show-config", help="print resolved configuration").set_defaults(func=cmd_show_config)

    demo_parser = sub.add_parser("demo", help="run the pipeline on SYNTHETIC data")
    demo_parser.add_argument("--matches", type=int, default=220)
    demo_parser.add_argument("--seed", type=int, default=42)
    demo_parser.add_argument("--model", choices=["logreg", "xgboost"], default="logreg")
    demo_parser.add_argument("--threshold", type=float, default=None)
    demo_parser.set_defaults(func=cmd_demo)

    back = sub.add_parser("backtest", help="walk-forward backtest on stored/CSV matches")
    back.add_argument("--csv", default=None, help="read matches from this CSV instead of the database")
    back.add_argument("--model", choices=["logreg", "xgboost"], default="logreg")
    back.add_argument("--threshold", type=float, default=None)
    back.add_argument("--stem", default="backtest", help="report file stem")
    back.add_argument(
        "--include-synthetic",
        dest="include_synthetic",
        action="store_true",
        help="also include synthetic demo rows (off by default)",
    )
    back.add_argument(
        "--require-odds", dest="require_odds", action="store_true",
        help="only use matches priced on both sides (use while the odds backfill is partial)",
    )
    back.add_argument(
        "--save-predictions",
        dest="save_predictions",
        action="store_true",
        help="store the run's predictions so they can be settled later",
    )
    back.add_argument(
        "--feature-set",
        dest="feature_set",
        choices=sorted(features.FEATURE_SETS),
        default="filtered_novig",
        help="filtered_novig = validated R1/R2 key rules in vig-free space (recommended); filtered = same in raw prices; v1 = original 30; v2 = raw R1-R6",
    )
    back.add_argument(
        "--predicted-before-minutes",
        dest="predicted_before_minutes",
        type=float,
        default=None,
        help="prediction clock in minutes before kick-off (default: the 24h clock); a run using closing features must sit inside the closing window",
    )
    back.add_argument(
        "--use-closing-features",
        dest="use_closing_features",
        action="store_true",
        help="fill the closing-price flags (fav_no_drop / dog_no_drop); refused when the prediction clock is earlier than the closing window",
    )
    back.add_argument(
        "--blacklist-markets",
        dest="blacklist_markets",
        default=None,
        help="comma-separated list of markets to blacklist (e.g. OU_over)",
    )
    back.add_argument(
        "--allowed-markets",
        dest="allowed_markets",
        default=None,
        help="comma-separated list of allowed markets (e.g. AH_dog,OU_under)",
    )
    back.set_defaults(func=cmd_backtest)

    fetch = sub.add_parser("fetch-results", help="download results (labels) from an API")
    fetch.add_argument("--provider", choices=["football-data-org", "api-football"], required=True)
    fetch.add_argument("--competition", default="PL", help="football-data.org competition code")
    fetch.add_argument("--league", default="EPL", help="api-football league name or id")
    fetch.add_argument("--season", default=None)
    fetch.add_argument(
        "--refresh",
        action="store_true",
        help="ignore the stored raw payload and hit the network again",
    )
    fetch.set_defaults(func=cmd_fetch_results)

    probe = sub.add_parser("probe-odds", help="README step 9.1: inspect a real OddsPapi response")
    probe.add_argument("--fixture-id", default=None)
    probe.set_defaults(func=cmd_probe_odds)

    probe5 = sub.add_parser(
        "probe-5dollar",
        help="check the 5DollarFootballAPI plan, quota, league ids and a real odds block",
    )
    probe5.add_argument("--fixture-id", dest="fixture_id", default=None)
    probe5.set_defaults(func=cmd_probe_five_dollar)

    five = sub.add_parser(
        "fetch-5dollar",
        help="pull a league's results + opening odds from 5DollarFootballAPI",
    )
    five.add_argument("--league", default=None, help="league id (see probe-5dollar)")
    five.add_argument("--season", default=None, help="season label, e.g. 26/27 or 2026")
    five.add_argument(
        "--bookmaker", default=None, help="bookmaker slug (default: FB_BOOKMAKER_SLUG)"
    )
    five.add_argument("--max-pages", dest="max_pages", type=int, default=20)
    five.add_argument(
        "--max-fixtures", dest="max_fixtures", type=int, default=None,
        help="stop after this many fixtures",
    )
    five.add_argument(
        "--no-odds", dest="no_odds", action="store_true",
        help="results only; skip the odds include",
    )
    five.add_argument(
        "--per-fixture-odds", dest="per_fixture_odds", action="store_true",
        help="fetch odds one fixture at a time (needed on Free/Community)",
    )
    five.add_argument(
        "--max-odds", dest="max_odds", type=int, default=None,
        help="cap the per-fixture odds attempts in this run (cached ones are instant)",
    )
    five.add_argument(
        "--cache-only", dest="cache_only", action="store_true",
        help="never call the API; rebuild the database from cached responses only",
    )
    five.add_argument("--refresh", action="store_true")
    five.set_defaults(func=cmd_fetch_five_dollar)

    scrape = sub.add_parser("scrape-thscore", help="opt-in Thscore scrape (supplementary source)")
    scrape.add_argument("--day", default="today", help="today | f1..f7 | n1..n7")
    scrape.add_argument("--date", default=None, help="ISO date to stamp the kick-offs with")
    scrape.add_argument("--save", action="store_true", help="store the scores (odds are never guessed)")
    scrape.add_argument("--headed", action="store_true", help="show the browser window")
    scrape.add_argument(
        "--i-understand-robots",
        dest="i_understand_robots",
        action="store_true",
        help="acknowledge that robots.txt disallows the site's data endpoints",
    )
    scrape.set_defaults(func=cmd_scrape_thscore)

    link = sub.add_parser("link", help="join odds rows to result rows by team + kick-off time")
    link.add_argument(
        "--tolerance",
        type=float,
        default=linking.DEFAULT_TOLERANCE_HOURS,
        help="hours of kick-off slack",
    )
    link.add_argument("--dry-run", action="store_true")
    link.set_defaults(func=cmd_link)

    settle = sub.add_parser("settle", help="fill in outcome/profit for a recorded run")
    settle.add_argument("--run-id", dest="run_id", required=True, help="run id to settle")
    settle.set_defaults(func=cmd_settle)

    audit = sub.add_parser(
        "audit-rules",
        help="descriptive per-rule hit rate for rule set 2 (R1-R6); in-sample only",
    )
    audit.add_argument(
        "--novig",
        action="store_true",
        help="use the vig-free thresholds against vig-free prices",
    )
    audit.add_argument(
        "--filtered",
        action="store_true",
        help="audit only the validated rule set (R1 and R2 on key handicaps; cuts failed Over and R5/R6 rules)",
    )
    audit.add_argument(
        "--water",
        action="store_true",
        help=(
            "audit only the owner's 2026-10-21 rule R7 (handicap 0.5/1, "
            "favourite AH water HK 0.80-0.88 = decimal 1.80-1.88, favourite "
            "1X2 <= 1.87) on the AH favourite market"
        ),
    )
    audit.add_argument("--csv", default=None, help="read matches from this CSV")
    audit.add_argument("--stem", default="rule_audit")
    audit.set_defaults(func=cmd_audit_rules)

    forward = sub.add_parser("predict-forward", help="plan B: predict matches with no result yet")
    forward.add_argument("--model", required=True, help="path to a saved ModelBundle")
    forward.add_argument("--csv", default=None)
    forward.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="edge threshold (default: config edge_threshold)",
    )
    forward.add_argument(
        "--blacklist-markets",
        dest="blacklist_markets",
        default=None,
        help="comma-separated list of markets to blacklist (e.g. OU_over)",
    )
    forward.add_argument(
        "--allowed-markets",
        dest="allowed_markets",
        default=None,
        help="comma-separated list of allowed markets (e.g. AH_dog,OU_under)",
    )
    forward.add_argument(
        "--save",
        action="store_true",
        help="save paper-trading predictions into database so they can be settled later",
    )
    forward.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="custom run id for paper-trading run (default: paper-YYYYMMDD-HHMMSS)",
    )
    forward.add_argument(
        "--include-synthetic",
        dest="include_synthetic",
        action="store_true",
        help="also include synthetic demo rows (off by default)",
    )
    train = sub.add_parser("train-model", help="train and save a frozen model bundle into models/")
    train.add_argument("--model", choices=["logreg", "xgboost"], default="logreg")
    train.add_argument("--feature-set", choices=sorted(features.FEATURE_SETS), default="filtered_novig")
    train.add_argument("--output", "--model-path", "--out", dest="out", default=None, help="output path for the model .pkl")
    train.add_argument("--csv", default=None)
    train.add_argument("--rolling-window", type=int, default=features.DEFAULT_ROLLING_WINDOW)
    train.add_argument("--require-odds", action="store_true", default=True)
    train.add_argument("--use-closing-features", action="store_true")
    train.add_argument("--include-synthetic", action="store_true")
    train.set_defaults(func=cmd_train_model)

    verify = sub.add_parser("verify-model", help="check a saved model against its hash sidecar")
    verify.add_argument("--model", required=True)
    verify.set_defaults(func=cmd_verify_model)

    run_1x2_p = sub.add_parser(
        "run-1x2",
        help="run academic Data Science 1X2 match outcome prediction experiment",
    )
    run_1x2_p.add_argument("--csv", default=None, help="load matches from custom CSV")
    run_1x2_p.add_argument("--rolling-window", type=int, default=5, help="rolling match window (default: 5)")
    run_1x2_p.add_argument("--train-ratio", type=float, default=0.8, help="train split ratio (default: 0.8)")
    run_1x2_p.add_argument("--no-odds", action="store_true", help="exclude market odds features (use match stats only)")
    run_1x2_p.set_defaults(func=cmd_run_1x2)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config()
    try:
        return int(args.func(args, cfg))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        import traceback

        traceback.print_exc()
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
