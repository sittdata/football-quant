"""Scoring and the written report.

The report is deliberately blunt about sample size. README section 8.5 is
right: 20 test matches cannot demonstrate an edge, so the report says so in
plain language instead of dressing up a lucky run.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import price_pool_rules
from .config import Config
from .db import utcnow_iso
from .features import FEATURE_COLUMNS, missing_feature_report

EPS = 1e-12

#: the user's original hand-count this audit was built to test, as
#: (low, high, total). Kept as a named reference so the report can show the
#: difference between that estimate and what the full priced set measures.
HAND_CHECK_REFERENCE = (13, 17, 20)

#: columns whose prices feed the format check (spec section 9.3): a decimal
#: price is never below 1.0, so any price below it proves a non-decimal format.
FORMAT_CHECK_PRICE_COLUMNS = (
    "pool_fav_open",
    "draw_ft_odds",
    "pool_dog_open",
    "pool_fav_close",
    "draw_ft_close",
    "pool_dog_close",
    "ah_fav_price",
    "ah_dog_price",
    "ou_over_price",
    "ou_under_price",
    "btts_open",
    "btts_close",
)


# --------------------------------------------------------------------------- #
# probability scores
# --------------------------------------------------------------------------- #
def binary_scores(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    mask = ~np.isnan(y_true) & ~np.isnan(probability)
    if mask.sum() == 0:
        return {"n": 0, "log_loss": math.nan, "brier": math.nan, "accuracy": math.nan}

    y = y_true[mask].astype(float)
    p = np.clip(probability[mask].astype(float), EPS, 1.0 - EPS)
    return {
        "n": int(mask.sum()),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p >= 0.5).astype(float) == y)),
        "base_rate": float(np.mean(y)),
    }


def head_comparison(predictions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Model vs the market's own implied probability, head by head."""
    out: dict[str, dict[str, Any]] = {}

    pairs = {
        "AH (favourite covers)": ("ah_label", "p_fav_cover", "fair_fav_ah"),
        "O/U (over)": ("ou_label", "p_over", "fair_over"),
    }
    for label, (target, model_col, baseline_col) in pairs.items():
        y = pd.to_numeric(predictions.get(target), errors="coerce").to_numpy(dtype="float64")
        model_p = pd.to_numeric(predictions.get(model_col), errors="coerce").to_numpy(dtype="float64")
        base_p = pd.to_numeric(predictions.get(baseline_col), errors="coerce").to_numpy(dtype="float64")

        model_scores = binary_scores(y, model_p)
        baseline_scores = binary_scores(y, base_p)

        better = None
        if model_scores["n"] and baseline_scores["n"]:
            better = model_scores["log_loss"] < baseline_scores["log_loss"]

        out[label] = {
            "model": model_scores,
            "market_baseline": baseline_scores,
            "model_beats_market_logloss": better,
        }
    return out


# --------------------------------------------------------------------------- #
# betting
# --------------------------------------------------------------------------- #
def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    cumulative = profit.cumsum()
    peak = cumulative.cummax()
    return float((peak - cumulative).max())


def betting_summary(predictions: pd.DataFrame) -> dict[str, Any]:
    bets = predictions[predictions["market"].notna()].copy()
    if bets.empty:
        return {
            "n_bets": 0,
            "profit": 0.0,
            "staked": 0.0,
            "roi": math.nan,
            "hit_rate": math.nan,
            "avg_edge": math.nan,
            "max_drawdown": 0.0,
            "by_market": {},
        }

    staked = float(bets["stake"].sum())
    profit = float(bets["profit"].sum())
    outcome = pd.to_numeric(bets["outcome_unit"], errors="coerce")

    by_market: dict[str, dict[str, Any]] = {}
    for market, group in bets.groupby("market"):
        by_market[str(market)] = {
            "n": int(len(group)),
            "profit": float(group["profit"].sum()),
            "roi": float(group["profit"].sum() / group["stake"].sum())
            if group["stake"].sum()
            else math.nan,
            "avg_edge": float(pd.to_numeric(group["edge"], errors="coerce").mean()),
        }

    return {
        "n_bets": int(len(bets)),
        "profit": profit,
        "staked": staked,
        "roi": float(profit / staked) if staked else math.nan,
        "hit_rate": float((outcome > 0).mean()),
        "push_rate": float((outcome == 0).mean()),
        "avg_edge": float(pd.to_numeric(bets["edge"], errors="coerce").mean()),
        "max_drawdown": max_drawdown(pd.to_numeric(bets["profit"], errors="coerce").fillna(0.0)),
        "by_market": by_market,
    }


def period_breakdown(predictions: pd.DataFrame, n_periods: int = 4) -> list[dict[str, Any]]:
    """Profit in equal time slices - README asks for stability across periods."""
    bets = predictions[predictions["market"].notna()].copy()
    if bets.empty:
        return []

    bets = bets.sort_values("kickoff_at", kind="mergesort").reset_index(drop=True)
    # NOTE: np.array_split on a DataFrame returns numpy arrays and loses the
    # column names, so slice with .iloc instead.
    n_chunks = max(1, min(n_periods, len(bets)))
    edges = np.linspace(0, len(bets), n_chunks + 1).astype(int)
    chunks = [bets.iloc[edges[i] : edges[i + 1]] for i in range(n_chunks)]

    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        rows.append(
            {
                "period": index,
                "from": str(chunk["kickoff_at"].min()),
                "to": str(chunk["kickoff_at"].max()),
                "n_bets": int(len(chunk)),
                "profit": float(chunk["profit"].sum()),
                "roi": float(chunk["profit"].sum() / chunk["stake"].sum())
                if chunk["stake"].sum()
                else math.nan,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# caveats
# --------------------------------------------------------------------------- #
def caveats(predictions: pd.DataFrame, cfg: Config) -> list[str]:
    notes: list[str] = []
    n_test = int(len(predictions))
    n_bets = int(predictions["market"].notna().sum()) if "market" in predictions else 0

    if n_test < 100:
        # README 8.5: +/-11% uncertainty at n=20
        margin = 100.0 / math.sqrt(max(n_test, 1)) if n_test else float("nan")
        notes.append(
            f"Test window is only {n_test} matches. At this size a hit rate is "
            f"uncertain to roughly +/-{margin:.0f}%, so this is a proof of concept "
            f"for the pipeline, NOT evidence of an edge. Increase to 200+ matches "
            f"before claiming anything."
        )
    if n_bets < 30:
        notes.append(
            f"Only {n_bets} matches cleared the edge threshold of {cfg.edge_threshold:.3f}. "
            "Profit or loss on so few bets is noise."
        )
    if "is_synthetic" in predictions.columns and predictions["is_synthetic"].fillna(0).astype(int).max() == 1:
        notes.append(
            "These rows are SYNTHETIC demo data. They exercise the code path only and "
            "must never be presented as fetched evidence."
        )
    if "fav_side" in predictions.columns:
        unknown = int(predictions["fav_side"].isna().sum())
        if unknown:
            notes.append(
                f"{unknown} matches had no explicit favourite side; they defaulted to "
                "'home', which is an assumption you should verify."
            )
    betting = betting_summary(predictions)
    avg_edge = betting.get("avg_edge")
    if betting["n_bets"] and isinstance(avg_edge, float) and math.isfinite(avg_edge) and avg_edge > 0.10:
        notes.append(
            f"Average edge is {avg_edge:.1%}, which is implausibly large. Real edges are "
            "usually 1-5%. A number this big almost always means the model is "
            "over-confident (miscalibrated) rather than that the market is wrong - "
            "compare the log loss column above before believing it."
        )

    notes.append(
        "Odds are used as supplied by the provider. Confirm they are the OPENING price "
        "and note the bookmaker; a closing price would make the backtest meaningless."
    )
    return notes


# --------------------------------------------------------------------------- #
# spec checks (sections 7.4 and 9.x), measured on the frame the run used
# --------------------------------------------------------------------------- #
def rule_performance_rows(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Per-rule hit rate table (spec section 7.4).

    Descriptive and in-sample: the rules, the thresholds and the data all come
    from the same period. See :func:`fbpredict.price_pool_rules.audit_rules`.
    """
    if frame is None or frame.empty:
        return []
    audit = price_pool_rules.audit_rules(frame)
    return audit.to_dict(orient="records")


def overround_comparison(frame: pd.DataFrame | None) -> dict[str, Any]:
    """Measured 1X2 margin vs the margins implied by the rule images (spec 9.4).

    Both sides use the same definition (the sum of inverse prices), so the
    numbers are comparable.
    """
    measured: dict[str, Any] = {}
    needed = ("pool_fav_open", "draw_ft_odds", "pool_dog_open")
    if frame is not None and not frame.empty and all(c in frame.columns for c in needed):
        fav = pd.to_numeric(frame["pool_fav_open"], errors="coerce")
        draw = pd.to_numeric(frame["draw_ft_odds"], errors="coerce")
        dog = pd.to_numeric(frame["pool_dog_open"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = 1.0 / fav + 1.0 / draw + 1.0 / dog
        factor = factor.replace([np.inf, -np.inf], np.nan).dropna()
        if len(factor):
            measured = {
                "n": int(len(factor)),
                "min": float(factor.min()),
                "mean": float(factor.mean()),
                "max": float(factor.max()),
            }
    table = price_pool_rules.rule_table_overrounds().to_dict(orient="records")
    return {"measured": measured, "table": table}


def odds_format_record(frame: pd.DataFrame | None) -> dict[str, Any]:
    """What odds format the data was in, raw and stored (spec sections 9.2/9.5)."""
    raw_values: list[str] = []
    stored_values: list[str] = []
    prices_checked = 0
    prices_below_one = 0
    if frame is not None and not frame.empty:
        for column, target in (
            ("odds_format_raw", raw_values),
            ("odds_format_stored", stored_values),
        ):
            if column in frame.columns:
                values = sorted({str(v) for v in frame[column].dropna().unique()})
                target.extend(values)
        for column in FORMAT_CHECK_PRICE_COLUMNS:
            if column not in frame.columns:
                continue
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            prices_checked += int(len(series))
            prices_below_one += int((series < 1.0).sum())
    return {
        "raw": raw_values,
        "stored": stored_values,
        "prices_checked": prices_checked,
        "prices_below_one": prices_below_one,
    }


def untestable_rules_record() -> list[dict[str, str]]:
    """Rules the plan cannot test, with the reason (spec section 6.2)."""
    rows: list[dict[str, str]] = []
    if price_pool_rules.R4A_UNTESTABLE:
        rows.append(
            {
                "rule_id": "R4a",
                "rule": "Ladbrokes Over HT (opening and closing price)",
                "reason": (
                    "needs Ladbrokes Over HT opening and closing prices; this plan has "
                    "no Ladbrokes feed, and the spec forbids substituting another "
                    "bookmaker, so the rule stays NaN instead of being guessed"
                ),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #
def evaluate(result, cfg: Config) -> dict[str, Any]:
    predictions = result.predictions
    frame = getattr(result, "data_frame", None)
    return {
        "generated_at": utcnow_iso(),
        "run_id": result.run_id,
        "config": {
            "league": cfg.league,
            "bookmaker": cfg.bookmaker,
            "edge_threshold": cfg.edge_threshold,
            "stake": cfg.stake,
            "train_ratio": cfg.train_ratio,
            "min_train": cfg.min_train,
            "refit_every": cfg.refit_every,
        },
        "sizes": {
            "train": result.train_size,
            "test": result.test_size,
            "models_fitted": result.n_models_fitted,
            "taken": result.taken,
            "skipped": result.skipped,
        },
        "features": {
            "feature_set": getattr(result, "feature_set", "v1"),
            "predicted_before_minutes": getattr(result, "predicted_before_minutes", None),
            "active": len(getattr(result, "active_features", []) or []),
            "dropped_empty": list(getattr(result, "dropped_features", []) or []),
        },
        "heads": head_comparison(predictions),
        "betting": betting_summary(predictions),
        "periods": period_breakdown(predictions),
        "feature_coverage": _feature_coverage(result, predictions),
        "rule_performance": rule_performance_rows(frame),
        "overround_comparison": overround_comparison(frame),
        "odds_format": odds_format_record(frame),
        "untestable_rules": untestable_rules_record(),
        "notes": list(result.notes) + caveats(predictions, cfg),
    }


def _feature_coverage(result, predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """Coverage of the features the model actually saw.

    The predictions frame carries only a handful of columns, so measuring it
    would report every feature as absent. Prefer the feature frame the
    backtest already built for its test window.
    """
    frame = getattr(result, "feature_coverage", None)
    if frame is None or frame.empty:
        frame = missing_feature_report(predictions)
    return frame.to_dict(orient="records")


def render_markdown(metrics: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Backtest report")
    add("")
    add(f"Generated: `{metrics['generated_at']}`  ")
    add(f"Run id: `{metrics['run_id']}`")
    add("")

    cfg = metrics["config"]
    add("## Setup")
    add("")
    add(f"- league: `{cfg['league']}`, bookmaker: `{cfg['bookmaker']}`")
    add(f"- edge threshold: `{cfg['edge_threshold']}`  (skip below this)")
    add(f"- flat stake: `{cfg['stake']}` per bet")
    add(f"- time-ordered split: `{cfg['train_ratio']:.0%}` train / rest test, never shuffled")
    features_block = metrics.get("features") or {}
    if features_block:
        add(
            f"- feature set: `{features_block.get('feature_set', 'v1')}`, "
            f"{features_block.get('active', 0)} features used by the last fitted model"
        )
        dropped = features_block.get("dropped_empty") or []
        if dropped:
            preview = ", ".join(str(name) for name in dropped[:6])
            if len(dropped) > 6:
                preview += ", ..."
            add(
                f"- dropped for having no value in the training window: "
                f"**{len(dropped)}** (`{preview}`)"
            )
        minutes = features_block.get("predicted_before_minutes")
        if minutes is not None:
            add(f"- prediction clock: **{minutes:g} minutes** before kick-off")
    add("")

    sizes = metrics["sizes"]
    add("## Sizes")
    add("")
    add("| | count |")
    add("|---|---|")
    add(f"| train matches | {sizes['train']} |")
    add(f"| test matches | {sizes['test']} |")
    add(f"| models fitted (walk-forward) | {sizes['models_fitted']} |")
    add(f"| matches bet | {sizes['taken']} |")
    add(f"| matches skipped | {sizes['skipped']} |")
    add("")

    add("## Model vs market baseline")
    add("")
    add("| head | n | model log loss | baseline log loss | model Brier | baseline Brier | model beats market |")
    add("|---|---|---|---|---|---|---|")
    for name, block in metrics["heads"].items():
        model = block["model"]
        base = block["market_baseline"]
        beat = block["model_beats_market_logloss"]
        beat_text = "yes" if beat else ("no" if beat is False else "n/a")
        add(
            f"| {name} | {model['n']} | {_fmt(model['log_loss'])} | {_fmt(base['log_loss'])} | "
            f"{_fmt(model['brier'])} | {_fmt(base['brier'])} | {beat_text} |"
        )
    add("")

    betting = metrics["betting"]
    add("## Betting (after the bookmaker's margin)")
    add("")
    add(f"- bets: **{betting['n_bets']}**")
    add(f"- profit: **{_fmt(betting['profit'], 3)}** units on {_fmt(betting['staked'], 3)} staked")
    add(f"- ROI: **{_pct(betting['roi'])}**")
    add(f"- hit rate: {_pct(betting['hit_rate'])}, pushes: {_pct(betting.get('push_rate'))}")
    add(f"- average edge: {_pct(betting['avg_edge'])}")
    add(f"- max drawdown: {_fmt(betting['max_drawdown'], 3)} units")
    add("")

    if betting["by_market"]:
        add("| market | bets | profit | ROI | avg edge |")
        add("|---|---|---|---|---|")
        for market, block in betting["by_market"].items():
            add(
                f"| {market} | {block['n']} | {_fmt(block['profit'], 3)} | "
                f"{_pct(block['roi'])} | {_pct(block['avg_edge'])} |"
            )
        add("")

    periods = metrics.get("periods") or []
    if periods:
        add("## Stability across the test window")
        add("")
        add("| period | from | to | bets | profit | ROI |")
        add("|---|---|---|---|---|---|")
        for block in periods:
            add(
                f"| {block['period']} | {block['from'][:10]} | {block['to'][:10]} | "
                f"{block['n_bets']} | {_fmt(block['profit'], 3)} | {_pct(block['roi'])} |"
            )
        add("")

    add("## Rule set 2 - per-rule performance (in-sample)")
    add("")
    rule_rows = metrics.get("rule_performance") or []
    if rule_rows:
        add(
            "Descriptive only - the thresholds, the rule selection and the data all "
            "come from the same period. Pushes are excluded from the hit rate: a "
            "returned stake is not a win."
        )
        add("")
        for line in _rule_table_lines(pd.DataFrame(rule_rows)):
            add(line)
    else:
        add("No rule-performance rows were available for this run.")
    add("")

    add("## Overround comparison (spec 9.4)")
    add("")
    overround = metrics.get("overround_comparison") or {}
    measured = overround.get("measured") or {}
    if measured:
        add(
            f"- measured on {measured['n']} matches of this run's own frame "
            f"(favourite / draw / underdog 1X2 opening): min {_fmt(measured['min'])} / "
            f"mean {_fmt(measured['mean'])} / max {_fmt(measured['max'])}"
        )
    else:
        add("- the frame carried no complete 1X2 opening triple, so nothing was measured")
    table = overround.get("table") or []
    if table:
        add("")
        add("| rule | printed prices (fav / draw / dog) | implied overround |")
        add("|---|---|---|")
        for row in table:
            prices = " / ".join(_fmt(float(value), 3) for value in row["prices"])
            add(f"| {row['key']} | {prices} | {_fmt(row['overround'], 4)} |")
    add("")
    if measured and table:
        implied = [float(row["overround"]) for row in table if row.get("overround")]
        if implied:
            add(
                f"The images imply a margin of {_fmt(min(implied), 4)}-{_fmt(max(implied), 4)}, "
                f"while this data's book averages {_fmt(measured['mean'], 4)}. The same "
                "printed thresholds therefore sit differently on a tighter book - which "
                "is exactly what the `v2_novig` variant exists to check."
            )
            add("")

    add("## Odds format record (spec 9.2 / 9.5)")
    add("")
    fmt = metrics.get("odds_format") or {}
    raw_text = ", ".join(fmt.get("raw") or []) or "not recorded"
    stored_text = ", ".join(fmt.get("stored") or []) or "not recorded"
    add(f"- odds_format_raw / odds_format_stored observed: **{raw_text} / {stored_text}**")
    add(
        f"- format check: {fmt.get('prices_checked', 0)} prices inspected, "
        f"{fmt.get('prices_below_one', 0)} below 1.0 - a decimal price cannot be below 1.0, "
        "so the stored columns are decimal"
    )
    add(
        "- traceability deviation: the spec asks for one row per price, while this schema "
        "keeps one row per match - bookmaker ~ `odds_source`, captured_at ~ "
        "`odds_fetched_at`, is_open_price ~ the `*_open` column suffix"
    )
    add(
        "- the Ladbrokes Over HT range \"0.50-0.75\" cannot be decimal; read as Hong Kong "
        "odds it is 1.50-1.75 decimal, which is the reading the spec's own note assumes"
    )
    add("")

    add("## Rules that stay untestable")
    add("")
    untestable = metrics.get("untestable_rules") or []
    if untestable:
        for row in untestable:
            add(f"- `{row['rule_id']}` ({row['rule']}): {row['reason']}")
    else:
        add("- none")
    add("")

    add("## What this run does NOT prove")
    add("")
    for note in metrics["notes"]:
        add(f"- {note}")
    add("")

    add("## Feature coverage in the test window")
    add("")
    add("| feature | present | missing |")
    add("|---|---|---|")
    for row in metrics["feature_coverage"]:
        add(f"| {row['feature']} | {row['present']} | {row['missing']} |")
    add("")

    return "\n".join(lines)


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(number) or math.isinf(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(number) or math.isinf(number):
        return "n/a"
    return f"{number * 100:.2f}%"


def save_report(metrics: dict[str, Any], cfg: Config, stem: str = "backtest") -> tuple[Path, Path]:
    cfg.ensure_dirs()
    md_path = cfg.reports_dir / f"{stem}.md"
    json_path = cfg.reports_dir / f"{stem}.json"
    md_path.write_text(render_markdown(metrics), encoding="utf-8")
    json_path.write_text(
        json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return md_path, json_path


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value


# --------------------------------------------------------------------------- #
# rule audit (rule set 2) - descriptive, in-sample
# --------------------------------------------------------------------------- #
MIN_MEANINGFUL_N = 30

RULE_AUDIT_CAVEATS = (
    "This is an IN-SAMPLE description, not a backtest. The rules and the data "
    "come from the same person and the same period, so nothing here is out of "
    "sample. It answers 'when this condition held, what actually happened?' - "
    "it does not establish an edge.",
    "A push returns the stake, so pushes are counted in their own column and "
    "excluded from the hit rate. Counting a push as a win inflates the hit rate.",
    "Break-even is 1 / the average decimal odds actually available, not 50%. "
    "Being right more often than half the time is not the same as being "
    "profitable: at 1.60 you need 62.5%, at 1.90 you need 52.6%.",
    f"Rules with n < {MIN_MEANINGFUL_N} cannot support a conclusion. They are "
    "listed for completeness and marked inconclusive.",
)


def _verdict(row: pd.Series) -> str:
    """Blunt verdict for one rule, driven by the interval rather than the point."""
    if row["n"] < MIN_MEANINGFUL_N:
        return f"inconclusive (n < {MIN_MEANINGFUL_N})"
    low, high, break_even = row["ci_low"], row["ci_high"], row["break_even"]
    if not all(isinstance(v, float) and math.isfinite(v) for v in (low, high, break_even)):
        return "n/a"
    if low > break_even:
        return "candidate (interval clears break-even)"
    if high < break_even:
        return "worse than break-even"
    return "not resolved (interval straddles break-even)"


def _rule_table_lines(audit: pd.DataFrame) -> list[str]:
    """Markdown table shared by the audit report and the backtest report."""
    lines = [
        "| rule | line | market | n | W / P / L | hit rate | 95% CI | avg odds | "
        "break-even | hit - break-even | ROI | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in audit.iterrows():
        lines.append(
            f"| {row['rule_id']} | {row['line']} | {row['market']} | {row['n']} | "
            f"{row['wins']} / {row['pushes']} / {row['losses']} | "
            f"{_pct(row['hit_rate'])} | {_pct(row['ci_low'])} - {_pct(row['ci_high'])} | "
            f"{_fmt(row['avg_odds'], 3)} | {_pct(row['break_even'])} | "
            f"{_pct(row['edge_vs_break_even'])} | {_pct(row['roi'])} | {_verdict(row)} |"
        )
    return lines


def rule_audit_markdown(audit: pd.DataFrame, *, context: dict[str, Any] | None = None) -> str:
    lines: list[str] = []
    add = lines.append

    water_audit = "WATER" in str((context or {}).get("rule_set", ""))
    if water_audit:
        add("# Rule audit - owner rule R7 (water 0.5/1, play the favourite)")
    else:
        add("# Rule audit - rule set 2 (R1-R6)")
    add("")
    if context:
        add(f"Generated: `{context.get('generated_at', '')}`  ")
        add(f"Matches audited: **{context.get('n_matches', 0)}**  ")
        add(f"Rule set: `{context.get('rule_set', 'RULES_V2')}`  ")
        add(f"Source: {context.get('source', 'local database')}")
        add("")

    add("## How to read this")
    add("")
    for caveat in RULE_AUDIT_CAVEATS:
        add(f"- {caveat}")
    add("")

    if audit.empty:
        add("No rule fired on any match that had usable prices.")
        return "\n".join(lines)

    add("## Per-rule results")
    add("")
    for line in _rule_table_lines(audit):
        add(line)
    add("")

    resolvable = audit[audit["n"] >= MIN_MEANINGFUL_N]
    if resolvable.empty:
        add(
            f"**No rule reached n >= {MIN_MEANINGFUL_N}, so none of them can be "
            "concluded on this dataset.**"
        )
    else:
        candidates = resolvable[resolvable["ci_low"] > resolvable["break_even"]]
        add(f"Rules with a usable sample: {len(resolvable)}")
        add("")
        if candidates.empty:
            add(
                "None of the rules with a usable sample has an interval that "
                "clears its own break-even price."
            )
        else:
            add("Rules whose 95% interval clears break-even:")
            add("")
            for _, row in candidates.iterrows():
                add(
                    f"- `{row['key']}` on {row['market']}: n={row['n']}, "
                    f"hit {_pct(row['hit_rate'])} vs break-even {_pct(row['break_even'])}"
                )
    add("")

    # ------------------------------------------------------------------ #
    # the comparison the plan's goal 1 asks for: the original hand-count.
    # Skipped for the owner-rule audit, which is not about the printed tables.
    # ------------------------------------------------------------------ #
    if HAND_CHECK_REFERENCE and not water_audit:
        low, high, total = HAND_CHECK_REFERENCE
        add("## Compared with the original hand-count")
        add("")
        add(
            f"The estimate this audit exists to test: the hand-picked rules were "
            f"right **{low}-{high} of {total}** (about {low / total:.0%}-{high / total:.0%})."
        )
        mass = audit[audit["n"] >= 10]
        if mass.empty:
            add(
                "- no rule fired on enough matches here to compare even loosely; "
                "the hand-count and this table answer different questions"
            )
        else:
            best = mass.loc[mass["hit_rate"].idxmax()]
            add(
                f"- the only rows carrying any weight here (n >= 10) hit "
                f"{_pct(mass['hit_rate'].min())}-{_pct(best['hit_rate'])} "
                f"(best: `{best['key']}` on {best['market']}, n={best['n']}, "
                f"95% CI {_pct(best['ci_low'])}-{_pct(best['ci_high'])})"
            )
            clears = mass[mass["ci_low"] > mass["break_even"]]
            if clears.empty:
                add(
                    "- none of those intervals clears its own break-even price, so "
                    "the hand-count's range is neither reproduced nor ruled out at "
                    "this sample size"
                )
            else:
                add(
                    f"- {len(clears)} row(s) with n >= 10 have an interval that clears "
                    "break-even, so part of the hand-count's range is at least "
                    "compatible with an edge at those thresholds"
                )
        add(
            f"- the hand-count looked at about {total} matches chosen by eye; this "
            "audit is every firing on the full priced set, so they are not the same "
            "measurement"
        )
        add("")

    notes = sorted({str(note) for note in audit["note"].dropna() if str(note)})
    if notes:
        add("## Rule caveats carried from the spec")
        add("")
        for note in notes:
            add(f"- {note}")
        add("")

    untestable = untestable_rules_record()
    if untestable:
        add("## Rules that stay untestable")
        add("")
        for row in untestable:
            add(f"- `{row['rule_id']}` ({row['rule']}): {row['reason']}")
        add("")

    return "\n".join(lines)


def save_rule_audit(
    audit: pd.DataFrame, cfg: Config, stem: str = "rule_audit", *, context: dict[str, Any] | None = None
) -> tuple[Path, Path]:
    cfg.ensure_dirs()
    md_path = cfg.reports_dir / f"{stem}.md"
    json_path = cfg.reports_dir / f"{stem}.json"
    md_path.write_text(rule_audit_markdown(audit, context=context), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {"context": _jsonable(context or {}), "rows": _jsonable(audit.to_dict(orient="records"))},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return md_path, json_path
