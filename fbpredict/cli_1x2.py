"""CLI command handler for the simple 1X2 Match Outcome Prediction project.

Executes feature extraction, chronological walk-forward model training,
and generates an academic Data Science report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from fbpredict import config, db
from fbpredict.features_1x2 import build_1x2_features
from fbpredict.model_1x2 import CLASS_NAMES, run_1x2_experiment


def format_confusion_matrix(cm) -> str:
    """Format 3x3 confusion matrix as ASCII markdown table."""
    lines = [
        "| Actual \\ Predicted | Away (0) | Draw (1) | Home (2) | Total |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for i, label in enumerate(CLASS_NAMES):
        row = cm[i]
        total = int(sum(row))
        lines.append(f"| **{label}** | {row[0]} | {row[1]} | {row[2]} | {total} |")
    return "\n".join(lines)


def generate_markdown_report(exp_res: dict, out_path: Path, league: str) -> None:
    """Write an academic report for Data Science presentation."""
    results = exp_res["results"]
    importances = exp_res["feature_importances"]
    n_train = exp_res["n_train"]
    n_test = exp_res["n_test"]

    lines = [
        "# Academic Data Science Project Report",
        "## Football Match Outcome Prediction (1X2: Home / Draw / Away)",
        "",
        f"- **League Analyzed**: {league}",
        f"- **Dataset Size**: {n_train + n_test} matches ({n_train} Train / {n_test} Test)",
        "- **Validation Methodology**: Temporal 80/20 Chronological Split (Zero Future Leakage)",
        "",
        "---",
        "",
        "### 1. Model Comparison & Benchmark Evaluation",
        "",
        "| Model | Accuracy | Macro F1 | Log Loss | Brier Score |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for res in results:
        lines.append(
            f"| **{res.name}** | {res.accuracy * 100:.2f}% | {res.macro_f1:.4f} | {res.log_loss_val:.4f} | {res.brier_score:.4f} |"
        )

    lines.extend([
        "",
        "> **Note on Benchmarks**: In competitive sports markets, bookmaker odds (Market Implied Baseline) reflect large aggregated information. Beating or matching the market log loss demonstrates genuine predictive signal.",
        "",
        "---",
        "",
        "### 2. Confusion Matrices (Test Set)",
        "",
    ])

    for res in results:
        lines.append(f"#### {res.name}")
        lines.append(format_confusion_matrix(res.confusion_mat))
        lines.append("")

    lines.extend([
        "---",
        "",
        "### 3. Feature Importance (Random Forest)",
        "",
        "| Rank | Feature | Description | Importance |",
        "| :---: | :--- | :--- | :---: |",
    ])

    descriptions = {
        "market_p_home": "Bookmaker implied probability of Home win",
        "market_p_away": "Bookmaker implied probability of Away win",
        "market_p_draw": "Bookmaker implied probability of Draw",
        "diff_gd_avg": "Home vs Away rolling goal difference differential",
        "diff_form_pts": "Home vs Away rolling form points differential",
        "home_gf_avg": "Home team rolling goals scored average (last 5)",
        "away_gf_avg": "Away team rolling goals scored average (last 5)",
        "home_ga_avg": "Home team rolling goals conceded average (last 5)",
        "away_ga_avg": "Away team rolling goals conceded average (last 5)",
        "home_form_pts": "Home team rolling points per game (last 5)",
        "away_form_pts": "Away team rolling points per game (last 5)",
        "home_gd_avg": "Home team rolling goal difference (last 5)",
        "away_gd_avg": "Away team rolling goal difference (last 5)",
        "diff_rest_days": "Difference in rest days between teams",
        "home_rest_days": "Days since Home team's previous match",
        "away_rest_days": "Days since Away team's previous match",
    }

    for rank, (feat, val) in enumerate(importances.items(), 1):
        desc = descriptions.get(feat, feat)
        lines.append(f"| {rank} | `{feat}` | {desc} | {val:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "### 4. Key Academic Takeaways",
        "1. **Temporal Integrity**: Rolling features computed with strict `.shift(1)` eliminate look-ahead bias.",
        "2. **Probability Calibration**: Cross-entropy (Log Loss) and Brier Score evaluate true calibrated probability distribution across Home, Draw, and Away.",
        "3. **Practical Utility**: Results clearly visualize whether team historical form adds complementary value over raw market pricing.",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def cmd_run_1x2(args, cfg) -> int:
    """Run the 1X2 Data Science experiment."""
    if getattr(args, "csv", None):
        print(f"Loading matches from CSV: {args.csv}")
        df = pd.read_csv(args.csv)
    else:
        con = db.init_db(cfg.db_path)
        try:
            df = db.fetch_matches_df(con, include_synthetic=False, require_result=True)
            if df.empty:
                print("No real matches with results in football.db, checking demo.db...")
                demo_db = cfg.data_dir / "demo.db"
                if demo_db.exists():
                    con_demo = db.init_db(demo_db)
                    try:
                        df = db.fetch_matches_df(con_demo, include_synthetic=True, require_result=True)
                    finally:
                        con_demo.close()
        finally:
            con.close()

    if df.empty:
        raise SystemExit(
            "No matches with results found. Please fetch matches or run demo first:\n"
            "  .\\run.ps1 demo\n"
            "  .\\run.ps1 fetch-results --provider football-data-org --competition PL"
        )

    print(f"Loaded {len(df)} matches. Building zero-leakage 1X2 features...")
    prepared = build_1x2_features(df, rolling_window=args.rolling_window)

    print(f"Running multi-model evaluation (Train/Test split: {args.train_ratio:.0%})...")
    exp_res = run_1x2_experiment(
        prepared,
        include_market_odds=not args.no_odds,
        train_ratio=args.train_ratio,
        seed=cfg.seed,
    )

    print("\n" + "=" * 70)
    print(" 1X2 MATCH OUTCOME PREDICTION EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"Train matches: {exp_res['n_train']} | Test matches: {exp_res['n_test']}")
    print("-" * 70)
    print(f"{'Model':<32} {'Accuracy':>10} {'Macro F1':>10} {'Log Loss':>10} {'Brier':>10}")
    print("-" * 70)

    for res in exp_res["results"]:
        print(
            f"{res.name:<32} {res.accuracy * 100:>9.2f}% {res.macro_f1:>10.4f} "
            f"{res.log_loss_val:>10.4f} {res.brier_score:>10.4f}"
        )
    print("=" * 70)

    report_path = cfg.reports_dir / "simple_1x2_project_report.md"
    generate_markdown_report(exp_res, report_path, league=cfg.league)
    print(f"\nAcademic report saved to: {report_path}\n")

    return 0
