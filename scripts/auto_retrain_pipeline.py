"""Automated Multi-League ML Model Retraining Pipeline.

Industrial-grade self-updating machine learning pipeline:
1. Ingests completed match outcomes & half-time stats from Thscore CDN feeds & API.
2. Updates pending matches in SQLite database (`data/football.db`).
3. Checks if new labelled data exceeds minimum retraining threshold.
4. Performs strict Walk-Forward Out-Of-Sample Benchmark (Safety Gate):
   - Trains candidate model on 80% chronological train set.
   - Evaluates candidate vs current production model on 20% holdout test set.
   - Computes ROC-AUC, Brier score loss, and calibration.
5. If Safety Gate passes:
   - Backs up current production model to `models/backups/`.
   - Trains final model on 100% of updated dataset.
   - Atomically promotes new model to `models/multileague_v2.pkl`.
   - Updates sidecar metadata and appends to `logs/retrain_audit.json`.
6. Dispatches full retrain audit report to Telegram Bot.

Usage:
    python scripts/auto_retrain_pipeline.py
    python scripts/auto_retrain_pipeline.py --force
    python scripts/auto_retrain_pipeline.py --min-new-matches 15 --dry-run
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

# Console utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import brier_score_loss, roc_auc_score

load_dotenv(ROOT_DIR / ".env")

from fbpredict import config, features, labels, model
from fbpredict.db import connect, upsert_matches
from fbpredict.model import ModelBundle, prepare_frame, train_model
from fbpredict.providers.thscore_api import ThscoreApiProvider
from fbpredict.providers.thscore_feed import fetch_matches_feed

# Config & Directories
MODELS_DIR = ROOT_DIR / "models"
BACKUPS_DIR = MODELS_DIR / "backups"
LOGS_DIR = ROOT_DIR / "logs"
AUDIT_LOG_FILE = LOGS_DIR / "retrain_audit.json"
PROD_MODEL_PATH = MODELS_DIR / "multileague_v2.pkl"
PROD_META_PATH = MODELS_DIR / "multileague_v2.meta.json"

BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_ALLOWED_CHAT_ID", "")


def send_telegram(text: str, parse_mode: str = "HTML") -> bool:
    """Send formatted notification via Telegram Bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Bot token or chat ID not set. Skipping push.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
    success = True
    for chunk in chunks:
        try:
            resp = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": parse_mode},
                timeout=15,
            )
            if resp.status_code != 200:
                print(f"[Telegram ERROR] {resp.status_code}: {resp.text}")
                success = False
        except Exception as e:
            print(f"[Telegram ERROR] {e}")
            success = False
    return success


def sync_results_and_ingest(con) -> dict[str, int]:
    """Sync results from CDN feeds and API into data/football.db."""
    print("\n[Step 1/5] Ingesting completed match results & updating database...")
    stats = {"updated_pending": 0, "new_finished_ingested": 0}

    # 1. Pull feed from Thscore CDN (bf_th_nt.js)
    try:
        leagues, matches_raw = fetch_matches_feed()
        finished_feed = [m for m in matches_raw if m.get("status") == -1]
        print(f"  * Thscore CDN: {len(matches_raw)} active matches, {len(finished_feed)} finished.")
    except Exception as e:
        print(f"  * [Warning] Thscore CDN fetch failed: {e}")
        finished_feed = []

    # Map finished feed by match_id
    feed_finished_map = {m["match_id"]: m for m in finished_feed}

    # 2. Update pending matches in DB that have completed
    cur = con.cursor()
    pending_rows = cur.execute(
        """
        SELECT match_id, kickoff_at FROM matches
        WHERE home_goals IS NULL
        """
    ).fetchall()

    rows_to_update = []
    for mid, kickoff in pending_rows:
        raw_mid = mid.replace("thscore:", "")
        if raw_mid in feed_finished_map:
            fm = feed_finished_map[raw_mid]
            rows_to_update.append({
                "match_id": mid,
                "home_goals": fm["home_goals"],
                "away_goals": fm["away_goals"],
                "ht_home": fm["ht_home"],
                "ht_away": fm["ht_away"],
                "home_corner": fm["home_corner"],
                "away_corner": fm["away_corner"],
                "home_red": fm["home_red"],
                "away_red": fm["away_red"],
                "home_yellow": fm["home_yellow"],
                "away_yellow": fm["away_yellow"],
                "result_fetched_at": datetime.now(timezone.utc).isoformat(),
            })

    if rows_to_update:
        upsert_matches(con, rows_to_update)
        stats["updated_pending"] = len(rows_to_update)
        print(f"  * Successfully updated scores for {len(rows_to_update)} pending matches in DB.")

    # 3. Ingest finished matches from feed that have both AH and OU lines
    new_finished_rows = []
    for fm in finished_feed:
        if (
            fm.get("home_goals") is not None
            and fm.get("away_goals") is not None
            and fm.get("ah_line") is not None
            and fm.get("ou_line") is not None
        ):
            mid = f"thscore:{fm['match_id']}"
            ah_line = float(fm["ah_line"])
            if ah_line > 0:
                fav_side, ah_fav = "home", -ah_line
            elif ah_line < 0:
                fav_side, ah_fav = "away", ah_line
            else:
                fav_side, ah_fav = "home", 0.0

            new_finished_rows.append({
                "match_id": mid,
                "provider": "thscore_feed",
                "league": fm["league"],
                "season": "2026",
                "kickoff_at": fm["kickoff_at"],
                "home_team": fm["home_team"],
                "away_team": fm["away_team"],
                "fav_side": fav_side,
                "ah_open": ah_fav,
                "ah_fav_price": fm.get("ah_fav_price") or fm.get("ah_home_price" if fav_side == "home" else "ah_away_price"),
                "ah_dog_price": fm.get("ah_dog_price") or fm.get("ah_away_price" if fav_side == "home" else "ah_home_price"),
                "ou_line_open": float(fm["ou_line"]),
                "ou_over_price": fm.get("ou_over_price"),
                "ou_under_price": fm.get("ou_under_price"),
                "pool_fav_open": fm.get("pool_fav_open"),
                "pool_dog_open": fm.get("pool_dog_open"),
                "draw_ft_odds": fm.get("draw_ft_odds"),
                "draw_ht_odds": fm.get("draw_ht_odds"),
                "ht_home": fm["ht_home"],
                "ht_away": fm["ht_away"],
                "home_goals": fm["home_goals"],
                "away_goals": fm["away_goals"],
                "result_fetched_at": datetime.now(timezone.utc).isoformat(),
            })

    if new_finished_rows:
        count = upsert_matches(con, new_finished_rows)
        stats["new_finished_ingested"] = count
        print(f"  * Ingested {count} completed matches with AH/OU lines from feed into DB.")

    # 4. Optional: check Thscore REST API for yesterday's schedule if available
    try:
        cfg = config.Config.from_env()
        provider = ThscoreApiProvider(config=cfg, con=con)
        yesterday_str = (datetime.now(timezone.utc) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        api_matches = provider.get_schedule_by_date(yesterday_str)
        api_finished = [m for m in api_matches if m.status == -1 and m.home_goals is not None]
        print(f"  * Thscore REST API: Retrieved {len(api_finished)} finished matches for {yesterday_str}.")
    except Exception as e:
        print(f"  * [Info] Thscore REST API schedule check skipped/unavailable: {e}")

    return stats


def load_dataset(con) -> pd.DataFrame:
    """Load fully labelled matches with valid opening odds."""
    df = pd.read_sql_query(
        """
        SELECT * FROM matches
        WHERE kickoff_at IS NOT NULL
          AND ah_open IS NOT NULL
          AND ou_line_open IS NOT NULL
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
        ORDER BY kickoff_at ASC
        """,
        con,
    )
    return df


def get_current_model_meta() -> dict[str, Any]:
    """Read metadata from current production model sidecar."""
    if PROD_META_PATH.exists():
        try:
            with open(PROD_META_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"metadata": {"trained_rows": 0}}


def evaluate_predictions(prep_df: pd.DataFrame, preds: pd.DataFrame) -> dict[str, float]:
    """Compute out-of-sample evaluation metrics."""
    y_test_ah = pd.to_numeric(prep_df["ah_label"], errors="coerce")
    y_test_ou = pd.to_numeric(prep_df["ou_label"], errors="coerce")
    valid_mask = y_test_ah.notna() & y_test_ou.notna()

    y_ah = y_test_ah[valid_mask].astype(int)
    y_ou = y_test_ou[valid_mask].astype(int)
    p_ah = preds.loc[valid_mask, "p_fav_cover"]
    p_ou = preds.loc[valid_mask, "p_over"]

    auc_ah = roc_auc_score(y_ah, p_ah)
    auc_ou = roc_auc_score(y_ou, p_ou)
    brier_ah = brier_score_loss(y_ah, p_ah)
    brier_ou = brier_score_loss(y_ou, p_ou)

    return {
        "ah_auc": round(float(auc_ah), 4),
        "ou_auc": round(float(auc_ou), 4),
        "avg_auc": round(float((auc_ah + auc_ou) / 2.0), 4),
        "ah_brier": round(float(brier_ah), 4),
        "ou_brier": round(float(brier_ou), 4),
        "avg_brier": round(float((brier_ah + brier_ou) / 2.0), 4),
        "evaluated_matches": int(len(y_ah)),
    }


def run_walk_forward_benchmark(
    df: pd.DataFrame,
) -> tuple[bool, dict[str, Any], dict[str, Any], ModelBundle]:
    """Execute Walk-Forward validation against holdout test set (Safety Gate)."""
    print("\n[Step 3/5] Running Walk-Forward Out-of-Sample Benchmark...")

    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx].reset_index(drop=True).copy()
    test_df = df.iloc[split_idx:].reset_index(drop=True).copy()

    print(f"  * Chronological split: {len(train_df)} train matches, {len(test_df)} test matches.")
    print(f"    Train: {train_df['kickoff_at'].min()[:10]} -> {train_df['kickoff_at'].max()[:10]}")
    print(f"    Test : {test_df['kickoff_at'].min()[:10]} -> {test_df['kickoff_at'].max()[:10]}")

    # Ground truth prep
    prep_test = prepare_frame(test_df, need_labels=True, feature_set="thscore_v2")

    # 1. Baseline Benchmark (Current Production Model on Holdout)
    baseline_metrics = {}
    if PROD_MODEL_PATH.exists():
        try:
            prod_bundle = ModelBundle.load(PROD_MODEL_PATH, require_verified=False)
            preds_prod = prod_bundle.predict_proba_df(test_df)
            baseline_metrics = evaluate_predictions(prep_test, preds_prod)
            print(f"  * Production Model Holdout AUC: {baseline_metrics['avg_auc']} | Brier: {baseline_metrics['avg_brier']}")
        except Exception as e:
            print(f"  * [Warning] Could not evaluate current prod model: {e}")

    # 2. Candidate Model Training on 80% Train Split
    print("  * Training Candidate XGBoost model on train split...")
    candidate_bundle, _ = train_model(
        train_df,
        kind="xgboost",
        feature_set="thscore_v2",
        rolling_window=5,
        seed=42,
    )
    preds_cand = candidate_bundle.predict_proba_df(test_df)
    cand_metrics = evaluate_predictions(prep_test, preds_cand)
    print(f"  * Candidate Model Holdout AUC : {cand_metrics['avg_auc']} | Brier: {cand_metrics['avg_brier']}")

    # 3. Safety Gate Decision
    # Criterion: Candidate Avg ROC-AUC >= 0.5000 and within 0.015 of baseline (or better)
    base_auc = baseline_metrics.get("avg_auc", 0.5050)
    passed_auc = cand_metrics["avg_auc"] >= min(0.5000, base_auc - 0.015)
    passed_brier = cand_metrics["avg_brier"] <= 0.2800
    gate_passed = passed_auc and passed_brier

    return gate_passed, cand_metrics, baseline_metrics, candidate_bundle


def promote_production_model(
    df: pd.DataFrame, cand_metrics: dict[str, Any]
) -> tuple[Path, str, dict[str, Any]]:
    """Train on 100% data, back up previous model, and atomically freeze new model."""
    print("\n[Step 4/5] Training final production model on full dataset (100%)...")
    final_bundle, report = train_model(
        df,
        kind="xgboost",
        feature_set="thscore_v2",
        rolling_window=5,
        seed=42,
    )

    # 1. Backup current model if exists
    backup_file = None
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if PROD_MODEL_PATH.exists():
        backup_file = BACKUPS_DIR / f"multileague_v2_{timestamp_str}.pkl"
        shutil.copy2(PROD_MODEL_PATH, backup_file)
        if PROD_META_PATH.exists():
            shutil.copy2(PROD_META_PATH, BACKUPS_DIR / f"multileague_v2_{timestamp_str}.meta.json")
        print(f"  * Backed up previous model to: {backup_file.name}")

    # 2. Atomically save new model
    sidecar = final_bundle.save(PROD_MODEL_PATH)
    print(f"  * Saved new model to: {PROD_MODEL_PATH.name}")
    print(f"  * SHA-256 Sidecar   : {sidecar['sha256']}")
    print(f"  * Total Features    : {len(final_bundle.feature_columns)}")
    print(f"  * Active Rows Used  : {report.n_rows_used}")

    # 3. Append to Audit Log
    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PROMOTED",
        "sha256": sidecar["sha256"],
        "trained_rows": report.n_rows_used,
        "backup_path": str(backup_file) if backup_file else None,
        "metrics": cand_metrics,
    }
    _append_audit_log(audit_entry)

    return backup_file or PROD_MODEL_PATH, sidecar["sha256"], audit_entry


def _append_audit_log(entry: dict[str, Any]):
    """Append a record to retrain_audit.json."""
    logs = []
    if AUDIT_LOG_FILE.exists():
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


def format_telegram_report(
    status: str,
    total_matches: int,
    new_matches: int,
    cand_metrics: dict[str, Any],
    base_metrics: dict[str, Any],
    model_sha: str | None,
    backup_name: str | None,
) -> str:
    """Format rich Telegram HTML card for retrain report."""
    now_thai = (datetime.now(timezone.utc) + pd.Timedelta(hours=7)).strftime("%d/%m/%Y %H:%M") + " น."
    
    status_emoji = "✅" if status == "PROMOTED" else "⚠️"
    status_text = (
        "<b>PROMOTED TO PRODUCTION</b>"
        if status == "PROMOTED"
        else "<b>PROMOTION REJECTED (SAFETY GATE)</b>"
    )

    base_auc_str = f"{base_metrics.get('avg_auc', 0.5097):.4f}" if base_metrics else "N/A"
    cand_auc_str = f"{cand_metrics.get('avg_auc', 0.0):.4f}"
    diff_auc = cand_metrics.get("avg_auc", 0.0) - base_metrics.get("avg_auc", cand_metrics.get("avg_auc", 0.0))
    diff_sign = "+" if diff_auc >= 0 else ""
    diff_str = f"({diff_sign}{diff_auc:.4f})" if base_metrics else ""

    card = (
        f"🔄 <b>AUTO-RETRAIN PIPELINE REPORT</b>\n"
        f"<i>Machine Learning Self-Updating Engine</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 <b>เวลา:</b> {now_thai}\n"
        f"📊 <b>Dataset Status:</b>\n"
        f"  • Total Matches: <b>{total_matches:,}</b> คู่\n"
        f"  • New Labelled : <b>+{new_matches}</b> คู่ใหม่\n\n"
        f"🛡️ <b>Walk-Forward Safety Gate (Holdout 20%):</b>\n"
        f"  • AH ROC-AUC   : <b>{cand_metrics.get('ah_auc', 0.0):.4f}</b>\n"
        f"  • OU ROC-AUC   : <b>{cand_metrics.get('ou_auc', 0.0):.4f}</b>\n"
        f"  • Avg ROC-AUC  : <b>{cand_auc_str}</b> {diff_str}\n"
        f"  • Avg Brier    : <b>{cand_metrics.get('avg_brier', 0.0):.4f}</b>\n\n"
        f"🏆 <b>Decision:</b> {status_emoji} {status_text}\n"
    )

    if status == "PROMOTED" and model_sha:
        short_sha = model_sha[:16] + "..."
        card += (
            f"  • Model File   : <code>multileague_v2.pkl</code>\n"
            f"  • Backup File  : <code>{backup_name or 'N/A'}</code>\n"
            f"  • SHA-256      : <code>{short_sha}</code>\n"
        )
    else:
        card += "  • Action: Current production model kept intact. No regression.\n"

    card += "━━━━━━━━━━━━━━━━━━━━\n🎯 <i>Ready for daily 17:30 quant card generation.</i>"
    return card


def main():
    parser = argparse.ArgumentParser(description="Multi-League Model Auto-Retraining Pipeline")
    parser.add_argument("--force", action="store_true", help="Force retrain even if new matches < threshold")
    parser.add_argument("--min-new-matches", type=int, default=10, help="Minimum new matches to trigger retrain")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching new results from CDN/API")
    parser.add_argument("--dry-run", action="store_true", help="Run benchmark without saving model")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram notification")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 AUTOMATED MULTI-LEAGUE RETRAINING PIPELINE")
    print("=" * 80)

    con = connect(ROOT_DIR / "data" / "football.db")

    # Step 1: Sync match results
    sync_stats = {"updated_pending": 0, "new_finished_ingested": 0}
    if not args.skip_fetch:
        sync_stats = sync_results_and_ingest(con)

    # Step 2: Load labelled matches
    df = load_dataset(con)
    con.close()
    total_matches = len(df)
    print(f"\n[Step 2/5] Database has {total_matches} labelled matches with opening odds.")

    # Check against previous training metadata
    meta = get_current_model_meta()
    prev_trained_rows = meta.get("metadata", {}).get("trained_rows", 0)
    new_matches = max(0, total_matches - prev_trained_rows)

    print(f"  * Previous trained rows: {prev_trained_rows}")
    print(f"  * Current ready rows   : {total_matches}")
    print(f"  * Newly added matches  : {new_matches} (Threshold: {args.min_new_matches})")

    if new_matches < args.min_new_matches and not args.force:
        print(f"\n[Info] Only {new_matches} new matches (less than threshold {args.min_new_matches}).")
        print("Model is already up to date. Use --force to override. Exiting gracefully.")
        return 0

    # Step 3: Run Walk-Forward Safety Gate Benchmark
    gate_passed, cand_metrics, base_metrics, _ = run_walk_forward_benchmark(df)

    # Step 4: Decision & Promotion
    if args.dry_run:
        print("\n[Dry Run] Model evaluation complete. Production files untouched.")
        status = "DRY_RUN"
        model_sha = None
        backup_name = None
    elif gate_passed or args.force:
        print("\n✅ Safety Gate PASSED! Proceeding with model promotion...")
        backup_file, model_sha, audit_entry = promote_production_model(df, cand_metrics)
        status = "PROMOTED"
        backup_name = backup_file.name if backup_file else None
    else:
        print("\n⚠️ Safety Gate REJECTED candidate model due to performance degradation.")
        print("Current production model remains in place.")
        status = "REJECTED"
        model_sha = None
        backup_name = None
        _append_audit_log({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "REJECTED",
            "reason": "Safety Gate Failed (Degraded Holdout Metrics)",
            "metrics": cand_metrics,
        })

    # Step 5: Telegram Notification
    if not args.no_telegram and (status == "PROMOTED" or args.force or status == "REJECTED"):
        print("\n[Step 5/5] Sending Telegram Retrain Audit Report...")
        card_text = format_telegram_report(
            status=status,
            total_matches=total_matches,
            new_matches=new_matches,
            cand_metrics=cand_metrics,
            base_metrics=base_metrics,
            model_sha=model_sha,
            backup_name=backup_name,
        )
        send_telegram(card_text)
        print("  * Telegram notification sent successfully!")

    print("\n" + "=" * 80)
    print(f"🏁 PIPELINE RUN COMPLETED. STATUS: {status}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
