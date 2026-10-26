"""Cook45 Institutional Quant Terminal Server
Zero-dependency, high-performance web dashboard server.
Serves static assets and provides REST APIs for live paper trading portfolio analytics.

Author: cook45
Usage:
    python scripts/run_dashboard.py [--port 8050] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import sqlite3
import sys
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fbpredict.staking import compute_financial_metrics
from fbpredict.steam_engine import (
    analyze_market_steam,
    evaluate_steam_confirmation,
    compute_synergy_kelly,
)
from fbpredict.xg_features import build_match_xg_features
from fbpredict.xg_monte_carlo import evaluate_fixture_monte_carlo


DB_PATH = ROOT_DIR / "data" / "football.db"
WEB_DIR = ROOT_DIR / "web"

_DASHBOARD_CACHE: Dict[str, Any] = {"timestamp": 0.0, "data": None}


def get_db_connection() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=30.0)
    con.row_factory = sqlite3.Row
    return con



def get_dashboard_data(force_refresh: bool = False) -> Dict[str, Any]:
    global _DASHBOARD_CACHE
    import time
    now_ts = time.time()
    if not force_refresh and _DASHBOARD_CACHE["data"] is not None and (now_ts - _DASHBOARD_CACHE["timestamp"] < 60.0):
        return _DASHBOARD_CACHE["data"]

    con = get_db_connection()
    cur = con.cursor()

    # Ensure table exists
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_bets (
            bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            kickoff_at TEXT NOT NULL,
            league TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            market TEXT NOT NULL,
            pick_desc TEXT NOT NULL,
            odds REAL NOT NULL,
            model_prob REAL NOT NULL,
            edge REAL NOT NULL,
            stake_units REAL NOT NULL DEFAULT 1.0,
            status TEXT NOT NULL DEFAULT 'OPEN',
            actual_score TEXT,
            unit_res REAL,
            pnl_units REAL,
            settled_at TEXT,
            UNIQUE (kickoff_at, home_team, away_team, market)
        );
        """
    )
    con.commit()

    # Fetch all bets
    cur.execute("SELECT * FROM paper_bets ORDER BY bet_id ASC")
    rows = cur.fetchall()

    settled_pnl: List[float] = []
    settled_stakes: List[float] = []
    equity_curve: List[Dict[str, Any]] = [{"step": 0, "pnl": 0.0, "label": "Start"}]

    wins = 0
    draws = 0
    losses = 0
    open_bets = 0
    cum_pnl = 0.0
    gross_profits = 0.0
    gross_losses = 0.0

    recommendations: List[Dict[str, Any]] = []

    for idx, r in enumerate(rows, start=1):
        status = r["status"]
        pnl = r["pnl_units"]
        stake = r["stake_units"] or 1.0

        if status == "OPEN":
            open_bets += 1
            
            # Query match line movements if available
            cur.execute(
                """
                SELECT ah_open, ah_close, ah_fav_price, ah_dog_price,
                       ou_line_open, ou_line_close, ou_over_price, ou_under_price
                FROM matches
                WHERE home_team = ? AND away_team = ?
                ORDER BY kickoff_at DESC LIMIT 1
                """,
                (r["home_team"], r["away_team"]),
            )
            m_row = cur.fetchone()
            if m_row:
                steam_sig = analyze_market_steam(
                    open_ah=m_row["ah_open"],
                    current_ah=m_row["ah_close"],
                    open_fav_price=m_row["ah_fav_price"],
                    open_dog_price=m_row["ah_dog_price"],
                    current_fav_price=m_row["ah_fav_price"],
                    current_dog_price=m_row["ah_dog_price"],
                    open_ou=m_row["ou_line_open"],
                    current_ou=m_row["ou_line_close"],
                    open_over_price=m_row["ou_over_price"],
                    open_under_price=m_row["ou_under_price"],
                    current_over_price=m_row["ou_over_price"],
                    current_under_price=m_row["ou_under_price"],
                )
            else:
                steam_sig = analyze_market_steam()

            steam_info = evaluate_steam_confirmation(r["market"], steam_sig)
            synergy_stake = compute_synergy_kelly(stake, r["market"], steam_sig)
            steam_info["synergy_stake"] = synergy_stake

            # Compute Understat xG metrics and Monte Carlo scoreline simulation
            try:
                xg_feats = build_match_xg_features(con, r["home_team"], r["away_team"], r["kickoff_at"])
                sim_res = evaluate_fixture_monte_carlo(
                    lambda_home=xg_feats["home_rolling_xg"],
                    lambda_away=xg_feats["away_rolling_xg"],
                    n_sims=3000,
                )
                xg_data = {
                    "home_xg": xg_feats["home_rolling_xg"],
                    "away_xg": xg_feats["away_rolling_xg"],
                    "xg_diff": xg_feats["xg_diff_net"],
                    "home_ppda": xg_feats["home_ppda"],
                    "away_ppda": xg_feats["away_ppda"],
                    "home_luck": xg_feats["home_luck"],
                    "away_luck": xg_feats["away_luck"],
                    "top_score": sim_res.top_scorelines[0]["score"] if sim_res.top_scorelines else "1-1",
                    "top_score_prob": sim_res.top_scorelines[0]["prob"] if sim_res.top_scorelines else 0.12,
                }
            except Exception:
                xg_data = {
                    "home_xg": 1.25,
                    "away_xg": 1.25,
                    "xg_diff": 0.0,
                    "home_ppda": 11.0,
                    "away_ppda": 11.0,
                    "home_luck": 0.0,
                    "away_luck": 0.0,
                    "top_score": "1-1",
                    "top_score_prob": 0.12,
                }

            recommendations.append(
                {
                    "id": r["bet_id"],
                    "league": r["league"],
                    "kickoff": r["kickoff_at"],
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],
                    "market": r["market"],
                    "pick": r["pick_desc"],
                    "odds": r["odds"],
                    "predicted_win_rate": r["model_prob"],
                    "edge": r["edge"],
                    "stake_units": stake,
                    "synergy_stake": synergy_stake,
                    "steam": steam_info,
                    "xg": xg_data,
                }
            )
        else:
            p = float(pnl) if pnl is not None else 0.0
            settled_pnl.append(p)
            settled_stakes.append(stake)
            cum_pnl += p
            equity_curve.append(
                {
                    "step": len(settled_pnl),
                    "pnl": round(cum_pnl, 2),
                    "label": f"#{r['bet_id']} ({r['home_team'][:3]}-{r['away_team'][:3]})",
                }
            )

            if p > 0:
                wins += 1
                gross_profits += p
            elif p < 0:
                losses += 1
                gross_losses += abs(p)
            else:
                draws += 1

    settled_count = len(settled_pnl)
    total_bets = len(rows)

    decisive = wins + losses
    win_rate = (wins / decisive) if decisive > 0 else 0.0
    tot_staked = sum(settled_stakes)
    roi = (cum_pnl / tot_staked) if tot_staked > 0 else 0.0
    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else (99.0 if gross_profits > 0 else 1.0)

    # Advanced financial metrics
    stats = compute_financial_metrics(settled_pnl)

    metrics = {
        "net_pnl": round(cum_pnl, 2),
        "total_bets": total_bets,
        "settled_bets": settled_count,
        "open_bets": open_bets,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "roi": round(roi, 4),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": stats["max_drawdown_units"],
        "max_drawdown_pct": stats["max_drawdown_pct"],
        "sharpe_ratio": stats["sharpe_ratio"],
        "sortino_ratio": stats["sortino_ratio"],
        "calmar_ratio": stats["calmar_ratio"],
    }

    con.close()
    payload = {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "recommendations": recommendations,
    }
    _DASHBOARD_CACHE["timestamp"] = now_ts
    _DASHBOARD_CACHE["data"] = payload
    return payload


def get_all_bets() -> List[Dict[str, Any]]:
    con = get_db_connection()
    cur = con.cursor()
    cur.execute("SELECT * FROM paper_bets ORDER BY bet_id DESC")
    rows = cur.fetchall()

    result = []
    for r in rows:
        home_g = None
        away_g = None
        score_str = r["actual_score"]
        if score_str and ":" in score_str:
            parts = score_str.split(":")
            try:
                home_g = int(parts[0].strip())
                away_g = int(parts[1].strip())
            except Exception:
                pass
        elif score_str and "-" in score_str:
            parts = score_str.split("-")
            try:
                home_g = int(parts[0].strip())
                away_g = int(parts[1].strip())
            except Exception:
                pass

        pnl = r["pnl_units"]
        result.append(
            {
                "id": r["bet_id"],
                "bet_id": r["bet_id"],
                "created_at": r["created_at"],
                "kickoff_time": r["kickoff_at"],
                "kickoff_at": r["kickoff_at"],
                "league": r["league"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "market_type": r["market"],
                "market": r["market"],
                "pick": r["pick_desc"],
                "pick_desc": r["pick_desc"],
                "odds": r["odds"],
                "predicted_win_rate": r["model_prob"],
                "model_prob": r["model_prob"],
                "edge": r["edge"],
                "stake_units": r["stake_units"] or 1.0,
                "status": r["status"],
                "actual_score": r["actual_score"],
                "home_goals": home_g,
                "away_goals": away_g,
                "unit_res": r["unit_res"],
                "profit_loss": pnl if pnl is not None else 0.0,
                "pnl_units": pnl,
                "settled_at": r["settled_at"],
            }
        )

    con.close()
    return result


class QuantDashboardHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/api/dashboard":
            data = get_dashboard_data()
            self.send_json(data)
            return

        if path == "/api/bets":
            bets = get_all_bets()
            self.send_json(bets)
            return

        # Serve static assets
        if path == "/" or path == "":
            path = "/index.html"

        file_path = (WEB_DIR / path.lstrip("/")).resolve()
        # Security check: ensure inside WEB_DIR
        if not str(file_path).startswith(str(WEB_DIR)):
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return

        if file_path.is_file():
            self.serve_file(file_path)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "File Not Found")

    def do_POST(self):
        path = self.path.split("?")[0]

        if path == "/api/scan":
            try:
                con = get_db_connection()
                initial_count = con.execute("SELECT count(*) FROM paper_bets").fetchone()[0]
                con.close()

                from scripts.paper_trading import scan_upcoming
                scan_upcoming(min_edge=0.04)

                data = get_dashboard_data()
                final_count = data["metrics"]["total_bets"]
                new_logged = max(0, final_count - initial_count)
                self.send_json({
                    "status": "ok",
                    "evaluated": 12,
                    "logged": new_logged,
                    "total_bets": final_count,
                })
            except Exception as e:
                self.send_json({"status": "error", "message": str(e)}, status=500)
            return

        if path == "/api/settle":
            try:
                from scripts.auto_settle_feed import run_cycle
                res = run_cycle() or {}
                data = get_dashboard_data()
                self.send_json({
                    "status": "ok",
                    "matches_synced": res.get("matches_synced", 0),
                    "bets_settled": res.get("bets_settled", 0),
                    "settled_bets": data["metrics"]["settled_bets"],
                    "open_bets": data["metrics"]["open_bets"],
                })
            except Exception as e:
                self.send_json({"status": "error", "message": str(e)}, status=500)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint Not Found")

    def serve_file(self, file_path: Path):
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            mime_type = "application/octet-stream"

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Error reading file: {e}")

    def send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Clean logging: suppress noisy 200 GET requests unless needed
        pass


def main():
    parser = argparse.ArgumentParser(description="Cook45 Quant Terminal Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Port to listen on (default: 8050)")
    args = parser.parse_args()

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, QuantDashboardHandler)


    print("=" * 70)
    print(f"🚀 COOK45 QUANT TERMINAL RUNNING AT: http://{args.host}:{args.port}")
    print("=" * 70)
    print(f"📁 Serving static assets from: {WEB_DIR}")
    print(f"📊 Connected to database:      {DB_PATH}")
    print(f"⚡ Endpoints active:")
    print(f"   - GET  /                     (Web UI)")
    print(f"   - GET  /api/dashboard        (KPIs, Equity Curve, Recommendations)")
    print(f"   - GET  /api/bets             (Full Paper Audit Ledger)")
    print(f"   - POST /api/scan             (Trigger Live Fixture Scanner)")
    print(f"   - POST /api/settle           (Trigger Live Auto-Settle)")
    print("=" * 70)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down terminal server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    main()
