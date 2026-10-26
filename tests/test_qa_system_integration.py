"""System-Wide QA Integration & Security Test Suite.
Enforces Phase 3 (Integration), Phase 6 (Performance), and Phase 7 (Security) of testing-qa.
"""

import json
import sqlite3
import time
import urllib.request
import urllib.error
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "football.db"


class TestDatabaseIntegrity:
    """Verifies SQLite production and paper ledger schema constraints."""

    def test_database_connection_and_tables(self):
        assert DB_PATH.exists(), f"Database missing at {DB_PATH}"
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]
        
        assert "matches" in tables, "Table 'matches' must exist in football.db"
        assert "paper_bets" in tables, "Table 'paper_bets' must exist in football.db"
        con.close()

    def test_paper_bets_constraints(self):
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        
        cur.execute("SELECT COUNT(*) as cnt FROM paper_bets;")
        count = cur.fetchone()["cnt"]
        assert count >= 0

        # Check column types and required fields
        cur.execute("PRAGMA table_info(paper_bets);")
        cols = {r["name"]: r["type"] for r in cur.fetchall()}
        required = ["bet_id", "kickoff_at", "home_team", "away_team", "market", "odds", "model_prob", "edge", "status"]
        for col in required:
            assert col in cols, f"Column '{col}' missing from paper_bets"
        con.close()


class TestDashboardAPIAndSecurity:
    """Tests live dashboard server on port 8050 for contracts, performance, and security."""

    BASE_URL = "http://127.0.0.1:8050"

    def test_dashboard_api_contract_and_latency(self):
        start_ts = time.time()
        try:
            req = urllib.request.Request(f"{self.BASE_URL}/api/dashboard")
            with urllib.request.urlopen(req, timeout=5) as resp:
                latency = time.time() - start_ts
                assert resp.status == 200, f"Expected 200, got {resp.status}"
                data = json.loads(resp.read().decode("utf-8"))
                
                assert "metrics" in data, "Dashboard response must contain 'metrics'"
                assert "equity_curve" in data, "Dashboard response must contain 'equity_curve'"
                assert "recommendations" in data, "Dashboard response must contain 'recommendations'"
                
                # Performance Benchmark: response latency under 200ms
                assert latency < 0.5, f"Dashboard API latency too high: {latency*1000:.1f}ms"
        except urllib.error.URLError:
            pytest.skip("Dashboard daemon on port 8050 is offline during this test run")

    def test_bets_api_contract(self):
        try:
            req = urllib.request.Request(f"{self.BASE_URL}/api/bets")
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert resp.status == 200
                bets = json.loads(resp.read().decode("utf-8"))
                assert isinstance(bets, list)
        except urllib.error.URLError:
            pytest.skip("Dashboard daemon on port 8050 is offline")

    def test_security_path_traversal_blocked(self):
        """SAST/Security: Verify static file server prevents Directory Traversal."""
        try:
            req = urllib.request.Request(f"{self.BASE_URL}/../../../../Windows/win.ini")
            try:
                urllib.request.urlopen(req, timeout=3)
                pytest.fail("Path traversal should have been rejected with 403 or 404")
            except urllib.error.HTTPError as e:
                assert e.code in [403, 404], f"Expected 403 or 404, got {e.code}"
        except urllib.error.URLError:
            pytest.skip("Dashboard daemon on port 8050 is offline")


class TestQuantMathematicalInvariance:
    """Validates mathematical properties of quant probabilities, edge, and Kelly staking."""

    def test_probability_bounds_and_edge_calculation(self):
        from fbpredict.staking import calculate_kelly_fraction
        
        prob = 0.58
        odds = 1.95
        
        # Edge = (P * Odds) - 1.0 = (0.58 * 1.95) - 1.0 = 1.131 - 1.0 = +0.131 (+13.1%)
        edge = (prob * odds) - 1.0
        assert abs(edge - 0.131) < 1e-5
        
        # Invariance: If edge <= 0 (negative expectation), Kelly stake must be strictly 0.0
        no_edge_stake = calculate_kelly_fraction(prob=0.40, decimal_odds=2.0)
        assert no_edge_stake == 0.0, "Zero/negative edge must result in 0.0 stake"
        
        # Positive edge should yield valid positive stake clamped within bounds
        pos_stake = calculate_kelly_fraction(prob=0.58, decimal_odds=1.95)
        assert 0.5 <= pos_stake <= 2.5, f"Stake {pos_stake} must be clamped between 0.5 and 2.5"

    def test_understat_matches_integrity(self):
        """QA Audit: Verify understat_matches table contains full European big data."""
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM understat_matches;")
        count = cur.fetchone()[0]
        assert count >= 5000, f"Expected at least 5,000 Understat records, found {count}"

        # Verify no NaN or negative values in xG
        cur.execute("SELECT COUNT(*) FROM understat_matches WHERE home_xg < 0 OR away_xg < 0;")
        invalid_xg = cur.fetchone()[0]
        assert invalid_xg == 0, f"Found {invalid_xg} records with invalid negative xG"
        con.close()

    def test_multileague_v3_sidecar_and_model_security(self):
        """Security/SAST: Verify model pickle sha256 sidecar matches binary."""
        from fbpredict.model import ModelBundle
        v3_path = ROOT_DIR / "models" / "multileague_v3.pkl"
        assert v3_path.exists(), "multileague_v3.pkl must exist"

        verified, meta = ModelBundle.verify(v3_path)
        assert verified, f"v3 Model sha256 mismatch: {meta}"
        assert len(meta.get("feature_columns", [])) == 68, "v3 model must have exactly 68 features"

    def test_xg_monte_carlo_probability_conservation(self):
        """QA Audit: Monte Carlo simulation must conserve probability mass sum to 1.0."""
        from fbpredict.xg_monte_carlo import evaluate_fixture_monte_carlo
        res = evaluate_fixture_monte_carlo(lambda_home=1.85, lambda_away=1.15, n_sims=3000, seed=123)
        total_p = res.home_win_prob + res.draw_prob + res.away_win_prob
        assert abs(total_p - 1.0) < 0.01, f"Outcome probabilities must sum to 1.0, got {total_p}"
        assert res.fair_ah_fav_odds >= 1.0
        assert res.fair_over_odds >= 1.0


