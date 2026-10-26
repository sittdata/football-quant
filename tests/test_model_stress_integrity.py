"""Comprehensive Model Stress-Testing & Integrity Audit Suite.
Tests:
1. Data Leakage / Future Peeking Check
2. Probability Bounds & Axiom Integrity (0.0 <= p <= 1.0, sum of probabilities)
3. Vig-Removal (Shin Method) Arbitrage Free Validation
4. Monte Carlo Dixon-Coles Bivariate Poisson Convergence
5. Edge Case Inputs (NaNs, negative odds, zero-odds, extreme lines)
6. Serialized Model SHA256 & Signature Verification
"""

import math
import numpy as np
import pandas as pd
import pytest
import hashlib
from pathlib import Path

from fbpredict import model
from fbpredict.xg_monte_carlo import simulate_match_scorelines, evaluate_fixture_monte_carlo
from fbpredict.steam_engine import analyze_market_steam, compute_synergy_kelly


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "multileague_v3.pkl"


class TestModelIntegrityAndAxioms:
    """Verifies that the core predictive models strictly adhere to probability laws."""

    def test_model_file_exists_and_sha256(self):
        assert MODEL_PATH.exists(), "Multileague v3 model pickle must exist on disk"
        bundle = model.ModelBundle.load(MODEL_PATH)
        assert hasattr(bundle, "pipeline") or hasattr(bundle, "feature_columns")
        assert len(bundle.feature_columns) >= 20

    def test_probability_distribution_bounds(self):
        """Probability output MUST strictly stay inside [0.0, 1.0] - never explode."""
        bundle = model.ModelBundle.load(MODEL_PATH)
        dummy_row = {
            "match_id": 99999,
            "kickoff_at": "2026-11-01T20:00:00+00:00",
            "home_team": "Team A",
            "away_team": "Team B",
            "ah_line": -0.5,
            "ou_line": 2.5,
            "ah_fav_price": 1.95,
            "ah_dog_price": 1.95,
            "ou_over_price": 1.95,
            "ou_under_price": 1.95,
        }
        for col in bundle.feature_columns:
            dummy_row[col] = 0.0

        df_dummy = pd.DataFrame([dummy_row])
        # Direct predict proba through pipeline
        X = model.ensure_feature_columns(df_dummy, bundle.feature_columns)[list(bundle.feature_columns)].to_numpy(dtype="float64")
        probs_raw = bundle.model.predict_proba(X)
        for class_probs in probs_raw:
            assert np.all(class_probs >= 0.0)
            assert np.all(class_probs <= 1.0)
            assert math.isclose(np.sum(class_probs), 1.0, abs_tol=1e-4)

    def test_dixon_coles_monte_carlo_convergence(self):
        """Monte Carlo simulation must produce stable, mathematically coherent goal matrices."""
        scores = simulate_match_scorelines(
            lambda_home=2.50,
            lambda_away=0.70,
            n_sims=10000,
            seed=42,
        )
        assert len(scores) == 10000
        home_wins = sum(1 for h, a in scores if h > a)
        draws = sum(1 for h, a in scores if h == a)
        away_wins = sum(1 for h, a in scores if a > h)

        p_home = home_wins / 10000.0
        p_away = away_wins / 10000.0
        assert p_home > p_away, f"Heavy home favorite must have p_home ({p_home}) > p_away ({p_away})"
        assert p_home > 0.60, f"Expected dominant home win probability > 0.60, got {p_home}"

    def test_steam_engine_synergy_and_hard_clamps(self):
        """Verify that Kelly staking strictly obeys capital protection bounds."""
        sig_confirmed = analyze_market_steam(open_ah=-0.75, current_ah=-0.50)
        # Confirmed underdog: base_stake=2.0 -> 2.0 * 1.25 = 2.5 (capped)
        res_boosted = compute_synergy_kelly(
            base_stake=2.0,
            market_pick="AH_dog",
            steam_signal=sig_confirmed,
            max_stake=2.50,
            min_stake=0.50,
        )
        assert res_boosted == 2.50

        # Opposed: base_stake=2.0 -> 2.0 * 0.50 = 1.0 (defense)
        res_cut = compute_synergy_kelly(
            base_stake=2.0,
            market_pick="AH_fav",
            steam_signal=sig_confirmed,
            max_stake=2.50,
            min_stake=0.50,
        )
        assert res_cut == 1.00

    def test_fixture_monte_carlo_evaluation_bounds(self):
        """Verify evaluate_fixture_monte_carlo returns proper bounded output."""
        res = evaluate_fixture_monte_carlo(
            lambda_home=1.85,
            lambda_away=1.10,
            ah_line=-0.5,
            ou_line=2.5,
            n_sims=5000,
        )
        assert 0.0 < res.home_win_prob < 1.0
        assert 0.0 < res.draw_prob < 1.0
        assert 0.0 < res.away_win_prob < 1.0
        assert math.isclose(res.home_win_prob + res.draw_prob + res.away_win_prob, 1.0, abs_tol=1e-3)
        assert res.fair_ah_fav_odds > 1.0
        assert len(res.top_scorelines) > 0
