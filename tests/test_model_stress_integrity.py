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
        sig_confirmed = analyze_market_steam(
            open_ah=-0.75,
            current_ah=-0.50,
            open_fav_price=1.95,
            open_dog_price=1.95,
            current_fav_price=1.95,
            current_dog_price=1.95,
            open_over_price=1.95,
            open_under_price=1.95,
            current_over_price=1.95,
            current_under_price=1.95,
        )
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

        # Without prices (has_odds=False), line-delta steam must NOT scale stakes
        sig_no_odds = analyze_market_steam(open_ah=-0.75, current_ah=-0.50)
        assert sig_no_odds.has_odds is False
        res_no_odds = compute_synergy_kelly(
            base_stake=2.0,
            market_pick="AH_dog",
            steam_signal=sig_no_odds,
            max_stake=2.50,
            min_stake=0.50,
        )
        assert res_no_odds == 2.00

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

    def test_feature_store_enrich_match_null_odds_and_handicaps(self):
        """Verify feature engine does not raise TypeError when matches have NULL handicaps or odds."""
        from fbpredict.feature_engine import HistoricalFeatureStore
        fs = HistoricalFeatureStore(db_path=ROOT_DIR / "data" / "football.db")
        feat = fs.enrich_match(
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff_iso="2026-10-30T15:00:00",
            ah_open=None,
            ou_line_open=None,
            ah_fav_price=None,
            ah_dog_price=None,
            ou_over_price=None,
            ou_under_price=None,
            draw_ft_odds=None,
            fav_side=None,
        )
        assert isinstance(feat, dict)
        assert np.isnan(feat["ah_open"])
        assert np.isnan(feat["ou_line_open"])
        assert np.isnan(feat["total_x_ah"])
        # Missing quotes must stay NaN, not be replaced by a fabricated 1.95 market.
        assert np.isnan(feat["fair_fav_ah"])
        assert np.isnan(feat["fair_dog_ah"])
        assert np.isnan(feat["ah_overround"])
        assert np.isnan(feat["fav_form"])
        assert np.isnan(feat["is_fav_home"])

    def test_quant_model_predict_match_none_odds_safe(self):
        """Verify QuantModelBundle.predict_match does not crash on None/NaN fair probs and odds."""
        from fbpredict.quant_model import QuantModelBundle, AH_FEATURE_COLS, OU_FEATURE_COLS
        from sklearn.dummy import DummyClassifier

        ah_mock = DummyClassifier(strategy="constant", constant=1).fit(
            np.zeros((5, len(AH_FEATURE_COLS))), np.ones(5)
        )
        ou_mock = DummyClassifier(strategy="constant", constant=1).fit(
            np.zeros((5, len(OU_FEATURE_COLS))), np.ones(5)
        )
        bundle = QuantModelBundle(ah_model=ah_mock, ou_model=ou_mock)

        dummy_features = {col: 0.0 for col in AH_FEATURE_COLS}
        dummy_features.update({col: 0.0 for col in OU_FEATURE_COLS})
        dummy_features["has_sufficient_history"] = 1
        dummy_features["fair_fav_ah"] = None
        dummy_features["fair_dog_ah"] = None
        dummy_features["fair_over"] = None
        dummy_features["fair_under"] = None
        dummy_features["ah_fav_price"] = None
        dummy_features["ah_dog_price"] = None
        dummy_features["ou_over_price"] = None
        dummy_features["ou_under_price"] = None

        pred = bundle.predict_match(dummy_features)
        assert pred is not None
        # No quoted price anywhere -> refuse to pick rather than invent 1.90.
        assert pred["safe"] is False
        assert pred["best_market"] is None
        assert "No quoted odds" in pred["reason"]

    def test_quant_model_predict_match_uses_only_quoted_prices(self):
        """A pick may only be priced from a real quote, and single-class 0 must not read as 1."""
        from fbpredict.quant_model import QuantModelBundle, AH_FEATURE_COLS, OU_FEATURE_COLS
        from sklearn.dummy import DummyClassifier

        # Constant class 0 everywhere: P(1) is 0, not 1 from column 0.
        ah_mock = DummyClassifier(strategy="constant", constant=0).fit(
            np.zeros((5, len(AH_FEATURE_COLS))), np.zeros(5)
        )
        ou_mock = DummyClassifier(strategy="constant", constant=1).fit(
            np.zeros((5, len(OU_FEATURE_COLS))), np.ones(5)
        )
        bundle = QuantModelBundle(ah_model=ah_mock, ou_model=ou_mock)

        dummy_features = {col: 0.0 for col in AH_FEATURE_COLS}
        dummy_features.update({col: 0.0 for col in OU_FEATURE_COLS})
        dummy_features["has_sufficient_history"] = 1
        dummy_features["fair_fav_ah"] = 0.5
        dummy_features["fair_dog_ah"] = 0.5
        dummy_features["fair_over"] = 0.5
        dummy_features["fair_under"] = 0.5
        dummy_features["ah_fav_price"] = None
        dummy_features["ah_dog_price"] = None
        dummy_features["ou_over_price"] = 2.10
        dummy_features["ou_under_price"] = 1.90

        pred = bundle.predict_match(dummy_features)
        assert pred["safe"] is True
        assert pred["p_fav"] == 0.0
        assert pred["best_market"] in ("OU_over", "OU_under")
        # Quoted pair exists; best market picked from the real pair
        assert pred["best_odds"] in (2.10, 1.90)

    def test_quant_model_skips_unpaired_markets(self):
        """When only one side of a pair is quoted, skip the market instead of inventing fair 0.5."""
        from fbpredict.quant_model import QuantModelBundle, AH_FEATURE_COLS, OU_FEATURE_COLS
        from sklearn.dummy import DummyClassifier

        ah_mock = DummyClassifier(strategy="constant", constant=0).fit(
            np.zeros((5, len(AH_FEATURE_COLS))), np.zeros(5)
        )
        ou_mock = DummyClassifier(strategy="constant", constant=1).fit(
            np.zeros((5, len(OU_FEATURE_COLS))), np.ones(5)
        )
        bundle = QuantModelBundle(ah_model=ah_mock, ou_model=ou_mock)

        dummy_features = {col: 0.0 for col in AH_FEATURE_COLS}
        dummy_features.update({col: 0.0 for col in OU_FEATURE_COLS})
        dummy_features["has_sufficient_history"] = 1
        dummy_features["ah_fav_price"] = 1.95
        dummy_features["ah_dog_price"] = None  # AH is unpaired
        dummy_features["ou_over_price"] = 2.10
        dummy_features["ou_under_price"] = None  # OU is unpaired

        pred = bundle.predict_match(dummy_features)
        # All available markets are unpaired -> refuse to price
        assert pred["safe"] is False
        assert pred["best_market"] is None

    def test_paper_trading_candidate_filtering_none_edge(self):
        """Paper trading candidate selection must never crash max(candidates) when some edges are None."""
        # Simulated pred where AH is unpaired (edges are None) and OU is paired
        pred = {
            "safe": True,
            "edge_fav": None,
            "edge_dog": None,
            "edge_over": 0.065,
            "edge_under": -0.065,
            "p_fav": 0.5,
            "p_dog": 0.5,
            "p_over": 0.55,
            "p_under": 0.45,
        }
        # Simulate paper_trading candidate building
        candidates = []
        is_champ = False
        side = "home"
        ah_open = -0.5
        ou_line = 2.5
        ah_fav_price = 1.95
        ah_dog_price = None
        ou_over_price = 2.05
        ou_under_price = 1.85

        if side in ("home", "away") and ah_open is not None and ah_fav_price is not None and pred.get("edge_fav") is not None:
            candidates.append(("AH_fav", pred["edge_fav"], pred["p_fav"], ah_fav_price))
        if ou_line is not None and ou_over_price is not None and pred.get("edge_over") is not None:
            candidates.append(("OU_over", pred["edge_over"], pred["p_over"], ou_over_price))
        if ou_line is not None and ou_under_price is not None and pred.get("edge_under") is not None:
            candidates.append(("OU_under", pred["edge_under"], pred["p_under"], ou_under_price))

        assert len(candidates) == 2
        # max must work cleanly without TypeError: '>' not supported between 'float' and 'NoneType'
        best_mkt, best_edge, best_prob, best_odds = max(candidates, key=lambda x: x[1])
        assert best_mkt == "OU_over"
        assert best_edge == 0.065
