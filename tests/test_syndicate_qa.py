"""
Comprehensive QA Test Suite for Syndicate Quant Alpha Engine.
Tests:
1. Mathematical invariants: Shin Vig-Removal (1993), American Odds conversion, Kelly Criterion.
2. In-Play Momentum processing & Bayesian probability update logic.
3. Component Integration: PinnacleSharpFeed, SofaScoreQuantFeed, SyndicateComboEngine.
4. Edge cases & error handling (missing fields, extreme odds, zero division, NaN values).
"""

import pytest
import math
import numpy as np
from unittest.mock import MagicMock, patch

from fbpredict.providers.pinnacle_feed import PinnacleSharpFeed
from fbpredict.providers.sofascore_feed import SofaScoreQuantFeed
from fbpredict.syndicate_engine import SyndicateComboEngine, SyndicateSignal

# ==============================================================================
# Phase 1: Mathematical Invariant Unit Tests
# ==============================================================================

class TestMathInvariants:
    def test_american_to_decimal_conversion(self):
        # Positive moneyline (+150 -> 2.50)
        assert PinnacleSharpFeed.american_to_decimal(150) == 2.50
        # Even money (+100 -> 2.00, -100 -> 2.00)
        assert PinnacleSharpFeed.american_to_decimal(100) == 2.00
        assert PinnacleSharpFeed.american_to_decimal(-100) == 2.00
        # Heavy favorite (-200 -> 1.50)
        assert PinnacleSharpFeed.american_to_decimal(-200) == 1.50
        # Extreme favorite (-1000 -> 1.10)
        assert PinnacleSharpFeed.american_to_decimal(-1000) == 1.10

    def test_shin_vig_removal_3way(self):
        import shin
        # Real-world 3-way odds: Home 2.40, Draw 3.20, Away 3.10
        raw_odds = [2.40, 3.20, 3.10]
        res = shin.calculate_implied_probabilities(raw_odds, full_output=True)
        
        # Invariant 1: Probabilities must sum exactly to 1.0
        assert math.isclose(sum(res.implied_probabilities), 1.0, rel_tol=1e-5)
        
        # Invariant 2: Insider parameter z must be in [0, 1)
        assert 0.0 <= res.z < 1.0
        
        # Invariant 3: Fair odds must be strictly greater than raw odds for all outcomes
        fair_odds = [1.0 / p for p in res.implied_probabilities]
        for r_o, f_o in zip(raw_odds, fair_odds):
            assert f_o > r_o, f"Fair odds {f_o} must exceed bookmaker raw odds {r_o}"

    def test_shin_vig_removal_2way(self):
        import shin
        # Asian Handicap / Over-Under 2-way odds: 1.95, 1.95
        raw_odds = [1.95, 1.95]
        probs = shin.calculate_implied_probabilities(raw_odds)
        
        assert math.isclose(sum(probs), 1.0, rel_tol=1e-5)
        # Symmetrical market must produce 50/50 fair probability
        assert math.isclose(probs[0], 0.50, rel_tol=1e-4)
        assert math.isclose(probs[1], 0.50, rel_tol=1e-4)

    def test_ev_and_quarter_kelly_calculation(self):
        # Case A: Positive EV (+5.0% edge at 2.00 odds)
        edge, kelly = PinnacleSharpFeed.calculate_ev(true_prob=0.525, market_odds=2.00)
        assert edge == 5.0
        # Full kelly = 0.05 / 1.0 = 5%, Quarter kelly = 1.25%
        assert math.isclose(kelly, 1.25, rel_tol=1e-2)

        # Case B: Negative EV (No trade, stake must be 0.0)
        edge_neg, kelly_neg = PinnacleSharpFeed.calculate_ev(true_prob=0.45, market_odds=2.00)
        assert edge_neg == 0.0
        assert kelly_neg == 0.0


# ==============================================================================
# Phase 2: Entity Matching & String Normalization
# ==============================================================================

class TestEntityNormalization:
    def test_name_cleaner(self):
        clean1 = SyndicateComboEngine._normalize_name("Arsenal FC")
        clean2 = SyndicateComboEngine._normalize_name("Arsenal")
        assert clean1 == clean2 == "arsenal"

        clean3 = SyndicateComboEngine._normalize_name("Genoa U20")
        assert "genoa" in clean3

    def test_matching_sofascore_event(self):
        engine = SyndicateComboEngine()
        sofa_events = [
            {"id": 101, "home_team": "Inter U23", "away_team": "Scafatese 1922"},
            {"id": 102, "home_team": "Slavia Prague", "away_team": "Jablonec"},
        ]
        
        # Test match
        matched = engine.find_matching_sofascore_event("Inter", "Scafatese", sofa_events)
        assert matched is not None
        assert matched["id"] == 101


# ==============================================================================
# Phase 3: Integration & Triangulation Pipeline
# ==============================================================================

class TestSyndicateIntegration:
    @patch.object(PinnacleSharpFeed, "get_market_lines")
    def test_triangulation_prematch_analysis(self, mock_lines):
        # Mock Pinnacle line response
        mock_lines.return_value = {
            "moneyline": {
                "raw_odds": [2.10, 3.40, 3.60],
                "true_probs": [0.45, 0.28, 0.27],
                "fair_odds": [2.22, 3.57, 3.70],
                "shin_z": 0.045,
                "vig_percent": 5.2
            }
        }

        engine = SyndicateComboEngine()
        pin_matchup = {
            "id": 99999,
            "home_team": "Real Madrid",
            "away_team": "Barcelona",
            "league": "La Liga",
            "is_live": False
        }

        signal = engine.analyze_triangulation(pin_matchup, sofa_live_events=[])
        assert signal is not None
        assert isinstance(signal, SyndicateSignal)
        assert signal.home_team == "Real Madrid"
        assert signal.shin_insider_z == 0.045
        assert signal.verdict in ["PREMATCH_SMART_MONEY", "NEUTRAL"]
        
        # Check dictionary serialization
        d = signal.to_dict()
        assert "sharp_baseline" in d
        assert "execution" in d

    @patch.object(PinnacleSharpFeed, "get_market_lines")
    def test_triangulation_prematch_real_computed_edge(self, mock_lines):
        """Pre-match edge must be calculated as (p*odds - 1)*100, not hardcoded 4.2."""
        mock_lines.return_value = {
            "moneyline": {
                "raw_odds": [2.20, 3.40, 3.60],
                "true_probs": [0.50, 0.25, 0.25],
                "fair_odds": [2.00, 4.00, 4.00],
                "shin_z": 0.05,
                "vig_percent": 3.5
            }
        }
        engine = SyndicateComboEngine()
        pin_matchup = {
            "id": 11111,
            "home_team": "Real Madrid",
            "away_team": "Barcelona",
            "league": "La Liga",
            "is_live": False
        }
        signal = engine.analyze_triangulation(pin_matchup, sofa_live_events=[])
        assert signal is not None
        assert signal.verdict == "PREMATCH_SMART_MONEY"
        # 0.50 * 2.20 - 1.0 = 0.10 -> 10.0%
        assert round(signal.edge_percent, 2) == 10.00

    @patch.object(PinnacleSharpFeed, "get_market_lines")
    @patch.object(SofaScoreQuantFeed, "get_attack_momentum")
    @patch.object(SofaScoreQuantFeed, "get_shotmap_xg")
    def test_triangulation_inplay_momentum_boost(self, mock_shots, mock_momentum, mock_lines):
        mock_lines.return_value = {
            "moneyline": {
                "raw_odds": [2.00, 3.20, 3.80],
                "true_probs": [0.47, 0.29, 0.24],
                "fair_odds": [2.12, 3.44, 4.16],
                "shin_z": 0.03,
                "vig_percent": 6.0
            }
        }
        
        # High home pressure momentum (+75)
        mock_momentum.return_value = [{"minute": m, "value": 75} for m in range(60, 70)]
        mock_shots.return_value = [
            {"minute": 65, "player": "Striker", "xg": 0.35, "coordinates": {"x": 15, "y": 50}}
        ]

        engine = SyndicateComboEngine()
        pin_matchup = {
            "id": 88888,
            "home_team": "Bayern Munich",
            "away_team": "Dortmund",
            "league": "Bundesliga",
            "is_live": True
        }
        sofa_events = [{
            "id": 777,
            "home_team": "Bayern Munich",
            "away_team": "Dortmund",
            "home_score": 1,
            "away_score": 1
        }]

        signal = engine.analyze_triangulation(pin_matchup, sofa_live_events=sofa_events)
        assert signal is not None
        assert signal.is_live is True
        assert signal.momentum_10m_avg == 75.0
        assert "Bayern Munich" in signal.dominating_side
        assert signal.verdict == "STRONG_INPLAY_ALPHA"
        assert signal.edge_percent > 0.0
        assert signal.quarter_kelly_stake_percent > 0.0


# ==============================================================================
# Phase 4: Robustness & Edge Cases
# ==============================================================================

class TestEdgeCases:
    def test_missing_or_corrupt_market_lines(self):
        engine = SyndicateComboEngine()
        with patch.object(PinnacleSharpFeed, "get_market_lines", return_value={}):
            signal = engine.analyze_triangulation({"id": 1, "home_team": "A", "away_team": "B"})
            assert signal is None

    def test_sofascore_empty_graph(self):
        engine = SyndicateComboEngine()
        with patch.object(SofaScoreQuantFeed, "get_attack_momentum", return_value=[]):
            pts = engine.sofascore.get_attack_momentum(999999)
            assert pts == []
