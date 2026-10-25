"""Unit tests for Quantitative Staking and Risk Metrics.
Author: cook45
"""

import unittest
import numpy as np
from fbpredict.staking import calculate_kelly_fraction, calculate_portfolio_metrics


class TestStakingEngine(unittest.TestCase):
    def test_calculate_kelly_fraction_basic(self):
        # 55% win prob at decimal odds 2.0 (Even money, positive edge 5%)
        # b = 1.0, p = 0.55, q = 0.45 -> full kelly = (1*0.55 - 0.45)/1 = 0.10
        # raw_stake = 0.10 * 12.5 = 1.25 units
        stake = calculate_kelly_fraction(prob=0.55, decimal_odds=2.0, fraction=0.25)
        self.assertAlmostEqual(stake, 1.25, places=2)

    def test_calculate_kelly_fraction_clamps(self):
        # Extremely high edge clamped at max_stake (2.5)
        high_stake = calculate_kelly_fraction(prob=0.90, decimal_odds=2.0, fraction=0.25)
        self.assertEqual(high_stake, 2.5)

        # Marginal edge clamped at min_stake (0.5)
        low_stake = calculate_kelly_fraction(prob=0.51, decimal_odds=2.0, fraction=0.25)
        self.assertEqual(low_stake, 0.5)

    def test_calculate_kelly_fraction_zero_or_negative_edge(self):
        # Break-even or negative edge returns 0.0
        no_edge = calculate_kelly_fraction(prob=0.50, decimal_odds=2.0)
        self.assertEqual(no_edge, 0.0)

        neg_edge = calculate_kelly_fraction(prob=0.45, decimal_odds=2.0)
        self.assertEqual(neg_edge, 0.0)

    def test_calculate_kelly_fraction_nan_and_none_protection(self):
        self.assertEqual(calculate_kelly_fraction(None, 2.0), 0.0)
        self.assertEqual(calculate_kelly_fraction(0.55, None), 0.0)
        self.assertEqual(calculate_kelly_fraction(np.nan, 2.0), 0.0)
        self.assertEqual(calculate_kelly_fraction(0.55, np.nan), 0.0)
        self.assertEqual(calculate_kelly_fraction(-0.1, 2.0), 0.0)
        self.assertEqual(calculate_kelly_fraction(1.5, 2.0), 0.0)

    def test_portfolio_metrics_max_drawdown_from_zero_baseline(self):
        # Initial losing streak from 0 capital: -1.0, -1.0, then +2.0
        # Equity: [0.0 (start), -1.0, -2.0, 0.0]
        # Max drawdown from peak (0.0) must be exactly 2.0 units!
        pnl_series = [-1.0, -1.0, 2.0]
        metrics = calculate_portfolio_metrics(pnl_series)
        self.assertEqual(metrics["max_drawdown"], 2.0)
        self.assertEqual(metrics["total_pnl"], 0.0)

    def test_portfolio_metrics_winning_sequence(self):
        pnl_series = [1.0, 1.5, 0.8, -0.5, 1.2]
        metrics = calculate_portfolio_metrics(pnl_series)
        self.assertGreater(metrics["sharpe_ratio"], 0.0)
        self.assertEqual(metrics["max_drawdown"], 0.5)
        self.assertAlmostEqual(metrics["total_pnl"], 4.0, places=2)


if __name__ == "__main__":
    unittest.main()
