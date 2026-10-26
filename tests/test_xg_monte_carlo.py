"""Unit tests for xG Monte Carlo Bivariate Poisson Simulation Engine.
Author: cook45
"""

import pytest
from fbpredict.xg_monte_carlo import (
    simulate_match_scorelines,
    evaluate_fixture_monte_carlo,
    MonteCarloSimResult,
)


def test_simulate_match_scorelines():
    # Symmetric equal match (1.4 vs 1.4)
    scores = simulate_match_scorelines(lambda_home=1.4, lambda_away=1.4, n_sims=5000, seed=42)
    assert len(scores) == 5000
    home_wins = sum(1 for h, a in scores if h > a)
    away_wins = sum(1 for h, a in scores if a > h)
    draws = sum(1 for h, a in scores if h == a)

    # Home and away win percentages should be very close (~35-40% each)
    assert abs(home_wins - away_wins) < 300
    assert draws > 1000


def test_evaluate_fixture_monte_carlo():
    # Strong home favorite: Arsenal (xG 2.4) vs Ipswich (xG 0.5)
    res = evaluate_fixture_monte_carlo(
        lambda_home=2.4,
        lambda_away=0.5,
        ah_line=-1.5,
        ou_line=2.5,
        n_sims=5000,
        seed=42,
    )
    assert isinstance(res, MonteCarloSimResult)
    assert res.home_win_prob > 0.70
    assert res.away_win_prob < 0.15
    assert res.fair_ah_fav_odds < 2.50
    assert len(res.top_scorelines) == 5
    # Most likely scoreline should be something like 2-0 or 3-0
    top_score = res.top_scorelines[0]["score"]
    assert top_score in ["2-0", "3-0", "2-1", "3-1"]
