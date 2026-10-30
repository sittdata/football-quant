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


def test_evaluate_fixture_monte_carlo_away_favorite():
    # Strong away favorite: Man City (away xG 2.4) at Ipswich (home xG 0.5)
    res = evaluate_fixture_monte_carlo(
        lambda_home=0.5,
        lambda_away=2.4,
        ah_line=-1.5,
        ou_line=2.5,
        fav_side="away",
        n_sims=5000,
        seed=42,
    )
    assert isinstance(res, MonteCarloSimResult)
    assert res.away_win_prob > 0.70
    assert res.home_win_prob < 0.15
    # The away favourite covering -1.5 must have high probability (> 50%) and fair odds < 2.50
    assert res.ah_fav_cover_prob > 0.50
    assert res.fair_ah_fav_odds < 2.50


def test_quarter_credit_mapping():
    """Quarter lines must award exact split-stake credit, not an arbitrary clamp."""
    from fbpredict.xg_monte_carlo import _quarter_credit

    assert _quarter_credit(0.75) == 1.0
    assert _quarter_credit(0.25) == 0.75  # half win
    assert _quarter_credit(0.0) == 0.5  # push
    assert _quarter_credit(-0.25) == 0.25  # half loss
    assert _quarter_credit(-0.75) == 0.0  # full loss


def test_evaluate_fixture_monte_carlo_quarter_lines():
    """On a fixed seed the quarter-line probability must match the credit table exactly."""
    res = evaluate_fixture_monte_carlo(
        lambda_home=1.5,
        lambda_away=1.0,
        ah_line=-0.25,
        ou_line=2.25,
        fav_side="home",
        n_sims=5000,
        seed=42,
    )
    scores = simulate_match_scorelines(lambda_home=1.5, lambda_away=1.0, n_sims=5000, seed=42)
    from fbpredict.xg_monte_carlo import _quarter_credit

    expected_ah = sum(
        _quarter_credit(round(((h - a) + -0.25) * 4.0) / 4.0) for h, a in scores
    ) / len(scores)
    expected_ou = sum(
        _quarter_credit(round(((h + a) - 2.25) * 4.0) / 4.0) for h, a in scores
    ) / len(scores)

    assert res.ah_fav_cover_prob == pytest.approx(min(0.98, max(0.02, expected_ah)), abs=1e-4)
    assert res.ou_over_prob == pytest.approx(min(0.98, max(0.02, expected_ou)), abs=1e-4)


def test_evaluate_fixture_monte_carlo_rejects_unknown_fav_side():
    """An unrecognised favourite side must raise, not silently price a home favourite."""
    with pytest.raises(ValueError):
        evaluate_fixture_monte_carlo(
            lambda_home=1.4,
            lambda_away=1.1,
            fav_side="level",
            n_sims=100,
            seed=42,
        )
