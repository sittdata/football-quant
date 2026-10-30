"""Monte Carlo Dixon-Coles Bivariate Poisson Scoreline Simulation Engine.
Author: cook45

Simulates 10,000 matches using expected goals (xG) parameters with Dixon-Coles
low-score correlation adjustments (tau function for 0-0, 1-0, 0-1, 1-1).
Produces exact probability distributions for:
- 1X2 Match Outcome
- Asian Handicap lines (Favorite cover vs Underdog cover)
- Goal Total lines (Over vs Under)
- Most likely scorelines Matrix
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np


@dataclass
class MonteCarloSimResult:
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    ah_fav_cover_prob: float
    ou_over_prob: float
    fair_ah_fav_odds: float
    fair_ah_dog_odds: float
    fair_over_odds: float
    fair_under_odds: float
    expected_home_goals: float
    expected_away_goals: float
    top_scorelines: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "home_win_prob": round(self.home_win_prob, 4),
            "draw_prob": round(self.draw_prob, 4),
            "away_win_prob": round(self.away_win_prob, 4),
            "ah_fav_cover_prob": round(self.ah_fav_cover_prob, 4),
            "ou_over_prob": round(self.ou_over_prob, 4),
            "fair_ah_fav_odds": round(self.fair_ah_fav_odds, 2),
            "fair_ah_dog_odds": round(self.fair_ah_dog_odds, 2),
            "fair_over_odds": round(self.fair_over_odds, 2),
            "fair_under_odds": round(self.fair_under_odds, 2),
            "expected_home_goals": round(self.expected_home_goals, 2),
            "expected_away_goals": round(self.expected_away_goals, 2),
            "top_scorelines": self.top_scorelines,
        }


def _tau(x: int, y: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    """Dixon-Coles correlation adjustment factor for low-scoring states."""
    if x == 0 and y == 0:
        return 1.0 - lambda_h * lambda_a * rho
    elif x == 0 and y == 1:
        return 1.0 + lambda_h * rho
    elif x == 1 and y == 0:
        return 1.0 + lambda_a * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def simulate_match_scorelines(
    lambda_home: float,
    lambda_away: float,
    n_sims: int = 10000,
    seed: int = 42,
    rho: float = -0.06,
    max_goals: int = 10,
) -> List[Tuple[int, int]]:
    """Simulate match scorelines using Dixon-Coles bivariate Poisson distribution."""
    # Precompute probability matrix up to max_goals
    rng = np.random.default_rng(seed)

    lh = max(0.1, float(lambda_home))
    la = max(0.1, float(lambda_away))

    matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=np.float64)
    for x in range(max_goals + 1):
        p_x = (math.exp(-lh) * (lh**x)) / math.factorial(x)
        for y in range(max_goals + 1):
            p_y = (math.exp(-la) * (la**y)) / math.factorial(y)
            t = _tau(x, y, lh, la, rho)
            matrix[x, y] = max(0.0, p_x * p_y * t)

    # Normalize matrix to sum to 1.0
    matrix /= matrix.sum()

    # Flatten and sample using categorical distribution
    flat_probs = matrix.flatten()
    sampled_indices = rng.choice(len(flat_probs), size=n_sims, p=flat_probs)

    scores = []
    for idx in sampled_indices:
        h = int(idx // (max_goals + 1))
        a = int(idx % (max_goals + 1))
        scores.append((h, a))

    return scores


def _quarter_credit(margin: float) -> float:
    """Fraction of the stake won for a quarter-line margin (push counts as 0.5)."""
    if margin > 0.25:
        return 1.0
    if margin == 0.25:
        return 0.75
    if margin == 0.0:
        return 0.5
    if margin == -0.25:
        return 0.25
    return 0.0


def evaluate_fixture_monte_carlo(
    lambda_home: float,
    lambda_away: float,
    ah_line: float = -0.5,
    ou_line: float = 2.5,
    fav_side: str = "home",
    n_sims: int = 10000,
    seed: int = 42,
) -> MonteCarloSimResult:
    """Evaluate full Asian Handicap and Totals pricing via Monte Carlo simulation."""
    scores = simulate_match_scorelines(lambda_home, lambda_away, n_sims=n_sims, seed=seed)

    home_wins = 0
    draws = 0
    away_wins = 0

    fav_covers = 0.0
    overs = 0.0

    score_counter = collections.Counter(scores)

    side = str(fav_side).strip().lower() if fav_side is not None else ""
    if side not in ("home", "away"):
        raise ValueError(f"fav_side must be 'home' or 'away', got {fav_side!r}")
    is_fav_home = side == "home"

    for h, a in scores:
        diff = h - a
        fav_diff = diff if is_fav_home else -diff
        tot = h + a

        if diff > 0:
            home_wins += 1
        elif diff == 0:
            draws += 1
        else:
            away_wins += 1

        # Asian Handicap payout logic (quarter lines split the stake)
        net_margin = round((fav_diff + ah_line) * 4.0) / 4.0
        fav_covers += _quarter_credit(net_margin)

        # Over / Under
        ou_margin = round((tot - ou_line) * 4.0) / 4.0
        overs += _quarter_credit(ou_margin)

    p_hw = home_wins / n_sims
    p_d = draws / n_sims
    p_aw = away_wins / n_sims

    p_fav = min(0.98, max(0.02, fav_covers / n_sims))
    p_dog = 1.0 - p_fav

    p_over = min(0.98, max(0.02, overs / n_sims))
    p_under = 1.0 - p_over

    fair_fav_odds = round(1.0 / p_fav, 2)
    fair_dog_odds = round(1.0 / p_dog, 2)
    fair_over_odds = round(1.0 / p_over, 2)
    fair_under_odds = round(1.0 / p_under, 2)

    # Top 5 most likely scorelines
    top_scores = []
    for (h, a), count in score_counter.most_common(5):
        top_scores.append(
            {
                "score": f"{h}-{a}",
                "prob": round(count / n_sims, 4),
            }
        )

    return MonteCarloSimResult(
        home_win_prob=round(p_hw, 4),
        draw_prob=round(p_d, 4),
        away_win_prob=round(p_aw, 4),
        ah_fav_cover_prob=round(p_fav, 4),
        ou_over_prob=round(p_over, 4),
        fair_ah_fav_odds=fair_fav_odds,
        fair_ah_dog_odds=fair_dog_odds,
        fair_over_odds=fair_over_odds,
        fair_under_odds=fair_under_odds,
        expected_home_goals=lambda_home,
        expected_away_goals=lambda_away,
        top_scorelines=top_scores,
    )
