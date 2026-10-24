"""Synthetic data generator - for exercising the pipeline, nothing else.

**These rows are not evidence.** Every row carries ``is_synthetic = 1`` and a
``provider`` of ``synthetic``, and the report prints a warning when it sees
them. They exist so the code path can be run and debugged before any API key
exists, and so a marker can tell synthetic output apart from real fetches at a
glance.

The generator is internally consistent: goals are drawn from a Poisson model,
and the opening odds are derived from that same model plus a bookmaker margin.
That means a model can genuinely learn something from it - which is exactly why
its results must never be presented as a real backtest.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy.stats import poisson

MAX_GOALS = 9
DEFAULT_MARGIN = 0.05
HANDICAP_LINES = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25]
TOTAL_LINES = [1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]

SYNTHETIC_PROVIDER = "synthetic"
SYNTHETIC_MARKER = "SYNTHETIC-DEMO-DO-NOT-CITE"


def _score_grid(lambda_home: float, lambda_away: float) -> np.ndarray:
    goals = np.arange(MAX_GOALS)
    home = poisson.pmf(goals, lambda_home)
    away = poisson.pmf(goals, lambda_away)
    grid = np.outer(home, away)
    return grid / grid.sum()


def _diff_matrix() -> np.ndarray:
    goals = np.arange(MAX_GOALS)
    return goals[:, None] - goals[None, :]


def _total_matrix() -> np.ndarray:
    goals = np.arange(MAX_GOALS)
    return goals[:, None] + goals[None, :]


def _priced(probability: float, margin: float) -> float:
    """Turn a true probability into a decimal price with the bookmaker margin."""
    fair = min(max(probability, 1e-4), 1 - 1e-4)
    return round(1.0 / (fair * (1.0 + margin)), 2)


def _choose_handicap(grid: np.ndarray, margin: float) -> tuple[float, float, float]:
    diff = _diff_matrix()
    best_line, best_gap, best_prices = -0.25, 9.9, (1.9, 1.9)
    for line in HANDICAP_LINES:
        cover = float(grid[diff + line > 0].sum())
        gap = abs(cover - 0.5)
        if gap < best_gap:
            best_gap = gap
            best_line = line
            best_prices = (_priced(cover, margin), _priced(1.0 - cover, margin))
    return best_line, best_prices[0], best_prices[1]


def _choose_total(grid: np.ndarray, margin: float) -> tuple[float, float, float]:
    total = _total_matrix()
    best_line, best_gap, best_prices = 2.5, 9.9, (1.9, 1.9)
    for line in TOTAL_LINES:
        over = float(grid[total > line].sum())
        gap = abs(over - 0.5)
        if gap < best_gap:
            best_gap = gap
            best_line = line
            best_prices = (_priced(over, margin), _priced(1.0 - over, margin))
    return best_line, best_prices[0], best_prices[1]


def _drift_line(rng: np.random.Generator, line: float, move_chance: float = 0.35) -> float:
    """The closing line: mostly unchanged, sometimes moved a quarter goal.

    Having a closing line in the demo data means the drift features
    (``ah_drift`` / ``ah_not_drifted_to_075``) are actually exercised instead of
    arriving all-NaN.
    """
    roll = rng.random()
    if roll < move_chance:
        return round(line - 0.25, 2)  # moved further toward the favourite
    if roll < move_chance + 0.10:
        return round(line + 0.25, 2)  # drifted back
    return line


def generate(
    *,
    seed: int = 42,
    n_teams: int = 20,
    n_matches: int = 220,
    start: str = "2024-08-01",
    margin: float = DEFAULT_MARGIN,
) -> pd.DataFrame:
    """Return a synthetic match table shaped exactly like the ``matches`` table."""
    rng = np.random.default_rng(seed)

    teams = [f"Synthetic FC {index:02d}" for index in range(1, n_teams + 1)]
    attack = np.clip(rng.normal(1.15, 0.30, n_teams), 0.35, 2.4)
    defence = np.clip(rng.normal(1.15, 0.30, n_teams), 0.35, 2.4)
    base_rate = 1.30
    home_advantage = 0.22

    fixtures: list[tuple[int, int]] = []
    for home in range(n_teams):
        for away in range(n_teams):
            if home != away:
                fixtures.append((home, away))
    rng.shuffle(fixtures)
    fixtures = fixtures[:n_matches]

    start_date = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    rows: list[dict] = []

    for index, (home, away) in enumerate(fixtures):
        lambda_home = base_rate * attack[home] / defence[away] + home_advantage
        lambda_away = base_rate * attack[away] / defence[home]
        grid = _score_grid(lambda_home, lambda_away)

        goals = np.arange(MAX_GOALS)
        home_goals = int(rng.choice(goals, p=grid.sum(axis=1) / grid.sum()))
        away_goals = int(rng.choice(goals, p=grid.sum(axis=0) / grid.sum()))

        ht_lambda_home = lambda_home * 0.45
        ht_lambda_away = lambda_away * 0.45
        ht_home = min(home_goals, int(rng.poisson(ht_lambda_home)))
        ht_away = min(away_goals, int(rng.poisson(ht_lambda_away)))

        p_home = float(grid[np.tril_indices(MAX_GOALS, -1)].sum())
        p_away = float(grid[np.triu_indices(MAX_GOALS, 1)].sum())
        p_draw = float(np.trace(grid))

        odds_home = _priced(p_home, margin)
        odds_away = _priced(p_away, margin)
        odds_draw = _priced(p_draw, margin)

        fav_side = "home" if odds_home < odds_away else "away"
        pool_fav = min(odds_home, odds_away)
        pool_dog = max(odds_home, odds_away)

        # Closing 1X2 prices: a small drift around the opening ones, so the
        # closing-dependent features (fav_no_drop / dog_no_drop) have real
        # values to read in the late-prediction mode. Like every other column
        # here, this is synthetic and must never be cited as evidence.
        pool_fav_close = round(max(1.01, pool_fav + float(rng.normal(0.0, 0.06))), 2)
        pool_dog_close = round(max(1.01, pool_dog + float(rng.normal(0.0, 0.20))), 2)
        draw_ft_close = round(max(1.01, odds_draw + float(rng.normal(0.0, 0.10))), 2)

        ah_line, ah_fav_price, ah_dog_price = _choose_handicap(grid, margin)
        ou_line, ou_over_price, ou_under_price = _choose_total(grid, margin)
        ah_close = _drift_line(rng, ah_line)
        ou_close = _drift_line(rng, ou_line)

        ht_grid = _score_grid(ht_lambda_home, ht_lambda_away)
        ht_draw = float(np.trace(ht_grid))
        draw_ht_odds = _priced(ht_draw, margin)

        kickoff = start_date + timedelta(days=index // 3, hours=(index % 3) * 4)

        rows.append(
            {
                "match_id": f"synthetic:{index:04d}",
                "provider": SYNTHETIC_PROVIDER,
                "league": "Synthetic League",
                "season": start[:4],
                "kickoff_at": kickoff.isoformat(),
                "home_team": teams[home],
                "away_team": teams[away],
                "fav_side": fav_side,
                "ah_open": ah_line,
                "ah_fav_price": ah_fav_price,
                "ah_dog_price": ah_dog_price,
                "pool_fav_open": pool_fav,
                "pool_dog_open": pool_dog,
                "draw_ft_odds": odds_draw,
                "pool_fav_close": pool_fav_close,
                "pool_dog_close": pool_dog_close,
                "draw_ft_close": draw_ft_close,
                "draw_ht_odds": draw_ht_odds,
                "ou_line_open": ou_line,
                "ou_over_price": ou_over_price,
                "ou_under_price": ou_under_price,
                "ah_close": ah_close,
                "ou_line_close": ou_close,
                "ht_home": ht_home,
                "ht_away": ht_away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "odds_source": f"{SYNTHETIC_MARKER} (Poisson + {margin:.0%} margin)",
                "result_source": SYNTHETIC_MARKER,
                "odds_fetched_at": None,
                "result_fetched_at": None,
                "is_synthetic": 1,
            }
        )

    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path) -> str:
    df.to_csv(path, index=False, encoding="utf-8")
    return str(path)
