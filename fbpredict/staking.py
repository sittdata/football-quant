"""Quantitative Staking & Capital Allocation Engine.
Author: cook45

Implements:
1. Fractional Kelly Criterion (Quarter-Kelly, Half-Kelly) for optimal geometric bankroll growth.
2. Dynamic risk caps (max drawdown protection, minimum and maximum unit clamp).
3. Risk-adjusted metrics (Sharpe ratio, Sortino ratio, Return over Max Drawdown).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_kelly_fraction(
    prob: float,
    decimal_odds: float,
    fraction: float = 0.25,
    min_stake: float = 0.5,
    max_stake: float = 2.5,
) -> float:
    """Calculate Fractional Kelly stake in units.

    Formula:
        b = decimal_odds - 1.0
        p = prob
        q = 1.0 - p
        full_kelly = (b * p - q) / b
        stake = fraction * full_kelly * 10.0 (normalized to 1.0 base unit)

    Clamped between min_stake and max_stake when edge > 0.
    """
    try:
        if prob is None or decimal_odds is None:
            return 0.0
        p = float(prob)
        d_odds = float(decimal_odds)
        if not np.isfinite(p) or not np.isfinite(d_odds):
            return 0.0
        if d_odds <= 1.0 or p <= 0.0 or p >= 1.0:
            return 0.0
    except (ValueError, TypeError):
        return 0.0

    b = d_odds - 1.0
    q = 1.0 - p
    full_kelly = (b * p - q) / b


    if full_kelly <= 0.0:
        return 0.0

    # Scale 5-10% full kelly into 1.0 - 2.5 units
    raw_stake = full_kelly * (fraction / 0.25) * 12.5
    clamped_stake = max(min_stake, min(max_stake, raw_stake))
    return round(float(clamped_stake), 2)


def calculate_portfolio_metrics(pnl_series: list[float] | np.ndarray) -> dict[str, float]:
    """Compute financial risk metrics (Sharpe, Sortino, Max Drawdown, Calmar)."""
    pnl = np.array(pnl_series, dtype=float)
    if len(pnl) < 2:
        return {
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_units": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar_ratio": 0.0,
        }

    total_pnl = float(np.sum(pnl))
    mean_return = float(np.mean(pnl))
    std_return = float(np.std(pnl))

    sharpe = float(mean_return / std_return * np.sqrt(len(pnl))) if std_return > 1e-6 else 0.0

    neg_returns = pnl[pnl < 0]
    if len(neg_returns) > 0:
        downside_std = float(np.std(neg_returns))
        sortino = float(mean_return / downside_std * np.sqrt(len(pnl))) if downside_std > 1e-6 else 0.0
    else:
        sortino = 99.0

    equity = np.insert(np.cumsum(pnl), 0, 0.0)
    peaks = np.maximum.accumulate(equity)
    drawdowns = peaks - equity
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    calmar = float(total_pnl / max_dd) if max_dd > 1e-6 else 99.0

    return {
        "total_pnl": round(total_pnl, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_units": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "calmar_ratio": round(calmar, 2),
    }


# Alias for backward and forward compatibility
compute_financial_metrics = calculate_portfolio_metrics

