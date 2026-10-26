# Product Context: Cook45 Quantitative Football Terminal

## Product Purpose
Institutional-grade Quantitative Football Hedge Fund & Syndicate Trading Terminal. 
Provides statistical edge analysis on top European football leagues using walk-forward validated machine learning models (Dixon-Coles bivariate Poisson, regularized XGBoost, and Shin vig-removal algorithms) combined with Fractional Kelly Criterion capital allocation.

## Primary Users & Roles
- **Syndicate Trader / Operator:** Evaluates high-alpha match opportunities, inspects implied vig-free edges, and monitors real-time market drift.
- **Risk Manager:** Tracks portfolio equity curves, drawdowns, win rates, and Kelly stake exposures.

## Core Workflows
1. **Fixture Scanning:** Fetching upcoming fixtures across European leagues, pricing them against bookmaker lines (AH & OU), and identifying positive expected value (+EV) bets exceeding safety thresholds.
2. **Portfolio Tracking:** Monitoring active open wagers, historical settled bets, and the out-of-sample bankroll growth trajectory.
3. **Automated Settlement:** Ingesting real completed match scores from data feeds and executing rule-based P&L settlement into the SQLite audit ledger.

## Technical Stack & Surface
- **Frontend:** Vanilla HTML5, High-Performance Modern CSS (Design Tokens, Container Queries), Vanilla JS, Chart.js for equity trajectories.
- **Backend:** Python Threading HTTP Server (port 8050), SQLite3 database (`football.db`), pandas / scikit-learn / xgboost inference pipeline.
- **Surface Mode:** `Operate` (High information density, immediate scanability, zero latency, zero decorative fluff).
