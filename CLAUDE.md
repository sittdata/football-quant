# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment & Commands

The project uses Python 3.10+ (specifically Python 3.13 on this Windows machine). The bare `python` command may resolve to the Windows Store alias. Always execute Python via `.\run.ps1 <command>` or the full path `C:\Users\sitti\AppData\Local\Programs\Python\Python313\python.exe`.

### Testing
- **Run all tests**: `.\run.ps1 test` or `python -m pytest -q`
- **Run a single test file**: `.\run.ps1 test tests/test_features.py`
- **Run a specific test**: `.\run.ps1 test tests/test_features.py -k test_rule_features`

### Core CLI Commands (`python -m fbpredict <command>`)
- **Academic Data Science 1X2 Experiment**: `.\run.ps1 run-1x2`
  Runs zero-leakage rolling feature extraction and multi-model benchmark evaluation (Logistic Regression, Random Forest, HistGradientBoosting vs Market Odds Baseline) on 1X2 outcomes (Home / Draw / Away), generating `reports/simple_1x2_project_report.md`.
- **Jupyter Notebook (`football_prediction_analysis.ipynb`)**:
  Interactive Data Science notebook that loads 100% real Premier League match data and trains calibrated **XGBoost** models to predict:
  1. Over/Under 2.5 Goal Probabilities (%)
  2. Asian Handicap Favorite vs Underdog Cover Probabilities (%)
  Includes EDA charts, feature importance, reliability calibration curves, and an interactive fixture simulator.
- **Demo pipeline (synthetic data)**: `.\run.ps1 demo`
  Generates `data/synthetic_matches.csv`, populates isolated `data/demo.db`, runs walk-forward backtest, outputs `reports/demo_backtest.md`.
- **View configuration & API status**: `.\run.ps1 show-config`
- **Database init**: `.\run.ps1 init-db` (creates schema in `data/football.db`).
- **Fetch data**:
  - 5DollarFootballAPI (results + opening odds in one call): `.\run.ps1 fetch-5dollar --league <id> --season 2025`
  - Probe 5DollarFootballAPI: `.\run.ps1 probe-5dollar`
  - Results from football-data.org: `.\run.ps1 fetch-results --provider football-data-org --competition PL`
  - Results from API-Football: `.\run.ps1 fetch-results --provider api-football --league EPL`
  - Supplementary Thscore scrape: `.\run.ps1 scrape-thscore --day today --i-understand-robots`
- **Link odds to results**: `.\run.ps1 link --tolerance 24.0` (joins independent odds and score rows by team and kickoff).
- **Audit rules**: `.\run.ps1 audit-rules --filtered --novig`
- **Train & freeze model bundle**: `.\run.ps1 train-model --model xgboost --feature-set filtered_novig` (saves `.pkl` alongside a SHA-256 sidecar).
- **Verify model artifact**: `.\run.ps1 verify-model --model models/<model>.pkl`
- **Walk-forward backtest**: `.\run.ps1 backtest --feature-set filtered_novig --threshold 0.06`
- **Forward predict (paper trading)**: `.\run.ps1 predict-forward --model models/<model>.pkl --save`
- **Settle recorded predictions**: `.\run.ps1 settle --run-id <run_id>`

---

## High-Level Architecture & Pipeline Flow

The repository implements a football betting research and trading pipeline for **Asian Handicap (AH)** and **Over/Under (O/U)** markets, focusing on opening price-pool rules (`tip.jpg`), market edge calculation, and quant syndicate alpha.

### 1. Data Ingestion & Caching (`fbpredict/providers/`, `fbpredict/db.py`)
- Raw payloads are strictly archived in `data/raw/` and stored in SQLite `raw_payload` with `fetched_at` timestamps to provide immutable evidence of external fetching and prevent duplicate quota consumption.
- Supported providers: `five_dollar.py` (preferred single-source for odds + scores), `football_data_org.py`, `api_football.py`, `oddspapi.py`, `thscore.py`, `thscore_feed.py`, `pinnacle_feed.py`, `sofascore_feed.py`, `understat.py`.
- Two database files: `data/football.db` (real data) and `data/demo.db` (synthetic test data only; rows flagged `is_synthetic = 1`).

### 2. Matching & Record Linking (`fbpredict/linking.py`)
- When odds and match scores originate from separate providers (e.g., OddsPapi + football-data.org), `link_results` maps them using team name similarity and kickoff time proximity within a configurable tolerance window.

### 3. Feature Engineering & Rule Audit (`fbpredict/features.py`, `fbpredict/price_pool_rules.py`)
- Features extract opening Asian Handicap, pool odds (1X2 for favourite), over/under lines, and rolling historical performance.
- **Zero Data Leakage Guard**: All rolling statistics use strict `.shift(1)` before window aggregation. Closing features are gated and only accessible within the defined closing window (`CLOSING_PRICE_LEAD_MINUTES = 45`).
- Supported feature sets: `v1` (baseline features), `v2` (raw R1-R6 rule sets), `filtered` (validated key rules), and `filtered_novig` (validated rules evaluated in margin-stripped vig-free space).

### 4. Label Computation & Market Settlement (`fbpredict/labels.py`)
- Calculates exact settlement payouts: half-win (+0.5), push (0.0), half-loss (-0.5), full win (+1.0), and full loss (-1.0) under Asian lines.
- Removes bookmaker margins (multiplicative and Shin's method) to derive true fair odds and probabilities.

### 5. Quantitative Modeling & Security (`fbpredict/model.py`, `fbpredict/quant_model.py`)
- Supports multi-output models (Logistic Regression, XGBoost, HistGradientBoosting) outputting `p_fav_cover` and `p_over`.
- Models utilize probability calibration (`CalibratedClassifierCV`) to suppress uncalibrated probability spikes.
- **Model Integrity Protection**: Saved models (`ModelBundle`) generate a SHA-256 sidecar JSON. Hash verification is executed *prior* to unpickling to prevent arbitrary code execution or tampered model states.

### 6. Validation & Backtesting (`fbpredict/backtest.py`)
- Strict chronological walk-forward validation (default 80% train / 20% test).
- `blind_view` masks `home_goals`, `away_goals`, `ht_home`, and `ht_away` for upcoming test dates to enforce blind prediction.
- Edge rule: Market edge is calculated as $\text{Edge} = P_{\text{model}} - P_{\text{implied\_novig}}$. Matches are skipped if no market exceeds the threshold (`edge_threshold`).

### 7. Advanced Syndicate & Microstructure Engine (`fbpredict/syndicate_engine.py`, `steam_engine.py`, `lineup_engine.py`)
- `steam_engine.py`: Tracks line movement velocity, probability drift, and Reverse Line Movement (RLM) against sharp books.
- `lineup_engine.py`: Monitors starting lineups 60 minutes prior to kickoff, applying quantitative lambda penalties to expected goals for missing Tier 1 players.
- `syndicate_engine.py`: Combines Pinnacle sharp baseline, SofaScore attack momentum/xG, and Quarter-Kelly staking.

### 8. Storage & Database Integrity Constraints (`fbpredict/db.py`)
- Predictions table enforces a SQL check constraint: `CHECK (predicted_at < kickoff_at)` preventing any accidental look-ahead data insertion.
- Tracks paper-trading runs and settles profit/loss once actual scores land.

### 9. Operating Strategy Defaults (`.agents/rules/user_betting_strategy.md`)
- Production model: `models/multileague_v2.pkl`
- Odds feed: Live Asian Bookmakers via Thscore feed
- Target edge: Positive edge $\ge 8\% - 10\%$ (skip unconfident matches)
- Staking: Dynamic flat stake based on daily bankroll divided by qualifying picks $N$.
