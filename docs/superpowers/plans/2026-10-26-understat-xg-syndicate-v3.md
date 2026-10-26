# Understat xG Syndicate Pipeline & Multileague v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform raw Understat xG / PPDA data into a complete quantitative edge pipeline: historical big data ingestion across top 5 European leagues, feature engineering for `multileague_v3` model, Monte Carlo scoreline simulation, and institutional dashboard tactical visuals.

**Architecture:**
- **Data Ingestion:** Ingest historical seasons (2021-2024) across top 5 European leagues into `understat_matches`.
- **Feature Pipeline & Training:** Build `fbpredict/xg_features.py` to link xG / PPDA rolling features with match odds; train and freeze `multileague_v3.pkl`.
- **Monte Carlo Engine:** Build `fbpredict/xg_monte_carlo.py` to simulate 10,000 Dixon-Coles Poisson runs per fixture.
- **Frontend Visuals:** Enhance Terminal dashboard (port 8050) with xG Tactical Cards, Dominance Meters, and Luck/Mean Reversion badges.

**Tech Stack:** Python 3.11, SQLite3, pandas, numpy, scikit-learn, xgboost, Chart.js, CSS Design Tokens.

---

### Task 1: Historical Big Data Ingestion (Top 5 European Leagues 2021-2024)
- [x] Extend `scripts/fetch_understat_xg.py` with multi-season backfill support (`--seasons 2021 2022 2023 2024`)
- [x] Run backfill for top 5 European leagues (EPL, La_liga, Bundesliga, Serie_A, Ligue_1)
- [x] Verify database has thousands of matched xG records in `data/football.db`

### Task 2: xG Feature Engineering & Dataset Linking
- [x] Write failing test in `tests/test_xg_features.py` for rolling xG differentials, PPDA ratios, and luck overperformance
- [x] Run test to confirm failure (Red)
- [x] Implement `fbpredict/xg_features.py` with fuzzy team name normalization and rolling metrics calculation
- [x] Run test to confirm passing (Green)

### Task 3: Train and Benchmark `multileague_v3` Model
- [x] Write training script `scripts/train_multileague_v3.py` incorporating xG features into XGBoost
- [x] Run walk-forward benchmark (80% train, 20% test) comparing v2 vs v3 ROC-AUC and Brier scores
- [x] Freeze trained model to `models/multileague_v3.pkl` with metadata sidecar

### Task 4: xG Monte Carlo Simulation Engine
- [x] Write failing test in `tests/test_xg_monte_carlo.py` testing Poisson match score simulation (10,000 iterations)
- [x] Run test to confirm failure (Red)
- [x] Implement `fbpredict/xg_monte_carlo.py` calculating fair Asian Handicap and Totals probabilities
- [x] Run test to confirm passing (Green)

### Task 5: Dashboard Tactical Visuals & API Integration
- [x] Wire xG metrics and Monte Carlo simulation into `scripts/run_dashboard.py` `/api/dashboard`
- [x] Update `web/index.html`, `web/style.css`, and `web/app.js` with Tactical Matchup Card (Dominance Meter, PPDA Style, Luck Warning)
- [x] Restart dashboard daemon on port 8050 and verify API responses

### Task 6: Full Verification & QA
- [x] Run full test suite across Football and Basketball (`pytest tests/`)
- [x] Confirm 0 regressions and 100% test pass
