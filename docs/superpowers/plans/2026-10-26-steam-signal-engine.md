# Market Microstructure & Steam Signal Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement sharp money steam detection and dynamic Kelly stake scaling across Football and Basketball platforms with visual badges on both dashboards.

**Architecture:**
- **Football:** Extend `fbpredict/steam_engine.py` with `compute_synergy_kelly()` and wire into `scripts/run_dashboard.py`.
- **Basketball:** Implement `nbapredict/steam_engine.py` for NBA Spread & Totals velocity tracking and wire into `basketball/scripts/run_dashboard.py`.
- **UI:** Render high-contrast Institutional Steam Badges (`🔥 STEAM: DOG` or `🛡️ RLM: UNDER`) on both dashboards.

---

### Task 1: Football Steam Engine Enhancement & Synergy Kelly Sizing
- [x] Write failing test in `tests/test_steam_engine.py` for `compute_synergy_kelly(base_stake, pick_market, steam_signal)`
- [x] Run `pytest tests/test_steam_engine.py` to confirm failure (Red)
- [x] Implement `compute_synergy_kelly()` in `fbpredict/steam_engine.py` (1.25x for confirmed, 0.5x for conflict, 1.0x neutral, clamped to [0.5, 2.5])
- [x] Run `pytest tests/test_steam_engine.py` to confirm passing (Green)

### Task 2: Basketball NBA Steam Engine Implementation
- [x] Write failing test in `basketball/tests/test_nba_steam.py` testing spread velocity and total line moves
- [x] Run `pytest tests/test_nba_steam.py` in basketball to confirm failure (Red)
- [x] Implement `nbapredict/steam_engine.py` in basketball with `analyze_nba_steam()` and `evaluate_nba_synergy()`
- [x] Run `pytest tests/test_nba_steam.py` to confirm passing (Green)

### Task 3: Dashboard API Integration (Football & Basketball)
- [x] In `football/scripts/run_dashboard.py`, compute steam signal for upcoming recommendations and attach `steam` object to `/api/dashboard`
- [x] In `basketball/scripts/run_dashboard.py`, compute NBA steam signals and attach `steam` object to predictions payload
- [x] Verify both `/api/dashboard` endpoints return valid steam objects

### Task 4: Frontend UI Institutional Steam Badges
- [x] Update `football/web/index.html`, `style.css`, and `app.js` to display steam badges on recommendations table
- [x] Update `basketball/web/index.html`, `style.css`, and `app.js` to display steam badges on actionable picks
- [x] Run `impeccable detect` on both ports (8050 and 8060) to verify 0 anti-patterns

### Task 5: System Verification
- [x] Run full test suite on Football (`pytest tests/`)
- [x] Run full test suite on Basketball (`pytest tests/`)
- [x] Confirm 100% test pass and zero regressions
