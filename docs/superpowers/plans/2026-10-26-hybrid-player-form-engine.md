# Hybrid Player Form & Positional Impact Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade Hybrid Player Form Engine combining Understat offensive data (xG, xA, Key Passes) and SofaScore ratings to dynamically calculate position-specific form factors and starter shock penalties for the 60-minute lineup engine.

**Architecture:**
- SQLite Table `player_match_stats` stores match-by-match player performance.
- `UnderstatPlayerProvider` extracts per-match xG, xA, key passes, minutes, and position.
- `PlayerFormAnalyzer` calculates rolling 5-game vs season positional KPIs and Starter Shock Factor.
- `LineupEngine` consumes dynamic player form factors to compute exact Poisson lambda penalties.

**Tech Stack:** Python 3.11, SQLite3, Requests/Urllib, Pytest, NumPy, Scikit-learn.

**Spec:** Approach 2 approved during brainstorming.

## Global Constraints
- All probabilities must remain strictly bounded in $[0.0, 1.0]$.
- Dynamic Form Factor must be clamped within $[0.60, 1.40]$ to prevent sample noise explosions.
- Zero data leakage: rolling 5 matches must only include matches strictly before the fixture kickoff time.

---

### Task 1: Database Schema & Migration for Player Match Stats
**Files:**
- Modify: `fbpredict/db.py`
- Test: `tests/test_db_player_stats.py`

- [ ] Write failing test `tests/test_db_player_stats.py` checking `init_player_stats_table` creates table and indexes.
- [ ] Implement `init_player_stats_table(con)` in `fbpredict/db.py`.
- [ ] Run `pytest tests/test_db_player_stats.py` and verify PASS.
- [ ] Commit.

### Task 2: Understat Player Match Scraper Provider
**Files:**
- Create: `fbpredict/providers/understat_player.py`
- Test: `tests/test_understat_player.py`

- [ ] Write failing test `tests/test_understat_player.py` with mock Understat player JSON payloads.
- [ ] Implement `parse_understat_player_history(raw_json)` and `fetch_player_stats(player_id)`.
- [ ] Implement batch ingestion into `player_match_stats`.
- [ ] Run `pytest tests/test_understat_player.py` and verify PASS.
- [ ] Commit.

### Task 3: Dynamic Positional Form & Starter Shock Calculator
**Files:**
- Create: `fbpredict/player_form.py`
- Test: `tests/test_player_form.py`

- [ ] Write failing test `tests/test_player_form.py` testing FW ($xG+xA$), MF (Key Passes), DF/GK (Ratings), and Starter Ratio ($0.0 \le S \le 1.0$).
- [ ] Implement `compute_positional_form_factor(player_stats, position)`.
- [ ] Implement `calculate_starter_shock_factor(starts_last_5)`.
- [ ] Implement `get_player_dynamic_penalty(player_name, team, match_date, con)`.
- [ ] Run `pytest tests/test_player_form.py` and verify PASS.
- [ ] Commit.

### Task 4: Integration with Lineup Engine & Monte Carlo
**Files:**
- Modify: `fbpredict/lineup_engine.py`
- Test: `tests/test_lineup_engine.py`

- [ ] Update `calculate_lineup_lambdas` in `fbpredict/lineup_engine.py` to accept dynamic form factors when database connection is provided.
- [ ] Verify star players in a slump receive reduced penalties, while in-form players receive amplified penalties.
- [ ] Run full test suite `pytest tests/test_lineup_engine.py` and verify PASS.
- [ ] Commit.
