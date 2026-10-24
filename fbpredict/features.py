"""Feature engineering.

Implements README.md section 3 plus the extra features it asks for, and adds
the leakage guards the brief insists on:

* every rolling team statistic is ``.shift(1)``-ed, so a match never sees its
  own result,
* rolling windows are computed per team on a strictly time-ordered frame.

The table from ``tip.jpg`` is treated as a **hypothesis**, exactly as the brief
says: ``rule_features`` turns it into numeric columns and nothing more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import labels, price_pool_rules

# --------------------------------------------------------------------------- #
# the rule table (tip.jpg / README 2.1)
# --------------------------------------------------------------------------- #
RULES = pd.DataFrame(
    [
        (-0.25, 2.06, 2.10),
        (-0.50, 1.79, 1.83),
        (-0.75, 1.60, 1.64),
        (-1.00, 1.47, 1.50),
        (-1.25, 1.30, 1.33),
        (-1.50, 1.27, 1.29),
        (-1.75, 1.18, 1.20),
        (-2.00, 1.15, 1.18),
    ],
    columns=["ah", "pool_min", "pool_max"],
)

# The red row in the image: -0.5 that never drifts to -0.75 pairs with a
# price-pool of 1.58-1.59 and "ต่อ+สูง" (favourite + over). Kept separate
# because it is a conditional rule, not a plain row.
SPECIAL_AH_NOT_DRIFTED = (-0.50, 1.58, 1.59)

DRAW_HT_HIGH = 2.35
DRAW_HT_VERY_HIGH = 2.45
DRAW_FT_CENTRE = 4.30
DRAW_FT_TOL = 0.05

#: How many previous matches a team's rolling form averages over. This is the
#: single source of truth: it is also recorded on the trained model so that
#: training and inference can never silently use different windows.
DEFAULT_ROLLING_WINDOW = 5

#: columns the model consumes (order matters for reproducibility)
FEATURE_COLUMNS: tuple[str, ...] = (
    "rule_match",
    "rule_near",
    "rule_dist",
    "special_ah_not_drifted",
    "draw_ht_high",
    "draw_ht_vhigh",
    "draw_ft_match",
    "ah_abs",
    "total_x_ah",
    "ah_drift",
    "ou_drift",
    "ah_not_drifted_to_075",
    "ou_line_open",
    "ou_over_price",
    "ou_under_price",
    "ah_overround",
    "ou_overround",
    "draw_ht_odds",
    "draw_ft_odds",
    "pool_fav_open",
    "home_gf_avg",
    "home_ga_avg",
    "home_total_avg",
    "home_over_rate",
    "away_gf_avg",
    "away_ga_avg",
    "away_total_avg",
    "away_over_rate",
    "goals_avg_diff",
    "expected_total",
)

#: closing prices are treated as known within this many minutes of kick-off.
#: A run that wants the closing-price features must predict at or inside this
#: window; anything earlier is refused rather than silently leaked. The value
#: is an explicit assumption: the provider does not timestamp its closing
#: stage, so the whole quarter-hour before kick-off is treated as "the closing
#: price exists by now".
CLOSING_PRICE_LEAD_MINUTES = 15

#: flags whose value depends on the closing price. In the normal (24h before
#: kick-off) mode these MUST stay NaN - the closing price does not exist yet,
#: and filling them would be look-ahead.
CLOSING_FEATURES: tuple[str, ...] = ("fav_no_drop", "dog_no_drop")
CLOSING_FEATURES_NOVIG: tuple[str, ...] = ("fav_no_drop_nv", "dog_no_drop_nv")

#: rule set 2 (R1-R6) blocks, kept strictly separate from the tip.jpg table
#: (spec section 1). v2 uses the printed prices; v2_novig uses the same
#: thresholds moved into vig-free probability space (spec section 9.4b).
RULE_SET_V2_FEATURES: tuple[str, ...] = tuple(price_pool_rules.rule_feature_names())
RULE_SET_V2_NOVIG_FEATURES: tuple[str, ...] = tuple(
    price_pool_rules.rule_feature_names(suffix="_nv")
)

#: the owner's 2026-10-21 water rule (R7), kept out of both printed sets.
#: The "water" feature set below carries ONLY this rule, which is what the
#: owner asked to train first with every other rule set aside.
RULE_WATER_FEATURES: tuple[str, ...] = tuple(
    price_pool_rules.rule_feature_names(table=price_pool_rules.RULES_WATER)
)

#: Filtered rule set features (R1 and R2 only; cuts R3, R4, R5, R6, R7)
RULE_SET_FILTERED_FEATURES: tuple[str, ...] = tuple(
    price_pool_rules.rule_feature_names(table=price_pool_rules.RULES_FILTERED)
)
RULE_SET_FILTERED_NOVIG_FEATURES: tuple[str, ...] = tuple(
    price_pool_rules.rule_feature_names(table=price_pool_rules.RULES_FILTERED_NOVIG, suffix="_nv")
)

#: Pure raw-price feature set — zero hand-coded rule signals.
#:
#: The model sees only real market numbers and rolling team stats so it can
#: discover price-relationship patterns that the written rules may have missed.
#: Deliberately excludes every rule_match / rule_near / rule_dist column so
#: the learned weights cannot be explained by our own rule hypotheses.
#:
#: Columns included:
#:   - raw opening prices: pool_fav_open, draw_ft_odds, ah_fav_price,
#:     ah_dog_price, ou_over_price, ou_under_price
#:   - market structure: ah_open (the handicap line itself), ou_line_open,
#:     ah_abs (absolute handicap magnitude), ah_overround, ou_overround
#:   - line movement: ah_drift, ou_drift (opening -> closing shift; NaN when
#:     the provider does not supply a closing line)
#:   - rolling team form (5-match, shift-1, no look-ahead):
#:     home/away GF avg, GA avg, total avg, over-rate
#:   - derived interaction: total_x_ah (ou_line * |ah|), goals_avg_diff,
#:     expected_total
RAW_FEATURE_COLUMNS: tuple[str, ...] = (
    # --- raw market prices (opening) ---
    "pool_fav_open",       # 1X2 price on the favourite side (Pinnacle pool)
    "draw_ft_odds",        # 1X2 draw price (full-time)
    "ah_fav_price",        # Asian-Handicap price on the favourite
    "ah_dog_price",        # Asian-Handicap price on the underdog
    "ou_over_price",       # Over price
    "ou_under_price",      # Under price
    # --- market structure ---
    "ah_open",             # handicap line (signed; favourite gives goals)
    "ou_line_open",        # total-goals line
    "ah_abs",              # |ah_open| — magnitude without direction
    "ah_overround",        # bookmaker margin on the AH market
    "ou_overround",        # bookmaker margin on the OU market
    # --- line movement (NaN when provider has no closing line) ---
    "ah_drift",            # ah_close - ah_open
    "ou_drift",            # ou_line_close - ou_line_open
    # --- rolling team form (shift-1, no look-ahead) ---
    "home_gf_avg",
    "home_ga_avg",
    "home_total_avg",
    "home_over_rate",
    "away_gf_avg",
    "away_ga_avg",
    "away_total_avg",
    "away_over_rate",
    # --- derived interactions ---
    "goals_avg_diff",      # home_gf_avg - away_gf_avg
    "expected_total",      # (home_gf + away_ga + away_gf + home_ga) / 2
    "total_x_ah",          # ou_line_open * |ah_open|  (scale interaction)
)

#: the selectable feature sets.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "v1": FEATURE_COLUMNS,
    "v2": FEATURE_COLUMNS + RULE_SET_V2_FEATURES + CLOSING_FEATURES,
    "v2_novig": FEATURE_COLUMNS + RULE_SET_V2_NOVIG_FEATURES + CLOSING_FEATURES_NOVIG,
    "water": RULE_WATER_FEATURES,
    "filtered": FEATURE_COLUMNS + RULE_SET_FILTERED_FEATURES + CLOSING_FEATURES,
    "filtered_novig": FEATURE_COLUMNS + RULE_SET_FILTERED_NOVIG_FEATURES + CLOSING_FEATURES_NOVIG,
    # Pure ML: no rule features at all — model learns from raw prices only.
    # Use --model xgboost with raw_xgb for non-linear pattern discovery.
    "raw": RAW_FEATURE_COLUMNS,
    "raw_xgb": RAW_FEATURE_COLUMNS,
    # Raw prices + filtered R1/R2 rules (vig-free) — best of both worlds.
    "combined": RAW_FEATURE_COLUMNS + RULE_SET_FILTERED_NOVIG_FEATURES + CLOSING_FEATURES_NOVIG,
}

THSCORE_FEATURES: tuple[str, ...] = (
    "home_ht_gf_avg",
    "home_ht_ga_avg",
    "home_ht_total_avg",
    "home_ht_over_rate",
    "away_ht_gf_avg",
    "away_ht_ga_avg",
    "away_ht_total_avg",
    "away_ht_over_rate",
    "ht_goals_avg_diff",
    "expected_ht_total",
    "ht_to_ft_ratio",
    "home_corner_avg",
    "away_corner_avg",
    "expected_corners",
    "corner_diff",
)

FEATURE_SETS["thscore_v2"] = (
    FEATURE_COLUMNS + RULE_SET_FILTERED_NOVIG_FEATURES + THSCORE_FEATURES + CLOSING_FEATURES_NOVIG
)
FEATURE_SETS["thscore_xgb"] = (
    RAW_FEATURE_COLUMNS + RULE_SET_FILTERED_NOVIG_FEATURES + THSCORE_FEATURES + CLOSING_FEATURES_NOVIG
)



def feature_set_columns(feature_set: str) -> tuple[str, ...]:
    """The full column list for a named feature set."""
    try:
        return FEATURE_SETS[feature_set]
    except KeyError:
        raise ValueError(
            f"unknown feature set {feature_set!r}; expected one of {sorted(FEATURE_SETS)}"
        ) from None


def _no_drop_flag(close: object, opened: object) -> np.ndarray:
    """``1`` when the closing value did not drop below the opening value.

    Accepts pandas Series or numpy arrays. NaN when either side is unknown:
    the flag cannot be evaluated, and a 0/1 guess would fabricate information.
    """
    close_arr = np.asarray(pd.to_numeric(close, errors="coerce"), dtype="float64")
    open_arr = np.asarray(pd.to_numeric(opened, errors="coerce"), dtype="float64")
    valid = np.isfinite(close_arr) & np.isfinite(open_arr)
    return np.where(valid, (close_arr >= open_arr).astype("float64"), np.nan)


def _add_closing_features(out: pd.DataFrame, *, novig: bool, fill: bool) -> pd.DataFrame:
    """The "price did not drop" flags (spec section 6.1, R3b's extra condition).

    ``fill=False`` leaves them as an all-NaN column on purpose: that is the
    normal mode. ``fill=True`` is the late-prediction mode, where the closing
    price is assumed to already exist.
    """
    names = CLOSING_FEATURES_NOVIG if novig else CLOSING_FEATURES
    if not fill:
        for name in names:
            out[name] = np.nan
        return out

    if novig:
        fav_open, _, dog_open, _ = labels.fair_three_way_array(
            _column_or_nan(out, "pool_fav_open"),
            _column_or_nan(out, "draw_ft_odds"),
            _column_or_nan(out, "pool_dog_open"),
        )
        fav_close, _, dog_close, _ = labels.fair_three_way_array(
            _column_or_nan(out, "pool_fav_close"),
            _column_or_nan(out, "draw_ft_close"),
            _column_or_nan(out, "pool_dog_close"),
        )
    else:
        fav_open = _column_or_nan(out, "pool_fav_open")
        dog_open = _column_or_nan(out, "pool_dog_open")
        fav_close = _column_or_nan(out, "pool_fav_close")
        dog_close = _column_or_nan(out, "pool_dog_close")

    fav_flag = _no_drop_flag(fav_close, fav_open)
    dog_flag = _no_drop_flag(dog_close, dog_open)
    if novig:
        out[CLOSING_FEATURES_NOVIG[0]] = fav_flag
        out[CLOSING_FEATURES_NOVIG[1]] = dog_flag
    else:
        out[CLOSING_FEATURES[0]] = fav_flag
        out[CLOSING_FEATURES[1]] = dog_flag
    return out


# --------------------------------------------------------------------------- #
# rule features
# --------------------------------------------------------------------------- #
def rule_features(ah_open: float, pool_fav: float, tol: float = 0.03) -> pd.Series:
    """How well does (opening handicap, favourite price) fit the tip table?

    ``rule_match``  exact fit inside the row's band
    ``rule_near``   within ``tol`` of the band (a softer version)
    ``rule_dist``   distance to the band in price units, ``NaN`` if no row
    """
    if ah_open is None or pool_fav is None or pd.isna(ah_open) or pd.isna(pool_fav):
        return pd.Series({"rule_match": 0, "rule_near": 0, "rule_dist": np.nan})

    distances = (RULES["ah"] - float(ah_open)).abs()
    if distances.min() > 1e-6:
        return pd.Series({"rule_match": 0, "rule_near": 0, "rule_dist": np.nan})

    row = RULES.loc[distances.idxmin()]
    lo, hi = float(row["pool_min"]), float(row["pool_max"])
    price = float(pool_fav)
    if lo <= price <= hi:
        dist = 0.0
    else:
        dist = min(abs(price - lo), abs(price - hi))

    return pd.Series(
        {
            "rule_match": int(dist == 0.0),
            "rule_near": int(dist <= tol),
            "rule_dist": dist,
        }
    )


def special_rule_features(ah_open: float, pool_fav: float) -> int:
    """The red row from ``tip.jpg``.

    "-0.5 that has not drifted to -0.75" paired with a price-pool of 1.58-1.59
    and the note ต่อ+สูง (favourite plus over). It is a conditional rule rather
    than a plain row of the table, so it gets its own flag.
    """
    if ah_open is None or pool_fav is None or pd.isna(ah_open) or pd.isna(pool_fav):
        return 0
    line, low, high = SPECIAL_AH_NOT_DRIFTED
    if abs(float(ah_open) - line) > 1e-6:
        return 0
    return int(low <= float(pool_fav) <= high)


# --------------------------------------------------------------------------- #
# favourite orientation
# --------------------------------------------------------------------------- #
def favourite_is_home(df: pd.DataFrame) -> pd.Series:
    """Which side carries the handicap.

    Uses the explicit ``fav_side`` column when present, otherwise falls back to
    the lower 1X2 price. Unknown rows default to ``home`` and are counted by
    :func:`unknown_fav_side_count` so the report can disclose it.
    """
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    if "fav_side" in df.columns:
        result = df["fav_side"].where(df["fav_side"].isin(["home", "away"]), other=pd.NA)

    if "pool_home_open" in df.columns and "pool_away_open" in df.columns:
        inferred = np.where(
            pd.to_numeric(df["pool_home_open"], errors="coerce")
            < pd.to_numeric(df["pool_away_open"], errors="coerce"),
            "home",
            "away",
        )
        result = result.fillna(pd.Series(inferred, index=df.index))

    return result.fillna("home")


def unknown_fav_side_count(df: pd.DataFrame) -> int:
    if "fav_side" not in df.columns:
        return len(df)
    return int((~df["fav_side"].isin(["home", "away"])).sum())


def goal_diff_favourite(df: pd.DataFrame) -> pd.Series:
    """Goals scored by the favourite minus goals by the underdog."""
    diff = pd.to_numeric(df["home_goals"], errors="coerce") - pd.to_numeric(
        df["away_goals"], errors="coerce"
    )
    is_home = favourite_is_home(df) == "home"
    return pd.Series(np.where(is_home, diff, -diff), index=df.index, dtype="float64")


def total_goals(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["home_goals"], errors="coerce") + pd.to_numeric(
        df["away_goals"], errors="coerce"
    )


# --------------------------------------------------------------------------- #
# odds -> fair probabilities (also used as the baseline)
# --------------------------------------------------------------------------- #
def _column_or_nan(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def add_fair_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Add vig-free probabilities for both two-way markets.

    Delegates to :func:`fbpredict.labels.fair_two_way_array`, so the margin
    rule has exactly one implementation and the reported baseline can never
    drift away from the rule used to pick bets.
    """
    out = df.copy()

    fair_fav, fair_dog, margin_ah = labels.fair_two_way_array(
        _column_or_nan(out, "ah_fav_price"), _column_or_nan(out, "ah_dog_price")
    )
    out["fair_fav_ah"] = fair_fav
    out["fair_dog_ah"] = fair_dog
    out["ah_overround"] = margin_ah

    fair_over, fair_under, margin_ou = labels.fair_two_way_array(
        _column_or_nan(out, "ou_over_price"), _column_or_nan(out, "ou_under_price")
    )
    out["fair_over"] = fair_over
    out["fair_under"] = fair_under
    out["ou_overround"] = margin_ou

    return out


# --------------------------------------------------------------------------- #
# rolling team form - shift(1), never the current match
# --------------------------------------------------------------------------- #
ROLLING_SOURCE_COLUMNS = ("gf", "ga", "total", "went_over")


def add_team_rolling(
    df: pd.DataFrame, window: int = DEFAULT_ROLLING_WINDOW, min_periods: int = 1
) -> pd.DataFrame:
    """Add per-team rolling averages of the *previous* matches only."""
    if df.empty:
        for side in ("home", "away"):
            for suffix in ("gf_avg", "ga_avg", "total_avg", "over_rate"):
                df[f"{side}_{suffix}"] = np.nan
        df["goals_avg_diff"] = np.nan
        df["expected_total"] = np.nan
        return df

    work = df.reset_index(drop=True).copy()
    work["_row"] = work.index

    has_ht = "ht_home" in work.columns and "ht_away" in work.columns
    has_corner = "home_corner" in work.columns and "away_corner" in work.columns

    home_view = work.assign(
        team=work["home_team"],
        gf=pd.to_numeric(work["home_goals"], errors="coerce"),
        ga=pd.to_numeric(work["away_goals"], errors="coerce"),
        side="home",
    )
    if has_ht:
        home_view["ht_gf"] = pd.to_numeric(work["ht_home"], errors="coerce")
        home_view["ht_ga"] = pd.to_numeric(work["ht_away"], errors="coerce")
    if has_corner:
        home_view["corner_gf"] = pd.to_numeric(work["home_corner"], errors="coerce")
        home_view["corner_ga"] = pd.to_numeric(work["away_corner"], errors="coerce")

    away_view = work.assign(
        team=work["away_team"],
        gf=pd.to_numeric(work["away_goals"], errors="coerce"),
        ga=pd.to_numeric(work["home_goals"], errors="coerce"),
        side="away",
    )
    if has_ht:
        away_view["ht_gf"] = pd.to_numeric(work["ht_away"], errors="coerce")
        away_view["ht_ga"] = pd.to_numeric(work["ht_home"], errors="coerce")
    if has_corner:
        away_view["corner_gf"] = pd.to_numeric(work["away_corner"], errors="coerce")
        away_view["corner_ga"] = pd.to_numeric(work["home_corner"], errors="coerce")

    long = pd.concat([home_view, away_view], ignore_index=True)
    long["total"] = long["gf"] + long["ga"]
    long["went_over"] = (long["total"] > pd.to_numeric(long["ou_line_open"], errors="coerce")).astype(
        "float64"
    )
    long.loc[long["total"].isna(), "went_over"] = np.nan

    active_rolling_cols = list(ROLLING_SOURCE_COLUMNS)
    nice = ["gf_avg", "ga_avg", "total_avg", "over_rate"]

    if has_ht and "ht_gf" in long:
        long["ht_total"] = long["ht_gf"] + long["ht_ga"]
        long["ht_went_over"] = (long["ht_total"] > 0.5).astype("float64")
        long.loc[long["ht_total"].isna(), "ht_went_over"] = np.nan
        active_rolling_cols.extend(["ht_gf", "ht_ga", "ht_total", "ht_went_over"])
        nice.extend(["ht_gf_avg", "ht_ga_avg", "ht_total_avg", "ht_over_rate"])

    if has_corner and "corner_gf" in long:
        long["corner_total"] = long["corner_gf"] + long["corner_ga"]
        active_rolling_cols.extend(["corner_gf", "corner_ga", "corner_total"])
        nice.extend(["corner_for_avg", "corner_against_avg", "corner_avg"])

    long = long.sort_values(["team", "kickoff_at", "_row"], kind="mergesort")
    grouped = long.groupby("team", sort=False)
    for column in active_rolling_cols:
        long[f"{column}_roll"] = grouped[column].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).mean()
        )

    stat_cols = [f"{c}_roll" for c in active_rolling_cols]

    home_stats = long[long["side"] == "home"].set_index("_row")[stat_cols]
    home_stats.columns = [f"home_{n}" for n in nice]
    away_stats = long[long["side"] == "away"].set_index("_row")[stat_cols]
    away_stats.columns = [f"away_{n}" for n in nice]

    out = work.join(home_stats, on="_row").join(away_stats, on="_row")
    out["goals_avg_diff"] = out["home_gf_avg"] - out["away_gf_avg"]
    out["expected_total"] = (
        out["home_gf_avg"] + out["away_ga_avg"] + out["away_gf_avg"] + out["home_ga_avg"]
    ) / 2.0

    if has_ht and "home_ht_gf_avg" in out.columns:
        out["ht_goals_avg_diff"] = out["home_ht_gf_avg"] - out["away_ht_gf_avg"]
        out["expected_ht_total"] = (
            out["home_ht_gf_avg"] + out["away_ht_ga_avg"] + out["away_ht_gf_avg"] + out["home_ht_ga_avg"]
        ) / 2.0
        out["ht_to_ft_ratio"] = out["expected_ht_total"] / (out["expected_total"] + 1e-4)
    else:
        out["home_ht_gf_avg"] = np.nan
        out["home_ht_ga_avg"] = np.nan
        out["home_ht_total_avg"] = np.nan
        out["home_ht_over_rate"] = np.nan
        out["away_ht_gf_avg"] = np.nan
        out["away_ht_ga_avg"] = np.nan
        out["away_ht_total_avg"] = np.nan
        out["away_ht_over_rate"] = np.nan
        out["ht_goals_avg_diff"] = np.nan
        out["expected_ht_total"] = np.nan
        out["ht_to_ft_ratio"] = np.nan

    if has_corner and "home_corner_avg" in out.columns:
        out["corner_diff"] = out["home_corner_avg"] - out["away_corner_avg"]
        out["expected_corners"] = (out["home_corner_avg"] + out["away_corner_avg"]) / 2.0
    else:
        out["home_corner_avg"] = np.nan
        out["away_corner_avg"] = np.nan
        out["expected_corners"] = np.nan
        out["corner_diff"] = np.nan

    return out.drop(columns=["_row"])


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add unit outcomes and binary targets for both markets."""
    out = df.copy()
    if out.empty:
        for col in ("goal_diff_fav", "total_goals", "ah_unit", "ou_unit_over", "ah_label", "ou_label"):
            out[col] = pd.Series(dtype="float64")
        return out

    diff = goal_diff_favourite(out)
    total = total_goals(out)
    out["goal_diff_fav"] = diff
    out["total_goals"] = total

    have_result = diff.notna() & total.notna() & pd.to_numeric(out["ah_open"], errors="coerce").notna()

    out["ah_unit"] = np.where(
        have_result,
        [
            labels.ah_unit(d, a) if (pd.notna(d) and pd.notna(a)) else np.nan
            for d, a in zip(diff, pd.to_numeric(out["ah_open"], errors="coerce"))
        ],
        np.nan,
    )

    line = pd.to_numeric(out["ou_line_open"], errors="coerce")
    have_ou = total.notna() & line.notna()
    out["ou_unit_over"] = np.where(
        have_ou,
        [
            labels.ou_unit(t, l, "over") if (pd.notna(t) and pd.notna(l)) else np.nan
            for t, l in zip(total, line)
        ],
        np.nan,
    )

    out["ah_label"] = pd.Series(
        [
            labels.ah_binary(d, a) if (pd.notna(d) and pd.notna(a)) else np.nan
            for d, a in zip(diff, pd.to_numeric(out["ah_open"], errors="coerce"))
        ],
        index=out.index,
        dtype="float64",
    )
    out["ou_label"] = pd.Series(
        [
            labels.ou_binary(t, l, "over") if (pd.notna(t) and pd.notna(l)) else np.nan
            for t, l in zip(total, line)
        ],
        index=out.index,
        dtype="float64",
    )

    return out


# --------------------------------------------------------------------------- #
# the full pipeline
# --------------------------------------------------------------------------- #
def build_features(
    df: pd.DataFrame,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    tol: float = 0.03,
    feature_set: str = "v1",
    use_closing_features: bool = False,
) -> pd.DataFrame:
    """Everything the model needs, in one call.

    ``feature_set`` selects the column block: ``v1`` is the original 30,
    ``v2`` adds the printed R1-R6 rule features and the closing-price flags,
    ``v2_novig`` adds instead the no-vig converted blocks. ``use_closing_features``
    fills the closing-price flags and must only be set by callers predicting
    inside the closing window (:data:`CLOSING_PRICE_LEAD_MINUTES`).
    """
    expected = feature_set_columns(feature_set)  # validates the name early
    if df.empty:
        out = add_labels(add_team_rolling(df))
        for column in expected:
            if column not in out.columns:
                out[column] = np.nan
        return out

    out = df.copy()
    if "kickoff_at" in out.columns and not pd.api.types.is_datetime64_any_dtype(out["kickoff_at"]):
        # ISO-8601 strings sort correctly, but the backtest does arithmetic on
        # kick-off, so normalise once here instead of at every call site.
        out["kickoff_at"] = pd.to_datetime(out["kickoff_at"], utc=True, format="ISO8601")
    out = out.sort_values("kickoff_at", kind="mergesort").reset_index(drop=True)

    rule_block = out.apply(
        lambda row: rule_features(row.get("ah_open"), row.get("pool_fav_open"), tol=tol),
        axis=1,
    )
    out = pd.concat([out, rule_block], axis=1)

    empty = pd.Series([None] * len(out), index=out.index)
    out["special_ah_not_drifted"] = [
        special_rule_features(ah, pool)
        for ah, pool in zip(
            out["ah_open"] if "ah_open" in out.columns else empty,
            out["pool_fav_open"] if "pool_fav_open" in out.columns else empty,
        )
    ]

    out["draw_ht_high"] = (pd.to_numeric(out["draw_ht_odds"], errors="coerce") >= DRAW_HT_HIGH).astype(int)
    out["draw_ht_vhigh"] = (pd.to_numeric(out["draw_ht_odds"], errors="coerce") >= DRAW_HT_VERY_HIGH).astype(int)
    draw_ft = pd.to_numeric(out["draw_ft_odds"], errors="coerce")
    out["draw_ft_match"] = draw_ft.between(
        DRAW_FT_CENTRE - DRAW_FT_TOL, DRAW_FT_CENTRE + DRAW_FT_TOL, inclusive="both"
    ).astype(int)

    ah = pd.to_numeric(out["ah_open"], errors="coerce")
    out["ah_abs"] = ah.abs()
    out["total_x_ah"] = pd.to_numeric(out["ou_line_open"], errors="coerce") * out["ah_abs"]

    # Opening -> closing movement. Only 5DollarFootballAPI supplies a closing
    # line so far; where it is missing these stay NaN and the model imputes.
    ah_close = _column_or_nan(out, "ah_close")
    ou_open = pd.to_numeric(out["ou_line_open"], errors="coerce")
    ou_close = _column_or_nan(out, "ou_line_close")
    out["ah_drift"] = ah_close - ah
    out["ou_drift"] = ou_close - ou_open

    # The red row of tip.jpg: "-0.5 that never drifted to -0.75". This is the
    # real version of the rule, using the closing line; special_ah_not_drifted
    # below is the price-band proxy for when no closing line exists.
    out["ah_not_drifted_to_075"] = (
        np.isclose(ah.to_numpy(dtype="float64"), -0.5, atol=1e-9)
        & (ah_close.to_numpy(dtype="float64") > -0.75)
    ).astype(int)

    out = add_fair_probs(out)
    out = add_team_rolling(out, window=rolling_window)
    out = add_labels(out)

    if feature_set == "v2":
        out = price_pool_rules.add_rule_features(out)
        out = _add_closing_features(out, novig=False, fill=use_closing_features)
    elif feature_set == "v2_novig":
        out = price_pool_rules.add_rule_features(out, suffix="_nv", novig=True)
        out = _add_closing_features(out, novig=True, fill=use_closing_features)
    elif feature_set == "filtered":
        out = price_pool_rules.add_rule_features(out, table=price_pool_rules.RULES_FILTERED)
        out = _add_closing_features(out, novig=False, fill=use_closing_features)
    elif feature_set in ("filtered_novig", "thscore_v2"):
        out = price_pool_rules.add_rule_features(
            out, table=price_pool_rules.RULES_FILTERED_NOVIG, suffix="_nv", novig=True
        )
        out = _add_closing_features(out, novig=True, fill=use_closing_features)
    elif feature_set == "thscore_xgb":
        out = price_pool_rules.add_rule_features(
            out, table=price_pool_rules.RULES_FILTERED_NOVIG, suffix="_nv", novig=True
        )
        out = _add_closing_features(out, novig=True, fill=use_closing_features)
        if "ah_open" not in out.columns:
            out["ah_open"] = np.nan
    elif feature_set == "water":
        # only the owner's 2026-10-21 rule; every printed rule stays out
        out = price_pool_rules.add_rule_features(out, table=price_pool_rules.RULES_WATER)
    elif feature_set in ("raw", "raw_xgb"):
        # Pure ML: no rule features at all. ah_open is included as a raw
        # number (not binned) so the model can find non-linear relationships
        # between the handicap size and the outcome by itself.
        if "ah_open" not in out.columns:
            out["ah_open"] = np.nan
        # ah_abs and total_x_ah are already computed above; nothing else needed.
    elif feature_set == "combined":
        # raw prices + filtered R1/R2 rules (vig-free) — best of both worlds
        out = price_pool_rules.add_rule_features(
            out, table=price_pool_rules.RULES_FILTERED_NOVIG, suffix="_nv", novig=True
        )
        out = _add_closing_features(out, novig=True, fill=use_closing_features)
        if "ah_open" not in out.columns:
            out["ah_open"] = np.nan
    return out


def missing_feature_report(
    df: pd.DataFrame, columns: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """How much of each feature is actually present (for the report).

    ``columns`` defaults to the v1 block; pass a feature set's columns to
    measure the v2 blocks too.
    """
    columns = tuple(columns) if columns is not None else FEATURE_COLUMNS
    rows = []
    for column in columns:
        if column not in df.columns:
            rows.append({"feature": column, "present": 0, "missing": len(df)})
            continue
        missing = int(pd.isna(df[column]).sum())
        rows.append({"feature": column, "present": len(df) - missing, "missing": missing})
    return pd.DataFrame(rows)
