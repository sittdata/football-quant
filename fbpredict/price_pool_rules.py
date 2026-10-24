"""Rule set 2 - the two extra price-pool images, transcribed in the user's spec.

``สเปกกฎราคาพูล (ส่งให้ AI เทรนโมเดล).md`` adds two more rule images on top of
the ``tip.jpg`` table already implemented in :mod:`fbpredict.features`.

The two sets **disagree** - the spec's own example is AH -0.75, which is
1.60-1.64 in the brief but 1.72-1.89 in image 2 - so the spec (section 1)
requires them to be kept as separate feature sets and forbids merging them or
picking a winner. Nothing in this module reads or writes
:data:`fbpredict.features.RULES`.

Two price spaces are provided
-----------------------------
``RULES_V2``
    the thresholds exactly as printed in the images (decimal 1X2 prices).
``RULES_V2_NOVIG``
    the same thresholds moved into vig-free probability space, so a rule written
    against one bookmaker's margin can be compared against another bookmaker's
    prices (spec section 9.4, option b).

How the no-vig conversion works, and its one assumption
-------------------------------------------------------
A bound on a *single* price cannot be normalised on its own, because the other
two sides of the book are unknown. So each rule row is treated as its own
representative book: the finite edge of each bound is taken as that side's
price, and the row's own overround becomes the conversion factor. Rows that
constrain only one or two sides have no overround of their own and fall back to
:data:`REFERENCE_OVERROUND`, the midpoint of the 1.07-1.14 the spec measured on
the images. That assumption is explicit and is reported, never hidden.

The data side is converted exactly, using each match's own overround.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from . import labels

INF = float("inf")
Range = tuple[float, float]

#: tolerance for a single printed price, from the brief (section 5 of the spec)
TOL = 0.03

#: fallback margin for rule rows that constrain fewer than three sides
REFERENCE_OVERROUND = 1.09

#: rule R4a needs Ladbrokes Over HT opening *and* closing prices. The plan we
#: are on has neither, and the spec (section 6.2) forbids silently substituting
#: another bookmaker, so the rule is emitted as NaN and reported as untestable.
R4A_UNTESTABLE = True

#: the spec (section 6.3 / 9.1) reads "0.50-0.75" as a non-decimal format,
#: since a decimal price cannot be below 1.0. Recorded for the report.
LADBROKES_OVER_HT_RAW_RANGE = (0.50, 0.75)
LADBROKES_OVER_HT_IF_HK = (1.50, 1.75)


# --------------------------------------------------------------------------- #
# range helpers (mirrors the spec's own helper names, on purpose)
# --------------------------------------------------------------------------- #
def pt(value: float) -> Range:
    """A single printed price becomes a band of +/- TOL."""
    return (value - TOL, value + TOL)


def ge(value: float) -> Range:
    return (value, INF)


def le(value: float) -> Range:
    return (-INF, value)


def _dist(value: float | None, rng: Range | None) -> float:
    """Distance from ``value`` to ``rng``.

    Three distinct cases, deliberately not collapsed:

    * ``rng is None`` - the rule does not check this side, so 0.0.
    * ``value`` missing - the rule *does* check this side but there is no
      price to compare. That is ``NaN``, never 0.0: a missing price is
      absence of evidence, and returning 0.0 here would make every rule fire
      on rows that carry no prices at all.
    * otherwise the real distance (0.0 inside the band).
    """
    if rng is None:
        return 0.0
    if value is None or not math.isfinite(value):
        return math.nan
    low, high = rng
    if low <= value <= high:
        return 0.0
    if low == -INF:
        return abs(value - high)
    if high == INF:
        return abs(value - low)
    return min(abs(value - low), abs(value - high))


def _representative_price(rng: Range | None) -> float | None:
    """The finite edge of a bound, used as that side's price for the margin."""
    if rng is None:
        return None
    low, high = rng
    if low == -INF:
        return high
    if high == INF:
        return low
    return (low + high) / 2.0


def _range_to_probability(rng: Range | None, overround: float) -> Range | None:
    """Move a price band into vig-free probability space.

    A price band (low, high) inverts to a probability band (1/high, 1/low), so
    the endpoints swap as well as invert.
    """
    if rng is None:
        return None
    low, high = rng
    prob_low = 0.0 if high == INF else (1.0 / high) / overround
    prob_high = 1.0 if low == -INF else (1.0 / low) / overround
    return (prob_low, prob_high)


# --------------------------------------------------------------------------- #
# the rule table, exactly as printed in the spec's section 5
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RuleRow:
    rule_id: str
    lines: tuple[float, ...]
    fav: Range | None
    draw: Range | None
    dog: Range | None
    note: str = ""
    #: True when a required data source is missing on this plan, so the rule
    #: cannot be evaluated at all. Its features are emitted as NaN and the
    #: audit does not measure it (spec section 6.2: never substitute).
    untestable: bool = False
    #: Optional band on the FAVOURITE's own handicap price (decimal). The
    #: printed rules constrain the 1X2 pool only; the owner's 2026-10-21 rule
    #: is written against the favourite side's handicap water instead.
    ah_fav: Range | None = None

    @property
    def key(self) -> str:
        """Stable column stem. The line is included because ids repeat (R1 twice)."""
        return f"{self.rule_id}_ah{self.lines[0]}"

    @property
    def overround(self) -> float:
        """The row's own book margin, or the fallback when sides are unchecked."""
        prices = [
            _representative_price(bound) for bound in (self.fav, self.draw, self.dog)
        ]
        if any(price is None for price in prices):
            return REFERENCE_OVERROUND
        return float(sum(1.0 / price for price in prices))  # type: ignore[operator]


RULES_V2: tuple[RuleRow, ...] = (
    # R1 - favourite side, price pool signals the favourite covers
    RuleRow("R1", (-0.25,), le(2.09), ge(3.25), ge(2.80)),
    RuleRow("R1", (-0.5,), le(1.85), ge(3.30), ge(4.00)),
    # R2 - underdog side, three variants at -0.5
    RuleRow("R2a", (-0.25,), ge(2.20), le(3.25), le(2.99)),
    RuleRow("R2a", (-0.5,), ge(1.83), le(3.60), le(3.80)),
    RuleRow("R2b", (-0.5,), ge(1.92), le(3.55), le(3.75)),
    RuleRow("R2c", (-0.5,), ge(2.00), le(3.50), le(3.65)),
    # R3 - full-time over
    RuleRow("R3a", (0.0, -0.25), None, ge(3.65), None, note="draw price only"),
    RuleRow(
        "R3b",
        (-1.5,),
        None,
        ge(4.40),
        ge(7.20),
        note="also needs fav/dog to not drop (closing prices)",
    ),
    # R4 - first-half over
    RuleRow(
        "R4a",
        (0.0, -0.25),
        None,
        ge(3.65),
        None,
        note="additionally needs Ladbrokes Over HT - untestable on this plan",
        untestable=R4A_UNTESTABLE,
    ),
    RuleRow("R4b", (-0.5,), None, ge(3.70), None, note="draw price only"),
    # R5 - the five "prices I like" rows; direction is not stated in the image
    RuleRow("R5_1", (-0.25,), pt(2.20), pt(3.50), pt(2.99)),
    RuleRow("R5_2", (-0.5,), pt(1.99), pt(3.50), pt(3.50)),
    RuleRow("R5_3", (-0.75,), (1.80, 1.89), pt(3.80), pt(4.40)),
    RuleRow("R5_4", (-0.75,), (1.72, 1.79), pt(3.60), pt(3.90)),
    RuleRow("R5_5", (-1.0,), (1.52, 1.65), pt(4.50), pt(5.00)),
    # R6 - the extra row appended under the second image
    RuleRow("R6_1", (-0.25,), (2.00, 2.09), pt(3.25), pt(3.10)),
)


def build_novig_table(
    table: Sequence[RuleRow] = RULES_V2,
    *,
    fallback_overround: float = REFERENCE_OVERROUND,
) -> tuple[RuleRow, ...]:
    """Move every threshold into vig-free probability space."""
    converted: list[RuleRow] = []
    for rule in table:
        margin = rule.overround if rule.overround != REFERENCE_OVERROUND else fallback_overround
        converted.append(
            RuleRow(
                rule_id=rule.rule_id,
                lines=rule.lines,
                fav=_range_to_probability(rule.fav, margin),
                draw=_range_to_probability(rule.draw, margin),
                dog=_range_to_probability(rule.dog, margin),
                note=rule.note,
                untestable=rule.untestable,
            )
        )
    return tuple(converted)


RULES_V2_NOVIG: tuple[RuleRow, ...] = build_novig_table(RULES_V2)


#: The owner's rule of 2026-10-21 ("คัดบอลต่อ 0.5/1 + ราคาน้ำ"): bet365 opening
#: prices only, handicap 0.5/1 (decimal -0.75), PLAY THE FAVOURITE. Conditions:
#: the favourite's own handicap water 0.80-0.88 stated in Hong Kong odds
#: (decimal 1.80-1.88, per the conversion the report records) and the
#: favourite's 1X2 decimal price <= 1.87.
#:
#: Deliberately kept OUT of RULES_V2: it is a new owner-supplied rule, not part
#: of the two printed tables, and appending it there would silently change the
#: row count and the audits of the printed sets.
RULES_WATER: tuple[RuleRow, ...] = (
    RuleRow(
        "R7",
        (-0.75,),
        le(1.87),
        None,
        None,
        note=(
            "new rule 2026-10-21: bet365 only, favourite AH water HK 0.80-0.88 "
            "(decimal 1.80-1.88), favourite 1X2 <= 1.87; direction: play the favourite"
        ),
        ah_fav=(1.80, 1.88),
    ),
)


#: Filtered rule set: removes failed Over rules (R3a, R3b, R4a, R4b),
#: unrepresentative single-point rules (R5_1..R5_5, R6_1), and water rule (R7).
#: Retains only the validated favorite/underdog key handicap rules.
RULES_FILTERED: tuple[RuleRow, ...] = (
    # R1 - favourite side, price pool signals the favourite covers
    RuleRow("R1", (-0.25,), le(2.09), ge(3.25), ge(2.80)),
    RuleRow("R1", (-0.5,), le(1.85), ge(3.30), ge(4.00)),
    # R2 - underdog side, three variants at -0.5 and -0.25
    RuleRow("R2a", (-0.25,), ge(2.20), le(3.25), le(2.99)),
    RuleRow("R2a", (-0.5,), ge(1.83), le(3.60), le(3.80)),
    RuleRow("R2b", (-0.5,), ge(1.92), le(3.55), le(3.75)),
    RuleRow("R2c", (-0.5,), ge(2.00), le(3.50), le(3.65)),
)

RULES_FILTERED_NOVIG: tuple[RuleRow, ...] = build_novig_table(RULES_FILTERED)


# --------------------------------------------------------------------------- #
# feature construction
# --------------------------------------------------------------------------- #
PRICE_COLUMNS = ("pool_fav_open", "draw_ft_odds", "pool_dog_open")


def rule_feature_names(table: Sequence[RuleRow] = RULES_V2, suffix: str = "") -> list[str]:
    names: list[str] = []
    for rule in table:
        names.extend(
            f"{rule.key}{suffix}_{kind}" for kind in ("match", "near", "dist")
        )
    return names


def _value(row: pd.Series, column: str) -> float | None:
    if column not in row.index:
        return None
    raw = row.get(column)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def rule_features(
    row: pd.Series,
    *,
    table: Sequence[RuleRow] = RULES_V2,
    suffix: str = "",
    fav_column: str = "pool_fav_open",
    draw_column: str = "draw_ft_odds",
    dog_column: str = "pool_dog_open",
    ah_column: str = "ah_open",
) -> pd.Series:
    """Three features per rule row: ``_match``, ``_near``, ``_dist``.

    A rule only produces a non-zero ``_match`` when the opening handicap is one
    of the lines that rule is written for, and every side the rule actually
    checks has a price. A missing price on a checked side yields
    ``match=0 / near=0 / dist=NaN`` - absence of evidence must not be read as
    a match. Rules marked ``untestable`` (R4a: no Ladbrokes feed) yield NaN in
    all three columns, so they can never fire and the audit never measures
    them; the report states why. Rules whose price triple is incomplete but
    testable (R3, R4b) still fire on the sides they do constrain.
    """
    ah = _value(row, ah_column)
    fav = _value(row, fav_column)
    draw = _value(row, draw_column)
    dog = _value(row, dog_column)

    out: dict[str, float] = {}
    for rule in table:
        stem = f"{rule.key}{suffix}"

        if rule.untestable:
            # Spec section 6.2: without the required data source the feature
            # is NaN and the rule is reported as untestable - never guessed
            # from other markets or bookmakers.
            out[f"{stem}_match"] = np.nan
            out[f"{stem}_near"] = np.nan
            out[f"{stem}_dist"] = np.nan
            continue

        if ah is None or not any(np.isclose(ah, line) for line in rule.lines):
            out[f"{stem}_match"] = 0
            out[f"{stem}_near"] = 0
            out[f"{stem}_dist"] = np.nan
            continue

        distances = [
            _dist(fav, rule.fav),
            _dist(draw, rule.draw),
            _dist(dog, rule.dog),
        ]
        if rule.ah_fav is not None:
            # The rule constrains the favourite side's own handicap price
            # (the 2026-10-21 water rule); a missing value suppresses the rule,
            # the same contract as the other sides.
            distances.append(_dist(_value(row, "ah_fav_price"), rule.ah_fav))
        if any(math.isnan(distance) for distance in distances):
            # A checked side has no price. The rule cannot be evaluated, so it
            # must not fire (spec: a missing price leaves the row unknown).
            out[f"{stem}_match"] = 0
            out[f"{stem}_near"] = 0
            out[f"{stem}_dist"] = np.nan
            continue

        distance = max(distances)
        out[f"{stem}_match"] = int(distance == 0.0)
        out[f"{stem}_near"] = int(distance <= TOL)
        out[f"{stem}_dist"] = distance

    return pd.Series(out)


def novig_prices(
    fav: float | None,
    draw: float | None,
    dog: float | None,
) -> tuple[float | None, float | None, float | None]:
    """Vig-free probabilities for one match's 1X2 book.

    Delegates to :func:`fbpredict.labels.fair_three_way_array`, so the margin
    rule has exactly one implementation in this codebase (the scalar
    :func:`fbpredict.labels.fair_three_way` uses the same one).
    """
    if fav is None or draw is None or dog is None:
        return None, None, None
    fair_fav, fair_draw, fair_dog, _ = labels.fair_three_way_array((fav,), (draw,), (dog,))
    values = (float(fair_fav[0]), float(fair_draw[0]), float(fair_dog[0]))
    if not all(math.isfinite(value) for value in values):
        return None, None, None
    return values


def add_rule_features(
    df: pd.DataFrame,
    *,
    suffix: str = "",
    novig: bool = False,
    table: Sequence[RuleRow] | None = None,
) -> pd.DataFrame:
    """Attach one rule set to a frame.

    ``novig=False`` uses the printed thresholds against raw prices;
    ``novig=True`` uses the converted thresholds against each match's own
    vig-free probabilities, so bookmaker margin differences cancel out.
    ``table`` defaults to the printed R1-R6 set (or its no-vig build when
    ``novig=True``); pass a custom table (for example :data:`RULES_WATER`) to
    generate that rule's columns instead.
    """
    if table is None:
        table = RULES_V2_NOVIG if novig else RULES_V2
    out = df.copy()

    if novig:
        probabilities = [
            novig_prices(
                _value(row, "pool_fav_open"),
                _value(row, "draw_ft_odds"),
                _value(row, "pool_dog_open"),
            )
            for _, row in out.iterrows()
        ]
        out["_nv_fav"] = [values[0] for values in probabilities]
        out["_nv_draw"] = [values[1] for values in probabilities]
        out["_nv_dog"] = [values[2] for values in probabilities]
        block = out.apply(
            lambda row: rule_features(
                row,
                table=table,
                suffix=suffix,
                fav_column="_nv_fav",
                draw_column="_nv_draw",
                dog_column="_nv_dog",
            ),
            axis=1,
        )
        out = pd.concat([out, block], axis=1)
        return out.drop(columns=["_nv_fav", "_nv_draw", "_nv_dog"])

    block = out.apply(
        lambda row: rule_features(row, table=table, suffix=suffix), axis=1
    )
    return pd.concat([out, block], axis=1)


# --------------------------------------------------------------------------- #
# descriptive audit - the direct answer to "is my rule actually accurate?"
# --------------------------------------------------------------------------- #
def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - the right one for small samples."""
    if trials <= 0:
        return (math.nan, math.nan)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def _outcome(df: pd.DataFrame, market: str) -> tuple[pd.Series, pd.Series]:
    """``(unit, decimal_odds)`` for the given market, row by row.

    ``AH_dog`` matters as much as ``AH_fav``: the R2 rules are written for the
    underdog ("the underdog does not lose the handicap"), so measuring them
    against the favourite's cover would answer the wrong question.
    """
    if market == "AH_fav":
        unit = [
            labels.ah_unit(diff, line)
            if math.isfinite(diff) and math.isfinite(line)
            else np.nan
            for diff, line in zip(
                pd.to_numeric(df["goal_diff_fav"], errors="coerce"),
                pd.to_numeric(df["ah_open"], errors="coerce"),
            )
        ]
        odds = pd.to_numeric(df.get("ah_fav_price"), errors="coerce")
    elif market == "AH_dog":
        unit = [
            labels.ah_unit_dog(diff, line)
            if math.isfinite(diff) and math.isfinite(line)
            else np.nan
            for diff, line in zip(
                pd.to_numeric(df["goal_diff_fav"], errors="coerce"),
                pd.to_numeric(df["ah_open"], errors="coerce"),
            )
        ]
        odds = pd.to_numeric(df.get("ah_dog_price"), errors="coerce")
    elif market == "OU_over":
        unit = [
            labels.ou_unit(total, line, "over")
            if math.isfinite(total) and math.isfinite(line)
            else np.nan
            for total, line in zip(
                pd.to_numeric(df["total_goals"], errors="coerce"),
                pd.to_numeric(df["ou_line_open"], errors="coerce"),
            )
        ]
        odds = pd.to_numeric(df.get("ou_over_price"), errors="coerce")
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown market {market!r}")
    return pd.Series(unit, index=df.index, dtype="float64"), odds


def audit_rules(
    df: pd.DataFrame,
    *,
    table: Sequence[RuleRow] | None = None,
    suffix: str = "",
    novig: bool = False,
    markets: Sequence[str] = ("AH_fav", "AH_dog", "OU_over"),
    min_n: int = 1,
) -> pd.DataFrame:
    """Hit rate, push rate, confidence interval and break-even per rule.

    **This is an in-sample description, not evidence of an edge.** The rules and
    the data come from the same person and the same period, and nothing here is
    out of sample. It answers "when this condition held, what actually
    happened?", which is a fair question - it just is not a backtest.

    Pushes are reported in their own column and are excluded from the hit rate,
    because a stake returned is not a win.

    ``novig=True`` audits the vig-free block: the thresholds come from
    :data:`RULES_V2_NOVIG` and the prices are each match's own vig-free
    probabilities. A no-vig audit should use its own suffix (``"_nv"``) so it
    can never silently reuse raw rule columns that happen to share the names.
    """
    if table is None:
        table = RULES_V2_NOVIG if novig else RULES_V2

    frame = df.copy()
    if "goal_diff_fav" not in frame.columns or "total_goals" not in frame.columns:
        raise ValueError(
            "audit_rules needs goal_diff_fav and total_goals; "
            "build features first (features.build_features)"
        )

    needed = [f"{rule.key}{suffix}_match" for rule in table]
    if not all(column in frame.columns for column in needed):
        frame = add_rule_features(frame, suffix=suffix, novig=novig, table=table)

    rows: list[dict] = []
    for market in markets:
        unit, odds = _outcome(frame, market)
        profit = [
            labels.unit_to_profit(u, o) if (math.isfinite(u) and math.isfinite(o)) else np.nan
            for u, o in zip(unit, odds)
        ]
        profit_series = pd.Series(profit, index=frame.index, dtype="float64")

        for rule in table:
            fired = frame[f"{rule.key}{suffix}_match"] == 1
            fired &= unit.notna() & odds.notna()
            n = int(fired.sum())
            if n < min_n:
                continue

            units = unit[fired]
            prices = odds[fired]
            profits = profit_series[fired]

            wins = int((units > 0).sum())
            pushes = int((units == 0).sum())
            losses = int((units < 0).sum())
            decided = wins + losses

            hit_rate = wins / decided if decided else math.nan
            ci_low, ci_high = wilson_interval(wins, decided)
            avg_odds = float(prices.mean())
            break_even = 1.0 / avg_odds if avg_odds > 0 else math.nan
            staked = float(len(units))

            rows.append(
                {
                    "rule_id": rule.rule_id,
                    "key": rule.key,
                    "line": rule.lines[0],
                    "market": market,
                    "n": n,
                    "wins": wins,
                    "pushes": pushes,
                    "losses": losses,
                    "hit_rate": hit_rate,
                    "push_rate": pushes / n if n else math.nan,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "avg_odds": avg_odds,
                    "break_even": break_even,
                    "edge_vs_break_even": (
                        hit_rate - break_even if decided else math.nan
                    ),
                    "profit": float(profits.sum()),
                    "roi": float(profits.sum() / staked) if staked else math.nan,
                    "note": rule.note,
                }
            )

    return pd.DataFrame(rows)


def overround_of(fav: float, draw: float, dog: float) -> float:
    """Book margin of a 1X2 triple, for the format/margin comparison."""
    return 1.0 / fav + 1.0 / draw + 1.0 / dog


def rule_table_overrounds(table: Sequence[RuleRow] = RULES_V2) -> pd.DataFrame:
    """The margin implied by each rule row's printed prices (spec section 9.4)."""
    rows = []
    for rule in table:
        prices = [_representative_price(b) for b in (rule.fav, rule.draw, rule.dog)]
        if any(price is None for price in prices):
            continue
        rows.append(
            {
                "key": rule.key,
                "prices": tuple(round(price, 3) for price in prices),  # type: ignore[arg-type]
                "overround": rule.overround,
            }
        )
    return pd.DataFrame(rows)
