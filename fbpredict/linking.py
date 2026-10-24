"""Joining odds rows to results rows.

Odds and results almost always come from two different providers, so the two
halves of the dataset have to be matched up. Matching on team names plus
kick-off time is fragile, which is exactly why every match here reports how it
was linked and how far apart the kick-off times were.

Rules:

* team names are normalised (case, accents, punctuation, common club suffixes),
* the pair must match in the same orientation (home/away are not swapped),
* kick-off times must be within ``tolerance_hours``,
* anything ambiguous is reported, never silently guessed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

CLUB_NOISE = {
    "fc", "cf", "sc", "ac", "afc", "cd", "ud", "sv", "vfb", "vfl", "tsg", "bsc",
    "as", "ss", "ssc", "rc", "rcd", "ca", "club", "de", "futbol", "calcio",
    "the", "if", "bk", "fk", "sk", "nk", "hk", "kf", "ku", "il", "ik",
}

_PUNCT = re.compile(r"[^a-z0-9 ]+")

#: Default slack allowed between the two providers' kick-off times. Single
#: source of truth: the CLI default reads from here too.
DEFAULT_TOLERANCE_HOURS = 36.0


def normalise_team(name: str | None) -> str:
    """Aggressive, deterministic normalisation for name matching."""
    if name is None:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ")
    text = _PUNCT.sub(" ", text)
    tokens = [token for token in text.split() if token and token not in CLUB_NOISE]
    return " ".join(tokens)


@dataclass
class LinkReport:
    linked: pd.DataFrame
    unmatched_odds: list[str]
    unmatched_results: list[str]
    ambiguous: list[str]

    @property
    def summary(self) -> str:
        return (
            f"linked {len(self.linked)} rows, "
            f"{len(self.unmatched_odds)} odds rows unmatched, "
            f"{len(self.unmatched_results)} results unmatched, "
            f"{len(self.ambiguous)} ambiguous"
        )


def link_results(
    odds_df: pd.DataFrame,
    results_df: pd.DataFrame,
    *,
    tolerance_hours: float = DEFAULT_TOLERANCE_HOURS,
) -> LinkReport:
    """Attach result columns from ``results_df`` onto ``odds_df``."""
    if odds_df.empty or results_df.empty:
        return LinkReport(odds_df.copy(), list(odds_df.get("match_id", [])), [], [])

    results = results_df.copy()
    results["_home"] = results["home_team"].map(normalise_team)
    results["_away"] = results["away_team"].map(normalise_team)
    results["_kickoff"] = pd.to_datetime(results["kickoff_at"], utc=True, format="ISO8601")

    odds = odds_df.copy()
    odds["_home"] = odds["home_team"].map(normalise_team)
    odds["_away"] = odds["away_team"].map(normalise_team)
    odds["_kickoff"] = pd.to_datetime(odds["kickoff_at"], utc=True, format="ISO8601")

    linked_rows: list[dict] = []
    unmatched_odds: list[str] = []
    ambiguous: list[str] = []

    for _, row in odds.iterrows():
        candidates = results[
            (results["_home"] == row["_home"])
            & (results["_away"] == row["_away"])
            & (
                (results["_kickoff"] - row["_kickoff"]).abs()
                <= pd.Timedelta(hours=tolerance_hours)
            )
        ]

        if candidates.empty:
            unmatched_odds.append(str(row.get("match_id")))
            continue
        if len(candidates) > 1:
            # keep the closest in time, but say so
            candidates = candidates.assign(
                _delta=(candidates["_kickoff"] - row["_kickoff"]).abs()
            ).sort_values("_delta")
            ambiguous.append(str(row.get("match_id")))

        match = candidates.iloc[0]
        merged = row.to_dict()
        merged.update(
            {
                "_matched_result_id": match.get("match_id"),
                "ht_home": match.get("ht_home"),
                "ht_away": match.get("ht_away"),
                "home_goals": match.get("home_goals"),
                "away_goals": match.get("away_goals"),
                "result_source": match.get("result_source"),
                "result_fetched_at": match.get("result_fetched_at"),
            }
        )
        linked_rows.append(merged)

    # Linked rows keep the *odds* match_id, so the matched result ids have to be
    # tracked separately. Comparing odds ids against results["match_id"] would
    # report every single result as unmatched.
    matched_result_ids = {
        row.get("_matched_result_id") for row in linked_rows if row.get("_matched_result_id")
    }
    unmatched_results = [
        str(mid) for mid in results["match_id"] if mid not in matched_result_ids
    ]

    linked = pd.DataFrame(linked_rows)
    drop = [
        column
        for column in ("_home", "_away", "_kickoff", "_delta", "_matched_result_id")
        if column in linked.columns
    ]
    if drop:
        linked = linked.drop(columns=drop)

    return LinkReport(
        linked=linked,
        unmatched_odds=unmatched_odds,
        unmatched_results=unmatched_results,
        ambiguous=ambiguous,
    )
