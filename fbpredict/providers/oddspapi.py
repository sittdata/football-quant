"""Opening / historical odds from OddsPapi.

**Read this before using it.** The OddsPapi response schema is not hard-coded
here on purpose: the brief (README section 8.8) explicitly forbids trusting a
vendor's marketing article, and this module was written without a live key.
So the provider does three honest things:

1. ``probe()`` fetches a real response, archives it verbatim, and prints the
   structure it actually found (this is README step 9.1),
2. ``extract_odds()`` pulls opening/closing prices out of that payload using an
   explicit mapping you fill in after looking at the probe output,
3. nothing is guessed silently - every field that could not be found is
   returned in ``missing`` so the report can disclose it.

Register: https://oddspapi.io/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..config import Config
from .base import BaseProvider, ProviderError

BASE_URL = "https://api.oddspapi.io/v4"


# --------------------------------------------------------------------------- #
# structural probe
# --------------------------------------------------------------------------- #
def summarise_structure(obj: Any, *, max_depth: int = 4, max_paths: int = 80) -> list[str]:
    """Human-readable outline of a JSON payload, for building a mapping."""
    lines: list[str] = []

    def walk(node: Any, path: str, depth: int) -> None:
        if len(lines) >= max_paths or depth > max_depth:
            return
        if isinstance(node, dict):
            keys = list(node.keys())
            lines.append(f"{path or '$'} -> object, keys={keys[:12]}")
            for key in keys[:12]:
                walk(node[key], f"{path}.{key}" if path else key, depth + 1)
        elif isinstance(node, list):
            lines.append(f"{path or '$'} -> list, len={len(node)}")
            if node:
                walk(node[0], f"{path}[0]", depth + 1)
        else:
            preview = repr(node)
            lines.append(f"{path or '$'} -> {type(node).__name__} = {preview[:60]}")

    walk(obj, "", 0)
    return lines


def find_lists_of_dicts(obj: Any, *, path: str = "", depth: int = 0, limit: int = 12) -> list[str]:
    """Paths that look like a collection we could iterate over."""
    found: list[str] = []
    if depth > 6 or len(found) >= limit:
        return found
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            found.append(path or "$")
        for index, item in enumerate(obj[:1]):
            found.extend(find_lists_of_dicts(item, path=f"{path}[{index}]", depth=depth + 1, limit=limit))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(
                find_lists_of_dicts(value, path=f"{path}.{key}" if path else key, depth=depth + 1, limit=limit)
            )
    return found[:limit]


def get_path(obj: Any, dotted: str) -> Any:
    """Resolve ``a.b[0].c`` against ``obj``; returns ``None`` when absent."""
    if not dotted:
        return None
    node = obj
    for raw_part in dotted.split("."):
        part = raw_part.strip()
        while "[" in part and part.endswith("]"):
            head, _, tail = part.partition("[")
            if head:
                if not isinstance(node, dict) or head not in node:
                    return None
                node = node[head]
            index_text = tail[:-1]
            if not index_text.isdigit():
                return None
            index = int(index_text)
            if not isinstance(node, list) or index >= len(node):
                return None
            node = node[index]
            part = ""
        if part:
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
    return node


# --------------------------------------------------------------------------- #
# mapping
# --------------------------------------------------------------------------- #
@dataclass
class OddsMapping:
    """Where to find things inside a bookmaker's odds payload.

    ``bookmaker_name`` and ``market_name`` are the fields used to filter; the
    outcome fields describe the ``[{name, price, handicap}]`` list.
    """

    collection: str = "data"
    bookmaker_name: str = "bookmaker"
    market_name: str = "market"
    market_handicap: str = "handicap"
    outcomes: str = "outcomes"
    outcome_name: str = "name"
    outcome_price: str = "price"
    timestamp: str = "updatedAt"
    #: accepted spellings per market
    ah_names: tuple[str, ...] = ("asian handicap", "ah", "handicap")
    one_x_two_names: tuple[str, ...] = ("1x2", "match winner", "moneyline")
    over_under_names: tuple[str, ...] = ("over/under", "totals", "ou", "goals over/under")
    draw_ht_names: tuple[str, ...] = ("ht 1x2", "half time", "1st half 1x2")
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OddsMapping":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data and f != "extra"}
        mapping = cls(**known)
        for key, value in data.items():
            if key not in cls.__dataclass_fields__:
                mapping.extra[key] = value
        return mapping

    @classmethod
    def load(cls, path) -> "OddsMapping":
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})


def _matches_any(value: str | None, candidates: Iterable[str]) -> bool:
    if not value:
        return False
    lowered = str(value).strip().lower()
    return any(candidate in lowered for candidate in candidates)


@dataclass
class ExtractedOdds:
    values: dict[str, Any]
    missing: list[str]
    entries_scanned: int = 0

    @property
    def complete(self) -> bool:
        return not self.missing


def extract_odds(
    payload: Any,
    *,
    bookmaker: str,
    mapping: OddsMapping | None = None,
) -> ExtractedOdds:
    """Best-effort extraction of the six numbers the model needs.

    Returns opening prices (earliest timestamp) for the favourite side of the
    Asian handicap, the 1X2 favourite price, the total line, and the HT/FT
    draw prices. Anything it cannot find is listed in ``missing``.
    """
    mapping = mapping or OddsMapping()
    entries = get_path(payload, mapping.collection)
    if not isinstance(entries, list):
        # fall back: try the payload itself, then every list we can find
        if isinstance(payload, list):
            entries = payload
        else:
            for candidate in find_lists_of_dicts(payload):
                node = get_path(payload, candidate)
                if isinstance(node, list) and node:
                    entries = node
                    break

    if not isinstance(entries, list):
        return ExtractedOdds(
            values={}, missing=["collection (could not find a list of bookmaker entries)"]
        )

    wanted = {
        "ah": ("ah_open", "ah_fav_price", "ah_dog_price"),
        "1x2": ("pool_fav_open", "pool_dog_open"),
        "ou": ("ou_line_open", "ou_over_price", "ou_under_price"),
        "draw": ("draw_ht_odds", "draw_ft_odds"),
    }
    found: dict[str, Any] = {}
    timestamps: dict[str, str] = {}
    scanned = 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if bookmaker and not _matches_any(
            str(get_path(entry, mapping.bookmaker_name) or ""), (bookmaker.lower(),)
        ):
            continue
        scanned += 1

        stamp = get_path(entry, mapping.timestamp) or ""
        markets = _collect_markets(entry, mapping)
        for market in markets:
            name = get_path(market, mapping.market_name)
            outcomes = get_path(market, mapping.outcomes) or []
            if not isinstance(outcomes, list):
                continue
            prices = _outcome_prices(outcomes, mapping)
            handicap = get_path(market, mapping.market_handicap)

            if _matches_any(name, mapping.ah_names) and handicap is not None:
                _keep_earliest(found, timestamps, "ah_open", float(handicap), stamp)
                _keep_earliest(found, timestamps, "ah_prices", prices, stamp)
            elif _matches_any(name, mapping.one_x_two_names):
                _keep_earliest(found, timestamps, "1x2_prices", prices, stamp)
            elif _matches_any(name, mapping.over_under_names) and handicap is not None:
                _keep_earliest(found, timestamps, "ou_line_open", float(handicap), stamp)
                _keep_earliest(found, timestamps, "ou_prices", prices, stamp)
            elif _matches_any(name, mapping.draw_ht_names):
                _keep_earliest(found, timestamps, "draw_prices", prices, stamp)

    values: dict[str, Any] = {}
    missing: list[str] = []

    ah_prices = found.get("ah_prices")
    if found.get("ah_open") is not None and isinstance(ah_prices, list) and len(ah_prices) >= 2:
        values["ah_open"] = found["ah_open"]
        values["ah_fav_price"] = ah_prices[0]
        values["ah_dog_price"] = ah_prices[1]
    else:
        missing.append("ah_open / ah_fav_price / ah_dog_price")

    one_x_two = found.get("1x2_prices")
    if isinstance(one_x_two, list) and len(one_x_two) >= 3:
        values["draw_ft_odds"] = one_x_two[1]
        home, _, away = one_x_two[0], one_x_two[1], one_x_two[2]
        values["pool_fav_open"] = min(home, away)
        values["pool_dog_open"] = max(home, away)
    else:
        missing.append("pool_fav_open / pool_dog_open / draw_ft_odds")

    ou_prices = found.get("ou_prices")
    if found.get("ou_line_open") is not None and isinstance(ou_prices, list) and len(ou_prices) >= 2:
        values["ou_line_open"] = found["ou_line_open"]
        values["ou_over_price"] = ou_prices[0]
        values["ou_under_price"] = ou_prices[1]
    else:
        missing.append("ou_line_open / ou_over_price / ou_under_price")

    draw_prices = found.get("draw_prices")
    if isinstance(draw_prices, list) and len(draw_prices) >= 3:
        values["draw_ht_odds"] = draw_prices[1]
    else:
        missing.append("draw_ht_odds")

    return ExtractedOdds(values=values, missing=missing, entries_scanned=scanned)


def _collect_markets(entry: dict, mapping: OddsMapping) -> list[dict]:
    """Find the market list inside a bookmaker entry, wherever it lives."""
    direct = entry.get("markets")
    if isinstance(direct, list):
        return [m for m in direct if isinstance(m, dict)]
    out: list[dict] = []
    for path in find_lists_of_dicts(entry):
        node = get_path(entry, path)
        if isinstance(node, list):
            out.extend(m for m in node if isinstance(m, dict))
    return out


def _outcome_prices(outcomes: list, mapping: OddsMapping) -> list[float]:
    prices: list[float] = []
    for outcome in outcomes:
        if isinstance(outcome, dict):
            price = get_path(outcome, mapping.outcome_price)
        else:
            price = outcome
        try:
            prices.append(float(price))
        except (TypeError, ValueError):
            prices.append(float("nan"))
    return prices


def _keep_earliest(found: dict, stamps: dict, key: str, value: Any, stamp: str) -> None:
    previous = stamps.get(key)
    if previous is None or str(stamp) < str(previous):
        found[key] = value
        stamps[key] = stamp


# --------------------------------------------------------------------------- #
# provider
# --------------------------------------------------------------------------- #
class OddsPapiProvider(BaseProvider):
    def __init__(
        self, cfg: Config, con=None, use_cache: bool = True, refresh: bool = False
    ) -> None:
        super().__init__(
            name="oddspapi", cfg=cfg, con=con, use_cache=use_cache, refresh=refresh
        )

    def _params(self, **extra: Any) -> dict[str, Any]:
        params = {"apiKey": self.require_key(self.cfg.oddspapi_key, "ODDSPAPI_KEY")}
        params.update({k: v for k, v in extra.items() if v is not None})
        return params

    # ------------------------------------------------------------------ #
    def discover_bookmakers(self) -> Any:
        return self.request(
            f"{BASE_URL}/bookmakers",
            params=self._params(),
            request_key="bookmakers",
            endpoint="/bookmakers",
        ).data

    def discover_markets(self) -> Any:
        return self.request(
            f"{BASE_URL}/markets",
            params=self._params(),
            request_key="markets",
            endpoint="/markets",
        ).data

    def fetch_odds(self, fixture_id: str | int) -> Any:
        return self.request(
            f"{BASE_URL}/odds",
            params=self._params(fixtureId=fixture_id),
            request_key=f"odds-{fixture_id}",
            endpoint="/odds",
        ).data

    def fetch_historical_odds(self, fixture_id: str | int) -> Any:
        return self.request(
            f"{BASE_URL}/historical-odds",
            params=self._params(fixtureId=fixture_id),
            request_key=f"hist-odds-{fixture_id}",
            endpoint="/historical-odds",
        ).data

    # ------------------------------------------------------------------ #
    def probe(self, fixture_id: str | int | None = None) -> dict[str, Any]:
        """README step 9.1: fetch for real, then report what came back."""
        report: dict[str, Any] = {"fixtures": {}, "notes": []}

        for label, call in (
            ("bookmakers", self.discover_bookmakers),
            ("markets", self.discover_markets),
        ):
            try:
                payload = call()
            except ProviderError as exc:
                report["notes"].append(f"{label}: {exc}")
                continue
            report[label] = summarise_structure(payload, max_depth=3, max_paths=40)

        if fixture_id is None:
            report["notes"].append(
                "no fixture id supplied - pass one to probe /odds and /historical-odds"
            )
            return report

        for label, call in (
            ("odds", self.fetch_odds),
            ("historical_odds", self.fetch_historical_odds),
        ):
            try:
                payload = call(fixture_id)
            except ProviderError as exc:
                report["fixtures"][label] = f"error: {exc}"
                continue
            report["fixtures"][label] = {
                "structure": summarise_structure(payload),
                "candidate_collections": find_lists_of_dicts(payload),
                "has_score_field": _mentions_score(payload),
            }

        return report


def _mentions_score(payload: Any) -> list[str]:
    """README 6.1 asks: does the odds response sneak in a score?"""
    hits: list[str] = []

    def walk(node: Any, path: str, depth: int = 0) -> None:
        if depth > 6 or len(hits) >= 20:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                if any(word in lowered for word in ("score", "goal", "result")):
                    hits.append(f"{path}.{key} = {str(value)[:60]}")
                walk(value, f"{path}.{key}", depth + 1)
        elif isinstance(node, list):
            for index, item in enumerate(node[:3]):
                walk(item, f"{path}[{index}]", depth + 1)

    walk(payload, "$")
    return hits


def dumps_mapping_template() -> str:
    """A starter YAML the user edits after reading the probe output."""
    template = {
        "collection": OddsMapping.collection,
        "bookmaker_name": OddsMapping.bookmaker_name,
        "market_name": OddsMapping.market_name,
        "market_handicap": OddsMapping.market_handicap,
        "outcomes": OddsMapping.outcomes,
        "outcome_name": OddsMapping.outcome_name,
        "outcome_price": OddsMapping.outcome_price,
        "timestamp": OddsMapping.timestamp,
        "ah_names": list(OddsMapping.ah_names),
        "one_x_two_names": list(OddsMapping.one_x_two_names),
        "over_under_names": list(OddsMapping.over_under_names),
        "draw_ht_names": list(OddsMapping.draw_ht_names),
    }
    return json.dumps(template, indent=2, ensure_ascii=False)
