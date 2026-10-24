"""Thscore Direct Live Odds & Match Engine.

Parses official high-frequency live data feeds from Thscore:
- Match feed: /gf/data/bf_th_nt.js (Match status, HT scores, corners, cards, weather)
- Odds feeds:
    * /gf/data/odds/en/ch_goal{company_id}.xml (Opening and Current AH, OU, 1X2 odds)
    * /gf/data/odds/en/ch_runOddsData_{company_id}.txt (Streaming odds updates)

Companies:
    31: SBOBET (Asian Market Leader)
    8:  Bet365 (Global Odds Benchmark)
    3:  Crown (Traditional Macau/Asian Handicap)
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.thscore.vip/",
}
BASE_URL = "https://www.thscore.vip"


@dataclass
class ThscoreLiveMatch:
    match_id: str
    league: str
    home_team: str
    away_team: str
    kickoff_at: str
    status: int  # 0: Upcoming, >0: In progress, -1: Finished
    home_goals: int | None = None
    away_goals: int | None = None
    ht_home: int | None = None
    ht_away: int | None = None
    home_corner: int | None = None
    away_corner: int | None = None
    home_red: int | None = None
    away_red: int | None = None
    home_yellow: int | None = None
    away_yellow: int | None = None
    weather: str | None = None
    home_rank: str | None = None
    away_rank: str | None = None
    referee: str | None = None
    # Live Asian Odds (Primary / SBOBET)
    ah_line: float | None = None
    ah_home_price: float | None = None
    ah_away_price: float | None = None
    ou_line: float | None = None
    ou_over_price: float | None = None
    ou_under_price: float | None = None
    pool_home: float | None = None
    pool_draw: float | None = None
    pool_away: float | None = None
    # Bet365 Sharp Consensus (Company 8)
    b365_ah_line: float | None = None
    b365_ah_home: float | None = None
    b365_ah_away: float | None = None
    b365_ou_line: float | None = None
    b365_ou_over: float | None = None
    b365_ou_under: float | None = None
    # Crown Consensus (Company 3)
    crown_ah_line: float | None = None
    crown_ah_home: float | None = None
    crown_ah_away: float | None = None


def fetch_matches_feed() -> tuple[dict[int, str], list[dict[str, Any]]]:
    """Fetch all active and upcoming matches and leagues from bf_th_nt.js."""
    url = f"{BASE_URL}/gf/data/bf_th_nt.js"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    text = resp.text

    # 1. Parse Leagues (B array)
    leagues = {}
    for match in re.finditer(r"B\[(\d+)\]\s*=\s*\[([^\]]+)\];", text):
        idx = int(match.group(1))
        raw_items = match.group(2).split(",")
        # The Thai league name is typically at index 8 or 2
        name = raw_items[2].strip("'\" ") if len(raw_items) > 2 else "Unknown"
        if len(raw_items) > 8 and raw_items[8].strip("'\" "):
            name = raw_items[8].strip("'\" ")
        leagues[idx] = name

    # 2. Parse Matches (A array)
    matches_raw = []
    # Pattern to match: A[1]=[2997760,...]
    for match in re.finditer(r"A\[\d+\]\s*=\s*\[([^\]]+)\];", text):
        items = [i.strip("'\" ") for i in match.group(1).split(",")]
        if len(items) < 15:
            continue

        try:
            mid = items[0]
            lea_idx = int(items[1]) if items[1].isdigit() else 0
            lea_name = leagues.get(lea_idx, "Unknown")
            home = items[4]
            away = items[5]
            kickoff = items[6]
            status = int(items[8]) if items[8].isdigit() or items[8].startswith("-") else 0
            
            # Scores
            h_score = int(items[9]) if items[9].isdigit() else None
            a_score = int(items[10]) if items[10].isdigit() else None
            ht_h = int(items[11]) if items[11].isdigit() else None
            ht_a = int(items[12]) if items[12].isdigit() else None
            
            # Cards
            h_red = int(items[13]) if items[13].isdigit() else 0
            a_red = int(items[14]) if items[14].isdigit() else 0
            h_yel = int(items[15]) if items[15].isdigit() else 0
            a_yel = int(items[16]) if items[16].isdigit() else 0

            # Default initial AH and OU if embedded
            ah_line = float(items[21]) if len(items) > 21 and items[21].replace(".", "", 1).replace("-", "", 1).isdigit() else None
            ou_line = float(items[25]) if len(items) > 25 and items[25].replace(".", "", 1).isdigit() else None
            
            # Corners
            tot_corners = int(items[27]) if len(items) > 27 and items[27].isdigit() else None
            a_corners = int(items[28]) if len(items) > 28 and items[28].isdigit() else None
            h_corners = (tot_corners - a_corners) if (tot_corners is not None and a_corners is not None and tot_corners >= a_corners) else None

            # Standings / Referee / Weather
            h_rank = items[17].strip("'\" ") if len(items) > 17 and items[17].strip("'\" ") else None
            a_rank = items[18].strip("'\" ") if len(items) > 18 and items[18].strip("'\" ") else None
            referee = items[32].strip("'\" ") if len(items) > 32 and items[32].strip("'\" ") else None
            weather = items[33].strip("'\" ") if len(items) > 33 and items[33].strip("'\" ") else None

            matches_raw.append({
                "match_id": str(mid),
                "league": lea_name,
                "home_team": home,
                "away_team": away,
                "kickoff_at": kickoff,
                "status": status,
                "home_goals": h_score,
                "away_goals": a_score,
                "ht_home": ht_h,
                "ht_away": ht_a,
                "home_corner": h_corners,
                "away_corner": a_corners,
                "home_red": h_red,
                "away_red": a_red,
                "home_yellow": h_yel,
                "away_yellow": a_yel,
                "ah_line": ah_line,
                "ou_line": ou_line,
                "home_rank": h_rank,
                "away_rank": a_rank,
                "referee": referee,
                "weather": weather,
            })
        except Exception:
            continue

    return leagues, matches_raw


def fetch_company_odds(company_id: int = 31) -> dict[str, dict[str, float]]:
    """Fetch live Asian Handicap, Over/Under, and 1X2 odds from ch_goal{cid}.xml."""
    url = f"{BASE_URL}/gf/data/odds/en/ch_goal{company_id}.xml"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        if resp.status_code != 200 or not resp.text.startswith("<?xml"):
            return {}
        
        root = ET.fromstring(resp.text)
        odds_map = {}
        for m_elem in root.findall(".//m"):
            text = m_elem.text
            if not text:
                continue
            parts = [p.strip() for p in text.split(",")]
            if len(parts) < 13:
                continue
            
            mid = str(parts[0])
            # Index 2: AH Line, Index 3: AH Home Price, Index 4: AH Away Price
            def safe_float(v):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None

            ah_line = safe_float(parts[2])
            ah_home = safe_float(parts[3])
            ah_away = safe_float(parts[4])
            
            # Index 6: 1X2 Home, 7: Draw, 8: Away
            p_home = safe_float(parts[6])
            p_draw = safe_float(parts[7])
            p_away = safe_float(parts[8])

            # Index 10: OU Line, 11: Over Price, 12: Under Price
            ou_line = safe_float(parts[10])
            ou_over = safe_float(parts[11])
            ou_under = safe_float(parts[12])

            odds_map[mid] = {
                "ah_line": ah_line,
                "ah_home_price": ah_home,
                "ah_away_price": ah_away,
                "ou_line": ou_line,
                "ou_over_price": ou_over,
                "ou_under_price": ou_under,
                "pool_home": p_home,
                "pool_draw": p_draw,
                "pool_away": p_away,
            }
        return odds_map
    except Exception as e:
        print(f"[Warning] Failed to fetch company odds ({company_id}): {e}")
        return {}


def get_thscore_live_data(company_id: int = 31) -> list[ThscoreLiveMatch]:
    """Combine match feed with live multi-bookmaker odds (SBOBET, Bet365, Crown)."""
    _, raw_matches = fetch_matches_feed()
    sbo_odds = fetch_company_odds(company_id=31)
    b365_odds = fetch_company_odds(company_id=8)
    crown_odds = fetch_company_odds(company_id=3)

    results = []
    for m in raw_matches:
        mid = m["match_id"]
        # Primary odds: SBOBET -> Bet365 -> Crown -> embedded
        sbo = sbo_odds.get(mid, {})
        b365 = b365_odds.get(mid, {})
        crown = crown_odds.get(mid, {})

        primary = sbo or b365 or crown or {}

        ah_line = primary.get("ah_line") if primary.get("ah_line") is not None else m.get("ah_line")
        ou_line = primary.get("ou_line") if primary.get("ou_line") is not None else m.get("ou_line")

        match_obj = ThscoreLiveMatch(
            match_id=mid,
            league=m["league"],
            home_team=m["home_team"],
            away_team=m["away_team"],
            kickoff_at=m["kickoff_at"],
            status=m["status"],
            home_goals=m["home_goals"],
            away_goals=m["away_goals"],
            ht_home=m["ht_home"],
            ht_away=m["ht_away"],
            home_corner=m["home_corner"],
            away_corner=m["away_corner"],
            home_red=m["home_red"],
            away_red=m["away_red"],
            home_yellow=m["home_yellow"],
            away_yellow=m["away_yellow"],
            weather=m.get("weather"),
            home_rank=m.get("home_rank"),
            away_rank=m.get("away_rank"),
            referee=m.get("referee"),
            # Primary Odds
            ah_line=ah_line,
            ah_home_price=primary.get("ah_home_price"),
            ah_away_price=primary.get("ah_away_price"),
            ou_line=ou_line,
            ou_over_price=primary.get("ou_over_price"),
            ou_under_price=primary.get("ou_under_price"),
            pool_home=primary.get("pool_home"),
            pool_draw=primary.get("pool_draw"),
            pool_away=primary.get("pool_away"),
            # Bet365 Sharp Consensus
            b365_ah_line=b365.get("ah_line"),
            b365_ah_home=b365.get("ah_home_price"),
            b365_ah_away=b365.get("ah_away_price"),
            b365_ou_line=b365.get("ou_line"),
            b365_ou_over=b365.get("ou_over_price"),
            b365_ou_under=b365.get("ou_under_price"),
            # Crown Consensus
            crown_ah_line=crown.get("ah_line"),
            crown_ah_home=crown.get("ah_home_price"),
            crown_ah_away=crown.get("ah_away_price"),
        )
        results.append(match_obj)

    return results
