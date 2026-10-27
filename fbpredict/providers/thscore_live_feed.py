"""
Thscore Live Odds & Score Real-Time Stream Provider.
Reverse-Engineered directly from Thscore CDN stream (No API Key Required).

Extracts:
- Live match status & time
- Live scores & Half-Time scores
- Red cards & Corners
- Real-Time Asian Handicap Line (ราคาบอลแฮนดิแคป)
- Real-Time Over/Under Line (ราคาสูง-ต่ำ)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
import requests

STREAM_URL = "http://www.thscore.mobi/gf/phone/livechange_en.txt"

@dataclass
class LiveMatchOdds:
    match_id: str
    status: int
    kickoff_raw: str
    home_score: int
    away_score: int
    ht_home: int
    ht_away: int
    home_red: int
    away_red: int
    home_corner: int
    away_corner: int
    handicap_line: float | None
    over_under_line: float | None
    updated_at: float

class ThscoreLiveFeedProvider:
    """Zero-key, real-time live match and odds feed reverse-engineered from Thscore."""

    def __init__(self, timeout: float = 6.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15",
            "Accept": "*/*",
            "Referer": "http://www.thscore.mobi/"
        })

    def fetch_live_stream(self) -> list[LiveMatchOdds]:
        """Fetches and parses the latest CDN live delta stream."""
        # Add cache buster timestamp
        url = f"{STREAM_URL}?r={int(time.time() * 1000)}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()

        text = resp.text.strip()
        if not text:
            return []

        matches = []
        raw_items = text.split("!")
        now = time.time()

        for item in raw_items:
            if not item:
                continue
            fields = item.split("^")
            if len(fields) < 15:
                continue

            try:
                def _to_int(val: str, default: int = 0) -> int:
                    return int(val) if val and val.lstrip("-").isdigit() else default

                def _to_float(val: str) -> float | None:
                    if not val:
                        return None
                    try:
                        return float(val)
                    except ValueError:
                        return None

                match = LiveMatchOdds(
                    match_id=f"thscore:{fields[0]}",
                    status=_to_int(fields[1]),
                    kickoff_raw=fields[2],
                    home_score=_to_int(fields[4]),
                    away_score=_to_int(fields[5]),
                    ht_home=_to_int(fields[6]),
                    ht_away=_to_int(fields[7]),
                    home_red=_to_int(fields[8]),
                    away_red=_to_int(fields[9]),
                    home_corner=_to_int(fields[10]),
                    away_corner=_to_int(fields[11]),
                    handicap_line=_to_float(fields[12]),
                    over_under_line=_to_float(fields[14]),
                    updated_at=now
                )
                matches.append(match)
            except Exception:
                continue

        return matches

if __name__ == "__main__":
    provider = ThscoreLiveFeedProvider()
    print("[*] Fetching live stream from Thscore CDN...")
    live_items = provider.fetch_live_stream()
    print(f"[+] Total live matches retrieved: {len(live_items)}")
    
    # Print sample matches with odds
    for m in live_items[:5]:
        print(f"Match {m.match_id}: Score {m.home_score}-{m.away_score} (HT {m.ht_home}-{m.ht_away}) | Corners {m.home_corner}-{m.away_corner} | HDP Line: {m.handicap_line} | O/U Line: {m.over_under_line}")
