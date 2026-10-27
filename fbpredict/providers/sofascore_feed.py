"""
SofaScore Quant Live Feed Provider.
Bypasses Cloudflare JA3/JA4 WAF challenges via curl_cffi (safari184 impersonation)
and bypasses local ISP DNS poisoning via Cloudflare DNS-over-HTTPS (DoH).

Provides:
- get_live_events(): All live football events with current score, status, and IDs
- get_attack_momentum(event_id): Minute-by-minute attack momentum index (-100 to +100)
- get_shotmap_xg(event_id): Precise pitch coordinates (x,y), outcome, player, and Opta xG
- get_match_statistics(event_id): Live possession, dangerous attacks, passes, corners, cards
- get_live_odds(event_id): Current bookmaker markets and implied probabilities
"""

from typing import List, Dict, Any, Optional
from curl_cffi import requests
import logging

logger = logging.getLogger(__name__)

class SofaScoreQuantFeed:
    BASE_URL = "https://www.sofascore.com/api/v1"
    DOH_URL = "https://cloudflare-dns.com/dns-query"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session(impersonate="safari184")
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.sofascore.com/",
            "Origin": "https://www.sofascore.com"
        })

    def _get(self, path: str) -> Optional[Dict[str, Any]]:
        url = f"{self.BASE_URL}/{path.lstrip('/')}"
        try:
            r = self.session.get(url, doh_url=self.DOH_URL, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            logger.warning(f"SofaScore API returned status {r.status_code} for {path}")
            return None
        except Exception as e:
            logger.error(f"SofaScore request error on {path}: {e}")
            return None

    def get_live_events(self) -> List[Dict[str, Any]]:
        """Fetch all currently live football matches worldwide."""
        data = self._get("sport/football/events/live")
        if not data:
            return []
        events = data.get("events", [])
        results = []
        for e in events:
            results.append({
                "id": e.get("id"),
                "home_team": e.get("homeTeam", {}).get("name"),
                "away_team": e.get("awayTeam", {}).get("name"),
                "home_score": e.get("homeScore", {}).get("current", 0),
                "away_score": e.get("awayScore", {}).get("current", 0),
                "status": e.get("status", {}).get("description"),
                "tournament": e.get("tournament", {}).get("name"),
                "start_timestamp": e.get("startTimestamp")
            })
        return results

    def get_attack_momentum(self, event_id: int) -> List[Dict[str, Any]]:
        """Extract minute-by-minute attack momentum values (-100 to +100)."""
        data = self._get(f"event/{event_id}/graph")
        if not data:
            return []
        return data.get("graphPoints", [])

    def get_shotmap_xg(self, event_id: int) -> List[Dict[str, Any]]:
        """Extract Opta shotmap coordinates, body part, outcome, and calculated xG."""
        data = self._get(f"event/{event_id}/shotmap")
        if not data:
            return []
        shots = data.get("shotmap", [])
        clean_shots = []
        for s in shots:
            clean_shots.append({
                "minute": s.get("time"),
                "player": s.get("player", {}).get("name"),
                "shot_type": s.get("shotType"),
                "body_part": s.get("bodyPart"),
                "xg": s.get("xg"),
                "coordinates": s.get("playerCoordinates")
            })
        return clean_shots

    def get_match_statistics(self, event_id: int) -> Dict[str, Any]:
        """Extract in-play detailed statistics (possession, dangerous attacks, duels, saves)."""
        data = self._get(f"event/{event_id}/statistics")
        if not data:
            return {}
        return data.get("statistics", {})

    def get_live_odds(self, event_id: int) -> List[Dict[str, Any]]:
        """Extract real-time bookmaker markets (1X2, Asian handicap, totals)."""
        data = self._get(f"event/{event_id}/odds/1/all")
        if not data:
            return []
        return data.get("markets", [])
