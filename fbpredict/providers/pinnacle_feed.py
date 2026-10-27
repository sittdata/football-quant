"""
Pinnacle Sharp Market Depth & Fair Odds Provider.
Direct connection to Pinnacle Arcadia Gateway (Zero-authentication, high-throughput).

Integrates:
- Real-time Matchup Discovery (Football & Basketball)
- Straight Line Extraction (Moneyline 1X2, Asian Handicap Spreads, Over/Under Totals)
- Shin Vig-Removal Algorithm (1993) to strip bookmaker margin and extract
  unbiased true probabilities and insider trading volume (z)
- +EV Arbitrage Edge Calculator against soft bookmakers
- Optimal Kelly Staking recommendations
"""

from typing import List, Dict, Any, Optional, Tuple
from curl_cffi import requests
import shin
import logging

logger = logging.getLogger(__name__)

class PinnacleSharpFeed:
    BASE_URL = "https://guest.api.arcadia.pinnacle.com/0.1"
    DOH_URL = "https://cloudflare-dns.com/dns-query"

    def __init__(self, sport_id: int = 29, timeout: int = 10):
        # Sport 29 = Football (Soccer), Sport 4 = Basketball
        self.sport_id = sport_id
        self.timeout = timeout
        self.session = requests.Session(impersonate="safari184")
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Referer": "https://www.pinnacle.com/",
            "Origin": "https://www.pinnacle.com"
        })

    @staticmethod
    def american_to_decimal(american: float) -> float:
        if american > 0:
            return round(1.0 + american / 100.0, 4)
        elif american < 0:
            return round(1.0 + 100.0 / abs(american), 4)
        return 1.0

    def get_matchups(self) -> List[Dict[str, Any]]:
        """Fetch all upcoming and live matchups for the sport."""
        url = f"{self.BASE_URL}/sports/{self.sport_id}/matchups?withSpecials=false"
        try:
            r = self.session.get(url, doh_url=self.DOH_URL, timeout=self.timeout)
            if r.status_code != 200:
                logger.warning(f"Pinnacle matchups returned status {r.status_code}")
                return []
            data = r.json()
            results = []
            for m in data:
                parts = m.get("participants", [])
                if len(parts) >= 2:
                    home = next((p.get("name") for p in parts if p.get("alignment") == "home"), parts[0].get("name"))
                    away = next((p.get("name") for p in parts if p.get("alignment") == "away"), parts[1].get("name"))
                    results.append({
                        "id": m.get("id"),
                        "home_team": home,
                        "away_team": away,
                        "league": m.get("league", {}).get("name", "Unknown"),
                        "is_live": m.get("isLive", False),
                        "start_time": m.get("startTime")
                    })
            return results
        except Exception as e:
            logger.error(f"Error fetching Pinnacle matchups: {e}")
            return []

    def get_market_lines(self, matchup_id: int) -> Dict[str, Any]:
        """Fetch straight lines (Moneyline, Spreads, Totals) and apply Shin Vig-Removal."""
        url = f"{self.BASE_URL}/matchups/{matchup_id}/markets/straight"
        try:
            r = self.session.get(url, doh_url=self.DOH_URL, timeout=self.timeout)
            if r.status_code != 200:
                return {}
            markets = r.json()
            output = {"raw_markets": markets, "moneyline": None, "spreads": [], "totals": []}

            # 1. Moneyline
            ml = next((m for m in markets if m.get("type") == "moneyline" and m.get("period") == 0), None)
            if ml:
                prices = ml.get("prices", [])
                p_h = next((p for p in prices if p.get("designation") == "home"), None)
                p_d = next((p for p in prices if p.get("designation") == "draw"), None)
                p_a = next((p for p in prices if p.get("designation") == "away"), None)
                if p_h and p_a:
                    if p_d: # 3-way
                        raw_odds = [self.american_to_decimal(p_h["price"]), self.american_to_decimal(p_d["price"]), self.american_to_decimal(p_a["price"])]
                    else: # 2-way (e.g. Basketball)
                        raw_odds = [self.american_to_decimal(p_h["price"]), self.american_to_decimal(p_a["price"])]
                    
                    shin_res = shin.calculate_implied_probabilities(raw_odds, full_output=True)
                    output["moneyline"] = {
                        "raw_odds": raw_odds,
                        "true_probs": list(shin_res.implied_probabilities),
                        "fair_odds": [round(1.0 / p, 3) for p in shin_res.implied_probabilities],
                        "shin_z": shin_res.z,
                        "vig_percent": (sum([1.0/o for o in raw_odds]) - 1.0) * 100.0
                    }

            # 2. Spreads (Asian Handicap)
            spread_list = [m for m in markets if m.get("type") == "spread" and m.get("period") == 0]
            for sp in spread_list:
                prices = sp.get("prices", [])
                p_h = next((p for p in prices if p.get("designation") == "home"), None)
                p_a = next((p for p in prices if p.get("designation") == "away"), None)
                if p_h and p_a:
                    h_dec = self.american_to_decimal(p_h["price"])
                    a_dec = self.american_to_decimal(p_a["price"])
                    shin_p = shin.calculate_implied_probabilities([h_dec, a_dec])
                    output["spreads"].append({
                        "handicap": p_h.get("points", 0),
                        "home_odds": h_dec,
                        "away_odds": a_dec,
                        "home_fair": round(1.0 / shin_p[0], 3),
                        "away_fair": round(1.0 / shin_p[1], 3),
                        "home_true_prob": shin_p[0],
                        "away_true_prob": shin_p[1]
                    })

            # 3. Totals (Over / Under)
            total_list = [m for m in markets if m.get("type") == "total" and m.get("period") == 0]
            for tot in total_list:
                prices = tot.get("prices", [])
                p_o = next((p for p in prices if p.get("designation") == "over"), None)
                p_u = next((p for p in prices if p.get("designation") == "under"), None)
                if p_o and p_u:
                    o_dec = self.american_to_decimal(p_o["price"])
                    u_dec = self.american_to_decimal(p_u["price"])
                    shin_p = shin.calculate_implied_probabilities([o_dec, u_dec])
                    output["totals"].append({
                        "points": p_o.get("points", 0),
                        "over_odds": o_dec,
                        "under_odds": u_dec,
                        "over_fair": round(1.0 / shin_p[0], 3),
                        "under_fair": round(1.0 / shin_p[1], 3),
                        "over_true_prob": shin_p[0],
                        "under_true_prob": shin_p[1]
                    })

            return output
        except Exception as e:
            logger.error(f"Error fetching market lines for matchup {matchup_id}: {e}")
            return {}

    @staticmethod
    def calculate_ev(true_prob: float, market_odds: float) -> Tuple[float, float]:
        """
        Calculate Mathematical Expected Value (+EV) and Quarter-Kelly stake.
        Returns: (edge_percent, quarter_kelly_percent)
        """
        edge = (true_prob * market_odds - 1.0) * 100.0
        if market_odds <= 1.0 or edge <= 0:
            return 0.0, 0.0
        full_kelly = (true_prob * market_odds - 1.0) / (market_odds - 1.0)
        quarter_kelly = max(0.0, full_kelly * 0.25) * 100.0
        return round(edge, 2), round(quarter_kelly, 2)
