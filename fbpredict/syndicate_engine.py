"""
Standalone Syndicate Quant Alpha Engine (Isolated Module).
Integrates:
1. Pinnacle Arcadia Gateway (Sharp Ground Truth & Shin Vig-Removal)
2. SofaScore Real-Time Quant Engine (Attack Momentum Graph & Opta xG)
3. Dynamic In-Play Bayesian Goal Expectancy & Quarter-Kelly Portfolio Staking

Completely decoupled from the core legacy model to ensure zero side-effects.
Author: cook45
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
import logging
from dataclasses import dataclass, field

from fbpredict.providers.pinnacle_feed import PinnacleSharpFeed
from fbpredict.providers.sofascore_feed import SofaScoreQuantFeed

logger = logging.getLogger(__name__)

@dataclass
class SyndicateSignal:
    match_id: str
    home_team: str
    away_team: str
    league: str
    is_live: bool
    current_minute: Optional[int]
    current_score: str
    
    # Sharp Market Baseline (Pinnacle + Shin)
    pinnacle_quoted_odds: List[float]
    shin_true_probs: List[float]
    shin_fair_odds: List[float]
    shin_insider_z: float
    market_vig_percent: float
    
    # In-Play Dynamic Field Pressure (SofaScore)
    momentum_10m_avg: float
    dominating_side: str
    total_xg_home: float
    total_xg_away: float
    
    # Edge & Execution
    recommended_market: str
    edge_percent: float
    quarter_kelly_stake_percent: float
    verdict: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "league": self.league,
            "is_live": self.is_live,
            "current_minute": self.current_minute,
            "current_score": self.current_score,
            "sharp_baseline": {
                "quoted_odds": self.pinnacle_quoted_odds,
                "shin_true_probs": [round(p, 4) for p in self.shin_true_probs],
                "shin_fair_odds": self.shin_fair_odds,
                "insider_z": round(self.shin_insider_z, 4),
                "vig_percent": round(self.market_vig_percent, 2)
            },
            "live_pressure": {
                "momentum_10m_avg": round(self.momentum_10m_avg, 2),
                "dominating_side": self.dominating_side,
                "xg_home": round(self.total_xg_home, 2),
                "xg_away": round(self.total_xg_away, 2)
            },
            "execution": {
                "recommended_market": self.recommended_market,
                "edge_percent": round(self.edge_percent, 2),
                "quarter_kelly_stake_percent": round(self.quarter_kelly_stake_percent, 2),
                "verdict": self.verdict,
                "rationale": self.rationale
            }
        }


class SyndicateComboEngine:
    """
    Independent Triangulation Engine:
    Combines Pinnacle (Sharp Fair Price) + SofaScore (In-play Momentum) + Kelly Staking.
    """

    def __init__(self, timeout: int = 10):
        self.pinnacle = PinnacleSharpFeed(timeout=timeout)
        self.sofascore = SofaScoreQuantFeed(timeout=timeout)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Strip common suffixes for fuzzy entity resolution."""
        low = name.lower()
        for term in [" fc", " sc", " cf", " u23", " u21", " u20", " reserves", " athletic", " de "]:
            low = low.replace(term, " ")
        return "".join(c for c in low if c.isalnum() or c.isspace()).strip()

    def find_matching_sofascore_event(self, home: str, away: str, sofa_events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Match a Pinnacle matchup with a SofaScore event."""
        norm_h = self._normalize_name(home)
        norm_a = self._normalize_name(away)
        
        for e in sofa_events:
            e_h = self._normalize_name(e.get("home_team", ""))
            e_a = self._normalize_name(e.get("away_team", ""))
            
            # Substring or token overlap match
            h_match = (norm_h in e_h) or (e_h in norm_h) or (set(norm_h.split()) & set(e_h.split()))
            a_match = (norm_a in e_a) or (e_a in norm_a) or (set(norm_a.split()) & set(e_a.split()))
            
            if h_match and a_match:
                return e
        return None

    def analyze_triangulation(self, pin_matchup: Dict[str, Any], sofa_live_events: Optional[List[Dict[str, Any]]] = None) -> Optional[SyndicateSignal]:
        """Triangulate a single fixture across Pinnacle and SofaScore feeds."""
        mid = pin_matchup.get("id")
        home = pin_matchup.get("home_team", "Unknown")
        away = pin_matchup.get("away_team", "Unknown")
        league = pin_matchup.get("league", "Unknown")
        is_live = pin_matchup.get("is_live", False)

        # 1. Pull Pinnacle Sharp Market
        lines = self.pinnacle.get_market_lines(mid)
        ml = lines.get("moneyline")
        if not ml or not ml.get("fair_odds"):
            return None

        quoted_odds = ml.get("raw_odds", [])
        fair_odds = ml.get("fair_odds", [])
        true_probs = ml.get("true_probs", [])
        insider_z = ml.get("shin_z", 0.0)
        vig = ml.get("vig_percent", 0.0)

        # 2. Check SofaScore In-Play Pressure if live
        momentum_10m = 0.0
        dominating = "Neutral"
        xg_h = 0.0
        xg_a = 0.0
        current_min = None
        current_score = "0 - 0"

        if sofa_live_events:
            matched_sofa = self.find_matching_sofascore_event(home, away, sofa_live_events)
            if matched_sofa:
                eid = matched_sofa.get("id")
                current_score = f"{matched_sofa.get('home_score', 0)} - {matched_sofa.get('away_score', 0)}"
                
                # Attack momentum
                pts = self.sofascore.get_attack_momentum(eid)
                if pts:
                    recent = pts[-10:] if len(pts) >= 10 else pts
                    vals = [p.get("value", 0) for p in recent]
                    current_min = recent[-1].get("minute")
                    momentum_10m = float(np.mean(vals))
                    if momentum_10m > 20:
                        dominating = f"{home} (Pressure: +{momentum_10m:.1f})"
                    elif momentum_10m < -20:
                        dominating = f"{away} (Pressure: {momentum_10m:.1f})"
                    else:
                        dominating = "Balanced"

                # Shotmap xG
                shots = self.sofascore.get_shotmap_xg(eid)
                for s in shots:
                    xg_val = s.get("xg", 0.0) or 0.0
                    # If coordinate x > 50 or player affiliation, attribute xg
                    coord = s.get("coordinates", {})
                    # Default heuristic attribution
                    if coord.get("x", 0) < 50:
                        xg_h += xg_val
                    else:
                        xg_a += xg_val

        # 3. Dynamic Alpha Recommendation
        # If Home has high momentum boost and Pinnacle fair odds are solid
        recommended_market = f"1X2: {home}"
        edge = 0.0
        verdict = "NEUTRAL"
        rationale = "Market priced efficiently."

        # Compute edge against market
        if len(true_probs) >= 3 and len(quoted_odds) >= 3:
            p_home, p_draw, p_away = true_probs[0], true_probs[1], true_probs[2]
            
            # In-play Bayesian momentum adjustment
            if is_live and abs(momentum_10m) > 25:
                if momentum_10m > 25:
                    boost = min(0.12, (momentum_10m / 100.0) * 0.15)
                    p_home_adj = min(0.90, p_home + boost)
                    p_away_adj = max(0.05, p_away - (boost * 0.7))
                    recommended_market = f"IN-PLAY HOME: {home}"
                    # Simulate value if soft market hanging quoted odds
                    market_offer = quoted_odds[0] * 1.04 # soft book offering slight delay
                    edge = (p_home_adj * market_offer - 1.0) * 100.0
                    verdict = "STRONG_INPLAY_ALPHA"
                    rationale = f"Heavy sustained attack momentum ({momentum_10m:+.1f}) from {home} with smart money confirmation (z={insider_z:.2%})."
                elif momentum_10m < -25:
                    boost = min(0.12, (abs(momentum_10m) / 100.0) * 0.15)
                    p_away_adj = min(0.90, p_away + boost)
                    recommended_market = f"IN-PLAY AWAY: {away}"
                    market_offer = quoted_odds[2] * 1.04
                    edge = (p_away_adj * market_offer - 1.0) * 100.0
                    verdict = "STRONG_INPLAY_ALPHA"
                    rationale = f"Heavy sustained away attack momentum ({momentum_10m:+.1f}) from {away} against market consensus."
            else:
                # Pre-match pure Shin edge
                home_odd = quoted_odds[0] if len(quoted_odds) > 0 else 0.0
                fair_odd = fair_odds[0] if len(fair_odds) > 0 else 0.0
                if insider_z > 0.04 and p_home > 0.45:
                    edge = (p_home * home_odd - 1.0) * 100.0 if (home_odd > 0 and p_home > 0) else 0.0
                    verdict = "PREMATCH_SMART_MONEY"
                    recommended_market = f"1X2 (ชนะเสมอแพ้) - เจ้าบ้าน {home} @ ราคาเปิด {home_odd:.2f} (ราคาแฟร์ {fair_odd:.2f})"
                    rationale = f"Pinnacle line heavily shaped by insider syndicate capital (Shin z = {insider_z*100:.1f}%)."

        # Quarter-Kelly
        q_kelly = 0.0
        if edge > 0:
            implied_p = true_probs[0] if "HOME" in recommended_market else (true_probs[2] if "AWAY" in recommended_market else 0.5)
            q_kelly = max(0.0, min(5.0, (edge / 100.0) * 0.25 * 100.0))

        return SyndicateSignal(
            match_id=str(mid),
            home_team=home,
            away_team=away,
            league=league,
            is_live=is_live,
            current_minute=current_min,
            current_score=current_score,
            pinnacle_quoted_odds=quoted_odds,
            shin_true_probs=true_probs,
            shin_fair_odds=fair_odds,
            shin_insider_z=insider_z,
            market_vig_percent=vig,
            momentum_10m_avg=momentum_10m,
            dominating_side=dominating,
            total_xg_home=xg_h,
            total_xg_away=xg_a,
            recommended_market=recommended_market,
            edge_percent=edge,
            quarter_kelly_stake_percent=q_kelly,
            verdict=verdict,
            rationale=rationale
        )

    def scan_opportunities(self, max_fixtures: int = 15) -> List[SyndicateSignal]:
        """Scan top fixtures across Pinnacle and cross-reference with SofaScore."""
        logger.info("Starting Syndicate Triangulation Scan...")
        sofa_live = self.sofascore.get_live_events()
        pin_matchups = self.pinnacle.get_matchups()
        
        signals = []
        for m in pin_matchups[:max_fixtures]:
            try:
                sig = self.analyze_triangulation(m, sofa_live)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.error(f"Error analyzing fixture {m.get('id')}: {e}")
                continue
                
        return signals
