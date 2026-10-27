"""
Standalone CLI Runner for the Syndicate Quant Alpha Engine.
Runs completely isolated from the legacy fbpredict model.

Usage:
  python run_syndicate_scanner.py
  python run_syndicate_scanner.py --limit 10
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
from fbpredict.syndicate_engine import SyndicateComboEngine

def main():
    parser = argparse.ArgumentParser(description="Pinnacle + SofaScore Syndicate Alpha Scanner")
    parser.add_argument("--limit", type=int, default=8, help="Number of fixtures to scan")
    args = parser.parse_args()

    print("=" * 85)
    print(">>> [PINNACLE + SOFASCORE + SHIN] SYNDICATE QUANT ALPHA SCANNER <<<")
    print("     (Standalone Isolated Engine - Zero Impact on Core Legacy Model)")
    print("=" * 85)

    engine = SyndicateComboEngine()
    print(f"\n[*] Scanning top {args.limit} global fixtures across Pinnacle & SofaScore...")
    signals = engine.scan_opportunities(max_fixtures=args.limit)

    print(f"\n[+] Scan Complete! Extracted and Analyzed {len(signals)} Active Matchups:\n")
    print("-" * 85)

    for idx, sig in enumerate(signals):
        status_tag = f"[LIVE {sig.current_minute or 0}']" if sig.is_live else "[UPCOMING]"
        print(f"[{idx+1:2d}] {status_tag:<11} {sig.home_team:>20} vs {sig.away_team:<20} | {sig.league}")
        print(f"     Pinnacle Raw 1X2: {[round(o, 2) for o in sig.pinnacle_quoted_odds]} | Vig: {sig.market_vig_percent:.1f}% | Insider z: {sig.shin_insider_z*100:.1f}%")
        print(f"     Shin True Fair  : {[round(o, 2) for o in sig.shin_fair_odds]} (Prob: {[f'{p*100:.1f}%' for p in sig.shin_true_probs]})")
        
        if sig.is_live:
            print(f"     Field Momentum  : {sig.dominating_side} (10m Avg: {sig.momentum_10m_avg:+.1f}) | Score: {sig.current_score}")
        
        if sig.verdict != "NEUTRAL":
            print(f"     🔥 ACTIONABLE ALPHA : {sig.verdict} -> {sig.recommended_market}")
            print(f"        Edge (+EV)       : +{sig.edge_percent:.2f}% | Recommended Quarter-Kelly: {sig.quarter_kelly_stake_percent:.2f}%")
            print(f"        Rationale        : {sig.rationale}")
        else:
            print(f"     Verdict         : {sig.verdict} (Market fully priced)")
        print("-" * 85)

    print("\n[SUCCESS] Syndicate Alpha Scan completed cleanly!")

if __name__ == "__main__":
    main()
