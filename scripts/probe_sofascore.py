"""
SofaScore Live Quantitative Extraction Engine.
Reverse-Engineered via TLS Profile (safari184) + DoH (Cloudflare DNS-over-HTTPS)
Bypasses both Cloudflare JA3/JA4 WAF challenges and ISP DNS Poisoning.

Extracts:
1. Live Football Events across all leagues worldwide
2. Real-Time Minute-by-Minute Attack Momentum Graph (-100 to +100)
3. Shotmap with Coordinates (x, y), body parts, and Expected Goals (xG)
4. Live In-Play / Pre-Match Odds & Asian Handicap lines
5. Live In-Play Dangerous Attacks, Possession & Team Statistics
"""

from curl_cffi import requests
import json
import time
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_URL = "https://www.sofascore.com/api/v1"
DOH_URL = "https://cloudflare-dns.com/dns-query"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com"
}

def extract_sofascore():
    session = requests.Session(impersonate="safari184")
    session.headers.update(HEADERS)

    print("=" * 75)
    print(">>> [SOFASCORE REVERSE-ENGINEERED QUANT EXTRACTION ENGINE] <<<")
    print("=" * 75)

    # 1. Fetch all Live Matches
    print("\n[*] 1. Fetching Live Matches worldwide...")
    live_url = f"{BASE_URL}/sport/football/events/live"
    r = session.get(live_url, doh_url=DOH_URL, timeout=12)
    if r.status_code != 200:
        print(f"[-] Failed with status {r.status_code}: {r.text[:200]}")
        return

    data = r.json()
    events = data.get("events", [])
    print(f"[+] SUCCESS! Currently Live Matches: {len(events)} matches\n")

    selected_event = None
    selected_event_has_momentum = None

    for idx, e in enumerate(events[:12]):
        eid = e.get("id")
        home = e.get("homeTeam", {}).get("name", "Unknown")
        away = e.get("awayTeam", {}).get("name", "Unknown")
        hs = e.get("homeScore", {}).get("current", 0)
        as_ = e.get("awayScore", {}).get("current", 0)
        league = e.get("tournament", {}).get("name", "Unknown")
        status_desc = e.get("status", {}).get("description", "Unknown")
        print(f"  [{idx+1:2d}] ID {eid:<9} | {home:>20} {hs} - {as_} {away:<20} | {status_desc:<9} | {league}")
        
    selected_event = None
    print("  [Scanning for live match with rich Attack Momentum & In-play tracking...]")
    for e in events:
        eid = e.get("id")
        rg = session.get(f"{BASE_URL}/event/{eid}/graph", doh_url=DOH_URL, timeout=4)
        if rg.status_code == 200 and len(rg.json().get("graphPoints", [])) > 0:
            selected_event = e
            break

    if not selected_event and events:
        selected_event = events[0]

    if not selected_event:
        print("[-] No live events available right now.")
        return

    eid = selected_event.get("id")
    home = selected_event.get("homeTeam", {}).get("name")
    away = selected_event.get("awayTeam", {}).get("name")
    tournament = selected_event.get("tournament", {}).get("name")
    print("\n" + "=" * 75)
    print(f"[TARGET MATCH DEEP-DIVE]: {home} vs {away} (Event ID: {eid}) | {tournament}")
    print("=" * 75)

    # 2. Extract Attack Momentum Graph
    print("\n[*] 2. Extracting Live Attack Momentum Graph (Minute-by-minute)...")
    graph_url = f"{BASE_URL}/event/{eid}/graph"
    rg = session.get(graph_url, doh_url=DOH_URL, timeout=10)
    if rg.status_code == 200:
        pts = rg.json().get("graphPoints", [])
        print(f"[+] Total Momentum Minutes Tracked: {len(pts)} minutes")
        if pts:
            print("   Recent Pressure Timeline (Last 10 recorded minutes):")
            for pt in pts[-10:]:
                minute = pt.get("minute")
                val = pt.get("value")
                if val > 0:
                    side = f"{home[:14]} Dominating"
                    bar = "[HOME] " + ("#" * min(abs(val) // 6, 15))
                elif val < 0:
                    side = f"{away[:14]} Dominating"
                    bar = "[AWAY] " + ("#" * min(abs(val) // 6, 15))
                else:
                    side = "Neutral Balance"
                    bar = "[---]"
                print(f"    Min {minute:2d}' : Value {val:+3d} | {side:<24} | {bar}")
    else:
        print(f"[-] Graph not available for this event (Status {rg.status_code})")

    # 3. Extract Match Statistics (Dangerous attacks, possession, corners, etc.)
    print("\n[*] 3. Extracting Detailed Live In-Play Match Statistics...")
    stats_url = f"{BASE_URL}/event/{eid}/statistics"
    rst = session.get(stats_url, doh_url=DOH_URL, timeout=10)
    if rst.status_code == 200:
        periods = rst.json().get("statistics", [])
        for period in periods:
            pname = period.get("period")
            groups = period.get("groups", [])
            print(f"[+] Period Statistics: {pname}")
            for g in groups:
                gname = g.get("groupName")
                items = g.get("statisticsItems", [])
                print(f"    --- {gname} ---")
                for it in items[:6]:
                    name = it.get("name")
                    h_val = it.get("home")
                    a_val = it.get("away")
                    print(f"      {h_val:>8}  |  {name:<26}  |  {a_val:<8}")
            break # Show primary period
    else:
        print(f"[-] Statistics not yet published for this tier (Status {rst.status_code})")

    # 4. Extract Shotmap & Expected Goals (xG)
    print("\n[*] 4. Extracting Live Shotmap & xG Coordinates...")
    shot_url = f"{BASE_URL}/event/{eid}/shotmap"
    rs = session.get(shot_url, doh_url=DOH_URL, timeout=10)
    if rs.status_code == 200:
        shots = rs.json().get("shotmap", [])
        print(f"[+] Total Shots Tracked: {len(shots)}")
        for s in shots[:8]:
            pname = s.get("player", {}).get("name", "Unknown")
            shot_type = s.get("shotType")
            xg = s.get("xg")
            body = s.get("bodyPart")
            time_m = s.get("time")
            coord = s.get("playerCoordinates")
            print(f"    {time_m:2d}' Shot by {pname:<18} ({body:<9}) | Outcome: {shot_type:<10} | xG: {xg} | Pitch(x,y): {coord}")
    else:
        print(f"[-] Shotmap / xG not available for this specific tier (Status {rs.status_code})")

    # 5. Extract Real-Time Bookmaker Odds & Market Lines
    print("\n[*] 5. Extracting Real-Time Bookmaker Odds & Market Probabilities...")
    odds_url = f"{BASE_URL}/event/{eid}/odds/1/all"
    ro = session.get(odds_url, doh_url=DOH_URL, timeout=10)
    if ro.status_code == 200:
        markets = ro.json().get("markets", [])
        print(f"[+] Available Odds Markets: {len(markets)}")
        for m in markets[:5]:
            mname = m.get("marketName")
            choices = m.get("choices", [])
            choice_str = " | ".join([f"{c.get('name')}: {c.get('fractionalValue', c.get('initialFractionalValue', c.get('decimalValue', 'N/A')))}" for c in choices])
            print(f"    Market '{mname}': {choice_str}")
    else:
        print(f"[-] Odds not returned (Status {ro.status_code})")

    print("\n" + "=" * 75)
    print("[SUCCESS] SofaScore In-Play Extraction Pipeline completed cleanly!")
    print("=" * 75)

if __name__ == "__main__":
    extract_sofascore()
