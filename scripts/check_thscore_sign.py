import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from fbpredict.providers.thscore_feed import get_thscore_live_data

matches = get_thscore_live_data(company_id=31)
# Find matches with known strong teams
for m in matches:
    if any(k in m.home_team for k in ["แมนเชสเตอร์", "ชักตาร์", "บาเยิร์น", "บาร์เซโลน่า", "ลิเวอร์พูล"]):
        print(f"Match: {m.home_team} vs {m.away_team} | ah_line: {m.ah_line} | ah_home: {m.ah_home_price} | ah_away: {m.ah_away_price}")
    if any(k in m.away_team for k in ["แมนเชสเตอร์", "ชักตาร์", "บาเยิร์น", "บาร์เซโลน่า", "ลิเวอร์พูล"]):
        print(f"Match: {m.home_team} vs {m.away_team} | ah_line: {m.ah_line} | ah_home: {m.ah_home_price} | ah_away: {m.ah_away_price}")
