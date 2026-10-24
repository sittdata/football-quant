import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fbpredict.providers.thscore_feed import get_thscore_live_data

matches = get_thscore_live_data(company_id=31)

# Check our target matches
targets = [
    ("ราปิด เวียนนา", "ออสเตรีย ลัสเตนัว"),
    ("กาญารี่", "โบโลญญ่า"),
    ("เอซีดี วีร์ตุส เอ็นเตลล่า", "มานโตวา"),
    ("แฮร์ธ่า เบอร์ลิน", "อาร์มีเนีย บีเลเฟลด์"),
    ("แบล็คพูล", "แมนส์ฟิลด์ ทาวน์"),
]

for home_target, away_target in targets:
    found = [m for m in matches if home_target in m.home_team or away_target in m.away_team]
    for m in found:
        utc_dt = datetime.strptime(m.kickoff_at, "%Y-%m-%d %H:%M:%S")
        thai_dt = utc_dt + timedelta(hours=7)
        print(f"Match: [{m.league}] {m.home_team} vs {m.away_team}")
        time_str = thai_dt.strftime('%H:%M')
        date_str = thai_dt.strftime('%d/%m/%Y')
        print(f"   UTC: {m.kickoff_at} --> เวลาไทย: {time_str} น. ({date_str}) | Status: {m.status}")
