import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from fbpredict.providers.thscore_feed import get_thscore_live_data
from fbpredict.model import ModelBundle
from fbpredict import labels

matches = get_thscore_live_data(company_id=31)

targets = [
    ("แฮร์ธ่า เบอร์ลิน", "อาร์มีเนีย บีเลเฟลด์"),
    ("กาญารี่", "โบโลญญ่า"),
    ("เอซีดี วีร์ตุส เอ็นเตลล่า", "มานโตวา"),
    ("แบล็คพูล", "แมนส์ฟิลด์ ทาวน์"),
    ("ราปิด เวียนนา", "ออสเตรีย ลัสเตนัว"),
]

for h_pat, a_pat in targets:
    found = [m for m in matches if h_pat in m.home_team or a_pat in m.away_team]
    if not found:
        continue
    m = found[0]
    print(f"\n==========================================")
    print(f"Match: {m.home_team} (เหย้า) vs {m.away_team} (เยือน)")
    print(f"Raw Thscore AH Line: {m.ah_line}")
    # In Thscore convention:
    # ah_line > 0 means Home is Fav (-line)
    # ah_line < 0 means Away is Fav (-abs(line))
    # ah_line == 0 means Pick'em
    if m.ah_line is not None:
        if m.ah_line > 0:
            print(f"-> ความหมายจริง: เจ้าบ้าน [{m.home_team}] เป็นทีมต่อ (-{m.ah_line}) | เยือน [{m.away_team}] เป็นทีมรอง (+{m.ah_line})")
        elif m.ah_line < 0:
            print(f"-> ความหมายจริง: ทีมเยือน [{m.away_team}] เป็นทีมต่อ (-{abs(m.ah_line)}) | เจ้าบ้าน [{m.home_team}] เป็นทีมรอง (+{abs(m.ah_line)})")
        else:
            print(f"-> ราคาเสมอ (0.0)")
