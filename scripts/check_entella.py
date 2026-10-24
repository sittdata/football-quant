import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from fbpredict.providers.thscore_feed import get_thscore_live_data

matches = get_thscore_live_data(company_id=31)
m = [x for x in matches if 'เอ็นเตลล่า' in x.home_team or 'เอนเตลล่า' in x.home_team or 'มานโตวา' in x.away_team or 'มันโตวา' in x.away_team]
for x in m:
    print('Home:', x.home_team, 'Away:', x.away_team)
    print('ah_line:', x.ah_line, 'ah_home_price:', x.ah_home_price, 'ah_away_price:', x.ah_away_price)
    print('ou_line:', x.ou_line, 'ou_over_price:', x.ou_over_price, 'ou_under_price:', x.ou_under_price)
