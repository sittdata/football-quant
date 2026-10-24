import sys
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fbpredict.providers.thscore_feed import get_thscore_live_data
from fbpredict.model import ModelBundle
from fbpredict import labels

matches = get_thscore_live_data(company_id=31)

big_games = [
    ("อาร์เซน่อล", "เอฟเวอร์ตัน"),
    ("เชลซี", "ท็อตแน่ม ฮ็อทสเปอร์"),
    ("โคเวนทรี ซิตี้", "ฟูแล่ม"),
]

bundle = ModelBundle.load(ROOT_DIR / "models" / "multileague_v2.pkl", require_verified=True)

for h, a in big_games:
    found = [m for m in matches if h in m.home_team and a in m.away_team]
    if not found:
        continue
    m = found[0]
    
    # In Thscore: ah_line > 0 means Home fav, ah_line < 0 means Away fav
    fav_side = 'home' if m.ah_line >= 0 else 'away'
    ah_fav = -abs(float(m.ah_line))
    
    row = {
        'match_id': f'thscore:{m.match_id}', 'provider': 'thscore_feed', 'league': m.league,
        'season': '2026', 'kickoff_at': m.kickoff_at, 'home_team': m.home_team, 'away_team': m.away_team,
        'fav_side': fav_side, 'ah_open': ah_fav,
        'ah_fav_price': 1.90, 'ah_dog_price': 1.90,
        'ou_line_open': float(m.ou_line), 'ou_over_price': 1.90, 'ou_under_price': 1.90,
        'pool_fav_open': 1.60, 'pool_dog_open': 4.50, 'draw_ft_odds': 3.80, 'draw_ht_odds': 2.10,
        'home_goals': None, 'away_goals': None, 'ht_home': None, 'ht_away': None,
        'home_corner': None, 'away_corner': None, 'home_red': None, 'away_red': None,
        'home_yellow': None, 'away_yellow': None,
    }
    df = pd.DataFrame([row])
    preds = bundle.predict_proba_df(df)
    
    p_fav = float(preds.loc[0, 'p_fav_cover'])
    p_dog = 1.0 - p_fav
    p_over = float(preds.loc[0, 'p_over'])
    p_under = 1.0 - p_over
    
    fav_team = m.home_team if fav_side == 'home' else m.away_team
    dog_team = m.away_team if fav_side == 'home' else m.home_team
    
    print(f"\nMatch: {m.home_team} vs {m.away_team} (AH Line: {m.ah_line}, OU Line: {m.ou_line})")
    print(f"   Fav ({fav_team} -{abs(m.ah_line)}): {p_fav*100:.1f}%")
    print(f"   Dog ({dog_team} +{abs(m.ah_line)}): {p_dog*100:.1f}%")
    print(f"   Over ({m.ou_line}): {p_over*100:.1f}%")
    print(f"   Under ({m.ou_line}): {p_under*100:.1f}%")
