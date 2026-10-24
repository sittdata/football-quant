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
m_entella = [m for m in matches if 'เอ็นเตลล่า' in m.home_team or 'เอนเตลล่า' in m.home_team][0]

print("Match found:", m_entella.home_team, "vs", m_entella.away_team)
print("Thscore ah_line:", m_entella.ah_line)
print("Home price:", m_entella.ah_home_price, "Away price:", m_entella.ah_away_price)

# In real betting:
# If ah_line in Thscore is positive (0.25), Home gives handicap (-0.25) -> Home is FAVOURITE!
# Let's see what model predicts when Home is Fav vs Away is Fav.

# Case A: Home is Fav (Entella ต่อ -0.25)
row_correct = {
    'match_id': f'thscore:{m_entella.match_id}', 'provider': 'thscore_feed', 'league': m_entella.league,
    'season': '2026', 'kickoff_at': m_entella.kickoff_at, 'home_team': m_entella.home_team, 'away_team': m_entella.away_team,
    'fav_side': 'home', 'ah_open': -0.25,
    'ah_fav_price': 2.08, # 1.08 HK = 2.08 Dec
    'ah_dog_price': 1.82,
    'ou_line_open': 2.25, 'ou_over_price': 1.90, 'ou_under_price': 1.90,
    'pool_fav_open': 2.30, 'pool_dog_open': 3.10, 'draw_ft_odds': 3.00, 'draw_ht_odds': 2.00,
    'home_goals': None, 'away_goals': None, 'ht_home': None, 'ht_away': None,
    'home_corner': None, 'away_corner': None, 'home_red': None, 'away_red': None,
    'home_yellow': None, 'away_yellow': None,
}

bundle = ModelBundle.load(ROOT_DIR / "models" / "multileague_v2.pkl", require_verified=True)
df_correct = pd.DataFrame([row_correct])
preds_correct = bundle.predict_proba_df(df_correct)
print("\n--- Correct Model Inference (Entella is Fav -0.25) ---")
print("p_fav_cover (โอกาส เอนเตลล่า ชนะราคาต่อ -0.25):", float(preds_correct.loc[0, 'p_fav_cover']))
print("p_dog_cover (โอกาส มันโตวา ชนะราคารอง +0.25):", 1.0 - float(preds_correct.loc[0, 'p_fav_cover']))
print("p_over:", float(preds_correct.loc[0, 'p_over']))
print("p_under:", 1.0 - float(preds_correct.loc[0, 'p_over']))
