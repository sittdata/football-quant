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
from fbpredict.model import ModelBundle
from fbpredict import labels

matches = get_thscore_live_data(company_id=31)
upcoming = [m for m in matches if m.status == 0 and m.ah_line is not None and m.ou_line is not None]

rows = []
for m in upcoming:
    def to_dec(p): return (p + 1.0) if p and p <= 1.5 else (p or 1.90)
    h_p, a_p = to_dec(m.ah_home_price), to_dec(m.ah_away_price)
    o_p, u_p = to_dec(m.ou_over_price), to_dec(m.ou_under_price)
    fav_side = 'home' if m.ah_line >= 0 else 'away'
    rows.append({
        'match_id': f'thscore:{m.match_id}', 'league': m.league, 'home_team': m.home_team, 'away_team': m.away_team,
        'kickoff_at': m.kickoff_at, 'fav_side': fav_side, 'ah_open': -abs(float(m.ah_line)),
        'ah_fav_price': h_p if fav_side == 'home' else a_p, 'ah_dog_price': a_p if fav_side == 'home' else h_p,
        'ou_line_open': float(m.ou_line), 'ou_over_price': o_p, 'ou_under_price': u_p,
        'pool_fav_open': 1.90, 'pool_dog_open': 3.60, 'draw_ft_odds': 3.20, 'draw_ht_odds': 2.10,
        'home_goals': None, 'away_goals': None, 'ht_home': None, 'ht_away': None,
        'home_corner': None, 'away_corner': None, 'home_red': None, 'away_red': None, 'home_yellow': None, 'away_yellow': None,
    })

df = pd.DataFrame(rows)
bundle = ModelBundle.load(ROOT_DIR / "models" / "multileague_v2.pkl", require_verified=True)
preds = bundle.predict_proba_df(df)

tonight_picks = []
for idx, r in df.iterrows():
    p_fav = float(preds.loc[idx, 'p_fav_cover'])
    p_dog = 1.0 - p_fav
    p_over = float(preds.loc[idx, 'p_over'])
    p_under = 1.0 - p_over
    fair_fav, fair_dog = labels.fair_two_way(r['ah_fav_price'], r['ah_dog_price'])
    fair_over, fair_under = labels.fair_two_way(r['ou_over_price'], r['ou_under_price'])
    fav_team = r['home_team'] if r['fav_side'] == 'home' else r['away_team']
    dog_team = r['away_team'] if r['fav_side'] == 'home' else r['home_team']
    
    mkts = [
        (f"ต่อ {fav_team} (-{abs(r['ah_open']):.2f})", p_fav - fair_fav, p_fav, r['ah_fav_price']),
        (f"รอง {dog_team} (+{abs(r['ah_open']):.2f})", p_dog - fair_dog, p_dog, r['ah_dog_price']),
        (f"สูง Over {r['ou_line_open']}", p_over - fair_over, p_over, r['ou_over_price']),
        (f"ต่ำ Under {r['ou_line_open']}", p_under - fair_under, p_under, r['ou_under_price']),
    ]
    best = max(mkts, key=lambda x: x[1])
    
    utc_dt = datetime.strptime(r['kickoff_at'], '%Y-%m-%d %H:%M:%S')
    thai_dt = utc_dt + timedelta(hours=7)
    
    # Check tonight between 18:00 and 02:00
    if best[1] >= 0.08 and ((thai_dt.hour >= 18) or (thai_dt.hour < 2)):
        tonight_picks.append({
            'time': thai_dt.strftime('%H:%M') + " น.",
            'dt': thai_dt,
            'hour': thai_dt.hour,
            'league': r['league'],
            'home': r['home_team'],
            'away': r['away_team'],
            'pick': best[0],
            'prob': best[2],
            'edge': best[1],
            'odds': best[3],
        })

df_t = pd.DataFrame(tonight_picks).sort_values(by=['dt', 'edge'], ascending=[True, False])

# Group 1: 18:00 - 20:30
g1 = df_t[df_t['hour'].between(18, 20)]
print(f"=== [ช่วงที่ 1: หัวค่ำ 18:00 - 20:30 น.] (มี {len(g1)} คู่) ===")
for i, r in g1.reset_index(drop=True).iterrows():
    print(f"{i+1:2d}. [{r['time']}] [{r['league']}] {r['home']} vs {r['away']} | {r['pick']} | Win {r['prob']*100:.1f}% | Edge +{r['edge']*100:.1f}% | น้ำ {r['odds']:.2f}")

# Group 2: 21:00 - 22:30
g2 = df_t[df_t['hour'].between(21, 22)]
print(f"\n=== [ช่วงที่ 2: รอบดึก 21:00 - 22:30 น.] (มี {len(g2)} คู่) ===")
for i, r in g2.reset_index(drop=True).iterrows():
    print(f"{i+1:2d}. [{r['time']}] [{r['league']}] {r['home']} vs {r['away']} | {r['pick']} | Win {r['prob']*100:.1f}% | Edge +{r['edge']*100:.1f}% | น้ำ {r['odds']:.2f}")

# Group 3: 23:00 - 02:00
g3 = df_t[(df_t['hour'] >= 23) | (df_t['hour'] < 2)]
print(f"\n=== [ช่วงที่ 3: รอบดึกมาก 23:00 - 02:00 น.] (มี {len(g3)} คู่) ===")
for i, r in g3.reset_index(drop=True).iterrows():
    print(f"{i+1:2d}. [{r['time']}] [{r['league']}] {r['home']} vs {r['away']} | {r['pick']} | Win {r['prob']*100:.1f}% | Edge +{r['edge']*100:.1f}% | น้ำ {r['odds']:.2f}")
