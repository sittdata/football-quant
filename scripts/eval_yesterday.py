import sqlite3
import sys
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fbpredict.db import connect
from fbpredict.model import ModelBundle
from fbpredict import labels

con = connect(ROOT_DIR / "data" / "football.db")
query = """
SELECT match_id, league, season, kickoff_at, home_team, away_team, fav_side,
       ah_open, ah_fav_price, ah_dog_price, pool_fav_open, pool_dog_open,
       draw_ft_odds, draw_ht_odds, ou_line_open, ou_over_price, ou_under_price,
       home_goals, away_goals, ht_home, ht_away
FROM matches
WHERE kickoff_at LIKE '2026-10-23%'
"""
df = pd.read_sql_query(query, con)
print(f"Matches on 2026-10-23: {len(df)}")

bundle = ModelBundle.load(ROOT_DIR / "models" / "multileague_v2.pkl", require_verified=True)
preds = bundle.predict_proba_df(df)

results = []
for idx, r in df.iterrows():
    p_fav = float(preds.loc[idx, 'p_fav_cover'])
    p_dog = 1.0 - p_fav
    p_over = float(preds.loc[idx, 'p_over'])
    p_under = 1.0 - p_over
    
    fair_fav, fair_dog = labels.fair_two_way(r['ah_fav_price'], r['ah_dog_price'])
    fair_over, fair_under = labels.fair_two_way(r['ou_over_price'], r['ou_under_price'])
    
    fav_team = r['home_team'] if r['fav_side'] == 'home' else r['away_team']
    dog_team = r['away_team'] if r['fav_side'] == 'home' else r['home_team']
    
    # Calculate settlement
    h_g = r['home_goals']
    a_g = r['away_goals']
    tot_g = h_g + a_g
    fav_g = h_g if r['fav_side'] == 'home' else a_g
    dog_g = a_g if r['fav_side'] == 'home' else h_g
    
    diff = fav_g - dog_g
    ah_unit = labels.ah_unit(diff, r['ah_open'])
    ou_unit = labels.ou_unit(tot_g, r['ou_line_open'])
    
    mkts = [
        (f"ต่อ {fav_team} (AH {r['ah_open']})", p_fav - fair_fav, p_fav, r['ah_fav_price'], ah_unit),
        (f"รอง {dog_team} (AH {abs(r['ah_open'])})", p_dog - fair_dog, p_dog, r['ah_dog_price'], -ah_unit),
        (f"สูง (Over {r['ou_line_open']})", p_over - fair_over, p_over, r['ou_over_price'], ou_unit),
        (f"ต่ำ (Under {r['ou_line_open']})", p_under - fair_under, p_under, r['ou_under_price'], -ou_unit),
    ]
    best = max(mkts, key=lambda x: x[1])
    
    # Unit result: +1.0 full win, +0.5 half win, 0 push, -0.5 half loss, -1.0 full loss
    unit_res = best[4]
    if unit_res > 0:
        outcome = "WIN (เข้าเต็ม)" if unit_res == 1.0 else "HALF WIN (เข้าครึ่ง)"
        profit = (best[3] - 1.0) * 100 * unit_res
    elif unit_res < 0:
        outcome = "LOSS (ตายเต็ม)" if unit_res == -1.0 else "HALF LOSS (เสียครึ่ง)"
        profit = -100 * abs(unit_res)
    else:
        outcome = "PUSH (เสมอเจ๊า)"
        profit = 0.0
        
    results.append({
        'league': r['league'],
        'match': f"{r['home_team']} vs {r['away_team']}",
        'score': f"{int(h_g)} - {int(a_g)}",
        'pick': best[0],
        'edge': best[1],
        'prob': best[2],
        'odds': best[3],
        'unit': unit_res,
        'outcome': outcome,
        'profit': profit
    })

res_df = pd.DataFrame(results).sort_values(by='edge', ascending=False)
print("\n--- RESULTS OF YESTERDAY'S PICKS (2026-10-23) ---")
tot_profit = 0
for i, row in res_df.iterrows():
    print(f"{i+1}. [{row['league']}] {row['match']} | สกอร์จริง: {row['score']}")
    print(f"   ไม้แนะนำ: {row['pick']} | Odds: {row['odds']:.2f} | Edge: +{row['edge']*100:.1f}%")
    print(f"   ผลการแข่งขัน: >> {row['outcome']} << (กำไร/ขาดทุน: {row['profit']:+.1f} บ.)\n")
    tot_profit += row['profit']

print(f"Total Net Profit/Loss for 100 THB Flat Staking: {tot_profit:+.1f} THB")
