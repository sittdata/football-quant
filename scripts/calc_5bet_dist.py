import sys
from pathlib import Path
import pandas as pd
import numpy as np

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
       ah_open, ah_fav_price, ah_dog_price, ou_line_open, ou_over_price, ou_under_price,
       pool_fav_open, pool_dog_open, draw_ft_odds, draw_ht_odds,
       home_goals, away_goals
FROM matches
WHERE kickoff_at >= '2026-09-01'
  AND home_goals IS NOT NULL AND away_goals IS NOT NULL
  AND ah_open IS NOT NULL AND ou_line_open IS NOT NULL
ORDER BY kickoff_at ASC
"""
df = pd.read_sql_query(query, con)
con.close()

bundle = ModelBundle.load(ROOT_DIR / "models" / "multileague_v2.pkl", require_verified=True)
preds = bundle.predict_proba_df(df)

candidates = []
for idx, r in df.iterrows():
    p_fav = float(preds.loc[idx, 'p_fav_cover'])
    p_dog = 1.0 - p_fav
    p_over = float(preds.loc[idx, 'p_over'])
    p_under = 1.0 - p_over
    
    fair_fav, fair_dog = labels.fair_two_way(r['ah_fav_price'] or 1.90, r['ah_dog_price'] or 1.90)
    fair_over, fair_under = labels.fair_two_way(r['ou_over_price'] or 1.90, r['ou_under_price'] or 1.90)
    
    fav_scored = r['home_goals'] if r['fav_side'] == 'home' else r['away_goals']
    dog_scored = r['away_goals'] if r['fav_side'] == 'home' else r['home_goals']
    tot_goals = r['home_goals'] + r['away_goals']
    
    ah_unit = labels.ah_unit(fav_scored - dog_scored, r['ah_open'])
    ou_unit = labels.ou_unit(tot_goals, r['ou_line_open'])
    
    edges = [
        ("AH_fav", p_fav - fair_fav, ah_unit),
        ("AH_dog", p_dog - fair_dog, -ah_unit),
        ("OU_over", p_over - fair_over, ou_unit),
        ("OU_under", p_under - fair_under, -ou_unit),
    ]
    best = max(edges, key=lambda x: x[1])
    if best[1] >= 0.08: # Min Edge 8%
        candidates.append({
            'date': r['kickoff_at'][:10],
            'edge': best[1],
            'unit': best[2], # >0 win, <0 loss
            'win': 1 if best[2] > 0 else 0
        })

cand_df = pd.DataFrame(candidates)
print(f"Total qualified picks (Edge >= 8%): {len(cand_df)}")
print(f"Overall win rate: {cand_df['win'].mean()*100:.1f}%")

# Simulate rolling 5-match sessions (representing daily 5-bet flat staking)
sessions = []
for i in range(0, len(cand_df) - 4, 5):
    chunk = cand_df.iloc[i:i+5]
    n_wins = chunk['win'].sum()
    sessions.append(n_wins)

sess_series = pd.Series(sessions)
dist = sess_series.value_counts(normalize=True).sort_index(ascending=False) * 100
print("\nDistribution of Results in 5-Bet Flat Staking Sessions:")
for wins, pct in dist.items():
    print(f"  ถูก {wins}/5 คู่: {pct:.1f}% ของจำนวนวันทั้งหมด")
