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

def main():
    matches = get_thscore_live_data(company_id=31)
    
    # Upcoming matches that have not kicked off yet
    upcoming = [m for m in matches if m.status == 0 and m.ah_line is not None and m.ou_line is not None]
    
    rows = []
    for m in upcoming:
        def to_dec(p):
            if p is None: return 1.90
            return (p + 1.0) if p <= 1.5 else p
        h_p = to_dec(m.ah_home_price)
        a_p = to_dec(m.ah_away_price)
        o_p = to_dec(m.ou_over_price)
        u_p = to_dec(m.ou_under_price)
        
        # Correct Thscore convention:
        if m.ah_line > 0:
            fav_side = 'home'
            ah_fav = -float(m.ah_line)
            fav_price = h_p
            dog_price = a_p
        elif m.ah_line < 0:
            fav_side = 'away'
            ah_fav = float(m.ah_line)
            fav_price = a_p
            dog_price = h_p
        else:
            fav_side = 'home'
            ah_fav = 0.0
            fav_price = h_p
            dog_price = a_p
            
        rows.append({
            'match_id': f'thscore:{m.match_id}',
            'provider': 'thscore_feed',
            'league': m.league,
            'season': '2026',
            'kickoff_at': m.kickoff_at,
            'home_team': m.home_team,
            'away_team': m.away_team,
            'fav_side': fav_side,
            'ah_open': ah_fav,
            'ah_fav_price': fav_price,
            'ah_dog_price': dog_price,
            'ou_line_open': float(m.ou_line),
            'ou_over_price': o_p,
            'ou_under_price': u_p,
            'pool_fav_open': 1.90, 'pool_dog_open': 3.60,
            'draw_ft_odds': 3.20, 'draw_ht_odds': 2.10,
            'home_goals': None, 'away_goals': None,
            'ht_home': None, 'ht_away': None,
            'home_corner': None, 'away_corner': None,
            'home_red': None, 'away_red': None,
            'home_yellow': None, 'away_yellow': None,
        })
        
    df = pd.DataFrame(rows)
    bundle_path = ROOT_DIR / "models" / "multileague_v2.pkl"
    bundle = ModelBundle.load(bundle_path, require_verified=True)
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
        ah_line_val = abs(r['ah_open'])
        
        mkts = [
            (f"ต่อ {fav_team} (-{ah_line_val:.2f})", p_fav - fair_fav, p_fav, r['ah_fav_price']),
            (f"รอง {dog_team} (+{ah_line_val:.2f})", p_dog - fair_dog, p_dog, r['ah_dog_price']),
            (f"สูง (Over {r['ou_line_open']})", p_over - fair_over, p_over, r['ou_over_price']),
            (f"ต่ำ (Under {r['ou_line_open']})", p_under - fair_under, p_under, r['ou_under_price']),
        ]
        best = max(mkts, key=lambda x: x[1])
        
        utc_dt = datetime.strptime(r['kickoff_at'], "%Y-%m-%d %H:%M:%S")
        thai_dt = utc_dt + timedelta(hours=7)
        time_str = thai_dt.strftime("%H:%M") + " น."
        
        if best[1] >= 0.08: # Positive Edge >= 8%
            results.append({
                'league': r['league'],
                'home': r['home_team'],
                'away': r['away_team'],
                'kickoff_dt': thai_dt,
                'kickoff_thai': time_str,
                'pick': best[0],
                'edge': best[1],
                'prob': best[2],
                'odds': best[3],
            })
            
    res_df = pd.DataFrame(results).sort_values(by=['kickoff_dt', 'edge'], ascending=[True, False])
    
    print(f"Total Qualified Matches with Edge >= 8%: {len(res_df)}")
    
    # Let's inspect the distribution across time slots
    for i, r in res_df.iterrows():
        print(f"[{r['kickoff_thai']}] [{r['league']}] {r['home']} vs {r['away']} | แนะนำ: {r['pick']} | Win Prob: {r['prob']*100:.1f}% | Edge: +{r['edge']*100:.1f}% | Odds: {r['odds']:.2f}")

if __name__ == "__main__":
    main()
