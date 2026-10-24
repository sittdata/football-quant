import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Windows encoding
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
    
    # Status 0 = not started yet
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
        fav_side = 'home' if m.ah_line <= 0 else 'away'
        
        rows.append({
            'match_id': f'thscore:{m.match_id}',
            'provider': 'thscore_feed',
            'league': m.league,
            'season': '2026',
            'kickoff_at': m.kickoff_at,
            'home_team': m.home_team,
            'away_team': m.away_team,
            'fav_side': fav_side,
            'ah_open': float(m.ah_line),
            'ah_fav_price': h_p if fav_side == 'home' else a_p,
            'ah_dog_price': a_p if fav_side == 'home' else h_p,
            'ou_line_open': float(m.ou_line),
            'ou_over_price': o_p,
            'ou_under_price': u_p,
            'pool_fav_open': m.pool_home if fav_side == 'home' else m.pool_away,
            'pool_dog_open': m.pool_away if fav_side == 'home' else m.pool_home,
            'draw_ft_odds': m.pool_draw or 3.20,
            'draw_ht_odds': 2.10,
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
    major_keywords = [
        "พรีเมียร์", "ลาลีกา", "บุนเดส", "กัลโช่", "เซเรีย", "ลีกเอิง", "แชมเปี้ยนชิพ", 
        "เจลีก", "เนเธอร์แลนด์", "โปรตุเกส", "เบลเยียม", "สก็อต", "ยูฟ่า", "ไทย", "สเปน",
        "เยอรมัน", "อังกฤษ", "อิตาลี", "ฝรั่งเศส"
    ]
    
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
            (f"ต่อ {fav_team} (AH {r['ah_open']})", p_fav - fair_fav, p_fav, r['ah_fav_price']),
            (f"รอง {dog_team} (AH {abs(r['ah_open']) if r['ah_open']<0 else r['ah_open']})", p_dog - fair_dog, p_dog, r['ah_dog_price']),
            (f"สูง (Over {r['ou_line_open']})", p_over - fair_over, p_over, r['ou_over_price']),
            (f"ต่ำ (Under {r['ou_line_open']})", p_under - fair_under, p_under, r['ou_under_price']),
        ]
        best = max(mkts, key=lambda x: x[1])
        
        is_major = any(k in r['league'] for k in major_keywords)
        
        results.append({
            'league': r['league'],
            'home': r['home_team'],
            'away': r['away_team'],
            'kickoff': r['kickoff_at'],
            'best_market': best[0],
            'edge': best[1],
            'prob': best[2],
            'price': best[3],
            'is_major': is_major
        })
        
    res_df = pd.DataFrame(results).sort_values(by=['is_major', 'edge'], ascending=[False, False])
    
    print("\n[TOP 15 CANDIDATES FOR FLAT STAKING]")
    for i, row in res_df.head(15).reset_index(drop=True).iterrows():
        badge = "⭐ Major" if row['is_major'] else "📌 Normal"
        print(f"{i+1}. {badge} [{row['league']}] {row['home']} vs {row['away']}")
        print(f"   Kickoff: {row['kickoff']} | แนะนำ: {row['best_market']}")
        print(f"   Win Prob: {row['prob']*100:.1f}% | Edge: +{row['edge']*100:.1f}% | Odds: {row['price']:.2f}")

if __name__ == "__main__":
    main()
