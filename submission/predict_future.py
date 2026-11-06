import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. โหลดข้อมูลเดิมเพื่อใช้สเกลค่า Min-Max ให้เท่ากับตอนเทรน
d = pd.read_csv('football.csv').dropna()
X_raw = d.iloc[:, 0:6].to_numpy()
scaler = MinMaxScaler().fit(X_raw)

# 2. โหลดโมเดลที่ฝึกสอนไว้
load_model = pickle.load(open('xgboostmodel.sav', 'rb'))

def predict_upcoming_match(home_team, away_team, home_xg, away_xg, home_xga, away_xga, handicap, odds_fav):
    """
    ฟังก์ชันทำนายผลการแข่งขันในอนาคต:
    - home_xg: คุณภาพโอกาสยิงเฉลี่ย 5 นัดหลังสุดของเจ้าบ้าน
    - away_xg: คุณภาพโอกาสยิงเฉลี่ย 5 นัดหลังสุดของทีมเยือน
    - home_xga: โอกาสเสียประตูเฉลี่ยของเจ้าบ้าน
    - away_xga: โอกาสเสียประตูเฉลี่ยของทีมเยือน
    - handicap: ราคาต่อรอง (เช่น -0.75 คือเจ้าบ้านต่อครึ่งควบลูก)
    - odds_fav: ค่าน้ำทีมต่อ (เช่น 1.95)
    """
    row = np.array([[home_xg, away_xg, home_xga, away_xga, handicap, odds_fav]])
    row_scaled = scaler.transform(row)

    probs = load_model.predict_proba(row_scaled)[0]
    pred = load_model.predict(row_scaled)[0]

    p_away = probs[0] * 100.0
    p_draw = probs[1] * 100.0
    p_home = probs[2] * 100.0

    labels = [f"AwayWin ({away_team} Win)", "Draw", f"HomeWin ({home_team} Win)"]

    print("=" * 65)
    print(f"Upcoming Match Prediction: {home_team} (Home) vs {away_team} (Away)")
    print(f"Handicap Line: {handicap} | Odds: {odds_fav}")
    print("-" * 65)
    print(f"  - {away_team:<18} Win : {p_away:.1f}%")
    print(f"  - {'Draw':<18}     : {p_draw:.1f}%")
    print(f"  - {home_team:<18} Win : {p_home:.1f}%")
    print("-" * 65)
    print(f"  Predicted Winner      : {labels[pred]}")
    print("=" * 65 + "\n")

# =========================================================
# ทดสอบทำนายคู่บอลในอนาคต (ตัวอย่าง 2 คู่)
# =========================================================
if __name__ == '__main__':
    # คู่ที่ 1: ลิเวอร์พูล พบ แมนเชสเตอร์ ยูไนเต็ด
    predict_upcoming_match(
        home_team="Liverpool",
        away_team="Man United",
        home_xg=1.34,
        away_xg=1.17,
        home_xga=1.13,
        away_xga=1.77,
        handicap=-1.0,
        odds_fav=1.95
    )

    # คู่ที่ 2: อาร์เซน่อล พบ เชลซี
    predict_upcoming_match(
        home_team="Arsenal",
        away_team="Chelsea",
        home_xg=1.75,
        away_xg=0.98,
        home_xga=1.37,
        away_xga=1.55,
        handicap=-0.5,
        odds_fav=1.90
    )
