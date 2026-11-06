import numpy as np
import pandas as pd

# =========================================================
# 1. โหลดข้อมูลดิบ (Load Raw Data)
# =========================================================
print("Loading raw data from raw_football.csv...")
df = pd.read_csv('raw_football.csv')
print(f"Initial rows: {len(df):,}")

# =========================================================
# 2. การทำความสะอาดข้อมูล (Data Cleaning)
# =========================================================
# 2.1 ลบข้อมูลที่ซ้ำซ้อนกัน (Deduplication)
df = df.drop_duplicates(subset=['match_id'])

# 2.2 จัดการค่าว่าง (Handling Missing Values)
# กรองเอาเฉพาะแมตช์ที่แข่งเสร็จแล้ว (มีผลสกอร์)
df = df.dropna(subset=['home_goals', 'away_goals'])

# เติมค่าว่างของราคาต่อรองด้วยค่ากลางของตลาด
df['ah_open'] = df['ah_open'].fillna(0.0)
df['ah_fav_price'] = df['ah_fav_price'].fillna(1.90)

# 2.3 แปลงชนิดข้อมูลวันที่ และเรียงลำดับเวลาจริง (Temporal Ordering)
df['kickoff_at'] = pd.to_datetime(df['kickoff_at'], errors='coerce')
df = df.sort_values('kickoff_at').reset_index(drop=True)

# =========================================================
# 3. การแปลงรูปข้อมูล (Data Transformation & Feature Engineering)
# =========================================================
# 3.1 คำนวณสถิติ xG เกมรุก-เกมรับ ย้อนหลัง 5 นัด (Rolling Average)
# ใส่ .shift(1) เพื่อป้องกันไม่ให้ข้อมูลผลการแข่งในอนาคตรั่วไหล (Zero Data Leakage)
h_rows = pd.DataFrame({'idx': df.index, 'team': df['home_team'], 'date': df['kickoff_at'], 'xg_f': df['home_xg'], 'xg_a': df['away_xg'], 'is_home': 1})
a_rows = pd.DataFrame({'idx': df.index, 'team': df['away_team'], 'date': df['kickoff_at'], 'xg_f': df['away_xg'], 'xg_a': df['home_xg'], 'is_home': 0})
t_logs = pd.concat([h_rows, a_rows], ignore_index=True).sort_values(['team', 'date']).reset_index(drop=True)

grouped = t_logs.groupby('team')
t_logs['s_xg_f'] = grouped['xg_f'].shift(1)
t_logs['s_xg_a'] = grouped['xg_a'].shift(1)

roll = t_logs.groupby('team')[['s_xg_f', 's_xg_a']].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
t_logs['xg_f_avg'] = roll['s_xg_f']
t_logs['xg_a_avg'] = roll['s_xg_a']

h_feats = t_logs[t_logs['is_home'] == 1].set_index('idx')
a_feats = t_logs[t_logs['is_home'] == 0].set_index('idx')

# 3.2 สร้างฟีเจอร์สำหรับเข้าโมเดล (6 คุณลักษณะหลัก)
df['Home_xG'] = h_feats['xg_f_avg'].fillna(1.5).round(2)
df['Away_xG'] = a_feats['xg_f_avg'].fillna(1.2).round(2)
df['Home_xGA'] = h_feats['xg_a_avg'].fillna(1.2).round(2)
df['Away_xGA'] = a_feats['xg_a_avg'].fillna(1.5).round(2)
df['Handicap'] = df['ah_open'].round(2)
df['Odds_Fav'] = df['ah_fav_price'].round(2)

# 3.3 สร้างคลาสเป้าหมาย (Target Encoding): AwayWin, Draw, HomeWin
df['Result'] = np.where(df['home_goals'] > df['away_goals'], 'HomeWin',
               np.where(df['home_goals'] == df['away_goals'], 'Draw', 'AwayWin'))

# =========================================================
# 4. บันทึกชุดข้อมูลสะอาดลง football.csv
# =========================================================
features_and_target = ['Home_xG', 'Away_xG', 'Home_xGA', 'Away_xGA', 'Handicap', 'Odds_Fav', 'Result']
clean_df = df[features_and_target].dropna()

clean_df.to_csv('football.csv', index=False)
print("=" * 60)
print("Data Cleaning & Transformation completed successfully!")
print(f"- Clean dataset saved to: football.csv")
print(f"- Usable match rows: {len(clean_df):,}")
print(f"- Columns extracted: {list(clean_df.columns)}")
print("=" * 60)
