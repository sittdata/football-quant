import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# 1. โหลดชุดข้อมูลฟุตบอล
d = pd.read_csv('football.csv')
d = d.drop_duplicates()
d = d.dropna()

# 2. แปลงผลการแข่งขันเป็นตัวเลข: 0 = ทีมเยือนชนะ, 1 = เสมอ, 2 = เจ้าบ้านชนะ
d['Result'] = d['Result'].map({'AwayWin': 0, 'Draw': 1, 'HomeWin': 2})

# 3. แยก Features (X) 6 คอลัมน์แรก และ Target (y) คอลัมน์สุดท้าย
X = d.iloc[:, 0:6].copy()
y = d.iloc[:, 6:7].copy()
X = X.to_numpy()
y = y.to_numpy()
X = MinMaxScaler().fit_transform(X)

# 4. แบ่งชุดข้อมูล Train 80% และ Test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
y_train = np.array(np.ravel(y_train), dtype=np.int64)
y_test = np.array(np.ravel(y_test), dtype=np.int64)

# 5. เทรนโมเดล XGBoost Classifier
save_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
).fit(X_train, y_train)

# 6. บันทึกโมเดลลงไฟล์ xgboostmodel.sav
pickle.dump(save_model, open("xgboostmodel.sav", 'wb'))
print("Model training completed. Saved to xgboostmodel.sav")
