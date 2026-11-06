import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. โหลดชุดข้อมูลฟุตบอล
d = pd.read_csv('football.csv')
d = d.drop_duplicates()
d = d.dropna()

# 2. แปลงผลการแข่งขันเป็นตัวเลข: 0 = ทีมเยือนชนะ, 1 = เสมอ, 2 = เจ้าบ้านชนะ
d['Result'] = d['Result'].map({'AwayWin': 0, 'Draw': 1, 'HomeWin': 2})

# 3. แยก Features (X) และ Target (y)
X = d.iloc[:, 0:6].copy()
y = d.iloc[:, 6:7].copy()
X = X.to_numpy()
y = y.to_numpy()
X = MinMaxScaler().fit_transform(X)

# 4. แบ่งชุดข้อมูล Train 80% และ Test 20% (ใช้ random_state เดียวกัน)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
y_train = np.array(np.ravel(y_train), dtype=np.int64)
y_test = np.array(np.ravel(y_test), dtype=np.int64)

# 5. โหลดโมเดลที่บันทึกไว้
load_model = pickle.load(open("xgboostmodel.sav", 'rb'))

# 6. วนลูปทำนายผลลัพธ์ของแต่ละแถวในชุด Test พร้อมแสดงค่าความน่าจะเป็น [%Away, %Draw, %Home]
for i in range(len(X_test)):
    row = np.array(X_test[i]).reshape(1, -1)
    print(load_model.predict(row), "\t", y_test[i], "\t", np.round(load_model.predict_proba(row), 4))

# 7. แสดงคะแนนความแม่นยำ (Accuracy)
print(round(load_model.score(X_train, y_train), 2))
print(round(load_model.score(X_test, y_test), 2))
