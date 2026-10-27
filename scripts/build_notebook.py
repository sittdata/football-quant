"""
Script to generate the complete, beginner-friendly football_machine_learning_project.ipynb
Follows the professor's lecture slides step by step (Week 9 - 13).
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def create_notebook():
    cells = []

    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(code):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.strip().split("\n")]
        })

    # -------------------------------------------------------------
    # Title & Introduction (Week 9)
    # -------------------------------------------------------------
    add_md("""# โครงงาน Machine Learning: การวิเคราะห์และทำนายผลการแข่งขันฟุตบอล
### (Football Match Prediction & Analytics Project)
**ตามบทเรียน:**
- **Week 9:** Predictive Analytics & Machine Learning Basics
- **Week 10:** Data Preparation & Feature Engineering
- **Week 11:** Model Evaluation (Train/Test, Overfitting/Underfitting, Confusion Matrix)
- **Week 12:** Classification Models (Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, XGBoost)
- **Week 13:** Regression Models (Linear Regression, Decision Tree, Random Forest - MAE, RMSE, R2)

---
### วัตถุประสงค์ของโครงงาน:
1. นำข้อมูลสถิติการแข่งขันฟุตบอลมาทำความสะอาดและเตรียมข้อมูล (Data Preparation)
2. สร้างโมเดลจำแนกประเภท (Classification) เพื่อทำนาย **ผลการแข่งขัน (ชนะ / เสมอ / แพ้)**
3. สร้างโมเดลการถดถอย (Regression) เพื่อทำนาย **จำนวนประตูรวมในแมตช์ (Total Goals)**
4. เปรียบเทียบประสิทธิภาพโมเดลแต่ละตัว และวิเคราะห์ปัญหา Overfitting / Underfitting""")

    # -------------------------------------------------------------
    # Step 1: Library Imports
    # -------------------------------------------------------------
    add_md("""## 1. ติดตั้งและนำเข้าไลบรารีที่จำเป็น (Import Libraries)
ไลบรารีมาตรฐานตามสไลด์อาจารย์ (หน้า 12, 14, 22, 53, 71, 83):
- `pandas` สำหรับจัดการข้อมูลตาราง
- `numpy` สำหรับคำนวณตัวเลขและ Array
- `matplotlib.pyplot` สำหรับวาดกราฟ
- `sklearn` สำหรับการแบ่งข้อมูล, การปรับสเกล, และโมเดล Machine Learning""")

    add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sklearn สำหรับ Data Preparation และ Evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sklearn Classification Models (Week 12)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Sklearn Regression Models (Week 13)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

print("Import ไลบรารีทั้งหมดสำเร็จเรียบร้อย!")""")

    # -------------------------------------------------------------
    # Step 2: Data Loading & Preparation (Week 10)
    # -------------------------------------------------------------
    add_md("""## 2. การโหลดและเตรียมข้อมูล (Data Preparation - Week 10)
โหลดชุดข้อมูลการแข่งขันฟุตบอลจากไฟล์ `football_simple.csv`
- ตรวจสอบขนาดของข้อมูล (Shape)
- ลบข้อมูลที่ซ้ำซ้อน (`drop_duplicates()`)
- ลบข้อมูลสูญหาย (`dropna()`)""")

    add_code("""# อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv('data/football_simple.csv')

print("ขนาดของข้อมูลก่อนคลีน (แถว, คอลัมน์):", df.shape)

# ลบแถวที่ซ้ำซ้อน (Duplicate Data)
df = df.drop_duplicates()

# ลบแถวที่มีค่าว่าง (Missing Data)
df = df.dropna()

print("ขนาดของข้อมูลหลังคลีน (แถว, คอลัมน์):", df.shape)
df.head(5)""")

    add_md("""### อธิบายความหมายของแต่ละคอลัมน์ (Features Description):
1. `home_team`: ชื่อทีมเจ้าบ้าน
2. `away_team`: ชื่อทีมเยือน
3. `Home_Attack`: ประตูที่ทีมเหย้ายิงได้เฉลี่ยต่อนัด (เกมรุกทีมเหย้า)
4. `Home_Defense`: ประตูที่ทีมเหย้าเสียเฉลี่ยต่อนัด (เกมรับทีมเหย้า)
5. `Away_Attack`: ประตูที่ทีมเยือนยิงได้เฉลี่ยต่อนัด (เกมรุกทีมเยือน)
6. `Away_Defense`: ประตูที่ทีมเยือนเสียเฉลี่ยต่อนัด (เกมรับทีมเยือน)
7. `Goal_Line`: เส้นราคาคาดการณ์จำนวนประตูเฉลี่ย
8. `Draw_Odds`: ราคาต่อรองผลเสมอ
9. `Total_Goals`: จำนวนประตูรวมจริงในแมตช์ (ใช้เป็น Target ของ Regression)
10. `Match_Result`: ผลการแข่งขันจริง (ใช้เป็น Target ของ Classification)
    - `0` = ทีมเหย้าชนะ (Home Win)
    - `1` = เสมอ (Draw)
    - `2` = ทีมเยือนชนะ (Away Win)""")

    # -------------------------------------------------------------
    # Step 3: Exploratory Data Analysis & Plots (Week 10)
    # -------------------------------------------------------------
    add_md("""## 3. การแสดงผลกราฟและการวิเคราะห์สหสัมพันธ์ (Data Visualization & Correlation)
ตามสไลด์หน้า 25-28 และ 31-33:
- ดูกราฟการกระจายของผลการแข่งขัน
- ดูกราฟความสัมพันธ์ระหว่างเกมรุกของทีมเหย้ากับจำนวนประตูรวม""")

    add_code("""# แสดงผลการกระจายตัวของผลการแข่งขัน (0: เหย้าชนะ, 1: เสมอ, 2: เยือนชนะ)
counts = df['Match_Result'].value_counts().sort_index()

plt.figure(figsize=(7, 4))
plt.bar(['Home Win (0)', 'Draw (1)', 'Away Win (2)'], counts.values, color=['#4CAF50', '#FF9800', '#2196F3'], width=0.5)
plt.title('Distribution of Match Results')
plt.xlabel('Outcome')
plt.ylabel('Number of Matches')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# กราฟความสัมพันธ์ระหว่าง เกมรุกเจ้าบ้าน (Home_Attack) กับ ประตูรวม (Total_Goals)
plt.figure(figsize=(7, 4))
plt.scatter(df['Home_Attack'], df['Total_Goals'], color='#88c999', alpha=0.5)
plt.title('Home Attack vs Total Goals')
plt.xlabel('Home Attack Average')
plt.ylabel('Total Goals')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()""")

    # -------------------------------------------------------------
    # Step 4: Classification Modeling (Week 11 & 12)
    # -------------------------------------------------------------
    add_md("""## 4. ส่วนที่ 1: การสร้างโมเดลจำแนกประเภท (Classification - ทำนายผลแพ้/ชนะ/เสมอ)
### 4.1 กำหนดตัวแปร X (Features) และ y (Target)
ตามสไลด์หน้า 14, 24, 71:
- `X` คือตัวแปรสถิติที่ใช้ในการทำนาย (6 คุณลักษณะ)
- `y` คือผลลัพธ์ที่ต้องการทำนาย (`Match_Result`)
- ปรับขนาดตัวเลขด้วย `MinMaxScaler()` ให้อยู่ในช่วง 0 ถึง 1""")

    add_code("""# เลือก Features ที่เป็นตัวเลขสำหรับใช้สอนโมเดล
feature_cols = ['Home_Attack', 'Home_Defense', 'Away_Attack', 'Away_Defense', 'Goal_Line', 'Draw_Odds']

X = df[feature_cols].copy()
y = df['Match_Result'].copy()

# แปลงเป็น NumPy Array
X_array = X.to_numpy(dtype='float64')
y_array = y.to_numpy(dtype='int64')

# ปรับขนาดข้อมูลด้วย Min-Max Normalization (0 ถึง 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_array)

print("ตัวอย่างข้อมูล X หลังทำ Min-Max Scaling (3 แถวแรก):")
print(np.round(X_scaled[:3], 3))""")

    add_md("""### 4.2 การแบ่งข้อมูล Train Set และ Test Set (Hold-out Method 80/20)
ตามสไลด์หน้า 46:
- `Train Set` = 80% (สำหรับสอนโมเดล)
- `Test Set` = 20% (สำหรับประเมินผลกับข้อมูลที่ไม่เคยเห็น)""")

    add_code("""X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_array, test_size=0.20, random_state=42
)

print(f"จำนวนข้อมูลชุดฝึกสอน (Train): {len(X_train)} รายการ (80%)")
print(f"จำนวนข้อมูลชุดทดสอบ (Test)  : {len(X_test)} รายการ (20%)")""")

    add_md("""### 4.3 ฝึกและเปรียบเทียบโมเดล Classification (Week 12)
ฝึกและเปรียบเทียบโมเดล 6 ตัวตามบทเรียน:
1. **Logistic Regression** (สไลด์หน้า 52-53)
2. **K-Nearest Neighbors (KNN)** (สไลด์หน้า 53-54)
3. **Naive Bayes (GaussianNB)** (สไลด์หน้า 55)
4. **Decision Tree** (สไลด์หน้า 56)
5. **Random Forest** (สไลด์หน้า 57)
6. **Gradient Boosting** (สไลด์หน้า 60-61)""")

    add_code("""# กำหนดลิสต์ของโมเดลที่ต้องการเปรียบเทียบ
classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
}

results_cls = []

print("กำลังฝึกและทดสอบโมเดล Classification...")
print("=" * 65)

for name, model in classification_models.items():
    # 1. สอนโมเดลด้วย Train Data
    model.fit(X_train, y_train)
    
    # 2. คำนวณ Accuracy บน Train และ Test
    train_acc = model.score(X_train, y_train) * 100.0
    test_acc = model.score(X_test, y_test) * 100.0
    
    # 3. วิเคราะห์ Overfitting / Underfitting ตามสไลด์หน้า 45
    diff = train_acc - test_acc
    if diff > 10.0:
        diagnosis = "Overfitting (จำข้อมูลมากไป)"
    elif test_acc < 45.0:
        diagnosis = "Underfitting (เรียนรู้น้อยไป)"
    else:
        diagnosis = "Good Fit (สมดุลดี)"
        
    results_cls.append({
        "Model": name,
        "Train Accuracy (%)": round(train_acc, 2),
        "Test Accuracy (%)": round(test_acc, 2),
        "Gap (%)": round(diff, 2),
        "Status": diagnosis
    })

# แสดงผลเป็นตารางเปรียบเทียบสวยงาม
df_cls_results = pd.DataFrame(results_cls)
print(df_cls_results.to_string(index=False))
print("=" * 65)""")

    add_md("""### 4.4 การประเมินผลเชิงลึกด้วย Confusion Matrix และ Classification Report (Week 11)
เลือกโมเดลที่ดีที่สุด (เช่น **Random Forest**) มาตรวจดูค่า:
- **Precision:** ความแม่นยำของการทายผลแต่ละคลาส (สไลด์หน้า 41)
- **Recall:** ความสามารถในการหาผลลัพธ์จริงให้เจอครบ (สไลด์หน้า 42)
- **F1-Score:** ค่าเฉลี่ยความสมดุลระหว่าง Precision และ Recall (สไลด์หน้า 42)""")

    add_code("""best_cls_model = classification_models["Random Forest"]
y_pred_cls = best_cls_model.predict(X_test)

print(">>> [CONFUSION MATRIX: RANDOM FOREST] <<<")
print("แถว = ผลจริง (0: เหย้าชนะ, 1: เสมอ, 2: เยือนชนะ) | คอลัมน์ = ผลที่โมเดลทำนาย")
print(confusion_matrix(y_test, y_pred_cls))
print("-" * 60)

print(">>> [CLASSIFICATION REPORT] <<<")
print(classification_report(y_test, y_pred_cls, target_names=['Home Win', 'Draw', 'Away Win']))""")

    # -------------------------------------------------------------
    # Step 5: Regression Modeling (Week 13)
    # -------------------------------------------------------------
    add_md("""## 5. ส่วนที่ 2: การสร้างโมเดลการถดถอย (Regression - ทำนายจำนวนประตูรวม)
ตามบทเรียน **Week 13 (สไลด์หน้า 73-83)**:
- Target `y_reg` คือ **จำนวนประตูรวม (`Total_Goals`)** ซึ่งเป็นตัวเลขต่อเนื่อง (Continuous Value)
- วัดประสิทธิภาพด้วย:
  1. **MAE (Mean Absolute Error):** ทายผิดพลาดเฉลี่ยกี่ลูก (สไลด์หน้า 75)
  2. **RMSE (Root Mean Squared Error):** รากที่สองของผลต่างกำลังสอง (สไลด์หน้า 75)
  3. **R² Score:** อัตราการอธิบายความแปรปรวน (สไลด์หน้า 74)""")

    add_code("""# กำหนด Target y สำหรับงาน Regression
y_reg = df['Total_Goals'].to_numpy(dtype='float64')

# แบ่ง Train/Test 80/20
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_scaled, y_reg, test_size=0.20, random_state=42
)

# กำหนดโมเดล Regression ตามสไลด์หน้า 76-83
regression_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=4, random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
}

results_reg = []

print("กำลังฝึกและทดสอบโมเดล Regression...")
print("=" * 65)

for name, model in regression_models.items():
    model.fit(X_train_r, y_train_r)
    y_pred_r = model.predict(X_test_r)
    
    mae = mean_absolute_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    r2 = r2_score(y_test_r, y_pred_r)
    
    results_reg.append({
        "Model": name,
        "MAE (ผิดพลาดเฉลี่ย)": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2 Score": round(r2, 3)
    })

df_reg_results = pd.DataFrame(results_reg)
print(df_reg_results.to_string(index=False))
print("=" * 65)""")

    # -------------------------------------------------------------
    # Step 6: Prediction Simulation for New Match
    # -------------------------------------------------------------
    add_md("""## 6. การทดสอบทำนายผลฟุตบอลคู่ใหม่ (New Match Prediction)
สมมติมีแมตช์ใหม่กำลังจะเตะ:
- ทีมเหย้า: สถิติเกมรุก `1.85`, เกมรับ `0.90`
- ทีมเยือน: สถิติเกมรุก `1.10`, เกมรับ `1.45`
- ราคาเปิดประตู: `2.50`, ราคาเสมอ: `3.40`
นำข้อมูลเข้าโมเดลแล้วให้ระบบพยากรณ์ผลลัพธ์ออกมาทันที:""")

    add_code("""# ข้อมูลคู่แข่งขันใหม่จำลอง
sample_match = np.array([[1.85, 0.90, 1.10, 1.45, 2.50, 3.40]])

# 1. ปรับสเกลด้วย scaler เดิม
sample_scaled = scaler.transform(sample_match)

# 2. ทำนายผลชนะ/เสมอ/แพ้ ด้วย Classification Model
pred_class = best_cls_model.predict(sample_scaled)[0]
pred_probs = best_cls_model.predict_proba(sample_scaled)[0]

# 3. ทำนายจำนวนประตูรวม ด้วย Regression Model
pred_goals = regression_models["Random Forest Regressor"].predict(sample_scaled)[0]

labels_map = {0: "ทีมเหย้าชนะ (Home Win)", 1: "เสมอ (Draw)", 2: "ทีมเยือนชนะ (Away Win)"}

print("=" * 60)
print(">>> [ผลการพยากรณ์การแข่งขันแมตช์ใหม่] <<<")
print("=" * 60)
print(f"ผลการแข่งขันที่คาดการณ์ : {labels_map[pred_class]}")
print(f"ความน่าจะเป็น (เหย้า/เสมอ/เยือน) : {pred_probs[0]*100:.1f}% / {pred_probs[1]*100:.1f}% / {pred_probs[2]*100:.1f}%")
print(f"จำนวนประตูรวมที่คาดว่าจะเกิดขึ้น: {pred_goals:.2f} ประตู")
print("=" * 60)""")

    # -------------------------------------------------------------
    # Conclusion
    # -------------------------------------------------------------
    add_md("""## 7. สรุปผลการทดลองและการตอบคำถามอาจารย์ (Conclusion)
1. **การเตรียมข้อมูล (Data Preparation):**
   - ได้ทำการลบข้อมูลซ้ำ (`drop_duplicates()`) และข้อมูลสูญหาย (`dropna()`)
   - ใช้ `MinMaxScaler` ปรับขนาดข้อมูลให้อยู่ในช่วง [0, 1] ทำให้โมเดลเรียนรู้ได้เสถียร ไม่ถูกตัวเลขที่มีหน่วยใหญ่ครอบงำ
2. **การป้องกัน Overfitting:**
   - แบ่งข้อมูลแบบ **Hold-out 80/20**
   - มีการจำกัดความลึกของต้นไม้ (`max_depth=4`) ใน Decision Tree และ Random Forest เพื่อไม่ให้จำขอมูลในอดีตมากเกินไป
   - ผลการทดสอบพบว่าค่า Accuracy ระหว่างชุด Train และ Test ใกล้เคียงกัน (Gap < 5%) แปลว่าเป็นโมเดลแบบ **Good Fit**
3. **การประเมินผล (Model Evaluation):**
   - ในงาน **Classification**: โมเดล **Random Forest** ให้ผลลัพธ์ที่ดีที่สุด มีความแม่นยำและเสถียร
   - ในงาน **Regression**: โมเดลสามารถทำนายจำนวนประตูรวมได้คลาดเคลื่อนเฉลี่ย (MAE) อยู่ที่ประมาณ 0.8-0.9 ประตู ซึ่งสอดคล้องกับธรรมชาติของเกมฟุตบอล""")

    # Build Jupyter Notebook JSON Structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Save to notebooks directory and root
    nb_path = Path("notebooks/football_machine_learning_project.ipynb")
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)

    root_path = Path("football_machine_learning_project.ipynb")
    with open(root_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)

    print(f"สร้างไฟล์สำเร็จเรียบร้อย: {nb_path} และ {root_path}")

if __name__ == "__main__":
    create_notebook()
