# DSA Football Predictive Analytics & Machine Learning Project

โครงสร้างโปรเจกต์นี้ได้รับการพัฒนาขึ้นโดยอิงตามเนื้อหาหลักสูตร **Data Science & Analytics (DSA.pdf - Week 9 ถึง Week 14)** เพื่อประยุกต์ใช้โมเดล Machine Learning ในการทำนายและวิเคราะห์ข้อมูลฟุตบอลจริง

---

## โครงสร้างโฟลเดอร์ (Directory Structure)

```
dsa_project/
│
├── 01_data_prep_feature_selection/   # [Week 10] การเตรียมข้อมูลและการคัดเลือกตัวแปร
│   ├── prepare_data.py               # Data Cleaning, Outlier Capping, Scaling (MinMax, Robust, Standard)
│   └── feature_selection.py          # Filter (Correlation), Embedded (Lasso L1, RF Importance)
│
├── 02_unsupervised_clustering/       # [Week 14] การเรียนรู้แบบไม่มีผู้สอน
│   └── team_clustering.py            # K-Means (k=4), DBSCAN, GMM, และ PCA 2D Visualization
│
├── 03_regression_models/             # [Week 13] โมเดลทำนายตัวเลขต่อเนื่อง (Expected Goals)
│   └── train_regression.py           # Linear Reg, SVR, DecisionTree, RandomForest, GBDT, XGBoost, LightGBM
│
├── 04_classification_models/         # [Week 12] โมเดลจำแนกกลุ่ม (BTTS & 1X2 Match Result)
│   └── train_classification.py       # Logistic, KNN, Naive Bayes, DecisionTree, RF, SVM, AdaBoost, XGBoost
│
├── 05_overfitting_and_validation/    # [Week 11] การประเมินผลและการป้องกัน Overfitting
│   └── kfold_bias_variance.py        # 5-Fold Cross Validation, Bias-Variance Tradeoff Curve
│
├── data/                             # เก็บไฟล์ Dataset ที่ผ่านการ Clean แล้ว
│   └── clean_dsa_dataset.csv
│
├── models/                           # บันทึกโมเดลที่ดีที่สุด (.pkl) ด้วย pickle
│   ├── best_total_goals_regressor.pkl
│   ├── best_classifier_target_btts.pkl
│   └── best_classifier_target_1x2.pkl
│
├── reports/                          # รายงานสรุปผล และรูปกราฟที่บันทึกจากการทดลอง
│   ├── figures/                      # กราฟ Correlation, PCA, ROC Curve, Confusion Matrix
│   ├── dsa_project_report.md         # รายงานผลเปรียบเทียบทุกโมเดล
│   ├── regression_results.csv
│   └── classification_results_*.csv
│
├── run_all_dsa.py                    # Master Script สำหรับรันการทดลองทั้งหมดในคำสั่งเดียว
└── README.md                         # คำอธิบายโครงสร้างโปรเจกต์
```

---

## การจับคู่เนื้อหากับสไลด์เรียน (Curriculum Mapping)

| สัปดาห์ | หัวข้อในสไลด์ DSA | สิ่งที่พัฒนาขึ้นในโปรเจกต์ |
| :---: | :--- | :--- |
| **Week 9** | Predictive & Prescriptive Analytics | นำผลทำนาย Expected Goals ไปเปรียบเทียบกับเส้น Over/Under เพื่อหา Value |
| **Week 10** | Data Prep & Feature Engineering | Clean missing values, RobustScaler, Lasso L1 feature selection |
| **Week 11** | Evaluation & Overfitting/Underfitting | Confusion Matrix, F1, ROC-AUC, RMSE, $R^2$, K-Fold, Bias-Variance Curve |
| **Week 12** | Classification Models | Binary (BTTS) & Multi-class (1X2) เทียบ 9 โมเดล (Logistic, SVM, XGB, ฯลฯ) |
| **Week 13** | Regression Models | ทายจำนวนประตูรวม เทียบ Linear Regression, SVR, Random Forest Regressor |
| **Week 14** | Unsupervised Learning & PCA | จัดกลุ่มสไตล์ทีมฟุตบอลด้วย K-Means + ลดมิติ 2D ด้วย PCA |

---

## วิธีการรันโปรเจกต์ (How to Run)

### 1. รันการทดลองทั้งหมดในคำสั่งเดียว
```powershell
python dsa_project/run_all_dsa.py
```

### 2. รันแยกแต่ละโมดูล
```powershell
# 1. การเตรียมข้อมูลและ Feature Selection
python dsa_project/01_data_prep_feature_selection/feature_selection.py

# 2. การจัดกลุ่มทีมฟุตบอลและ PCA
python dsa_project/02_unsupervised_clustering/team_clustering.py

# 3. โมเดล Regression (ทำนายประตูรวม)
python dsa_project/03_regression_models/train_regression.py

# 4. โมเดล Classification (ทำนาย BTTS และ 1X2)
python dsa_project/04_classification_models/train_classification.py

# 5. วิเคราะห์ K-Fold และ Bias-Variance
python dsa_project/05_overfitting_and_validation/kfold_bias_variance.py
```
