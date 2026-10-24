"""Master Pipeline: Run all DSA Coursework Experiments (Week 9 - Week 14).

Executes:
1. Data Preparation & Cleaning (Week 10)
2. Feature Selection (Correlation, Lasso, Random Forest Importance) (Week 10)
3. Unsupervised Clustering & PCA 2D Projection (Week 14)
4. Regression Modeling: Total Goals Prediction (Week 13)
5. Classification Modeling: BTTS & 1X2 Prediction (Week 12)
6. Model Evaluation, K-Fold Validation & Bias-Variance Analysis (Week 11)
7. Generates comprehensive markdown report with charts.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add all subdirectories to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
SUBDIRS = [
    PROJECT_DIR / "01_data_prep_feature_selection",
    PROJECT_DIR / "02_unsupervised_clustering",
    PROJECT_DIR / "03_regression_models",
    PROJECT_DIR / "04_classification_models",
    PROJECT_DIR / "05_overfitting_and_validation"
]
for d in SUBDIRS:
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from prepare_data import load_and_clean_data
from feature_selection import get_selected_features
from team_clustering import build_team_profiles, run_team_clustering
from train_regression import train_and_evaluate_regressors
from train_classification import train_and_evaluate_classifiers
from kfold_bias_variance import run_full_validation_experiment


def run_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("[START] DSA COURSEWORK MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # 1. Data Prep
    print("\n>>> STEP 1: Data Preparation & Cleaning (Week 10)")
    df = load_and_clean_data()

    # 2. Feature Selection
    print("\n>>> STEP 2: Feature Selection & Correlation Analysis (Week 10)")
    get_selected_features(df, target_col="target_total_goals", top_n=15)

    # 3. Unsupervised Learning
    print("\n>>> STEP 3: Unsupervised Team Clustering & PCA 2D (Week 14)")
    profiles = build_team_profiles(df)
    run_team_clustering(profiles, n_clusters=4)

    # 4. Regression Models
    print("\n>>> STEP 4: Regression Models for Total Goals (Week 13)")
    reg_results = train_and_evaluate_regressors(df, target_col="target_total_goals")

    # 5. Classification Models
    print("\n>>> STEP 5.1: Binary Classification - BTTS (Week 12)")
    clf_btts_results = train_and_evaluate_classifiers(df, target_col="target_btts", is_multiclass=False)

    print("\n>>> STEP 5.2: Multi-class Classification - 1X2 Match Result (Week 12)")
    clf_1x2_results = train_and_evaluate_classifiers(df, target_col="target_1x2", is_multiclass=True)

    # 6. K-Fold & Overfitting
    print("\n>>> STEP 6: K-Fold Cross Validation & Bias-Variance Analysis (Week 11)")
    run_full_validation_experiment(df, target_col="target_btts")

    # 7. Generate Master Markdown Report
    print("\n>>> STEP 7: Generating Final DSA Project Report")
    report_path = PROJECT_DIR / "reports" / "dsa_project_report.md"
    generate_markdown_report(report_path, reg_results, clf_btts_results, clf_1x2_results)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[DONE] PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print(f"[REPORT] Report saved at: {report_path}")
    print("=" * 70)


def generate_markdown_report(report_path: Path, reg_df, btts_df, df_1x2):
    content = f"""# รายงานผลการทดลองโมเดล Machine Learning (DSA Course Project)

> โปรเจกต์นี้ถูกออกแบบและพัฒนาตามหลักสูตรวิชา Data Science & Analytics (สไลด์ Week 9 – Week 14) 
> โดยดึงข้อมูลฟุตบอลจริงและราคาต่อรองมาผ่านกระบวนการ Data Pipeline ครบทุกขั้นตอน

---

## 1. การเตรียมข้อมูลและคัดเลือกฟีเจอร์ (Week 10)
* **Data Cleaning:** จัดการข้อมูลซ้ำ (`drop_duplicates()`), เติมค่าว่างด้วยมัธยฐาน (Median Imputation), แปลงหน่วยเวลา Kickoff
* **Feature Selection:** 
  1. **Filter Method:** วิเคราะห์ค่าสหสัมพันธ์ Pearson Correlation ([correlation_target_total_goals.png](figures/correlation_target_total_goals.png))
  2. **Embedded Method (Lasso L1):** ลดมิติและตัดสัมประสิทธิ์ที่ไม่มีนัยสำคัญเป็น 0 ([lasso_selection_target_total_goals.png](figures/lasso_selection_target_total_goals.png))
  3. **Random Forest Importance:** วัดคะแนนความสำคัญของตัวแปร ([rf_importance_target_total_goals.png](figures/rf_importance_target_total_goals.png))

---

## 2. การจัดกลุ่มทีมฟุตบอลด้วย Unsupervised Learning (Week 14)
* **Algorithms:** เปรียบเทียบ K-Means (k=4), DBSCAN, และ GMM
* **Dimensionality Reduction:** ใช้ **PCA (Principal Component Analysis)** ลดมิติเหลือ 2D เพื่อแสดงผลสไตล์ของแต่ละทีม
* **กราฟแสดงผล:** [team_pca_clusters.png](figures/team_pca_clusters.png)

---

## 3. ผลการทดลองโมเดล Regression: ทำนายประตูรวม (Week 13)
เป้าหมายคือทำนาย **จำนวนประตูรวม (Expected Total Goals)** เพื่อใช้ต่อยอดการวิเคราะห์ Prescriptive ในการหา Edge

| Algorithm | MAE | MSE | RMSE (Test) | RMSE (Train) | R² Score | Overfit Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{chr(10).join([f"| {row['Algorithm']} | {row['MAE']} | {row['MSE']} | {row['RMSE (Test)']} | {row['RMSE (Train)']} | {row['R² Score']} | {row['Overfit Gap']} |" for _, row in reg_df.iterrows()])}

* **กราฟเปรียบเทียบ:** [regression_comparison.png](figures/regression_comparison.png)
* **กราฟ Actual vs Predicted:** [best_regression_actual_vs_pred.png](figures/best_regression_actual_vs_pred.png)

---

## 4. ผลการทดลองโมเดล Classification (Week 11 & 12)

### 4.1 Binary Classification: ทั้งสองทีมยิงประตูได้ (BTTS)
| Algorithm | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{chr(10).join([f"| {row['Algorithm']} | {row['Accuracy']} | {row['Precision']} | {row['Recall']} | {row['Specificity']} | {row['F1-Score']} | {row['ROC-AUC']} |" for _, row in btts_df.iterrows()])}

* **Confusion Matrix:** [confusion_matrix_target_btts.png](figures/confusion_matrix_target_btts.png)
* **ROC Curves:** [roc_curves_target_btts.png](figures/roc_curves_target_btts.png)

### 4.2 Multi-class Classification: ผล 1X2 (ชนะ/เสมอ/แพ้)
| Algorithm | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
{chr(10).join([f"| {row['Algorithm']} | {row['Accuracy']} | {row['Precision']} | {row['Recall']} | {row['F1-Score']} | {row['ROC-AUC']} |" for _, row in df_1x2.iterrows()])}

* **Confusion Matrix 1X2:** [confusion_matrix_target_1x2.png](figures/confusion_matrix_target_1x2.png)

---

## 5. การวิเคราะห์ Overfitting, Bias & Variance และ K-Fold Cross Validation (Week 11)
* **5-Fold Cross Validation:** ป้องกัน Data Leakage และวัดความเสถียรของโมเดลบนชุดข้อมูลที่ไม่เคยเห็น
* **Bias-Variance Trade-off:** ทดสอบการปรับค่า `max_depth` ของต้นไม้ตัดสินใจ (1 ถึง 15) เพื่อชี้จุดเกิด Overfitting
* **กราฟ Validation Curve:** [overfitting_validation_curve.png](figures/overfitting_validation_curve.png)
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    run_pipeline()
