# 🏢 HR Intelligence System
### Machine Learning on IBM HR Analytics Dataset

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

---

## 🎯 Problem Statement

HR departments face 3 critical challenges daily:

- **Attrition** — Which employees are likely to resign? (costs $50,000+ per resignation)
- **Job Level** — What career level does an employee belong to based on their profile?
- **Salary** — What is the fair market salary for an employee?

This project builds **3 independent ML models** from a single dataset to solve all 3 problems — enabling data-driven HR decision making through an interactive Streamlit dashboard.

---

## 📊 Models & Results

| # | Model | Target | Type | Algorithm | Score |
|---|---|---|---|---|---|
| 1 | 🚨 Attrition Predictor | Will employee resign? | Binary Classification | XGBoost + SMOTE | **ROC-AUC: 82%** |
| 2 | 🏢 Job Level Predictor | Career level (1-5) | Multiclass Classification | XGBoost | **F1: 93.9%** |
| 3 | 💰 Salary Predictor | Monthly income | Regression | XGBoost | **R²: 95%** |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| ML Models | XGBoost, Random Forest, Logistic Regression, Ridge |
| ML Pipeline | Scikit-learn Pipelines, ColumnTransformer |
| Class Balancing | SMOTE (imbalanced-learn) |
| Hyperparameter Tuning | RandomizedSearchCV |
| Validation | 5-Fold Cross Validation |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Deployment | Docker |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```
HR-Intelligence-System/
│
├── HR_Intelligence_System.ipynb   ← Complete ML notebook (all 3 models)
├── app.py                          ← Streamlit dashboard
│
├── attrition_model.pkl             ← Trained Model 1
├── joblevel_model.pkl              ← Trained Model 2
├── salary_model.pkl                ← Trained Model 3
│
├── WA_Fn-UseC_-HR-Employee-Attrition.csv  ← IBM HR Dataset
├── requirements.txt                ← Python dependencies
├── Dockerfile                      ← Docker configuration
└── README.md                       ← This file
```

---

## 📓 Notebook Structure

The notebook follows a complete ML workflow:

```
1.  Problem Statement
2.  Import Libraries
3.  Load Dataset
4.  Exploratory Data Analysis (EDA)
5.  Data Preprocessing & Feature Engineering
─── MODEL 1: ATTRITION ─────────────────────
6.  Train/Test Split (Stratified)
7.  Model Comparison (LR vs RF vs XGBoost)
8.  SMOTE — Handle Class Imbalance
9.  Hyperparameter Tuning (RandomizedSearchCV)
10. 5-Fold Cross Validation
11. Final Evaluation (Confusion Matrix + ROC Curve)
12. Feature Importance
─── MODEL 2: JOB LEVEL ─────────────────────
13. Train/Test Split
14. Model Comparison
15. Hyperparameter Tuning
16. Cross Validation
17. Confusion Matrix
18. Feature Importance
─── MODEL 3: SALARY ────────────────────────
19. Train/Test Split
20. Model Comparison
21. Hyperparameter Tuning
22. Cross Validation
23. Actual vs Predicted + Residuals Plot
24. Feature Importance
─────────────────────────────────────────────
25. Save All Models + Final Summary
```

---

## 🚀 How to Run Locally

### Step 1 — Clone the repository
```bash
git clone https://github.com/priyalk16/HR-Intelligence-System.git
cd HR-Intelligence-System
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the Streamlit app
```bash
streamlit run app.py
```

### Step 4 — Open in browser
```
http://localhost:8501
```

---

## 🐳 How to Run with Docker

### Step 1 — Build the Docker image
```bash
docker build -t hr-intelligence-system .
```

### Step 2 — Run the container
```bash
docker run -p 8501:8501 hr-intelligence-system
```

### Step 3 — Open in browser
```
http://localhost:8501
```

---

## 📱 App Features

The Streamlit dashboard allows HR managers to:

- Enter any employee's details (age, department, experience, satisfaction scores etc.)
- Get **3 instant predictions** simultaneously:
  - 🚨 Attrition risk percentage
  - 🏢 Predicted job level (Entry → Director)
  - 💰 Expected market salary
- Receive **automated HR recommendations** based on results

---

## 📈 Key Findings

- **Overtime** is the strongest predictor of employee attrition
- **Monthly Income** and **Total Working Years** are the top features for Job Level prediction
- **Job Level** alone explains 95% of salary variance
- Employees who travel frequently have 25% higher attrition rate
- SMOTE improved attrition model recall from 12% to 43%

---

## 📄 Dataset

**IBM HR Analytics Employee Attrition & Performance**
- Source: IBM Watson Analytics / Kaggle
- Size: 1,470 employees × 35 features
- Target Variables: Attrition, JobLevel, MonthlyIncome
- Missing Values: None
- Class Imbalance: 84% Stay vs 16% Leave (handled with SMOTE)

---

## 🧠 ML Approach Highlights

| Technique | Why Used |
|---|---|
| **SMOTE** | Only 16% employees resigned — synthetic oversampling fixes imbalance |
| **Stratified Split** | Maintains class ratio in train/test for reliable evaluation |
| **RandomizedSearchCV** | Faster than GridSearch, finds near-optimal hyperparameters |
| **5-Fold CV** | Ensures model generalizes — not just lucky on one split |
| **Pipeline** | Prevents data leakage between preprocessing and modeling |
| **ROC-AUC** | Better metric than accuracy for imbalanced classification |

---

## 👩‍💻 Author

**Priyal Kantharia**
- 🎓 B.E. Computer Engineering — Atharva College of Engineering, Mumbai
- 💼 ML Intern @ Intellectsia AI | Ex Generative AI Research Assistant @ Soul AI
- 🔗 [LinkedIn](https://www.linkedin.com/in/priyalkantharia)
- 🐙 [GitHub](https://github.com/priyalk16)
- 📧 kanthariapriyal16@gmail.com

---

*"Turning HR data into actionable intelligence — one model at a time."*
