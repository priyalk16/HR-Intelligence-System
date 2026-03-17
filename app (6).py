import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="HR Intelligence System", page_icon="🏢", layout="wide")

st.title("🏢 HR Intelligence System")
st.markdown("**3 ML Models | IBM HR Analytics | XGBoost + SMOTE | Priyal Kantharia**")
st.markdown("---")

@st.cache_resource
def load_models():
    attrition = joblib.load("attrition_model.pkl")
    joblevel = joblib.load("joblevel_model.pkl")
    salary = joblib.load("salary_model.pkl")
    return attrition, joblevel, salary

attrition_model, joblevel_model, salary_model = load_models()

# Sidebar - Model Info
with st.sidebar:
    st.markdown("## 📊 Model Performance")
    st.markdown("---")
    st.markdown("### 🚨 Attrition Predictor")
    st.markdown("**ROC-AUC: 80.9%**")
    st.markdown("Binary Classification")
    st.markdown("---")
    st.markdown("### 🏢 Job Level Predictor")
    st.markdown("**Accuracy: 93%**")
    st.markdown("Multiclass Classification")
    st.markdown("---")
    st.markdown("### 💰 Salary Predictor")
    st.markdown("**R² Score: 95%**")
    st.markdown("Regression")
    st.markdown("---")
    st.markdown("**Dataset:** IBM HR Analytics")
    st.markdown("**Algorithm:** XGBoost")
    st.markdown("**Balancing:** SMOTE")
    st.markdown("**Tuning:** RandomizedSearchCV")

st.subheader("👤 Enter Employee Details")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Info**")
    age = st.number_input("Age", 18, 65, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education = st.selectbox("Education Level", [1, 2, 3, 4, 5],
        format_func=lambda x: {1:"Below College",2:"College",
                                3:"Bachelor",4:"Master",5:"PhD"}[x])
    education_field = st.selectbox("Education Field",
        ["Life Sciences","Medical","Marketing",
         "Technical Degree","Human Resources","Other"])
    distance_from_home = st.number_input("Distance from Home (km)", 1, 30, 5)

with col2:
    st.markdown("**Work Details**")
    department = st.selectbox("Department",
        ["Sales","Research & Development","Human Resources"])
    job_role = st.selectbox("Job Role",
        ["Sales Executive","Research Scientist","Laboratory Technician",
         "Manufacturing Director","Healthcare Representative","Manager",
         "Sales Representative","Research Director","Human Resources"])
    business_travel = st.selectbox("Business Travel",
        ["Travel_Rarely","Travel_Frequently","Non-Travel"])
    overtime = st.selectbox("Works Overtime?", ["Yes", "No"])
    num_companies = st.number_input("Companies Worked Before", 0, 10, 2)
    stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    performance_rating = st.selectbox("Performance Rating", [3, 4],
        format_func=lambda x: {3:"Excellent",4:"Outstanding"}[x])

with col3:
    st.markdown("**Experience**")
    total_working_years = st.number_input("Total Working Years", 0, 40, 10)
    years_at_company = st.number_input("Years at Company", 0, 40, 5)
    years_in_role = st.number_input("Years in Current Role", 0, 20, 3)
    years_since_promotion = st.number_input("Years Since Last Promotion", 0, 15, 1)
    years_with_manager = st.number_input("Years with Current Manager", 0, 20, 3)
    training_times = st.number_input("Training Times Last Year", 0, 6, 2)
    percent_hike = st.number_input("Salary Hike %", 11, 25, 14)

st.markdown("---")
st.subheader("😊 Satisfaction Scores (1=Low → 4=High)")
sc1, sc2, sc3, sc4, sc5 = st.columns(5)
with sc1:
    env_sat = st.slider("Environment", 1, 4, 3)
with sc2:
    job_sat = st.slider("Job", 1, 4, 3)
with sc3:
    rel_sat = st.slider("Relationship", 1, 4, 3)
with sc4:
    job_inv = st.slider("Involvement", 1, 4, 3)
with sc5:
    wlb = st.slider("Work-Life Balance", 1, 4, 3)

st.markdown("---")
r1, r2, r3 = st.columns(3)
with r1:
    daily_rate = st.number_input("Daily Rate ($)", 100, 1500, 800)
with r2:
    hourly_rate = st.number_input("Hourly Rate ($)", 30, 100, 65)
with r3:
    monthly_rate = st.number_input("Monthly Rate ($)", 2000, 27000, 14000)

st.markdown("---")

if st.button("🔮 Analyze Employee", use_container_width=True):

    gender_enc = 1 if gender == "Male" else 0
    overtime_enc = 1 if overtime == "Yes" else 0

    # Base features (no JobLevel, no MonthlyIncome)
    base = {
        'Age': age, 'DailyRate': daily_rate,
        'DistanceFromHome': distance_from_home, 'Education': education,
        'EnvironmentSatisfaction': env_sat, 'HourlyRate': hourly_rate,
        'JobInvolvement': job_inv, 'JobSatisfaction': job_sat,
        'MonthlyRate': monthly_rate, 'NumCompaniesWorked': num_companies,
        'PercentSalaryHike': percent_hike, 'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': rel_sat, 'StockOptionLevel': stock_option,
        'TotalWorkingYears': total_working_years, 'TrainingTimesLastYear': training_times,
        'WorkLifeBalance': wlb, 'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_role, 'YearsSinceLastPromotion': years_since_promotion,
        'YearsWithCurrManager': years_with_manager,
        'OverTime_encoded': overtime_enc, 'Gender_encoded': gender_enc,
        'BusinessTravel': business_travel, 'Department': department,
        'EducationField': education_field, 'JobRole': job_role,
        'MaritalStatus': marital_status
    }

    # Model 1 — Attrition (needs MonthlyIncome)
    df_m1 = pd.DataFrame([{**base, 'MonthlyIncome': 6500}])
    attr_pred = attrition_model.predict(df_m1)[0]
    attr_prob = attrition_model.predict_proba(df_m1)[0][1]

    # Model 2 — Job Level (needs MonthlyIncome)
    df_m2 = pd.DataFrame([{**base, 'MonthlyIncome': 6500}])
    jl_pred = joblevel_model.predict(df_m2)[0]
    jl_labels = {0:"Entry Level",1:"Mid Level",
                 2:"Senior Level",3:"Lead Level",4:"Director Level"}

    # Model 3 — Salary (needs JobLevel)
    df_m3 = pd.DataFrame([{**base, 'JobLevel': jl_pred + 1}])
    salary_pred = salary_model.predict(df_m3)[0]

    # Results
    st.markdown("## 📊 Analysis Results")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 🚨 Attrition Risk")
        risk_pct = round(attr_prob * 100, 1)
        if attr_pred == 1:
            st.error(f"**HIGH RISK**")
            st.metric("Probability", f"{risk_pct}%")
            st.caption("⚠️ Employee likely to resign")
        else:
            st.success(f"**LOW RISK**")
            st.metric("Probability", f"{risk_pct}%")
            st.caption("✅ Employee likely to stay")

    with c2:
        st.markdown("### 🏢 Predicted Job Level")
        level_label = jl_labels[jl_pred]
        st.info(f"**{level_label}**")
        st.metric("Level", f"{jl_pred + 1} / 5")
        st.caption("Based on experience & skills")

    with c3:
        st.markdown("### 💰 Expected Salary")
        st.success(f"**${salary_pred:,.0f}/month**")
        st.metric("Annual Package", f"${salary_pred*12:,.0f}")
        st.caption("Market rate estimate")

    st.markdown("---")
    st.markdown("### 💡 HR Recommendations")

    recs = []
    if attr_pred == 1:
        recs.append("🔴 **High Priority:** Schedule retention discussion immediately")
        if overtime_enc == 1:
            recs.append("🔴 Reduce overtime — key attrition driver detected")
        if years_since_promotion > 3:
            recs.append("🟡 Review promotion eligibility — no hike in 3+ years")
        if job_sat <= 2:
            recs.append("🟡 Investigate job satisfaction concerns urgently")
        if distance_from_home > 20:
            recs.append("🟡 Consider remote/hybrid work arrangement")
    else:
        recs.append("🟢 Retention risk is low — continue current engagement")
        if wlb <= 2:
            recs.append("🟡 Work-life balance needs attention")
        if training_times < 2:
            recs.append("🟡 Increase training & development opportunities")
        if years_since_promotion > 2:
            recs.append("🟡 Consider recognition or career growth discussion")

    for rec in recs:
        st.markdown(f"- {rec}")

    st.markdown("---")
    st.caption("🏢 HR Intelligence System | Built by Priyal Kantharia | "
               "IBM HR Analytics Dataset | XGBoost + SMOTE + RandomizedSearchCV | "
               "Attrition: ROC-AUC 80.9% | Job Level: 93% | Salary: R² 95%")
