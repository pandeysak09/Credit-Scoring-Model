import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="💳",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# TITLE
# =========================

st.title("💳 Credit Scoring Dashboard")
st.markdown("### Machine Learning Based Credit Risk Prediction")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("📌 Project Information")

st.sidebar.info("""
Dataset: Kaggle Credit Risk Dataset

Algorithm Used:
Random Forest Classifier

Model Accuracy:
93.12%

ROC-AUC:
93.51%

Developer:
Sakshi Pandey
""")

# =========================
# METRICS
# =========================

col1, col2, col3, col4 = st.columns(4)

col1.metric("📊 Records", "32,411")
col2.metric("📁 Features", "13")
col3.metric("🎯 Accuracy", "93.12%")
col4.metric("📈 ROC-AUC", "93.51%")

st.divider()

# =========================
# INPUT SECTION
# =========================

st.subheader("📝 Customer Information")

left, right = st.columns(2)

with left:

    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

    person_income = st.number_input(
        "Annual Income",
        min_value=1000,
        value=50000
    )

    person_emp_length = st.number_input(
        "Employment Length (Years)",
        min_value=0,
        value=5
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length",
        min_value=0,
        value=5
    )

    home_ownership = st.selectbox(
        "Home Ownership",
        [
            "MORTGAGE",
            "OWN",
            "RENT",
            "OTHER"
        ]
    )

with right:

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=500,
        value=10000
    )

    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=1.0,
        max_value=50.0,
        value=11.0
    )

    loan_percent_income = st.slider(
        "Loan Percent of Income",
        0.01,
        1.00,
        0.20
    )

    loan_intent = st.selectbox(
        "Loan Purpose",
        [
            "DEBTCONSOLIDATION",
            "EDUCATION",
            "HOMEIMPROVEMENT",
            "MEDICAL",
            "PERSONAL",
            "VENTURE"
        ]
    )

    loan_grade = st.selectbox(
        "Loan Grade",
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G"
        ]
    )

    previous_default = st.selectbox(
        "Previous Default",
        [
            "No",
            "Yes"
        ]
    )

# =========================
# ENCODING MAPS
# =========================

home_map = {
    "MORTGAGE": 0,
    "OWN": 1,
    "RENT": 2,
    "OTHER": 3
}

intent_map = {
    "DEBTCONSOLIDATION": 0,
    "EDUCATION": 1,
    "HOMEIMPROVEMENT": 2,
    "MEDICAL": 3,
    "PERSONAL": 4,
    "VENTURE": 5
}

grade_map = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6
}

default_map = {
    "No": 0,
    "Yes": 1
}

# =========================
# FEATURE ENGINEERING
# =========================

loan_income_ratio = loan_amnt / person_income

# =========================
# PREDICT BUTTON
# =========================

st.divider()

if st.button("🔍 Predict Credit Risk", use_container_width=True):

    data = pd.DataFrame([[
        person_age,
        person_income,
        home_map[home_ownership],
        person_emp_length,
        intent_map[loan_intent],
        grade_map[loan_grade],
        loan_amnt,
        loan_int_rate,
        loan_percent_income,
        default_map[previous_default],
        cb_person_cred_hist_length,
        loan_income_ratio
    ]],
    columns=[
        'person_age',
        'person_income',
        'person_home_ownership',
        'person_emp_length',
        'loan_intent',
        'loan_grade',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_default_on_file',
        'cb_person_cred_hist_length',
        'loan_income_ratio'
    ])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    probability = model.predict_proba(data_scaled)[0]

    st.divider()

    st.subheader("📋 Prediction Result")

    if prediction == 0:

        st.success(
            "✅ LOW CREDIT RISK\n\n"
            "The applicant appears financially stable and "
            "has a lower probability of loan default."
        )

        confidence = probability[0] * 100

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        

    else:

        st.error(
            "⚠️ HIGH CREDIT RISK\n\n"
            "The applicant has a higher probability of loan default."
        )

        confidence = probability[1] * 100

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

st.divider()

st.caption(
    "Credit Scoring Model using Machine Learning | Random Forest Classifier"
)