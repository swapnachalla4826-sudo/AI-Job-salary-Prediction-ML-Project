import streamlit as st
import pandas as pd
import joblib
import os
from datetime import date

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Salary Prediction",
    page_icon="💼",
    layout="centered"
)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = "DecisionTree_salary_model.pkl"  # Change if your model filename is different


@st.cache_resource
def load_model():
    path = os.path.join(MODEL_DIR, MODEL_FILE)
    return joblib.load(path)


try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Couldn't find '{MODEL_FILE}' in the project folder.")
    st.stop()

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.title("💼 AI Job Salary Prediction App")
st.write("Fill in the job details below to estimate the salary (USD).")

st.divider()

# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
with st.form("salary_form"):

    st.subheader("Role")

    col1, col2 = st.columns(2)

    with col1:
        job_title = st.text_input("Job Title", "Data Scientist")

        experience_level = st.selectbox(
            "Experience Level",
            ["EN", "MI", "SE", "EX"],
            help="EN=Entry, MI=Mid, SE=Senior, EX=Executive"
        )

        employment_type = st.selectbox(
            "Employment Type",
            ["FT", "PT", "CT", "FL"],
            help="FT=Full-time, PT=Part-time, CT=Contract, FL=Freelance"
        )

    with col2:
        years_experience = st.number_input(
            "Years of Experience",
            min_value=0,
            max_value=40,
            value=5
        )

        education_required = st.selectbox(
            "Education Required",
            ["Associate", "Bachelor", "Master", "PhD"]
        )

        industry = st.text_input(
            "Industry",
            "Technology"
        )

    st.subheader("Company")

    col3, col4 = st.columns(2)

    with col3:
        company_name = st.text_input(
            "Company Name",
            "TechCorp Inc"
        )

        company_size = st.selectbox(
            "Company Size",
            ["S", "M", "L"]
        )

    with col4:
        company_location = st.text_input(
            "Company Location",
            "United States"
        )

        employee_residence = st.text_input(
            "Employee Residence",
            "United States"
        )

    st.subheader("Work Setup")

    col5, col6 = st.columns(2)

    with col5:
        remote_ratio = st.selectbox(
            "Remote Ratio (%)",
            [0, 50, 100],
            index=1
        )

        salary_currency = st.selectbox(
            "Salary Currency",
            ["USD", "EUR", "GBP", "CAD", "AUD", "INR"]
        )

    with col6:
        benefits_score = st.slider(
            "Benefits Score",
            0.0,
            10.0,
            6.5,
            step=0.1
        )

        job_description_length = st.number_input(
            "Job Description Length (characters)",
            min_value=0,
            value=1200
        )

    st.subheader("Skills & Timeline")

    required_skills = st.text_input(
        "Required Skills (comma-separated)",
        "Python, SQL, Machine Learning"
    )

    col7, col8 = st.columns(2)

    with col7:
        posting_date = st.date_input(
            "Posting Date",
            date.today()
        )

    with col8:
        application_deadline = st.date_input(
            "Application Deadline",
            date.today()
        )

    submitted = st.form_submit_button("Predict Salary")

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if submitted:

    input_data = pd.DataFrame({
        "job_title": [job_title],
        "salary_currency": [salary_currency],
        "experience_level": [experience_level],
        "employment_type": [employment_type],
        "company_location": [company_location],
        "company_size": [company_size],
        "employee_residence": [employee_residence],
        "remote_ratio": [remote_ratio],
        "required_skills": [required_skills],
        "education_required": [education_required],
        "years_experience": [years_experience],
        "industry": [industry],
        "posting_date": [posting_date.strftime("%Y-%m-%d")],
        "application_deadline": [application_deadline.strftime("%Y-%m-%d")],
        "job_description_length": [job_description_length],
        "benefits_score": [benefits_score],
        "company_name": [company_name]
    })

    try:
        prediction = model.predict(input_data)[0]

        st.success(
            f"💰 Predicted Salary: ${prediction:,.0f} USD"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Input sent to the model:")
        st.dataframe(input_data)