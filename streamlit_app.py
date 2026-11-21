import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import base64
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical aesthetic & dark mode compatibility
st.markdown("""
<style>
    /* Clinical color palette */
    :root {
        --primary-teal: #14b8a6;
        --primary-teal-dark: #0f766e;
        --primary-teal-light: #5eead4;
        --accent-green: #10b981;
        --accent-amber: #fbbf24;
        --accent-red: #ef4444;
        --neutral-white: #ffffff;
        --neutral-gray-50: #f8fafc;
        --neutral-gray-100: #f1f5f9;
        --neutral-gray-200: #e2e8f0;
        --neutral-gray-600: #475569;
        --neutral-gray-800: #1e293b;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
    }

    /* Main container styling */
    .stApp {
        background-color: var(--neutral-gray-50);
        color: var(--neutral-gray-800);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    [data-theme="dark"] .stApp {
        background-color: #0f172a !important;
        color: #f1f5f9 !important;
    }

    [data-theme="dark"] .metric-card,
    [data-theme="dark"] .input-section,
    [data-theme="dark"] .clinical-context-box,
    [data-theme="dark"] .ci-box {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: #334155 !important;
    }

    [data-theme="dark"] .input-section-title {
        color: var(--primary-teal-light) !important;
        border-bottom: 2px solid #334155 !important;
    }

    [data-theme="dark"] p,
    [data-theme="dark"] h1,
    [data-theme="dark"] h2,
    [data-theme="dark"] h3,
    [data-theme="dark"] h4,
    [data-theme="dark"] label,
    [data-theme="dark"] div[data-testid="stMarkdownContainer"] {
        color: #f1f5f9 !important;
    }

    /* Target download buttons and text inside them in sidebar dark mode */
    [data-theme="dark"] .export-button {
        color: #ffffff !important;
    }
    [data-theme="dark"] .export-button-secondary {
        color: var(--primary-teal-light) !important;
        border-color: var(--primary-teal-light) !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    [data-theme="dark"] .export-button-secondary:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
    }

    [data-theme="dark"] .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-color: #334155 !important;
        color: #94a3b8 !important;
    }

    [data-theme="dark"] .stTabs [aria-selected="true"] {
        background-color: var(--primary-teal) !important;
        color: var(--neutral-white) !important;
        border-color: var(--primary-teal) !important;
    }

    [data-theme="dark"] .stMetric [data-testid="metric-value"] {
        color: #f1f5f9 !important;
    }

    /* Professional card styling */
    .metric-card {
        background: var(--neutral-white);
        border-radius: 12px;
        padding: 24px;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--neutral-gray-200);
        transition: all 0.2s ease;
        margin-bottom: 16px;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-color: var(--primary-teal-light);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        background-color: var(--neutral-white);
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        margin-right: 4px;
        border: 1px solid var(--neutral-gray-200);
        border-bottom: none;
        font-weight: 500;
        color: var(--neutral-gray-600);
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--neutral-gray-50);
        color: var(--primary-teal-dark);
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-teal);
        color: var(--neutral-white);
        border-color: var(--primary-teal);
    }

    /* Risk gauge styling */
    .risk-gauge {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--neutral-white);
        margin: 0 auto 20px auto;
        position: relative;
        box-shadow: var(--shadow-lg);
    }

    .risk-low { background: linear-gradient(135deg, var(--accent-green), #059669); }
    .risk-moderate { background: linear-gradient(135deg, var(--accent-amber), #d97706); }
    .risk-high { background: linear-gradient(135deg, var(--accent-red), #dc2626); }

    /* Input section styling */
    .input-section {
        background-color: var(--neutral-white);
        border-radius: 8px;
        padding: 20px;
        border: 1px solid var(--neutral-gray-200);
        margin-bottom: 16px;
    }

    .input-section-title {
        color: var(--primary-teal-dark);
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--neutral-gray-100);
    }

    /* Enhanced metric styling */
    .stMetric {
        background: var(--neutral-white);
        border-radius: 8px;
        padding: 16px;
        border: 1px solid var(--neutral-gray-200);
        box-shadow: var(--shadow-sm);
    }

    .stMetric label {
        color: var(--neutral-gray-600) !important;
        font-weight: 500 !important;
    }

    .stMetric [data-testid="metric-value"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--neutral-gray-800) !important;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: var(--neutral-gray-50);
        padding: 20px;
    }

    /* Export button styling */
    .export-button {
        background-color: var(--primary-teal);
        color: var(--neutral-white);
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
        display: inline-block;
        margin: 8px;
    }

    .export-button:hover {
        background-color: var(--primary-teal-dark);
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
        color: var(--neutral-white);
    }

    .export-button-secondary {
        background-color: var(--neutral-white);
        color: var(--primary-teal-dark);
        border: 2px solid var(--primary-teal-dark);
    }

    .export-button-secondary:hover {
        background-color: var(--neutral-gray-50);
        color: var(--primary-teal-dark);
    }

    /* Clinical Context box styling */
    .clinical-context-box {
        background-color: var(--neutral-gray-100);
        border: 1px solid var(--neutral-gray-200);
        border-radius: 8px;
        padding: 16px;
        margin-top: 20px;
    }

    .ci-box {
        background: var(--neutral-gray-100);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }

    .stSuccess {
        background-color: #f0fdf4;
        border-color: var(--accent-green);
        color: #166534;
    }

    .stError {
        background-color: #fef2f2;
        border-color: var(--accent-red);
        color: #991b1b;
    }

    .stWarning {
        background-color: #fffbeb;
        border-color: var(--accent-amber);
        color: #92400e;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--neutral-gray-800);
        font-weight: 600;
        letter-spacing: -0.025em;
    }

    p {
        color: var(--neutral-gray-600);
        line-height: 1.6;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .risk-gauge {
            width: 150px;
            height: 150px;
            font-size: 1.8rem;
        }

        .metric-card {
            padding: 16px;
        }
    }
</style>

<!-- Google Fonts -->
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    """Train and cache the Logistic Regression model using joblib for efficiency."""
    data = pd.read_csv("health_diagnosis_dataset.csv")
    X = data.drop(columns="Diagnosis")
    y = data["Diagnosis"]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    model.fit(X, y)
    return model, X.columns

# Load model once
model, feature_names = train_model()

# Helper function to classify risk tiers
def classify_risk_tier(probability):
    """Classify risk probability into clinical tiers."""
    prob_pct = probability * 100
    if prob_pct < 20:
        return "Low Risk", "var(--accent-green)", "🟢"
    elif prob_pct <= 60:
        return "Moderate Risk", "var(--accent-amber)", "🟡"
    else:
        return "High Risk", "var(--accent-red)", "🔴"

# Helper function to generate reports
def generate_reports(patient_data, risk_prob, risk_tier, top_features):
    """Generate text and CSV reports for export."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate text report
    text_report = f"""TYPE 2 DIABETES RISK PREDICTION REPORT
===================================================
Generated: {timestamp}
Report Type: Clinical Risk Assessment

PATIENT METRICS:
"""
    for key, value in patient_data.items():
        if key in ['Gender', 'Smoking', 'Alcohol', 'FamilyHistory']:
            text_value = "Yes" if value == 1 else "No"
        else:
            text_value = str(value)
        text_report += f"  {key}: {text_value}\n"

    text_report += f"""
PREDICTION RESULTS:
  Risk Probability: {risk_prob:.2%}
  Risk Tier: {risk_tier}
  Diabetes Risk: {'POSITIVE' if risk_prob >= 0.5 else 'NEGATIVE'}

TOP CONTRIBUTING FACTORS:
"""
    for i, (feat, contrib) in enumerate(top_features.items(), 1):
        direction = "increases" if contrib > 0 else "decreases"
        text_report += f"  {i}. {feat}: {contrib:+.3f} ({direction} risk)\n"

    text_report += f"""
CLINICAL SUMMARY:
  Based on Logistic Regression model with StandardScaler preprocessing.
  Model trained on 15,000 synthetic patient records.
  ROC-AUC: 0.906 | Accuracy: 82.67%

DISCLAIMER:
  This is a demonstration model trained on synthetic data.
  Not for clinical use. Consult healthcare professionals for medical decisions.
"""

    # Generate CSV report
    csv_data = {
        'Timestamp': [timestamp],
        'Risk_Probability': [risk_prob],
        'Risk_Tier': [risk_tier],
        'Diabetes_Risk': ['POSITIVE' if risk_prob >= 0.5 else 'NEGATIVE']
    }

    for key, value in patient_data.items():
        csv_data[key] = [value]

    for i, (feat, contrib) in enumerate(top_features.items(), 1):
        csv_data[f'Top_Feature_{i}'] = [feat]
        csv_data[f'Contribution_{i}'] = [contrib]

    csv_df = pd.DataFrame(csv_data)

    return text_report, csv_df

# Sidebar export functionality
def sidebar_export(patient_data, risk_prob, risk_tier, top_features):
    """Create export functionality in sidebar."""
    st.sidebar.markdown("## 📄 Export Assessment")

    # Generate reports
    text_report, csv_df = generate_reports(
        patient_data, risk_prob, risk_tier, top_features
    )

    # Download buttons
    col1, col2 = st.sidebar.columns(2)

    with col1:
        # Text report download
        b64_text = base64.b64encode(text_report.encode()).decode()
        href_text = f"<a href='data:text/plain;base64,{b64_text}' download='diabetes_risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt' class='export-button'>📝 TXT Report</a>"
        st.sidebar.markdown(href_text, unsafe_allow_html=True)

    with col2:
        # CSV report download
        csv = csv_df.to_csv(index=False)
        b64_csv = base64.b64encode(csv.encode()).decode()
        href_csv = f"<a href='data:text/csv;base64,{b64_csv}' download='diabetes_risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv' class='export-button export-button-secondary'>📊 CSV Report</a>"
        st.sidebar.markdown(href_csv, unsafe_allow_html=True)

    # Display report preview
    st.sidebar.markdown("## 📋 Report Preview")

    with st.sidebar.expander("Text Report Preview"):
        st.text_area("Text Report", text_report[:1000] + "..." if len(text_report) > 1000 else text_report, height=200)

    with st.sidebar.expander("CSV Data Preview"):
        st.dataframe(csv_df.head(), use_container_width=True)

# Main application
st.title("🩺 Type 2 Diabetes Risk Predictor")

st.markdown("""
A **Logistic Regression** model trained on 15,000 synthetic patient records.
Adjust the sliders in each section to estimate a patient's risk.

**Disclaimer:** Demonstration model trained on synthetic data. Not for clinical use.
""")

st.divider()

# Create tabs for organized input sections
Tab1, Tab2, Tab3 = st.tabs([
    "👤 Patient Demographics",
    "🩺 Metabolic Blood Markers",
    "🏥 Lifestyle & Genetic History"
])

# Patient Demographics Tab (Fixed: removed empty placeholder container box above title)
with Tab1:
    st.markdown('<div class="input-section-title">Patient Demographics</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 90, 45, help="Patient's age in years")

    with col2:
        gender_options = ["Female", "Male"]
        gender = st.selectbox("Gender", gender_options, help="Patient's gender")

# Metabolic Blood Markers Tab
with Tab2:
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<div class="input-section-title">Metabolic Blood Markers</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            glucose = st.slider("Glucose (mg/dL)", 60, 250, 105, help="Blood glucose level")
            insulin = st.slider("Insulin (mu U/mL)", 10, 300, 90, help="Insulin level")
            blood_pressure = st.slider("Blood Pressure (mmHg)", 90, 200, 120, help="Systolic blood pressure")

        with col2:
            cholesterol = st.slider("Cholesterol (mg/dL)", 120, 350, 200, help="Total cholesterol")
            bmi = st.slider("BMI", 15.0, 50.0, 27.0, help="Body Mass Index")
            skin_thickness = st.slider("Skin Thickness (mm)", 5, 60, 20, help="Triceps skin fold thickness")
            heart_rate = st.slider("Heart Rate (bpm)", 50, 120, 75, help="Resting heart rate")

        st.markdown('</div>', unsafe_allow_html=True)

# Lifestyle & Genetic History Tab
with Tab3:
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<div class="input-section-title">Lifestyle & Genetic History</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            physical_activity = st.slider("Physical Activity (hrs/week)", 0.0, 10.0, 3.0, help="Weekly physical activity duration")
            smoking = st.selectbox("Smoking", ["No", "Yes"], help="Smoking status")
            alcohol = st.selectbox("Alcohol", ["No", "Yes"], help="Alcohol consumption")

        with col2:
            family_history = st.selectbox("Family History of Diabetes", ["No", "Yes"], help="Family diabetes history")
            genetic_risk = st.slider("Genetic Risk Score", 0.0, 1.0, 0.5, help="Genetic predisposition score (0-1)")

        st.markdown('</div>', unsafe_allow_html=True)

# Process patient data
patient_data = {
    "Age": age,
    "Gender": 1 if gender == "Male" else 0,
    "BMI": bmi,
    "BloodPressure": blood_pressure,
    "Glucose": glucose,
    "Insulin": insulin,
    "SkinThickness": skin_thickness,
    "HeartRate": heart_rate,
    "Cholesterol": cholesterol,
    "PhysicalActivity": physical_activity,
    "Smoking": 1 if smoking == "Yes" else 0,
    "Alcohol": 1 if alcohol == "Yes" else 0,
    "FamilyHistory": 1 if family_history == "Yes" else 0,
    "GeneticRiskScore": genetic_risk
}

input_data = pd.DataFrame([patient_data])
input_data = input_data[feature_names]

# Prediction
prob = model.predict_proba(input_data)[0][1]
pred = int(prob >= 0.5)
risk_tier, risk_color, risk_emoji = classify_risk_tier(prob)

# Get coefficients for contribution analysis
coef = model.named_steps["clf"].coef_[0]
contrib = input_data.iloc[0] * coef
contrib_sorted = contrib.sort_values(ascending=False)
top_features = contrib_sorted.head(6).to_dict()

# Enhanced Risk Display
st.subheader("🎯 Risk Assessment")

# Create three-column layout for metrics
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Risk gauge
    gauge_class = f"risk-gauge risk-{risk_tier.lower().replace(' ', '-')}"
    st.markdown(f'''
    <div class="{gauge_class}">
        {risk_emoji}
    </div>
    ''', unsafe_allow_html=True)

    # Risk tier display
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: {risk_color}; margin-bottom: 8px;">{risk_tier}</h2>
        <p style="color: var(--neutral-gray-600); font-size: 1.1rem;">{prob*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # Determine if prediction is positive or negative
    if pred == 1:
        st.error("⚠️ **Prediction: Positive** for diabetes risk")
        st.write("Consider clinical follow-up and further diagnostic testing.")
    else:
        st.success("✅ **Prediction: Negative** for diabetes risk")
        st.write("Maintain healthy lifestyle habits and routine screening.")

    # Probability gauge with confidence interval
    import math
    se = math.sqrt(prob * (1 - prob) / len(input_data))
    ci_lower = max(0, prob - 1.96 * se)
    ci_upper = min(1, prob + 1.96 * se)

    st.markdown(f"""
    <div class="ci-box">
        <p style="margin-bottom: 8px; font-size: 0.9rem;">95% Confidence Interval</p>
        <p style="font-size: 1.2rem; font-weight: 600;">
            <span style="color: var(--primary-teal);">{ci_lower*100:.1f}%</span> -
            <span style="color: #6366f1;">{ci_upper*100:.1f}%</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Clinical Context
    st.markdown("""
    <div class="clinical-context-box">
        <h4 style="color: var(--primary-teal-dark); margin-bottom: 8px;">Clinical Context</h4>
        <p style="font-size: 0.95rem;">
            This assessment is based on a Logistic Regression model trained on 15,000 synthetic patient records.
            It evaluates metabolic and demographic risk factors to provide a preliminary diabetes risk estimate.
            <br><br>
            <strong>High-risk indicators:</strong> Elevated glucose, higher age, family history, and increased BMI are primary contributors to higher risk scores.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Export functionality
sidebar_export(patient_data, prob, risk_tier, top_features)

# Model information footer
st.divider()
st.caption("""
Model: Logistic Regression (StandardScaler, max_iter=2000) |
ROC-AUC: 0.906 | Data: 15,000 synthetic records
""")
