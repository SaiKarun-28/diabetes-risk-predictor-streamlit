import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

st.title("Type 2 Diabetes Risk Predictor")
st.markdown(
    "A **Logistic Regression** model trained on 15,000 synthetic patient records. "
    "Adjust the sliders to estimate a patient's risk. "
    "**Disclaimer:** Demonstration model trained on synthetic data. Not for clinical use."
)

@st.cache_data
def train_model():
    data = pd.read_csv("health_diagnosis_dataset.csv")
    X = data.drop(columns="Diagnosis")
    y = data["Diagnosis"]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    model.fit(X, y)
    return model, X.columns

model, feature_names = train_model()

st.divider()
st.subheader("Patient Inputs")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 90, 45)
    bmi = st.slider("BMI", 15.0, 50.0, 27.0)
    glucose = st.slider("Glucose (mg/dL)", 60, 250, 105)
    insulin = st.slider("Insulin (mu U/mL)", 10, 300, 90)

with col2:
    blood_pressure = st.slider("Blood Pressure (mmHg)", 90, 200, 120)
    cholesterol = st.slider("Cholesterol (mg/dL)", 120, 350, 200)
    heart_rate = st.slider("Heart Rate (bpm)", 50, 120, 75)
    skin_thickness = st.slider("Skin Thickness (mm)", 5, 60, 20)

st.write("### Lifestyle & History")
col3, col4 = st.columns(2)
with col3:
    physical_activity = st.slider("Physical Activity (hrs/week)", 0.0, 10.0, 3.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol", ["No", "Yes"])
with col4:
    family_history = st.selectbox("Family History of Diabetes", ["No", "Yes"])
    genetic_risk = st.slider("Genetic Risk Score", 0.0, 1.0, 0.5)
    gender = st.selectbox("Gender", ["Female", "Male"])

input_data = pd.DataFrame([{
    "Age": age, "Gender": 1 if gender == "Male" else 0, "BMI": bmi,
    "BloodPressure": blood_pressure, "Glucose": glucose, "Insulin": insulin,
    "SkinThickness": skin_thickness, "HeartRate": heart_rate,
    "Cholesterol": cholesterol, "PhysicalActivity": physical_activity,
    "Smoking": 1 if smoking == "Yes" else 0,
    "Alcohol": 1 if alcohol == "Yes" else 0,
    "FamilyHistory": 1 if family_history == "Yes" else 0,
    "GeneticRiskScore": genetic_risk
}])

input_data = input_data[feature_names]

st.divider()
prob = model.predict_proba(input_data)[0][1]
pred = int(prob >= 0.5)

st.subheader("Prediction Result")
risk_col = "inverse" if prob < 0.5 else "normal"
st.metric("Diabetes Risk Probability", f"{prob*100:.1f}%")
st.progress(float(prob))

if pred == 1:
    st.error("Prediction: **Positive** for diabetes risk")
    st.write("Consider clinical follow-up and further diagnostic testing.")
else:
    st.success("Prediction: **Negative** for diabetes risk")
    st.write("Maintain healthy lifestyle habits and routine screening.")

st.divider()
st.subheader("Top Contributing Factors")
coef = model.named_steps["clf"].coef_[0]
contrib = input_data.iloc[0] * coef
contrib_sorted = contrib.sort_values(ascending=False)

for feat, val in contrib_sorted.head(6).items():
    direction = "increases risk" if val > 0 else "reduces risk"
    st.write(f"- **{feat}**: {val:+.2f} ({direction})")

st.divider()
st.caption("Model: Logistic Regression (StandardScaler, max_iter=2000) | ROC-AUC: 0.906 | Data: 15,000 synthetic records")
