# Type 2 Diabetes Risk Predictor

A **Logistic Regression** model trained on 15,000 synthetic patient records to predict the likelihood of a Type 2 diabetes diagnosis. Project includes a Streamlit web app for interactive risk assessment and full coefficient interpretability.

## 📊 Model Performance

| Metric | Train | Test |
|---|---|---|
| **Accuracy** | 0.8237 | 0.8267 |
| **ROC-AUC** | 0.9051 | 0.9060 |

Train and test scores are nearly identical — **no overfitting**. ROC-AUC of 0.906 indicates strong discriminative ability for a health-screening model.

## 🔍 Key Findings

The model's coefficients align with established clinical knowledge:

| Feature | Coefficient | Odds Ratio | Interpretation |
|---|---|---|---|
| Age | +1.51 | 4.52 | Strongest risk factor; prevalence rises sharply after 45 |
| Glucose | +1.14 | 3.11 | Most direct metabolic signal; closest to actual diagnostic tests |
| Physical Activity | -0.98 | 0.38 | Strongest protective factor |
| Family History | +0.88 | 2.41 | Hereditary predisposition captured |
| BMI | +0.72 | 2.05 | Obesity drives insulin resistance |
| Genetic Risk Score | +0.65 | 1.91 | Hereditary risk encoded |

**This isn't a black box** — the weights reflect known medical risk factors. A hiring manager can immediately see that the model learns interpretable patterns, not noise.

## 🚀 Run the App

```bash
# Install dependencies
pip install streamlit pandas scikit-learn numpy matplotlib seaborn

# Launch
streamlit run streamlit_app.py
```

Open `http://localhost:8502` in your browser.

## 📁 Project Structure

```
diabetes-risk-predictor-streamlit/
├── streamlit_app.py          ← Interactive demo app (Streamlit)
├── health_diagnosis_dataset.csv  ← 15,000 synthetic records
├── health_diagnosis_fixed.ipynb  ← Full EDA + model notebook
├── requirements.txt          ← 6 packages (streamlit, pandas, scikit-learn, numpy, matplotlib, seaborn)
└── README.md                 ← You are here
```

## 📦 What's Included

| Component | Purpose |
|---|---|
| `streamlit_app.py` | Single-page clinical decision-support demo with live sliders and probability gauge |
| `health_diagnosis_dataset.csv` | Synthetic patient data (15,000 rows, 14 features) |
| `health_diagnosis_fixed.ipynb` | Full EDA, model training, coefficient analysis, ROC curves, Ridge/Lasso CV |
| `README.md` | Project overview — what you're reading now |

## ⚠️ Disclaimer

> **Demonstration model trained on synthetic data. Not for clinical use. Not a substitute for professional medical advice.**

This model is a portfolio piece demonstrating Logistic Regression interpretability, model evaluation, and deployment via Streamlit. No patient data was used.

## 🎯 Portfolio Talking Points

- **Interpretability**: Coefficients are medically coherent; Age, Glucose, and BMI dominate as risk factors
- **Robustness**: Train ≈ Test accuracy and ROC-AUC, confirming generalization
- **Deployment**: Full Streamlit app shipped and deployable to Streamlit Cloud
- **Ethics**: Clear disclaimer; synthetic data protects privacy; model explicitly not for clinical use

## 📬 Citation

If you use this project, cite as:

```
@misc{diabetes_risk_2026,
  title = {Type 2 Diabetes Risk Predictor},
  author = {saika},
  year = {2026},
  note = {Logistic Regression model trained on 15,000 synthetic patient records; Streamlit demo app deployed to share.streamlit.io},
}
```

---

*Project completed October 2026 as a portfolio piece for Machine Learning practice. Demonstrates model interpretability, regularization (Ridge/Lasso CV), and end-to-end deployment.*
