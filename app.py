
import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and scaler
model  = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# --- Page Config ---
st.set_page_config(
    page_title="Clinical Decision Support",
    page_icon="🏥",
    layout="wide"
)

# --- Header ---
st.title("🏥 AI-Powered Clinical Decision Support System")
st.markdown("#### Diabetes Risk Prediction with Explainable AI")
st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header("🩺 Enter Patient Details")

pregnancies  = st.sidebar.slider("Pregnancies",           0, 17, 1)
glucose      = st.sidebar.slider("Glucose Level",         0, 200, 120)
blood_press  = st.sidebar.slider("Blood Pressure",        0, 122, 70)
skin_thick   = st.sidebar.slider("Skin Thickness",        0, 99, 20)
insulin      = st.sidebar.slider("Insulin Level",         0, 846, 80)
bmi          = st.sidebar.slider("BMI",                   0.0, 67.1, 25.0)
dpf          = st.sidebar.slider("Diabetes Pedigree Fn",  0.0, 2.5, 0.5)
age          = st.sidebar.slider("Age",                   21, 81, 30)

input_data = np.array([[pregnancies, glucose, blood_press, skin_thick,
                         insulin, bmi, dpf, age]])

# --- Predict Button ---
if st.sidebar.button("🔍 Predict"):

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    st.markdown("## 📋 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error(f"⚠️ **HIGH RISK** — Diabetes Detected")
        else:
            st.success(f"✅ **LOW RISK** — No Diabetes Detected")
        st.metric("Risk Probability", f"{probability*100:.1f}%")

    with col2:
        # Risk gauge bar
        st.markdown("#### Risk Level")
        st.progress(float(probability))

    st.markdown("---")

    # --- SHAP Explanation ---
    st.markdown("## 🔬 Why This Prediction? (SHAP Explanation)")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values         = shap_values[0],
            base_values    = explainer.expected_value,
            data           = input_scaled[0],
            feature_names  = feature_names
        ),
        show=False
    )
    st.pyplot(fig)

    st.markdown("---")

    # --- Patient Summary ---
    st.markdown("## 📊 Patient Summary")
    import pandas as pd
    summary = pd.DataFrame({
        "Feature" : feature_names,
        "Value"   : input_data[0]
    })
    st.dataframe(summary, use_container_width=True)

else:
    st.info("👈 Enter patient details in the sidebar and click **Predict**")
    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model",     "XGBoost")
    col2.metric("Accuracy",  "~80%")
    col3.metric("ROC-AUC",   "~0.86")
