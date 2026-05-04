import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Load Model
# -------------------------------
model = pickle.load(open("heart_model.pkl", "rb"))

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# -------------------------------
# Title Section
# -------------------------------
st.markdown("<h1 style='text-align:center;'>❤️ Heart Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:grey;'>AI-powered risk analysis system</p>", unsafe_allow_html=True)

st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=100)

st.write("")

# -------------------------------
# Input Card
# -------------------------------
st.markdown("""
<div style="background-color:#f5f5f5;padding:20px;border-radius:10px">
<h3>Enter Patient Details</h3>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Input Layout
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 100, 30)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)

with col2:
    chol = st.number_input("Cholesterol", 100, 600, 200)
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    oldpeak = st.slider("Heart Stress Level (Oldpeak)", 0.0, 6.0, 1.0)
    ca = st.selectbox("Major Vessels (0-3)", [0,1,2,3])

# -------------------------------
# Predict Button
# -------------------------------
st.write("")
predict_btn = st.button("🚀 Predict Risk")

# -------------------------------
# Prediction Logic
# -------------------------------
if predict_btn:

    # IMPORTANT: feature order must match training
    input_data = np.array([[age, sex, cp, trestbps, chol, 0, 0, thalach, 0, oldpeak, 0, ca, 2]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    # -------------------------------
    # Result Section
    # -------------------------------
    if prediction == 1:
        st.markdown("<h2 style='color:red;text-align:center;'>⚠️ High Risk of Heart Disease</h2>", unsafe_allow_html=True)
        st.write(f"### Confidence: {probability*100:.2f}%")
    else:
        st.markdown("<h2 style='color:green;text-align:center;'>✅ Low Risk of Heart Disease</h2>", unsafe_allow_html=True)
        st.write(f"### Confidence: {(1-probability)*100:.2f}%")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("About")
st.sidebar.info("""
This app predicts the risk of heart disease using Machine Learning.

Model Used:
- Logistic Regression / Random Forest

Note:
This is for educational purposes only.
""")