import streamlit as st
import pickle
import os

st.set_page_config(page_title="Multi-Disease Prediction App", layout="centered")
st.title("❤️ Multi-Disease Prediction App")
st.write("Predict Heart Disease, Diabetes, or Kidney Disease easily!")

# ===============================
# 1️⃣ Load the single multi-disease model
# ===============================
MODEL_PATH = "/mnt/data/f205e730-9a31-4caf-8769-b686e297f20c.pkl"

@st.cache_resource
def load_all_models(path):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)  # Expecting a dictionary of models

all_models = load_all_models(MODEL_PATH)
if all_models is None:
    st.stop()

# ===============================
# 2️⃣ Select Disease
# ===============================
disease = st.selectbox("Select Disease to Predict", ["Heart Disease", "Diabetes", "Kidney Disease"])

model = all_models.get(disease)
if model is None:
    st.warning("No model available for this disease")
    st.stop()

# ===============================
# 3️⃣ Input form
# ===============================
st.header(f"{disease} Prediction")

with st.form("disease_form"):
    if disease == "Heart Disease":
        age = st.number_input("Age", 1, 120, 50)
        cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 1)
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0=No,1=Yes)", [0,1])
        restecg = st.number_input("Resting ECG (0-2)", 0, 2, 1)
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina (0=No,1=Yes)", [0,1])
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
        submit = st.form_submit_button("Predict")

        if submit:
            features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
            prediction = model.predict(features)[0]
            st.success(f"Prediction: {'Heart Disease' if prediction==1 else 'No Heart Disease'}")

    elif disease == "Diabetes":
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose Level", 0, 300, 120)
        bp = st.number_input("Blood Pressure", 0, 200, 70)
        skinthickness = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin Level", 0, 900, 79)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.number_input("Age", 1, 120, 30)
        submit = st.form_submit_button("Predict")

        if submit:
            features = [[pregnancies, glucose, bp, skinthickness, insulin, bmi, dpf, age]]
            prediction = model.predict(features)[0]
            st.success(f"Prediction: {'Diabetic' if prediction==1 else 'Non-Diabetic'}")

    elif disease == "Kidney Disease":
        age = st.number_input("Age", 1, 120, 50)
        bp = st.number_input("Blood Pressure", 50, 200, 80)
        sg = st.number_input("Specific Gravity (1.005-1.025)", 1.005, 1.025, 1.010)
        al = st.number_input("Albumin (0-5)", 0, 5, 0)
        su = st.number_input("Sugar (0-5)", 0, 5, 0)
        rbc = st.selectbox("Red Blood Cells (Normal/Abnormal)", ["Normal","Abnormal"])
        pc = st.selectbox("Pus Cell (Normal/Abnormal)", ["Normal","Abnormal"])
        submit = st.form_submit_button("Predict")

        # Encode categorical features
        rbc_val = 1 if rbc=="Normal" else 0
        pc_val = 1 if pc=="Normal" else 0

        if submit:
            features = [[age, bp, sg, al, su, rbc_val, pc_val]]
            prediction = model.predict(features)[0]
            st.success(f"Prediction: {'Kidney Disease' if prediction==1 else 'No Kidney Disease'}")
