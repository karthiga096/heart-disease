import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Multi-Disease Prediction App", layout="wide")

st.title("❤️ Multi-Disease Prediction App")
st.markdown("Predict **Heart Disease**, **Diabetes**, or **Kidney Disease** easily!")

# Sidebar for disease selection
disease = st.sidebar.selectbox(
    "Select Disease to Predict",
    ["Heart Disease", "Diabetes", "Kidney Disease"]
)

# Load models
@st.cache_resource
def load_model(file):
    with open(file, "rb") as f:
        return pickle.load(f)

if disease == "Heart Disease":
    model = load_model("models/heart_disease_model.pkl")
elif disease == "Diabetes":
    model = load_model("models/diabetes_model.pkl")
else:
    model = load_model("models/kidney_disease_model.pkl")

st.subheader(f"{disease} Prediction")

# Input fields depending on disease
def user_input_heart():
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl (0=No,1=Yes)", [0,1])
    restecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
    thalach = st.number_input("Max Heart Rate", 50, 250, 150)
    exang = st.selectbox("Exercise Induced Angina (0=No,1=Yes)", [0,1])
    oldpeak = st.number_input("ST depression", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope (0-2)", [0,1,2])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])
    thal = st.selectbox("Thal (1=Normal,2=Fixed,3=Reversible)", [1,2,3])
    return np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

def user_input_diabetes():
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 50, 250, 120)
    blood_pressure = st.number_input("Blood Pressure", 50, 150, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin Level", 0, 900, 30)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 33)
    return np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

def user_input_kidney():
    age = st.number_input("Age", 1, 120, 50)
    bp = st.number_input("Blood Pressure", 50, 200, 80)
    sg = st.number_input("Specific Gravity", 1.0, 1.030, 1.020, step=0.001)
    al = st.number_input("Albumin", 0, 5, 1)
    su = st.number_input("Sugar", 0, 5, 0)
    bgr = st.number_input("Blood Glucose Random", 50, 500, 120)
    bu = st.number_input("Blood Urea", 5, 200, 30)
    sc = st.number_input("Serum Creatinine", 0.2, 20.0, 1.0)
    sod = st.number_input("Sodium", 100, 160, 135)
    pot = st.number_input("Potassium", 2.5, 6.5, 4.0)
    hemo = st.number_input("Hemoglobin", 5.0, 20.0, 13.0)
    pcv = st.number_input("Packed Cell Volume", 20, 60, 40)
    return np.array([[age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv]])

# Get user input
if disease == "Heart Disease":
    input_data = user_input_heart()
elif disease == "Diabetes":
    input_data = user_input_diabetes()
else:
    input_data = user_input_kidney()

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error(f"⚠️ High risk of {disease}! Be cautious and consult a doctor.")
    else:
        st.success(f"✅ Low risk of {disease}. Keep maintaining a healthy lifestyle!")

    if prob is not None:
        st.info(f"Prediction Probability: {prob*100:.2f}%")
