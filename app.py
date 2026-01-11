import streamlit as st
import pickle
import numpy as np
import os

# ---------------------------
# Load the single model safely
# ---------------------------
MODEL_PATH = "disease_model.pkl"  # Place your .pkl here (same folder as app.py)

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    st.error(f"Model file not found! Make sure {MODEL_PATH} exists.")
    st.stop()  # Stop the app if model is missing

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("❤️ Multi-Disease Prediction App")
st.write("Predict Heart Disease, Diabetes, or Kidney Disease easily!")

# Choose disease type
disease = st.selectbox("Select Disease to Predict", ["Heart Disease", "Diabetes", "Kidney Disease"])

# User input fields (example: you can customize features per disease)
age = st.number_input("Age", 1, 120, 30)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
# Example features (customize depending on your model)
feature1 = st.number_input("Feature 1", 0.0, 500.0, 100.0)
feature2 = st.number_input("Feature 2", 0.0, 500.0, 100.0)
feature3 = st.number_input("Feature 3", 0.0, 500.0, 50.0)

# Prepare input for model
user_input = np.array([[age, sex, feature1, feature2, feature3]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(user_input)[0]

    if prediction == 0:
        st.success(f"No {disease} detected ✅")
    else:
        st.error(f"{disease} detected ⚠️")
