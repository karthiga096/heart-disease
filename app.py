import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Title
st.title("❤️ Heart Disease Prediction App")

# Load dataset directly (example dataset from UCI Heart Disease)
DATA_URL = "https://raw.githubusercontent.com/atharvakale/Heart-Disease-Prediction-ML/master/heart.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    return df

df = load_data()
st.success("Dataset loaded successfully!")
st.dataframe(df.head())

# Prepare data
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Display accuracy
st.write(f"Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

# User input for prediction
st.subheader("Make a Prediction")
user_input = {}
for column in X.columns:
    user_input[column] = st.number_input(f"Enter {column}", value=float(df[column].mean()))

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)

if prediction[0] == 1:
    st.error("⚠️ High risk of heart disease!")
else:
    st.success("💚 Low risk of heart disease")
