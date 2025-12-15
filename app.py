import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Title
st.title("❤️ Heart Disease Prediction App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your heart.csv file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head())

    # Split data
    X = df.drop("target", axis=1)  # Replace 'target' with your label column name
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Accuracy
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
else:
    st.info("Please upload the heart.csv file to continue.")
