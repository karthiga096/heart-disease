import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from io import StringIO

st.title("❤️ Heart Disease Prediction App")

# Embedded CSV data (small example dataset)
csv_data = """age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2,1
41,0,1,130,204,0,0,172,0,1.4,2,0,2,1
56,1,1,120,236,0,1,178,0,0.8,2,0,2,1
57,0,0,120,354,0,1,163,1,0.6,2,0,2,1
57,1,0,140,192,0,1,148,0,0.4,1,0,1,1
56,0,1,140,294,0,0,153,0,1.3,1,0,2,1
44,1,1,120,263,0,1,173,0,0,2,0,3,1
52,1,2,172,199,1,1,162,0,0.5,2,0,3,1
"""

# Load dataset
df = pd.read_csv(StringIO(csv_data))
st.success("Dataset loaded successfully!")
st.dataframe(df)

# Prepare features and labels
X = df.drop("target", axis=1)
y = df["target"]

# Split data
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
