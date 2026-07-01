import streamlit as st
import joblib
import pandas as pd

# Load files
model = joblib.load("mental_health_model.pkl")
encoders = joblib.load("label_encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Student Mental Health Prediction System")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 15, 40, 20)
course = st.text_input("Course")
cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Course": [course],
        "CGPA": [cgpa]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Mental Health Risk Detected")
    else:
        st.success("No Mental Health Risk Detected")
