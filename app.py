import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("XGBoostClassifier_best.pkl")

# Streamlit App Title
st.title("Diabetes Prediction App")
st.write("Enter the details below to predict whether a person has diabetes.")

# Create input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=500, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Prediction Button
if st.button("Predict"):
    # Create a NumPy array for the input values
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

    # Make Prediction
    prediction = model.predict(input_data)

    # Display Result
    if prediction[0] == 1:
        st.error("🚨 The model predicts that the person **has diabetes.**")
    else:
        st.success("✅ The model predicts that the person **does not have diabetes.**")
