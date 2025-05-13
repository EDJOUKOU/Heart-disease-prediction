#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("best_lr.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Heart Disease Prediction App")

st.write("""
### Enter the patient details to check the risk of heart disease
""")

# Create input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest pain type (cp)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting blood pressure (trestbps)", value=120)
chol = st.number_input("Serum cholesterol in mg/dl (chol)", value=200)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", options=[0, 1])
restecg = st.selectbox("Resting electrocardiographic results (restecg)", options=[0, 1, 2])
thalach = st.number_input("Maximum heart rate achieved (thalach)", value=150)
exang = st.selectbox("Exercise induced angina (exang)", options=[0, 1])
oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", value=1.0, step=0.1, format="%.1f")
slope = st.selectbox("Slope of the peak exercise ST segment", options=[0, 1, 2])
ca = st.selectbox("Number of major vessels colored by fluoroscopy (ca)", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

# Prediction
if st.button("Predict"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(features)[0]
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    st.subheader("Prediction Result:")
    st.success(result)


# In[ ]:




