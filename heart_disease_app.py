#!/usr/bin/env python
# coding: utf-8

# In[1]:


# heart_disease_app.py
import pickle
import numpy as np
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Function to load the model with error handling
def load_model():
    try:
        with open("best_lr.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'best_lr.pkl' not found. Please ensure it's in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Load the model
model = load_model()

# App header
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This app predicts the probability of heart disease based on clinical parameters.
Enter the patient's details below and click **Predict**.
""")

# Input section with two columns
st.header("Patient Information")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])
    sex = sex[1]  # Get the numeric value
    
    st.subheader("Symptoms")
    cp = st.selectbox(
        "Chest Pain Type", 
        options=[("Typical angina", 0), ("Atypical angina", 1), 
                ("Non-anginal pain", 2), ("Asymptomatic", 3)],
        format_func=lambda x: x[0]
    )
    cp = cp[1]  # Get the numeric value
    
    exang = st.radio(
        "Exercise Induced Angina", 
        options=[("No", 0), ("Yes", 1)],
        format_func=lambda x: x[0]
    )
    exang = exang[1]  # Get the numeric value

with col2:
    st.subheader("Clinical Measurements")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.radio(
        "Fasting Blood Sugar > 120 mg/dl", 
        options=[("No", 0), ("Yes", 1)],
        format_func=lambda x: x[0]
    )
    fbs = fbs[1]  # Get the numeric value
    
    st.subheader("Test Results")
    restecg = st.selectbox(
        "Resting ECG", 
        options=[("Normal", 0), ("ST-T wave abnormality", 1), 
                ("Probable left ventricular hypertrophy", 2)],
        format_func=lambda x: x[0]
    )
    restecg = restecg[1]  # Get the numeric value
    
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment", 
        options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
        format_func=lambda x: x[0]
    )
    slope = slope[1]  # Get the numeric value
    
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "Thalassemia", 
        options=[("Normal", 0), ("Fixed defect", 1), 
                ("Reversible defect", 2), ("Unknown", 3)],
        format_func=lambda x: x[0]
    )
    thal = thal[1]  # Get the numeric value

# Prediction button
if st.button("Predict Heart Disease Risk", type="primary"):
    # Create feature array
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal]])
    
    # Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1] * 100
    
    # Display results
    st.divider()
    st.header("Prediction Results")
    
    if prediction[0] == 1:
        st.error(f"🚨 High risk of heart disease ({probability:.1f}% probability)")
        with st.expander("Recommendations for High Risk"):
            st.markdown("""
            - **Consult a cardiologist immediately**
            - Schedule a comprehensive cardiac evaluation including:
              - Stress test
              - Echocardiogram
              - Coronary angiography
            - Monitor for symptoms:
              - Chest pain/discomfort
              - Shortness of breath
              - Fatigue
            - Lifestyle modifications:
              - Quit smoking
              - Healthy diet (Mediterranean recommended)
              - Regular moderate exercise
            """)
    else:
        st.success(f"✅ Low risk of heart disease ({100-probability:.1f}% probability)")
        with st.expander("Recommendations for Low Risk"):
            st.markdown("""
            - **Maintain heart-healthy habits**:
              - Balanced diet (fruits, vegetables, whole grains)
              - Regular physical activity (150 mins/week)
              - Maintain healthy weight
            - **Regular checkups**:
              - Annual physical exam
              - Monitor blood pressure and cholesterol
              - Consider cardiac screening after age 40
            - **Risk reduction**:
              - Manage stress
              - Limit alcohol
              - Control blood sugar if diabetic
            """)
    
    # Visual indicator
    st.progress(int(probability), text=f"Risk Score: {probability:.1f}%")
    
    # Model information
    st.caption("""
    Model: Logistic Regression (C=0.1, max_iter=100, penalty='l2', solver='liblinear')
    Accuracy: ~85% (on test set)
    """)

# Footer
st.divider()
st.markdown("""
**Note**: This prediction is for informational purposes only and should not replace professional medical advice. 
Always consult with a healthcare provider for medical diagnosis and treatment.
""")


# In[ ]:




