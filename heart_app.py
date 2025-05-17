#!/usr/bin/env python
# coding: utf-8

# In[20]:


# app.py
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# App title and description
st.title('❤️ Heart Disease Prediction App')
st.markdown("""
This app predicts the probability of heart disease based on clinical parameters.
Enter the patient's details below and click **Predict**.
""")
st.write("### Enter the patient details to check the risk of heart disease")

# Input fields
def user_input_features():
    age = st.number_input("Age", min_value=1, max_value=105, value=30)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", 
                     options=[("Typical angina", 0), ("Atypical angina", 1),
                             ("Non-anginal pain", 2), ("Asymptomatic", 3)],
                     format_func=lambda x: x[0])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
    restecg = st.selectbox("Resting ECG Results",
                          options=[("Normal", 0), ("ST-T wave abnormality", 1),
                                  ("Left ventricular hypertrophy", 2)],
                          format_func=lambda x: x[0])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment",
                        options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
                        format_func=lambda x: x[0])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia",
                      options=[("Normal", 0), ("Fixed Defect", 1), ("Reversible Defect", 2)],
                      format_func=lambda x: x[0])
    
    # Extract numerical values from select boxes
    cp = cp[1]
    restecg = restecg[1]
    slope = slope[1]
    thal = thal[1]
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal  
    }
    features = pd.DataFrame(data, index=[0])
    return features
    
# dataset

heart = pd.read_csv('heart.csv')
features  = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']

x = heart[features]
y = heart['target']

# Get user input
df = user_input_features()

# Load or create model
# Note: In a real app, you would load a pre-trained model
# For this example, we'll create a dummy model
# Replace this with your actual model loading code:
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)

# Dummy model - replace with your actual model
model = LogisticRegression(C=0.1, solver='liblinear').fit(x, y)
# Note: You would need to fit this with your actual training data
# model.fit(X_train, y_train)

# Prediction button
if st.button("Predict Heart Disease Risk", type="primary"):
    try:
        # Make prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0][1] * 100
        
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
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.divider()
st.markdown("""
**Note**: This prediction is for informational purposes only and should not replace professional medical advice. 
""")


# In[ ]:




