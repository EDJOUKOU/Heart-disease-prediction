#!/usr/bin/env python
# coding: utf-8

# In[18]:


# heart_app.py
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys

# Verify sklearn installation
try:
    from sklearn import __version__ as sk_version
    st.sidebar.success(f"scikit-learn version: {sk_version}")
except ImportError:
    st.error("scikit-learn not installed! Run: pip install scikit-learn")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# App header
st.title('❤️ Heart Disease Prediction App')
st.markdown("""
Predict heart disease risk using clinical parameters.  
Enter patient details below and click **Predict**.
""")

# Load data function with caching
@st.cache_data
def load_data():
    try:
        # Load your dataset - adjust path as needed
        heart = pd.read_csv('heart.csv')
        return heart
    except FileNotFoundError:
        st.error("Heart dataset not found! Please ensure 'heart.csv' is in the same directory.")
        st.stop()

# Load data
heart = load_data()

# Features and target
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = heart[features]
y = heart['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Input fields with better organization
def user_input_features():
    st.subheader("Patient Details")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])
        cp = st.selectbox("Chest Pain Type", 
                         options=[("Typical angina", 0), ("Atypical angina", 1),
                                 ("Non-anginal pain", 2), ("Asymptomatic", 3)],
                         format_func=lambda x: x[0])
        trestbps = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=250, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        
    with col2:
        restecg = st.selectbox("Resting ECG",
                              options=[("Normal", 0), ("ST-T wave abnormality", 1),
                                      ("Left ventricular hypertrophy", 2)],
                              format_func=lambda x: x[0])
        thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
        exang = st.radio("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST",
                            options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
                            format_func=lambda x: x[0])
        ca = st.slider("Number of Major Vessels", 0, 4, 0)
        thal = st.selectbox("Thalassemia",
                          options=[("Normal", 0), ("Fixed Defect", 1), ("Reversible Defect", 2)],
                          format_func=lambda x: x[0])
    
    # Convert from tuples to values
    data = {
        'age': age,
        'sex': sex[1],
        'cp': cp[1],
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs[1],
        'restecg': restecg[1],
        'thalach': thalach,
        'exang': exang[1],
        'oldpeak': oldpeak,
        'slope': slope[1],
        'ca': ca,
        'thal': thal[1]  
    }
    return pd.DataFrame(data, index=[0])

# Create and train model
@st.cache_resource
def get_model():
    model = LogisticRegression(C=0.1, solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)
    return model

model = get_model()

# Get user input
input_df = user_input_features()

# Display user inputs
st.subheader("Patient Input Summary")
st.write(input_df)

# Prediction
if st.button("Predict Heart Disease Risk", type="primary"):
    try:
        # Make prediction
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0][1] * 100
        
        # Show results
        st.divider()
        st.subheader("Results")
        
        if prediction[0] == 1:
            st.error(f"🚨 High risk of heart disease ({proba:.1f}% probability)")
            with st.expander("Recommendations"):
                st.markdown("""
                - Consult a cardiologist immediately
                - Schedule comprehensive cardiac evaluation
                - Monitor symptoms closely
                - Adopt heart-healthy lifestyle changes
                """)
        else:
            st.success(f"✅ Low risk of heart disease ({100-proba:.1f}% probability)")
            with st.expander("Recommendations"):
                st.markdown("""
                - Maintain healthy lifestyle
                - Regular exercise and balanced diet
                - Annual cardiac checkups
                """)
        
        # Visual indicator
        st.progress(int(proba), text=f"Risk Score: {proba:.1f}%")
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Footer
st.divider()
st.caption("""
**Note**: This tool is for informational purposes only and not a substitute for professional medical advice.
Model accuracy: ~85% (Logistic Regression with C=0.1)
""")


# In[ ]:




