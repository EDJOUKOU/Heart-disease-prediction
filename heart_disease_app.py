import streamlit as st
import pandas as pd
import pickle
import numpy as np
import joblib
import os
from sklearn.exceptions import NotFittedError

# Set app configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Enhanced model loading function
@st.cache_resource  # Better for loading models than cache_data
def load_model():
    model_paths = ['best_lr.joblib', 'best_lr.pkl']  # Try joblib first, then pickle
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                if path.endswith('.joblib'):
                    model = joblib.load(path)
                else:
                    with open(path, 'rb') as file:
                        model = pickle.load(file)
                
                # Verify it's a proper model
                if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                    st.success(f"Model loaded successfully from {path}")
                    return model
                else:
                    st.warning(f"File {path} exists but doesn't contain a valid model")
                    
            except Exception as e:
                st.warning(f"Failed to load {path}: {str(e)}")
                continue
    
    # If we get here, no model loaded successfully
    st.error("""
    **Could not load model file.**
    
    Please ensure either:
    1. 'best_lr.joblib' or 'best_lr.pkl' exists in your app directory
    2. The file is properly committed to your repository
    3. The file isn't corrupted
    
    For best results:
    - Use joblib format: `joblib.dump(model, 'best_lr.joblib')`
    - Ensure the model was saved with the same scikit-learn version
    """)
    st.stop()

# Load model with error handling
try:
    model = load_model()
except Exception as e:
    st.error(f"Critical error loading model: {str(e)}")
    st.stop()

# App title and description
st.title("❤️ Heart Disease Risk Prediction")
st.write("""
This application predicts the likelihood of heart disease based on patient health metrics.
Please fill in the patient details below and click 'Predict' to see the results.
""")

# Create input form
with st.form("patient_details"):
    st.header("Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest Pain Type", [
            "Typical angina", 
            "Atypical angina", 
            "Non-anginal pain", 
            "Asymptomatic"
        ])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG Results", [
            "Normal", 
            "ST-T wave abnormality", 
            "Left ventricular hypertrophy"
        ])
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        
    with col3:
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", [
            "Upsloping", 
            "Flat", 
            "Downsloping"
        ])
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [
            "Normal", 
            "Fixed defect", 
            "Reversible defect"
        ])
    
    submitted = st.form_submit_button("Predict Heart Disease Risk")

# Process inputs and make prediction when form is submitted
if submitted:
    try:
        # Convert categorical inputs to numerical values
        sex = 1 if sex == "Male" else 0
        cp_mapping = {
            "Typical angina": 0,
            "Atypical angina": 1,
            "Non-anginal pain": 2,
            "Asymptomatic": 3
        }
        cp = cp_mapping[cp]
        
        fbs = 1 if fbs == "Yes" else 0
        
        restecg_mapping = {
            "Normal": 0,
            "ST-T wave abnormality": 1,
            "Left ventricular hypertrophy": 2
        }
        restecg = restecg_mapping[restecg]
        
        exang = 1 if exang == "Yes" else 0
        
        slope_mapping = {
            "Upsloping": 0,
            "Flat": 1,
            "Downsloping": 2
        }
        slope = slope_mapping[slope]
        
        thal_mapping = {
            "Normal": 1,
            "Fixed defect": 2,
            "Reversible defect": 3
        }
        thal = thal_mapping[thal]
        
        # Create feature array in the correct order
        features = np.array([[
            age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal
        ]])
        
        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1] * 100
        
        # Display results
        st.subheader("Prediction Results")
        result_container = st.container()
        
        if prediction[0] == 1:
            result_container.error(f"🚨 High risk of heart disease ({probability:.1f}% probability)")
            result_container.markdown("""
            **Recommendations:**
            - Consult a cardiologist immediately
            - Schedule a comprehensive cardiac evaluation
            - Monitor symptoms closely
            """)
        else:
            result_container.success(f"✅ Low risk of heart disease ({100-probability:.1f}% probability)")
            result_container.markdown("""
            **Recommendations:**
            - Maintain a heart-healthy lifestyle
            - Regular exercise and balanced diet
            - Annual cardiac checkups
            """)
        
        # Show probability gauge
        st.progress(int(probability))
        st.caption(f"Probability of heart disease: {probability:.1f}%")
        
        # Show feature importance if available
        if hasattr(model, 'coef_'):
            st.subheader("Key Factors Influencing Prediction")
            feature_names = [
                'Age', 'Sex', 'Chest Pain', 'Blood Pressure', 'Cholesterol',
                'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate',
                'Exercise Angina', 'ST Depression', 'Slope', 'Major Vessels', 'Thal'
            ]
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': model.coef_[0]
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            st.dataframe(importance.style.format({'Coefficient': '{:.3f}'}))
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.error("Please check your input values and try again.")

# Add footer
st.markdown("---")
st.caption("""
Note: This prediction is for informational purposes only and should not replace 
professional medical advice. Always consult with a healthcare provider for 
medical diagnoses and treatment.
""")
