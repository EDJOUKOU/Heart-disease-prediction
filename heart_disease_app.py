import pickle
import numpy as np
import streamlit as st

# Load the trained model
with open("best_lr.pkl", "rb") as f:
    best_lr = pickle.load(f)

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease presence.")

# Collect input from user
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=120)
chol = st.number_input("Serum Cholesterol in mg/dl (chol)", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
restecg = st.selectbox("Resting ECG results (restecg)", options=[0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved (thalach)", value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
oldpeak = st.number_input("Oldpeak", value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

# Prediction
if st.button("Predict"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    prediction = best_lr.predict(features)
    probability = best_lr.predict_proba(features)[0][1] * 100

    st.subheader("Prediction Results")

    if prediction[0] == 1:
        st.error(f"🚨 High risk of heart disease ({probability:.1f}% probability)")
        st.markdown("""
        **Recommendations:**
        - Consult a cardiologist immediately
        - Schedule a comprehensive cardiac evaluation
        - Monitor symptoms closely
        """)
    else:
        st.success(f"✅ Low risk of heart disease ({100-probability:.1f}% probability)")
        st.markdown("""
        **Recommendations:**
        - Maintain a heart-healthy lifestyle
        - Regular exercise and balanced diet
        - Annual cardiac checkups
        """)

    st.progress(int(probability))
    st.caption(f"Probability of heart disease: {probability:.1f}%")

st.write("Model: Logistic Regression with parameters C=0.1, max_iter=100, penalty='l2', solver='liblinear'")

