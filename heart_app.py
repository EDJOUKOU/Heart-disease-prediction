#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv('heart.csv')
df.head()


# In[3]:


# feature engineering
from sklearn.preprocessing import StandardScaler

df['cp_thalach'] = df['cp'] * df['thalach'] 

new_feature = ['cp_thalach', 'exang', 'oldpeak']  

x = df[new_feature]
y = df['target']

# standardise

scaler = StandardScaler()

Xs = scaler.fit_transform(x)


# In[5]:


from sklearn.model_selection import train_test_split

Xs_train, Xs_test, y_train,y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

print(f'The train set size is: {Xs_train.shape}, {y_train.shape}')
print('-----------------------')
print(f'The test set size is: {Xs_test.shape}, {y_test.shape}')


# In[6]:


# Train and save the model

import pickle

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(C=100, solver = 'liblinear').fit(Xs_train, y_train)

# save the model

with open('LR.pkl', 'wb') as file:
    pickle.dump(LR, file)


# In[7]:


import streamlit as st

# create an app.py file

# load the saved file

with open('LR.pkl', 'rb') as file:
    LR = pickle.load(file)

st.title('❤️ Heart Disease Prediction App')
st.markdown("""
This app predicts the probability of heart disease based on clinical parameters.
Enter the patient's details below and click **Predict**.
""")
st.write("""
### Enter the patient details to check the risk of heart disease
""")

# create fields

# Input fields - make sure ALL features used in your model are included
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
# predict button

if st.button("Predict Heart Disease Risk"):
    try:
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
        
        # Ensure the model is loaded
        if 'model' not in st.session_state:
            st.error("Model not loaded properly")
            return
            
        prediction = st.session_state.model.predict(features)
        probability = st.session_state.model.predict_proba(features)[0][1] * 100
        
        # Display results
        if prediction[0] == 1:
            st.error(f"🚨 High risk of heart disease ({probability:.1f}% probability)")
        else:
            st.success(f"✅ Low risk of heart disease ({100-probability:.1f}% probability)")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")


# In[ ]:




