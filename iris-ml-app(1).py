# iris_app.py
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

# App title and description
st.write("""
# Simple Iris Flower Prediction App

This app predicts the **Iris flower** type based on measurements!
""")

# User input parameters in sidebar
st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
df = user_input_features()

# Display user inputs
st.subheader('User Input Parameters')
st.write(df)

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Train model
clf = RandomForestClassifier()
clf.fit(X, Y)

# Make predictions
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Display results
st.subheader('Class Labels')
st.write(pd.DataFrame({
    'Index': [0, 1, 2],
    'Flower Type': iris.target_names
}))

st.subheader('Prediction')
st.success(f"**Predicted Iris Type:** {iris.target_names[prediction][0]}")

st.subheader('Prediction Probability (%)')
proba_df = pd.DataFrame(
    prediction_proba * 100,
    columns=iris.target_names,
    index=['Probability']
).round(1)
st.write(proba_df)

# Add some styling
st.markdown("""
<style>
    .stSlider>div>div>div>div {
        background: #4CAF50;
    }
    .st-b7 {
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


# In[ ]:




