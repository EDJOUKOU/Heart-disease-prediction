#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import zipfile
import kaggle
import matplotlib.pyplot as plt


# In[2]:


# download the dataset direct from kaggle

get_ipython().system('kaggle datasets download -d mragpavank/heart-diseaseuci')


# In[3]:


# name of the zipfile

zipfile_name = 'heart-diseaseuci.zip'

#extract the content of the zipfile
with zipfile.ZipFile(zipfile_name, 'r') as file:
    file.extractall()

print('Extraction completed')


# In[4]:


# check the extracted file

df = pd.read_csv('heart.csv')
df.head()


# ## Data preprocessing

# In[5]:


# let's check basic info about the dataset

df.info()


# In[6]:


# check the data types of each column
df.dtypes


# In[7]:


# check if there are any missing values

df.isnull().sum()


# In[8]:


df.columns


# In[9]:


# standardize the features

from sklearn.preprocessing import StandardScaler

features  = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']

X = df[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ## Exploration Data Analysis

# In[10]:


# let's melt the data frame for visualization purposes

scaled_feat = pd.DataFrame(X_scaled)
scaled_feat.columns = features

df_melt = scaled_feat.melt(var_name = 'attribute', value_name = 'value')


# In[11]:


# Let's visualize the data

import seaborn as sns

sns.histplot(x='attribute', y='value', data = df_melt, kde = False, bins = 30 )
plt.title('features distribution')
plt.xlabel('value')
plt.ylabel('Frequency')
plt.xticks(rotation = 45)
plt.show()


# In[12]:


sns.boxplot(x='attribute', y='value', data = df_melt)
plt.title('Box Plot of all features')
plt.xlabel('attribute')
plt.ylabel('value')
plt.xticks(rotation = 45)
plt.show()


# In[13]:


# let's check the correlation between features and the target values


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt = '.2f')
plt.title('Correlation matrix')
plt.tight_layout()
plt.savefig('correlation matrix')
plt.show()


# none of the features are correlated with the target values, thus we need other ones!

# In[14]:


# descriptive statistics

df.describe().T


# ## Feature Engineering

# In[15]:


# First reset the index to ensure no duplicates
df = df.reset_index(drop=True)

# Now perform all your feature engineering operations:

# Age feature binning
df['age_bin'] = pd.cut(df['age'], bins=[20, 30, 40, 50, 60, 70, 80],
                      labels=['young', 'adult', 'mature', 'senior', 'old', 'very_old'])

# Bin cholesterol levels
df['chol_bin'] = pd.qcut(df['chol'], q=4, labels=['low', 'medium', 'high', 'very high'])

# Based on domain knowledge in medicine
# Blood pressure to cholesterol ratio
df['bp_chol_ratio'] = df['trestbps'] / (df['chol'] + 1e-6)

# Heart rate stress index (thalach vs resting bp)
df['hr_stress_index'] = df['thalach'] / df['trestbps']

# Chest pain and exercise induced angina combination
df['cp_exang_combo'] = np.where((df['cp'] > 0) & (df['exang'] > 0), 1, 0)

# Age-adjusted maximum heart rate
df['age_adj_thalach'] = df['thalach'] / (220 - df['age'])

# Combine some features
df['cp_thalach'] = df['cp'] * df['thalach'] 
df['exang_oldpeak'] = df['exang'] * df['oldpeak']  
df['slope_thalach'] = df['slope'] * df['thalach'] 
df['age_chol'] = df['age'] * df['chol'] 

# Create ratio features
df['thalach_oldpeak_ratio'] = df['thalach'] / (df['oldpeak'] + 1e-6) 
df['cp_exang_ratio'] = df['cp'] / (df['exang'] + 1e-6)


# In[16]:


# Convert categorical variable to numeric
from sklearn.preprocessing import LabelEncoder

Le = LabelEncoder()
df['age_code'] = Le.fit_transform(df['age_bin'])
df['chol_code'] = Le.fit_transform(df['chol_bin'])


# In[17]:


new_features = ['age_code', 'bp_chol_ratio', 'hr_stress_index', 'cp_exang_combo', 'age_adj_thalach', 'cp_thalach', 'exang_oldpeak',
               'slope_thalach', 'thalach_oldpeak_ratio', 'age_chol', 'chol_code',  'cp_exang_ratio']

correlations = df[new_features + ['target']].corr()['target'].sort_values(ascending=False)

print(correlations)


# the features 'cp_thalach', 'exang' and 'oldpeak' are chosen for model development

# ## Model development

# In[18]:


# Redefine the features for the Logistic Regression

new_feature = ['cp_thalach', 'exang', 'oldpeak']  

x = df[new_feature]
y = df['target']

# standardise

scaler = StandardScaler()

Xs = scaler.fit_transform(x)


# In[19]:


# Let's split the data into a train and test split

from sklearn.model_selection import train_test_split
Xs_train, Xs_test, y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

print(f'The train set size is: {Xs_train.shape}, {y_train.shape}')
print('-----------------------')
print(f'The test set size is: {Xs_test.shape}, {y_test.shape}')


# In[20]:


# Let's fit a Logistic model to the dataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

LR = LogisticRegression(C=0.01, solver = 'liblinear').fit(Xs_train, y_train)

# let's predict over the test set

yhat  = LR.predict(Xs_test)
yhat


# In[21]:


# let's fit a decision tree model
from sklearn.tree import DecisionTreeClassifier
heartTree = DecisionTreeClassifier(criterion='entropy', max_depth = 3)

#fit the model to the dataset

heartTree.fit(Xs_train, y_train)

# Let's predict on the test set
predTree = heartTree.predict(Xs_test)
predTree


# ## Model Evaluation

# In[22]:


# Let's compute the accuracy for both models

from sklearn import metrics

print(f'The accuracy of the Logistic Regression model is: {metrics.accuracy_score(y_test, yhat)}')
print('................')
print(f'The accuracy of the Decision Tree model is: {metrics.accuracy_score(y_test, predTree)}')


# In[23]:


# Let's compute Precision, Recall, and F1-Score for both models

from sklearn.metrics import classification_report

print(f"The accuracy of the Logistic Regression model is: {classification_report(y_test, yhat, target_names=['No heart disease', 'heart disease'])}")
print('................')
print(f"The accuracy of the Decision Tree model is: {classification_report(y_test, predTree, target_names=['No heart disease', 'heart disease'])}")


# In[24]:


# let's compute ROC and AUC

from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

# get the probabilities for both models

y_prob_LR = LR.predict_proba(Xs_test)[:, 1]
y_prob_ht = heartTree.predict_proba(Xs_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob_LR)
auc_score = roc_auc_score(y_test, y_prob_LR)
print(f"The AUC Score of the Logistic Regression model is: {auc_score:.3f}")
print('......................................')
fpr, tpr, thresholds = roc_curve(y_test, y_prob_ht)
auc_score = roc_auc_score(y_test, y_prob_ht)
print(f"The AUC Score of the Decision tree model is: {auc_score:.3f}")


# In[25]:


# let's implement cross validation
from sklearn.model_selection import cross_val_score

# Compute F1 score with 5-fold cross validation on both models

f1_score_LR = cross_val_score(LR, Xs, y, cv=5, scoring='f1')
f1_score_ht = cross_val_score(heartTree, Xs, y, cv=5, scoring='f1')

print(f'F1 score of the Logistic Regression model is: {f1_score_LR}')
print(f'Mean F1 score of the Logistic Regression model is: {f1_score_LR.mean():.2f}')
print('..................................')
print(f'F1 score of the Decision Tree model is: {f1_score_ht}')
print(f'Mean F1 score of the Decison Tree model is: {f1_score_ht.mean():.2f}')


# In[26]:


# Let's use Gridsearch to tune the parameters of the Decision Tree model

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Initialize the Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='f1',  # Can change to 'accuracy', 'roc_auc', etc.
    n_jobs=-1,  # Use all available CPU cores
    verbose=1
)

# Fit GridSearchCV to the training data
grid_search.fit(Xs_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_dt = grid_search.best_estimator_

print(f"Best Parameters: {best_params}")

# Evaluate on test set
y_pred_ht = best_dt.predict(Xs_test)
print(classification_report(y_test, y_pred_ht))


# In[27]:


# Let's use Gridsearch to tune the parameters of the Logistic Regression model

# Define separate parameter grids for different solvers
lr_param_grid = [
    # For liblinear which supports l1 and l2
    {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear'],
        'max_iter': [100, 200, 500]
    },
    # For saga which supports all penalties including elasticnet
    {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['saga'],
        'max_iter': [100, 200, 500],
        'l1_ratio': [0.1, 0.5, 0.9]  # Only used with elasticnet
    },
    # For newton-cg, lbfgs, sag which only support l2
    {
        'penalty': ['l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['newton-cg', 'lbfgs', 'sag'],
        'max_iter': [100, 200, 500]
    }
]
lr = LogisticRegression(random_state=42)
lr_grid = GridSearchCV(LR, lr_param_grid, cv=5, scoring='f1', n_jobs=-1)
lr_grid.fit(Xs_train, y_train)

# Get the best parameters and best estimator
best_params_lr = lr_grid.best_params_
best_lr = lr_grid.best_estimator_

print(f"Best Parameters: {best_params_lr}")

# Evaluate on test set
y_pred_lr = best_lr.predict(Xs_test)
print(classification_report(y_test, y_pred_lr))


# # Model comparison
# As we are dealing with detecting heart disease, the ability of the model to identify all 
# Positive instances are crucial, so we are going to compare the model using the Recall, which is mathematically defined as the True Positive Rate and translated with the following equation:
# 
# Recall = TP/(TP + FN)
# 
# Recall values for the **Logistic Regression** after grid search are (0.83 & 0.88) against (0.8 & 0.74), therefore, the Logistic Regression is considered better at predicting which patient is more likely to get heart disease or not.

# ## Model deployment

# In[34]:


# Train and save the model

import pickle

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(C=0.1, solver = 'liblinear').fit(Xs_train, y_train)

# save the model

with open('LR.pkl', 'wb') as file:
    pickle.dump(LR, file)


# In[35]:


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

age = st.number_input("Age", min_value=1, max_value=105, value=30)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest pain type (cp)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting blood pressure (trestbps)", min_value=80, max_value=250, value=120)
chol = st.number_input("Serum cholesterol in mg/dl (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", options=[0, 1])
restecg = st.selectbox("Resting ECG", 
        options=[("Normal", 0), ("ST-T wave abnormality", 1), 
                ("Probable left ventricular hypertrophy", 2)],
        format_func=lambda x: x[0])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise induced angina (exang)", options=[0, 1])
oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", value=1.0, step=0.1, format="%.1f")
slope = st.selectbox("Slope of the peak exercise ST segment", options=[("Upsloping", 0), ("Flat", 1), 
                                                                       ("Downsloping", 2)], 
                     format_func=lambda x: x[0])
ca = st.selectbox("Number of major vessels colored by fluoroscopy (ca)", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", 
        options=[("Normal", 0), ("Fixed defect", 1), 
                ("Reversible defect", 2), ("Unknown", 3)],
        format_func=lambda x: x[0])

# predict button

if st.button("Predict Heart Disease Risk", type="primary"):

    #prepare input as a 2D array
    
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal]])

    # make prediction

    prediction = LR.predict(featuures)
    probability  = model.predict_proba(features)[0][1] * 100
    
    # output result

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




