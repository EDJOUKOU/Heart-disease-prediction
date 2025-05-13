# Heart Disease Prediction Analysis

## Project Overview
This project focuses on predictive analysis of heart disease using machine learning techniques. The analysis leverages logistic regression and decision tree algorithms to classify patients based on their risk of heart disease, using clinical and demographic features.

## Dataset
- **Source**: UCI Heart Disease Dataset from Kaggle  
- **Collection Date**: May 12, 2025  
- **Features**: 14 clinical attributes including age, sex, chest pain type, resting blood pressure, cholesterol levels, and more  
- **Target**: Presence of heart disease (binary classification)  
- **Access**: [Dataset on Kaggle](https://www.kaggle.com/datasets/mragpavank/heart-diseaseuci)  

## Tools & Technologies
- **Primary IDE**: Jupyter Notebook for data analysis and model development  
- **Visualization**: Matplotlib and Seaborn for EDA  
- **Machine Learning**: Scikit-learn for model implementation  
- **Deployment**: Streamlit for web application  

## Data Preprocessing
The dataset underwent comprehensive cleaning and preparation:  
1. **Initial Inspection**: Verified data structure and quality  
2. **Missing Values**: Confirmed complete dataset with no null values  
3. **Data Types**: Ensured proper numeric formatting for all features  
4. **Feature Scaling**: Standardized continuous variables using StandardScaler  
5. **Feature Engineering**: Created meaningful derived features based on medical domain knowledge  

## Exploratory Data Analysis
Key findings from the EDA process:  
- Most features showed limited direct correlation with the target variable  
- Significant features identified through analysis:  
  - Chest pain type (cp)  
  - Maximum heart rate achieved (thalach)  
  - Exercise induced angina (exang)  
  - ST depression induced by exercise (oldpeak)  
- Visualizations included:  
  - Feature distribution plots  
  - Box plots for outlier detection  
  - Correlation analysis  

## Model Development
### Approach:  
1. **Algorithm Selection**:  
   - Logistic Regression (baseline)  
   - Decision Tree Classifier  
2. **Train-Test Split**: 70-30 stratified split to maintain class distribution  
3. **Feature Selection**: Focused on clinically relevant features ('cp_thalach', 'exang', 'oldpeak')  

### Evaluation Metrics:  
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

## Results & Deployment
- **Best Performing Model**: Logistic Regression demonstrated superior recall (sensitivity), crucial for medical diagnosis  
- **Model Interpretation**: Analyzed coefficients to understand feature importance  
- **Deployment**: Implemented as a web application using Streamlit for practical clinical use  

The final model achieved strong performance in identifying true positive cases while maintaining reasonable precision, making it suitable for preliminary heart disease risk assessment.  

[![Open in Streamlit](http://localhost:8501/))  

