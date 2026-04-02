CardioAI – Heart Attack Risk Prediction System

Overview:

CardioAI is a machine learning-based system that predicts the risk of heart attacks using lifestyle, demographic, and medical data. The system uses the XGBoost classifier for prediction and Streamlit for the user interface.
It also integrates Groq AI to provide medical explanations and a chatbot for cardiovascular health queries.

Features:

Heart attack risk prediction using Machine Learning
Risk score displayed on 1–10 scale
Risk category: Low / Medium / High
Probability-based prediction
AI-powered medical explanation using Groq AI
Interactive Streamlit UI
Technologies Used
Python
Scikit-learn
XGBoost
Pandas & NumPy
Matplotlib & Seaborn
Streamlit
Groq AI

Dataset Features:
Age
Gender
BMI
Smoking
Alcohol
Physical Activity
Diet
Sleep Hours
Stress
Diabetes / Hypertension
Family History
Model Performance
Training Accuracy: 96.07%
Testing Accuracy: 90.90%
Cross Validation Accuracy: 89.66%
Run the Project

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py

Disclaimer:

This project is developed for educational purposes and should not replace professional medical advice.
