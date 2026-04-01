# -*- coding: utf-8 -*-
"""Importing the Dependencies"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

"""Data Collection

Data Visualization

Data Pre-Processing
"""

# loading the excel file to a Pandas DataFrame
heartattack = pd.read_excel('/content/heart_attack.xlsx')

# print first 5 rows of the dataset
heartattack.head()

# print last 5 rows of the dataset
heartattack.tail()

# number of rows and columns in the dataset
heartattack.shape

# Getting some information about the data
heartattack.info()

# Checking for missing values
heartattack.isnull().sum()

# Statistical measures about the data
heartattack.describe()

"""Data Visualization"""

sns.set()

# BMI Distribution
plt.figure(figsize=(6,4))
sns.histplot(heartattack['BMI'], bins=20, kde=True, color='teal')
plt.title('BMI Distribution')
plt.xlabel('BMI')
plt.ylabel('Count')
plt.show()

# BMI vs Heart Attack
plt.figure(figsize=(6,4))
sns.boxplot(x='BMI', y='HeartAttack', data=heartattack, palette=['darkblue', 'orange'])
plt.title('BMI vs Heart Attack')
plt.ylabel('Heart Attack')
plt.show()

# Age Distribution
fig, ax = plt.subplots()
fig.set_size_inches(10,5)
plt.title('Age Distribution')
plt.ylabel('Count')
ax = sns.countplot(x="Age", data=heartattack, palette='pink', edgecolor='black')
plt.show()

# Heart Attack Count by Age
fig, ax = plt.subplots()
fig.set_size_inches(12,5)
plt.title('Heart Attack Count by Age')
plt.ylabel('Count')
sns.countplot(x="Age", hue="HeartAttack", data=heartattack, palette=['grey', 'brown'], edgecolor='black')
plt.show()

# Gender Distribution
fig, ax = plt.subplots()
fig.set_size_inches(5,5)
plt.title('Gender Distribution')
plt.ylabel('Count')
ax = sns.countplot(x="Gender", data=heartattack, palette=['#1E88E5', '#C2185B'], edgecolor='black', linewidth=1)
plt.show()

# Heart Attack Count by Gender
fig, ax = plt.subplots()
fig.set_size_inches(6,6)
plt.title('Heart Attack Count by Gender')
plt.ylabel('Count')
sns.countplot(x="Gender", hue="HeartAttack", data=heartattack, palette=['Grey', 'Orange'], edgecolor='black')
plt.show()

# Family History Distribution
fig, ax = plt.subplots()
fig.set_size_inches(5,5)
plt.title('Family History Distribution')
plt.ylabel('Count')
plt.xlabel('Family History')
ax = sns.countplot(x="Family History_Yes_No", data=heartattack, palette=['#00897B', '#FB8C00'], edgecolor='black', linewidth=1)
plt.show()

# Heart Attack Count by Family History
fig, ax = plt.subplots()
fig.set_size_inches(6,6)
plt.title('Heart Attack Count by Family History')
plt.ylabel('Count')
plt.xlabel('Family History')
sns.countplot(x="Family History_Yes_No", hue="HeartAttack", data=heartattack, palette=['Grey', 'darkgreen'], edgecolor='black')
plt.show()

# Family Members History Distribution
fig, ax = plt.subplots()
fig.set_size_inches(12,5)
plt.title('Family Members History Distribution')
plt.xticks(rotation=45)
plt.ylabel('Count')
ax = sns.countplot(x="Family Members History", data=heartattack, palette='BrBG', edgecolor='black', linewidth=1)
plt.show()

# Heart Attack Count by Family Members History
fig, ax = plt.subplots()
fig.set_size_inches(12,5)
plt.title('Heart Attack Count by Family Members History')
plt.xticks(rotation=45)
plt.ylabel('Count')
ax = sns.countplot(x="Family Members History", hue='HeartAttack', data=heartattack, palette=['Grey', 'darkblue'], edgecolor='black', linewidth=1)
plt.show()

# Smoking Distribution
fig, ax = plt.subplots()
fig.set_size_inches(5,5)
plt.title('Smoking Distribution')
plt.ylabel('Count')
ax = sns.countplot(x="Smoking", data=heartattack, palette=['darkRed', 'Orange'], edgecolor='black', linewidth=1)
plt.show()

# Heart Attack Count by Smoking
fig, ax = plt.subplots()
fig.set_size_inches(6,6)
plt.title('Heart Attack Count by Smoking')
plt.ylabel('Count')
ax = sns.countplot(x="Smoking", hue='HeartAttack', data=heartattack, palette=['Grey', 'darkred'], edgecolor='black', linewidth=1)
plt.show()

# Alcohol Distribution
fig, ax = plt.subplots()
fig.set_size_inches(5,5)
plt.title('Alcohol Distribution')
plt.ylabel('Count')
ax = sns.countplot(x="Alcohol", data=heartattack, palette=['#612D53', 'skyblue'], edgecolor='black', linewidth=1)
plt.show()

# Heart Attack Count by Alcohol
fig, ax = plt.subplots()
fig.set_size_inches(6,6)
plt.title('Heart Attack Count by Alcohol')
plt.ylabel('Count')
ax = sns.countplot(x="Alcohol", hue='HeartAttack', data=heartattack, palette=['Grey', 'lightyellow'], edgecolor='black', linewidth=1)
plt.show()

# Physical Activity Distribution
fig, ax = plt.subplots()
fig.set_size_inches(5,5)
plt.title('Physical Activity Distribution')
plt.ylabel('Count')
plt.xlabel('Physical Activity')
ax = sns.countplot(x="Physical activity", data=heartattack, palette=['brown', 'lightblue'], edgecolor='black', linewidth=1)
plt.show()

# Heart Attack Count by Physical Activity
fig, ax = plt.subplots()
fig.set_size_inches(6,6)
plt.title('Heart Attack Count by Physical Activity')
plt.ylabel('Count')
plt.xlabel('Physical Activity')
ax = sns.countplot(x="Physical activity", hue='HeartAttack', data=heartattack, palette=['Grey', '#BABF94'], edgecolor='black', linewidth=1)
plt.show()

# Yoga Distribution
fig, ax = plt.subplots()
fig.set_size_inches(5,5)
plt.title('Yoga Distribution')
plt.ylabel('Count')
ax = sns.countplot(x="Yoga", data=heartattack, palette=['#D92C54', '#F1E2E2'], edgecolor='black', linewidth=1)
plt.show()

# Heart Attack Count by Yoga
fig, ax = plt.subplots()
fig.set_size_inches(6,6)
plt.title('Heart Attack Count by Yoga')
plt.ylabel('Count')
ax = sns.countplot(x="Yoga", hue='HeartAttack', data=heartattack, palette=['Grey', '#56DFCF'], edgecolor='black', linewidth=1)
plt.show()

# Diet Distribution
fig, ax = plt.subplots()
fig.set_size_inches(5,5)
plt.title('Diet Distribution')
plt.ylabel('Count')
ax = sns.countplot(x="Diet", data=heartattack, palette=['Green', 'Red'], edgecolor='black', linewidth=1)
plt.show()

# Heart Attack Count by Diet
fig, ax = plt.subplots()
fig.set_size_inches(7,7)
plt.title('Heart Attack Count by Diet')
plt.ylabel('Count')
ax = sns.countplot(x="Diet", hue='HeartAttack', data=heartattack, palette=['#DA6C6C', '#819A91'], edgecolor='black', linewidth=1)
plt.show()

# Sleep Hours Distribution
fig, ax = plt.subplots()
fig.set_size_inches(5,5)
plt.title('Sleep Hours Distribution')
plt.ylabel('Count')
plt.xlabel('Sleep Hours')
ax = sns.countplot(x="Sleep_Hours", data=heartattack, palette=['#D98324', '#443627', '#EFDCAB'], edgecolor='black', linewidth=1)
plt.show()

# Heart Attack Count by Sleep Hours
fig, ax = plt.subplots()
fig.set_size_inches(7,7)
plt.title('Heart Attack Count by Sleep Hours')
plt.ylabel('Count')
plt.xlabel('Sleep Hours')
ax = sns.countplot(x="Sleep_Hours", hue='HeartAttack', data=heartattack, palette=['#295F98', '#C6E7FF'], edgecolor='black', linewidth=1)
plt.show()

# Stress Distribution
fig, ax = plt.subplots()
fig.set_size_inches(5,5)
plt.title('Stress Distribution')
plt.ylabel('Count')
ax = sns.countplot(x="Stress", data=heartattack, palette=['#DB1A1A', '#03AED2'], edgecolor='black', linewidth=1)
plt.show()

# Heart Attack Count by Stress
fig, ax = plt.subplots()
fig.set_size_inches(7,7)
plt.title('Heart Attack Count by Stress')
plt.ylabel('Count')
ax = sns.countplot(x="Stress", hue='HeartAttack', data=heartattack, palette=['#FFDE42', '#7FB77E'], edgecolor='black', linewidth=1)
plt.show()

# Diabetes HyperTension Distribution
fig, ax = plt.subplots()
fig.set_size_inches(10,5)
plt.title('Diabetes HyperTension Distribution')
plt.ylabel('Count')
plt.xlabel('Diabetes HyperTension')
ax = sns.countplot(x="Diabetes_HyperTension", data=heartattack, palette=['#5478FF', '#91D06C', '#FF4400', '#890596'], edgecolor='black', linewidth=1)
plt.show()

# Heart Attack Count by Diabetes HyperTension
fig, ax = plt.subplots()
fig.set_size_inches(10,5)
plt.title('Heart Attack Count by Diabetes HyperTension')
plt.ylabel('Count')
plt.xlabel('Diabetes HyperTension')
ax = sns.countplot(x="Diabetes_HyperTension", hue='HeartAttack', data=heartattack, palette=['#36064D', '#FFDBFD'], edgecolor='black', linewidth=1)
plt.show()

# Heart Attack Distribution ( Bar Chart )
fig, ax = plt.subplots()
fig.set_size_inches(5,5)
plt.title('Heart Attack Distribution')
plt.ylabel('Count')
plt.xlabel('Heart Attack')
ax = sns.countplot(x="HeartAttack", data=heartattack, palette=['#BF1A1A', '#CBF3BB'], edgecolor='black', linewidth=1)
plt.show()

# Heart Attack Distribution ( Pie Chart )
counts = heartattack['HeartAttack'].value_counts()

plt.pie(
    counts,
    labels=counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#237227', 'maroon']
)

plt.title('Heart Attack Distribution')
plt.show()

"""Data Pre-Processing

    Label Encoding
"""

# Checking the distribution of the Age Column
heartattack['Age'].value_counts()

heartattack.replace({'Age':{'20 - 30 years':0, '31 - 40 years':1, '41 - 50 years':2, '51 - 60 years':3, '61 - 70 years':4, '>70 years':5}}, inplace=True)

# Checking the distribution of the State Column
heartattack['State'].value_counts()

heartattack.replace({'State':{'Other States':0, 'Gujarat':1}}, inplace=True)

# Checking the distribution of the Gender Column
heartattack['Gender'].value_counts()

heartattack.replace({'Gender':{'Female':0, 'Male':1}}, inplace=True)

# Checking the distribution of the Family History_Yes_No Column
heartattack['Family History_Yes_No'].value_counts()

heartattack.replace({'Family History_Yes_No':{'No':0, 'Yes':1}}, inplace=True)

# Checking the distribution of the Family Members History Column
heartattack['Family Members History'].value_counts()

heartattack.replace({'Family Members History':{'No one':0, 'Father':1, 'Mother':2, 'Sibling':3, 'Father+Mother':4, 'Father+Sibling':5, 'Mother+Sibling':6, 'Father+Mother+Sibling':7}}, inplace=True)

# Checking the distribution of the Smoking Column
heartattack['Smoking'].value_counts()

heartattack.replace({'Smoking':{'No':0, 'Yes':1}}, inplace=True)

# Checking the distribution of the Alcohol Column
heartattack['Alcohol'].value_counts()

heartattack.replace({'Alcohol':{'No':0, 'Yes':1}}, inplace=True)

# Checking the distribution of the Physical activity Column
heartattack['Physical activity'].value_counts()

heartattack.replace({'Physical activity':{'No':0, 'Yes':1}}, inplace=True)

# Checking the distribution of the Yoga Column
heartattack['Yoga'].value_counts()

heartattack.replace({'Yoga':{'No':0, 'Yes':1}}, inplace=True)

# Checking the distribution of the Diet Column
heartattack['Diet'].value_counts()

heartattack.replace({'Diet':{'Non-veg':0, 'Veg':1}}, inplace=True)

# Checking the distribution of the Sleep_Hours Column
heartattack['Sleep_Hours'].value_counts()

heartattack.replace({'Sleep_Hours':{'<6 Hrs':0, '6-8 Hrs':1, '>8 Hrs':2}}, inplace=True)

# Checking the distribution of the Stress Column
heartattack['Stress'].value_counts()

heartattack.replace({'Stress':{'No':0, 'Yes':1}}, inplace=True)

# Checking the distribution of the Diabetes_HyperTension Column
heartattack['Diabetes_HyperTension'].value_counts()

heartattack.replace({'Diabetes_HyperTension':{'No':0, 'Diabetes':1, 'Hypertension':2, 'Diabetes+Hypertension':3}}, inplace=True)

# Checking the distribution of the HeartAttack Column
heartattack['HeartAttack'].value_counts()

heartattack.replace({'HeartAttack':{'No':0, 'Yes':1}}, inplace=True)

heartattack.drop('State', axis=1, inplace=True)

corr_matrix = heartattack.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

"""Feature Engineering"""

heartattack['Lifestyle_Risk'] = heartattack['Smoking'] + heartattack['Alcohol']

heartattack['Mental_Risk'] = heartattack['Stress'] - heartattack['Sleep_Hours']

heartattack['Activity_Score'] = heartattack['Physical activity'] + heartattack['Yoga']

# Age x BMI: older age combined with high BMI strongly predicts heart attack
heartattack['Age_BMI_Risk'] = heartattack['Age'] * heartattack['BMI']

# Diabetes/Hypertension x BMI: metabolic risk amplified by obesity
heartattack['Dia_BMI'] = heartattack['Diabetes_HyperTension'] * heartattack['BMI']

heartattack

corr_matrix = heartattack.corr()

plt.figure(figsize=(14,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

"""Splitting the Features and Target"""

X = heartattack.drop(columns=['HeartAttack'])
Y = heartattack['HeartAttack'].astype(int)

print(X)

print(Y)

"""Splitting the Data into Training Data and Test Data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y, random_state=42)

print(Y_train.value_counts())
print(Y_test.value_counts())

print(X.shape, X_train.shape, X_test.shape)

"""Model Training : XGBClassifiaction"""

xgb_model = XGBClassifier(
    learning_rate=0.05,
    n_estimators=60,
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=3,
    subsample=0.85,
    max_features='sqrt',
    random_state=42
)

xgb_model.fit(X_train, Y_train)

"""Model Evaluation

*   Accuracy Score
*   Confusion Matrix


"""

# training data accuracy
training_data_prediction = xgb_model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, training_data_prediction)

print(training_data_accuracy)

THRESHOLD = 0.35

# testing data accuracy
testing_data_proba = xgb_model.predict_proba(X_test)[:, 1]
testing_data_prediction = (testing_data_proba >= THRESHOLD).astype(int)
testing_data_accuracy = accuracy_score(Y_test, testing_data_prediction)

print(testing_data_accuracy)


print("Accuracy:", accuracy_score(Y_test, testing_data_prediction))
print("\nClassification Report:\n", classification_report(Y_test, testing_data_prediction))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, testing_data_prediction))


print("Accuracy:", accuracy_score(Y_train, training_data_prediction))
print("\nClassification Report:\n", classification_report(Y_train, training_data_prediction))
print("\nConfusion Matrix:\n", confusion_matrix(Y_train, training_data_prediction))

heartattack.head()

heartattack.tail()

"""Building a Predictive System"""

input_data = (2, 25.00, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 25.00, 25.00, 0)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# use threshold 0.35 — same as model evaluation
prediction = xgb_model.predict_proba(input_data_reshaped)[0][1]
print(prediction)

risk_score = round(prediction * 10)
print('Heart Attack Risk Score :', risk_score, "/ 10")

if risk_score <= 3:
    risk_level = "Low Risk"
elif risk_score <= 6:
    risk_level = "Medium Risk"
else:
    risk_level = "High Risk"
print("Risk Category :", risk_level)

"""Saving the trained model"""

import pickle

filename = 'heartattack_model.sav'
pickle.dump(xgb_model, open(filename, 'wb'))

# loading the saved model
loaded_model = pickle.load(open('heartattack_model.sav', 'rb'))

input_data = (2, 25.00, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 25.00, 25.00, 0)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# use threshold 0.30 — same as model evaluation
prediction = xgb_model.predict_proba(input_data_reshaped)[0][1]
print(prediction)

risk_score = round(prediction * 10)
print('Heart Attack Risk Score :', risk_score, "/ 10")

if risk_score <= 3:
    risk_level = "Low Risk"
elif risk_score <= 6:
    risk_level = "Medium Risk"
else:
    risk_level = "High Risk"
print("Risk Category :", risk_level)

"""Feature Importance & Confusion Matrix"""

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
})

importance = importance.sort_values(by='Importance', ascending=False)

print(importance)

importance.sort_values('Importance').plot.barh(x='Feature', y='Importance')
plt.show()

# confusion matrix for training data prediction
cm = confusion_matrix(Y_train, training_data_prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot(cmap='Reds')
plt.title("Confusion Matrix on Train Data")
plt.show()

# confusion matrix for test data prediction
cm = confusion_matrix(Y_test, testing_data_prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot(cmap='Reds')
plt.title("Confusion Matrix on Test Data")
plt.show()

"""Cross Validation"""

xgb_model = XGBClassifier(
    learning_rate=0.05,
    n_estimators=60,
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=3,
    subsample=0.85,
    max_features='sqrt',
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(xgb_model, X, Y, cv=cv, scoring='accuracy')

print("Cross Validation Scores:", scores)
print("Mean Accuracy:", np.mean(scores))