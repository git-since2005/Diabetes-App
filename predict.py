import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

def app(df):
	st.title("This app predicts a person having diabetes or not on the basis of feature values.")
	# 1. Glucose
	glucose = st.slider(label = "Glucose", min_value = min(df['Glucose']), max_value = max(df['Glucose']))
	# 2. Blood Pressure
	bp = st.slider(label = 'Blood Pressure', min_value = min(df['BloodPressure']), max_value = max(df['BloodPressure']))
	# 3. Insulin
	insulin = st.slider(label = 'Insulin', min_value = min(df['Insulin']), max_value = max(df['Insulin']))
	# 4. BMI
	bmi = st.slider(label = 'BMI', min_value = min(df['BMI']), max_value = max(df['BMI']))
	# 5. Pedigree
	pedigree = st.slider(label = 'Pedigree', min_value = min(df['DiabetesPedigreeFunction']), max_value = max(df['DiabetesPedigreeFunction']))
	# 6. Age
	age = st.slider(label = 'Age', min_value = min(df['Age']), max_value = max(df['Age']))
	if st.button("Predict"):
		pred, acc = d_tree_pred(df, glucose, bp, insulin, bmi, pedigree, age)
		st.write(f'Prediction: {pred}')
		st.write(f'Score: {acc}')
def d_tree_pred(df, glucose, bp, insulin, bmi, pedigree, age):
	features = [i for i in list(df.columns) if i not in ['SkinThickness', 'Pregnancies', 'Outcome']]
	# features.remove('Skin Thickness')
	# features.remove('Pregnancies')
	# features.remove('Outcome')
	x_train, x_test, y_train, y_test = train_test_split(df[features], df['Outcome'], random_state = 42, test_size = 0.3)
	dtree_clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 3)
	dtree_clf.fit(x_train, y_train)
	y_train_pred = dtree_clf.predict(x_train)
	y_test_pred = dtree_clf.predict(x_test)
	prediction = dtree_clf.predict([[glucose, bp, insulin, bmi, pedigree, age]])
	prediction = prediction[0]

	score = round(metrics.accuracy_score(y_train, y_train_pred), 3)

	return prediction, score