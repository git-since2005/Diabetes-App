import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from io import StringIO
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
import warnings

def app(df):
	warnings.filterwarnings('ignore')
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.title('Visualise the Diabetes Prediction Web app')

	if st.checkbox("Show the correlation map"):
		st.subheader('Correlation Heatmap')
		plt.figure(figsize = (17, 6))
		sns.heatmap(df.corr(), annot = True)
		st.pyplot()

	st.subheader("Predictor Selection")

	plot_select = st.selectbox("Select the Classifier to Visualise the Diabetes Prediction", ("Decisoin Tree Classifier", "GridSearchCV Best Tree Classifier"))

	if plot_select == 'Decisoin Tree Classifier':
		feature_columns = list(df.columns)

		feature_columns.remove('SkinThickness')
		feature_columns.remove('Pregnancies')
		feature_columns.remove('Outcome')

		x_train, x_test, y_train, y_test = train_test_split(df[feature_columns], df['Outcome'], test_size = 0.3, random_state = 42)
		dtree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
		dtree.fit(x_train, y_train)
		y_train_pred = dtree.predict(x_train)
		y_test_pred = dtree.predict(x_test)

		if st.checkbox("Plot Confusion Matrix"):
			plt.figure(figsize = (17, 6))
			plot_confusion_matrix(dtree, x_train, y_train, values_format = 'd')
			st.pyplot()
		if st.checkbox('Plot Decision Tree'):
			dot = tree.export_graphviz(decision_tree = dtree, max_depth = 3, out_file = None, filled = True, rounded = True, feature_names = feature_columns, class_names = ['0', '1'])
			st.graphviz_chart(dot.Graph)

	elif plot_select == 'GridSearchCV Best Tree Classifier':
		feature_column = [i for i in df.columns if i not in ['SkinThickness', 'Pregnancies', 'Outcome']]
		param_grid =  {
				'criterion' : ['entropy', 'gini'],
				'random_state' : [42],
				'max_depth' : np.arange(4, 21)
			}
		x_train, x_test, y_train, y_test = train_test_split(df[feature_column], df['Outcome'], test_size = 0.3, random_state = 42)
		grid = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = 'roc_auc', n_jobs = -1)
		grid.fit(x_train, y_train)
		best_tree = grid.best_estimator_

		grid.fit(x_train, y_train)
		y_train_pred = grid.predict(x_train)
		y_test_pred = grid.predict(x_test)

		if st.checkbox('Plot confusion matrix'):
			plt.figure(figsize = (17, 6))
			plot_confusion_matrix(grid, x_train, y_train, values_format = 'd')
			st.pyplot()
		if st.checkbox('Plot Decision Tree'):
			dot_data = tree.export_graphviz(decision_tree = best_tree, out_file = None, rounded = True, filled = True, class_names = ['0', '1'], feature_names = feature_column)
			st.graphviz_chart(dot_data)
			sns.heatmap(df.corr())
			st.pyplot()