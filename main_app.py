import streamlit as st
import numpy as np
import pandas as pd
import home
import dtree
import predict

st.set_page_config(page_title = 'Early Diabetes Prediction Web App',
	page_icon = 'random', 
	layout = 'wide',
	initial_sidebar_state = 'auto')

@st.cache()
def load_data():
	df = pd.read_csv('df.csv')
	df.head()

	df.dropna(inplace = True)

	return df

diabetes_df = load_data()

st.sidebar.title("Navigate")
pages_dict = {"Home" : home,
			"Predict Diabetes" : predict,
			"Visualise Decision Tree" : dtree}
user_choice = st.sidebar.radio("Go to", ("Home", "Predict Diabetes", "Visualise Decision Tree"))
if user_choice == "Home":
	home.app(diabetes_df)
elif user_choice != "Home":
	selected_page = pages_dict[user_choice]
	selected_page.app(diabetes_df)