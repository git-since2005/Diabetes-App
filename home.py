import streamlit as st

def app(df):
	st.title("Early Diabetes Prediction Web App")
	# st.write("<p style = 'color : red'>Diabetes is a chronic (long-lasting) condition that affects how your body turns food into energy. There isn't a cure for diabetes, but losing weight, eating healthy food, and being active can really help in reducing the impact of diabetes. This web app will help you to predict whether a person has diabetes or is prone to get diabetes in future by analysing the values of several features using the Decision Tree Classifier.</p>", unsafe_allow_html=True)
	st.markdown("Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy.There isn’t a cure yet for diabetes, but losing weight, eating healthy food, and being active can really help in reducing the impact of diabetes.This Web app will help you to predict whether a person has diabetes or is prone to get diabetes in future by analysing the values of several features using the Decision Tree Classifier.")
	st.header("View Data")
	with st.expander("View Data"):
		st.table(df)
	st.subheader("Columns Description:")
	col1, col2, col3 = st.columns(3)
	with col1:
		if st.checkbox("Show all column names"):
			st.table(df.columns)
	with col2:
		if st.checkbox("View column data-type"):
			col = st.selectbox("Select a column", df.columns)
			st.write(df[col].dtype)
	with col3:
		if st.checkbox("View column data"):
			data_col = st.selectbox("Select a column", df.columns)
			st.table(df[data_col])
	if st.checkbox("Show summary"):
		st.table(df.describe())