import streamlit as st
import pandas as pd
import joblib
import shap
import os
import matplotlib.pyplot as plt  # Import matplotlib


# Define paths to the model and training columns
MODEL_PATH = os.path.join("models", "mental_health_model.pkl")
TRAIN_COLUMNS_PATH = os.path.join("models", "train_columns.pkl")

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Load the training columns
@st.cache_resource
def load_train_columns():
    return joblib.load(TRAIN_COLUMNS_PATH)

train_columns = load_train_columns()

# Load the SHAP explainer
@st.cache_resource
def load_explainer():
    dummy_data = pd.DataFrame([{col: 0 for col in train_columns}])  # Provide a valid masker
    return shap.LinearExplainer(model, masker=shap.maskers.Independent(dummy_data))

explainer = load_explainer()

# Streamlit app
st.title("Mental Health Prediction Tool")
st.write("Enter your details to predict if you need mental health treatment.")

# Input fields
st.sidebar.header("Input Features")

# Define input fields based on your feature list
work_interfere = st.sidebar.selectbox("Work Interference", ['Never', 'Rarely', 'Sometimes', 'Often', 'Unknown'])
family_history = st.sidebar.selectbox("Family History", ['Yes', 'No'])
Age = st.sidebar.slider("Age", 18, 100, 30)
state = st.sidebar.text_input("State", "California")
care_options = st.sidebar.selectbox("Care Options", ['Yes', 'No'])
Gender = st.sidebar.selectbox("Gender", ['Male', 'Female', 'Non-binary', 'Other'])
no_employees = st.sidebar.selectbox("Number of Employees", ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'])
Country = st.sidebar.text_input("Country", "United States")
leave = st.sidebar.selectbox("Leave", ['Very easy', 'Somewhat easy', 'Somewhat difficult', 'Very difficult'])
benefits = st.sidebar.selectbox("Benefits", ['Yes', 'No', "Don't know"])
supervisor = st.sidebar.selectbox("Supervisor Support", ['Yes', 'No'])
coworkers = st.sidebar.selectbox("Coworker Support", ['Yes', 'No'])
phys_health_interview = st.sidebar.selectbox("Discuss Physical Health in Interview", ['Yes', 'No'])
mental_vs_physical = st.sidebar.selectbox("Mental vs Physical Health Importance", ['Yes', 'No'])
mental_health_consequence = st.sidebar.selectbox("Mental Health Consequence", ['Yes', 'No'])
seek_help = st.sidebar.selectbox("Seek Help", ['Yes', 'No'])
anonymity = st.sidebar.selectbox("Anonymity", ['Yes', 'No'])
wellness_program = st.sidebar.selectbox("Wellness Program", ['Yes', 'No'])
phys_health_consequence = st.sidebar.selectbox("Physical Health Consequence", ['Yes', 'No'])
remote_work = st.sidebar.selectbox("Remote Work", ['Yes', 'No'])
mental_health_interview = st.sidebar.selectbox("Discuss Mental Health in Interview", ['Yes', 'No'])
obs_consequence = st.sidebar.selectbox("Observed Consequence", ['Yes', 'No'])
tech_company = st.sidebar.selectbox("Tech Company", ['Yes', 'No'])
self_employed = st.sidebar.selectbox("Self Employed", ['Yes', 'No'])

# Create a dictionary for input data
input_data = {
    'work_interfere': work_interfere,
    'family_history': family_history,
    'Age': Age,
    'state': state,
    'care_options': care_options,
    'Gender': Gender,
    'no_employees': no_employees,
    'Country': Country,
    'leave': leave,
    'benefits': benefits,
    'supervisor': supervisor,
    'coworkers': coworkers,
    'phys_health_interview': phys_health_interview,
    'mental_vs_physical': mental_vs_physical,
    'mental_health_consequence': mental_health_consequence,
    'seek_help': seek_help,
    'anonymity': anonymity,
    'wellness_program': wellness_program,
    'phys_health_consequence': phys_health_consequence,
    'remote_work': remote_work,
    'mental_health_interview': mental_health_interview,
    'obs_consequence': obs_consequence,
    'tech_company': tech_company,
    'self_employed': self_employed
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# One-hot encode the input data
input_df = pd.get_dummies(input_df)

# Ensure all columns are present (add missing columns with 0)
input_df = input_df.reindex(columns=train_columns, fill_value=0)

# Predict button
if st.sidebar.button("Predict"):
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    # Display prediction
    st.subheader("Prediction")
    st.write(f"**Prediction:** {'Needs Treatment' if prediction == 1 else 'No Treatment Needed'}")
    st.write(f"**Probability:** {prediction_proba:.2f}")

    # # SHAP explanation
    # st.subheader("Feature Importance")
    # shap_values = explainer.shap_values(input_df)
    # shap.force_plot(explainer.expected_value, shap_values[0, :], input_df.iloc[0, :], matplotlib=True)
    # st.pyplot()