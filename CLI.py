import os
import pandas as pd
import joblib
import shap
import numpy as np
import argparse

# Define model paths
MODEL_PATH = os.path.join("models", "mental_health_model.pkl")
TRAIN_COLUMNS_PATH = os.path.join("models", "train_columns.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

# Load model, scaler, and training columns
lrc = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
train_columns = joblib.load(TRAIN_COLUMNS_PATH)

# SHAP explainer
explainer = shap.LinearExplainer(lrc, masker=shap.maskers.Independent(np.zeros((1, len(train_columns)))))

def predict_mental_health(input_data):
    """ Predicts whether a person needs mental health treatment based on input data. """
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encode and align columns with training data
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=train_columns, fill_value=0)

    # Scale Age
    input_df['Age'] = scaler.transform(input_df[['Age']])

    # Predict
    prediction = lrc.predict(input_df)[0]
    prediction_proba = lrc.predict_proba(input_df)[0][1]

    # SHAP explanation
    shap_values = explainer.shap_values(input_df)
    
    # Display SHAP explanation
    shap.force_plot(explainer.expected_value, shap_values[0, :], input_df.iloc[0, :])

    return prediction, prediction_proba

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mental Health Prediction CLI")
    
    parser.add_argument("--age", type=int, required=True, help="Age of the person")
    parser.add_argument("--gender", type=str, required=True, help="Gender (Male, Female, Non-binary, Other)")
    parser.add_argument("--country", type=str, required=True, help="Country of the person")
    parser.add_argument("--state", type=str, required=False, help="State (if applicable)")
    parser.add_argument("--self_employed", type=str, required=True, help="Self-employed (Yes/No)")
    parser.add_argument("--family_history", type=str, required=True, help="Family history of mental illness (Yes/No)")
    parser.add_argument("--work_interfere", type=str, required=True, help="Work interference due to mental health (Sometimes, Rarely, Often, Never, Unknown)")
    parser.add_argument("--no_employees", type=str, required=True, help="Number of employees in the company")
    parser.add_argument("--remote_work", type=str, required=True, help="Works remotely? (Yes/No)")
    parser.add_argument("--tech_company", type=str, required=True, help="Works in a tech company? (Yes/No)")
    parser.add_argument("--benefits", type=str, required=True, help="Mental health benefits provided? (Yes/No)")
    parser.add_argument("--care_options", type=str, required=True, help="Are care options available? (Yes/No)")
    parser.add_argument("--wellness_program", type=str, required=True, help="Is there a wellness program? (Yes/No)")
    parser.add_argument("--seek_help", type=str, required=True, help="Can the person seek help? (Yes/No)")
    parser.add_argument("--anonymity", type=str, required=True, help="Is anonymity protected? (Yes/No)")
    parser.add_argument("--leave", type=str, required=True, help="Ease of taking leave for mental health (Very easy, Somewhat easy, Neutral, Somewhat difficult, Very difficult)")
    parser.add_argument("--mental_health_consequence", type=str, required=True, help="Consequences for discussing mental health (Yes/No)")
    parser.add_argument("--phys_health_consequence", type=str, required=True, help="Consequences for discussing physical health (Yes/No)")
    parser.add_argument("--coworkers", type=str, required=True, help="Can you discuss with coworkers? (Yes/No)")
    parser.add_argument("--supervisor", type=str, required=True, help="Can you discuss with your supervisor? (Yes/No)")
    parser.add_argument("--mental_health_interview", type=str, required=True, help="Would mental health disclosure affect job interviews? (Yes/No)")
    parser.add_argument("--phys_health_interview", type=str, required=True, help="Would physical health disclosure affect job interviews? (Yes/No)")
    parser.add_argument("--mental_vs_physical", type=str, required=True, help="Is mental health as important as physical health? (Yes/No)")
    parser.add_argument("--obs_consequence", type=str, required=True, help="Observed negative consequences of mental health disclosure? (Yes/No)")

    args = parser.parse_args()

    # Convert arguments to dictionary
    input_data = vars(args)

    # Predict and explain
    prediction, probability = predict_mental_health(input_data)
    print(f"\nPrediction: {'Needs Treatment' if prediction == 1 else 'No Treatment Needed'}")
    print(f"Probability: {probability:.2f}")
