# Mental Health Prediction in the Tech Workplace

## Overview

This project analyzes a dataset from a 2014 survey measuring attitudes toward mental health and the frequency of mental health disorders in the tech workplace. The model predicts whether an individual may need mental health treatment based on various workplace and personal factors.

## Dataset

The dataset contains responses related to workplace mental health awareness and support. Key features include:

- **Demographics**: Age, Gender, Country, State
- **Employment details**: Self-employed, Tech company, Number of employees, Remote work
- **Workplace mental health support**: Mental health benefits, Anonymity, Leave policies, Wellness programs
- **Mental health perceptions**: Stigma, Consequences of discussing mental health issues, Willingness to discuss with coworkers and supervisors
- **Previous mental health conditions**: Family history, Previous treatment

## Dataset Preprocessing

1. **Cleaning and Handling Missing Values**
   - Removed unnecessary columns (`comments`, `Timestamp`)
   - Filled missing values in `state` with corresponding `Country`
   - Filled missing values in `work_interfere` with 'unknown'
   - Removed rows with missing `self_employed` values
   - Filtered age values to be between 18 and 100
2. **Gender Normalization**
   - Standardized gender categories into `Male`, `Female`, `Non-binary`, and `Other`
3. **Encoding Categorical Features**
   - One-hot encoding applied to categorical variables
   - Dropped the first category of each feature to avoid multicollinearity
4. **Feature Scaling**
   - Applied MinMax Scaling to normalize `Age`

## Model Selection and Training

The model used for classification is **Logistic Regression** with `l1` penalty and `liblinear` solver. The dataset was split into an 80-20 train-test ratio.

### Performance Metrics:

- **Accuracy**: 84.21%
- **Precision**: 82.19%
- **ROC-AUC Score**: 0.89

## Explainability: SHAP Analysis

SHAP (SHapley Additive exPlanations) was used to interpret the model:

- Features like `work_interfere`, `family_history`, and `seek_help` had the highest impact.
- The most influential feature was `work_interfere_Sometimes`, indicating work interference plays a crucial role in mental health conditions.

## How to Run the Model

### 1. Install Dependencies

Ensure you have Python installed and install required libraries:

```bash
pip install pandas numpy scikit-learn shap joblib matplotlib
```

### 2. Load Pretrained Model

Download the trained model (`mental_health_model.pkl`), scaler (`scaler.pkl`), and training columns (`train_columns.pkl`).


### 3. Running Inference

To run the model using the command line interface, use the following command:

```bash
python CLI.py --age 30 --gender "Male" --country "United States" --state "California" \ 
    --self_employed "No" --family_history "Yes" --work_interfere "Sometimes" --no_employees "6-25" \
    --remote_work "Yes" --tech_company "Yes" --benefits "Yes" --care_options "Yes" --wellness_program "Yes" \
    --seek_help "Yes" --anonymity "Yes" --leave "Somewhat easy" --mental_health_consequence "No" \
    --phys_health_consequence "No" --coworkers "Yes" --supervisor "Yes" --mental_health_interview "Yes" \
    --phys_health_interview "Yes" --mental_vs_physical "Yes" --obs_consequence "No"

    The output will be:

```bash
    Prediction: No Treatment Needed
    Probability: 0.11
## UI/CLI Usage Instructions

- The model will output:
  - **Prediction**: Whether the user is likely to need treatment.
  - **Probability Score**: Confidence of the prediction.
  - **SHAP Force Plot**: Visualizing feature contributions.

## Future Improvements

- Fine-tuning the model with additional features.
- Expanding the dataset with newer survey results.
- Deploying the model as a web-based tool for workplace mental health awareness.

## Acknowledgments

- Dataset sourced from Open Sourcing Mental Illness (OSMI).
- Inspired by real-world challenges in workplace mental health advocacy.

---

This project aims to bridge the gap between data science and mental health, providing insights for a more supportive workplace.

