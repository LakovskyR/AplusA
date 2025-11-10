"""
generate_data.py - Creates fake patient data for testing
"""

import pandas as pd
import numpy as np

# Set seed so we get the same data each time
np.random.seed(42)

# How many fake patients?
n_patients = 1000

# Create patient information
patients_features = pd.DataFrame({
    'PatientID': [f'P{str(i).zfill(4)}' for i in range(1, n_patients + 1)],
    'Age': np.random.randint(25, 80, n_patients),
    'Sex': np.random.choice(['M', 'F'], n_patients),
    'Pathology': np.random.choice(['NSCLC', 'Melanoma', 'Breast Cancer', 'CRC'], n_patients, p=[0.4, 0.3, 0.2, 0.1]),
    'n_visits': np.random.randint(1, 30, n_patients),
    'visits_per_month': np.random.uniform(0.5, 5.0, n_patients),
    'months_active': np.random.uniform(1, 24, n_patients),
    'days_since_last_visit': np.random.randint(0, 180, n_patients),
    'adherence': np.random.uniform(0.3, 1.0, n_patients),
    'satisfaction': np.random.uniform(1, 5, n_patients),
    'response_mean': np.random.uniform(0, 1, n_patients),
    'response_std': np.random.uniform(0, 0.5, n_patients),
    'biomarker_mean': np.random.uniform(50, 150, n_patients),
    'biomarker_std': np.random.uniform(5, 30, n_patients),
    'biomarker_min': np.random.uniform(30, 100, n_patients),
    'biomarker_max': np.random.uniform(100, 200, n_patients),
})

# Create predictions
y_true = np.random.choice([0, 1], n_patients, p=[0.65, 0.35])
y_proba = np.random.beta(2, 5, n_patients)
y_proba = np.where(y_true == 1, y_proba * 0.5 + 0.5, y_proba * 0.5)

risk_category = pd.cut(y_proba, bins=[0, 0.33, 0.67, 1.0], labels=['Low', 'Medium', 'High'])
pred_label = (y_proba > 0.5).astype(int)

predictions = pd.DataFrame({
    'PatientID': patients_features['PatientID'],
    'y_true': y_true,
    'y_proba': y_proba,
    'pred_label': pred_label,
    'risk_category': risk_category,
    'pred_high_sensitivity': (y_proba > 0.35).astype(int),
    'pred_high_precision': (y_proba > 0.70).astype(int),
    'shap_top_feature': np.random.choice(['visits_per_month', 'days_since_last_visit', 'response_mean'], n_patients)
})

# Create feature importance
features = [
    'visits_per_month', 'days_since_last_visit', 'months_active', 'n_visits',
    'response_mean', 'response_std', 'biomarker_min', 'biomarker_mean',
    'biomarker_max', 'biomarker_std', 'adherence', 'satisfaction',
    'Age', 'Sex_M', 'Pathology_NSCLC'
]

importance_values = np.array([0.150, 0.149, 0.107, 0.094, 0.091, 0.083, 0.071, 
                              0.051, 0.046, 0.037, 0.035, 0.030, 0.025, 0.018, 0.013])

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': importance_values
})

# Create model scores
model_scores = pd.DataFrame({
    'model': ['Random Forest', 'Logistic Regression', 'XGBoost'],
    'auc': [0.936, 0.921, 0.928],
    'precision': [0.667, 0.404, 0.632],
    'recall': [0.545, 0.955, 0.521],
    'f1': [0.600, 0.568, 0.571],
    'pr_auc': [0.638, 0.477, 0.612],
    'brier': [0.063, 0.081, 0.069]
})

# Save everything
patients_features.to_csv('data/patients_features.csv', index=False)
predictions.to_csv('data/predictions_full_bestmodel.csv', index=False)
feature_importance.to_csv('data/feature_importance_rf.csv', index=False)
model_scores.to_csv('data/model_scores.csv', index=False)

print("✅ Created 4 data files!")
print("   - patients_features.csv")
print("   - predictions_full_bestmodel.csv")
print("   - feature_importance_rf.csv")
print("   - model_scores.csv")