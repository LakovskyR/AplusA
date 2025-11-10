"""
data_loader.py - Loads the data files
"""

import streamlit as st
import pandas as pd
from config import DATA_DIR

@st.cache_data
def load_patients_features():
    """Load patient features"""
    df = pd.read_csv(DATA_DIR / "patients_features.csv")
    return df

@st.cache_data
def load_predictions():
    """Load predictions"""
    df = pd.read_csv(DATA_DIR / "predictions_full_bestmodel.csv")
    return df

@st.cache_data
def load_feature_importance():
    """Load feature importance"""
    df = pd.read_csv(DATA_DIR / "feature_importance_rf.csv")
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df

@st.cache_data
def load_model_scores():
    """Load model scores"""
    df = pd.read_csv(DATA_DIR / "model_scores.csv")
    return df

@st.cache_data
def load_all_data():
    """Load everything and combine"""
    patients = load_patients_features()
    predictions = load_predictions()
    df_merged = patients.merge(predictions, on="PatientID", how="inner")
    return df_merged, load_feature_importance(), load_model_scores()

def get_data_summary():
    """Get quick stats"""
    df, _, _ = load_all_data()
    summary = {
        "total_patients": len(df),
        "high_risk": len(df[df["risk_category"] == "High"]),
        "medium_risk": len(df[df["risk_category"] == "Medium"]),
        "low_risk": len(df[df["risk_category"] == "Low"]),
        "avg_probability": df["y_proba"].mean(),
        "pathologies": df["Pathology"].nunique(),
    }
    return summary