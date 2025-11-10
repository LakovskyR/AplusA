"""
utils.py - Helper functions
"""

import pandas as pd
from datetime import datetime

def format_percentage(value, decimals=1):
    """Turn 0.667 into 66.7%"""
    return f"{value * 100:.{decimals}f}%"

def format_number(value, decimals=0):
    """Turn 1000 into 1,000"""
    return f"{value:,.{decimals}f}"

def export_to_csv(df):
    """Convert data to CSV for download"""
    return df.to_csv(index=False).encode("utf-8")

def filter_dataframe(df, filters):
    """Apply filters to data"""
    df_filtered = df.copy()
    
    if "risk_categories" in filters and filters["risk_categories"]:
        df_filtered = df_filtered[df_filtered["risk_category"].isin(filters["risk_categories"])]
    
    if "pathologies" in filters and filters["pathologies"]:
        df_filtered = df_filtered[df_filtered["Pathology"].isin(filters["pathologies"])]
    
    if "age_range" in filters:
        min_age, max_age = filters["age_range"]
        df_filtered = df_filtered[(df_filtered["Age"] >= min_age) & (df_filtered["Age"] <= max_age)]
    
    if "probability_threshold" in filters:
        df_filtered = df_filtered[df_filtered["y_proba"] >= filters["probability_threshold"]]
    
    return df_filtered

def get_model_metrics(model_scores_df, model_name="rf"):
    """Get metrics for a specific model"""
    model_row = model_scores_df[model_scores_df["model"] == model_name].iloc[0]
    
    metrics = {
        "AUC": model_row["auc"],
        "Precision": model_row["precision"],
        "Recall": model_row["recall"],
        "F1 Score": model_row["f1"],
        "PR-AUC": model_row["pr_auc"],
        "Brier Score": model_row["brier"],
    }
    return metrics

def get_last_refresh_date():
    """Get current date/time"""
    return datetime.now().strftime("%B %d, %Y at %H:%M")