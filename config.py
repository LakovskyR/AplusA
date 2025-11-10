"""
config.py - Settings for your app
"""

from pathlib import Path

# Paths (where to find stuff)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# App Settings
APP_TITLE = "AplusA Patient Response Prediction Platform"
APP_ICON = "🏥"
PAGE_LAYOUT = "wide"

# Colors (like choosing paint colors!)
COLORS = {
    "primary": "#003366",      # Dark blue
    "high_risk": "#D32F2F",    # Red
    "medium_risk": "#FF9800",  # Orange
    "low_risk": "#4CAF50",     # Green
    "accent": "#1976D2",       # Blue
    "background": "#F5F5F5",   # Light gray
}

# Risk Categories
RISK_CATEGORIES = ["Low", "Medium", "High"]
RISK_COLOR_MAP = {
    "Low": COLORS["low_risk"],
    "Medium": COLORS["medium_risk"],
    "High": COLORS["high_risk"],
}

# What each feature means (like a dictionary!)
FEATURE_DEFINITIONS = {
    "visits_per_month": "Average number of patient visits per month",
    "days_since_last_visit": "Days elapsed since most recent contact",
    "months_active": "Total duration patient has been in study",
    "n_visits": "Total number of visits recorded",
    "response_mean": "Average historical response rate",
    "response_std": "Variability in patient responses",
    "biomarker_mean": "Average biomarker level",
    "biomarker_std": "Biomarker variability",
    "adherence": "Treatment adherence rate (0-1)",
    "satisfaction": "Patient satisfaction score (1-5)",
}