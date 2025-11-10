"""
app.py - The main application file
"""

import streamlit as st
from config import APP_TITLE, APP_ICON, PAGE_LAYOUT, COLORS, FEATURE_DEFINITIONS
from data_loader import load_all_data, get_data_summary
from visualizations import (
    create_risk_funnel,
    create_risk_donut,
    create_feature_importance_bar,
    create_probability_histogram,
    create_confusion_matrix,
)
from utils import (
    format_percentage,
    format_number,
    get_model_metrics,
    get_last_refresh_date,
    filter_dataframe,
    export_to_csv
)

# Page setup
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(f"""
<style>
    .main {{
        background-color: {COLORS['background']};
    }}
    h1, h2, h3 {{
        color: {COLORS['primary']};
    }}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_resource
def load_app_data():
    return load_all_data()

df_merged, df_importance, df_model_scores = load_app_data()
data_summary = get_data_summary()

# Sidebar
with st.sidebar:
    st.markdown(f"<h1 style='color: {COLORS['primary']}; text-align: center;'>AplusA</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {COLORS['accent']}; text-align: center;'>Patient Platform</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "**Navigate:**",
        ["🏠 Dashboard", "🔍 Patient Explorer", "📊 Feature Importance", "🎯 Model Performance"]
    )
    
    st.markdown("---")
    st.caption(f"**Last Refresh:** {get_last_refresh_date()}")
    st.caption(f"**Total Patients:** {format_number(data_summary['total_patients'])}")
    st.caption(f"**High Risk:** {data_summary['high_risk']}")
    st.caption(f"**Medium Risk:** {data_summary['medium_risk']}")
    st.caption(f"**Low Risk:** {data_summary['low_risk']}")

# Main title
st.title(f"{APP_ICON} {APP_TITLE}")

# PAGE 1: DASHBOARD
if page == "🏠 Dashboard":
    st.header("Executive Dashboard")
    st.markdown("Overview of patient response prediction model.")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    # This will use "rf" as default, based on your utils.py
    metrics = get_model_metrics(df_model_scores) 
    
    with col1:
        st.metric("Total Patients", format_number(data_summary["total_patients"]))
    with col2:
        st.metric("Model AUC", f"{metrics['AUC']:.3f}", delta="Excellent")
    with col3:
        st.metric("Precision", format_percentage(metrics['Precision']))
    with col4:
        st.metric("Recall", format_percentage(metrics['Recall']))
    
    st.markdown("---")
    
    # Charts
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(create_risk_funnel(df_merged), use_container_width=True)
    with col_right:
        st.plotly_chart(create_risk_donut(df_merged), use_container_width=True)
    
    st.plotly_chart(create_probability_histogram(df_merged), use_container_width=True)
    
    # Table
    st.subheader("Risk Breakdown")
    risk_summary = df_merged.groupby("risk_category").agg({
        "PatientID": "count",
        "y_proba": "mean",
        "Age": "mean",
    }).reset_index()
    risk_summary.columns = ["Risk", "Count", "Avg Prob", "Avg Age"]
    risk_summary["Avg Prob"] = risk_summary["Avg Prob"].apply(format_percentage)
    risk_summary["Avg Age"] = risk_summary["Avg Age"].apply(lambda x: f"{x:.1f}")
    st.dataframe(risk_summary, use_container_width=True, hide_index=True)

# PAGE 2: PATIENT EXPLORER
elif page == "🔍 Patient Explorer":
    st.header("Patient Explorer")
    st.markdown("Filter and explore patients.")
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.multiselect("Risk:", ["Low", "Medium", "High"], default=["High"])
    with col2:
        pathology_filter = st.multiselect("Pathology:", df_merged["Pathology"].unique().tolist(), default=df_merged["Pathology"].unique().tolist())
    with col3:
        age_filter = st.slider("Age:", int(df_merged["Age"].min()), int(df_merged["Age"].max()), (int(df_merged["Age"].min()), int(df_merged["Age"].max())))
    
    prob_threshold = st.slider("Min Probability:", 0.0, 1.0, 0.0, 0.05, format="%.2f")
    
    # Apply filters
    filters = {
        "risk_categories": risk_filter,
        "pathologies": pathology_filter,
        "age_range": age_filter,
        "probability_threshold": prob_threshold,
    }
    df_filtered = filter_dataframe(df_merged, filters)
    
    st.info(f"**{len(df_filtered)}** patients match filters (out of {len(df_merged)})")
    
    # Table
    st.subheader("Patients")
    
    # --- FIX 1: Using your capitalized column names ---
    display_cols = ["PatientID", "Age", "Sex", "Pathology", "risk_category", "y_proba", "n_visits", "Adherence", "Satisfaction"]
    
    df_display = df_filtered[display_cols].copy()
    
    # --- FIX 2: Renaming columns to match ---
    df_display.columns = ["Patient", "Age", "Sex", "Path", "Risk", "Prob", "Visits", "Adherence", "Satisfaction"]
    
    df_display["Prob"] = df_display["Prob"].apply(format_percentage)
    
    # --- FIX 3: Formatting the correct columns ---
    df_display["Adherence"] = df_display["Adherence"].apply(lambda x: f"{x:.2f}")
    df_display["Satisfaction"] = df_display["Satisfaction"].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # Export
    st.subheader("Export")
    csv_data = export_to_csv(df_filtered[display_cols])
    st.download_button("📥 Download CSV", csv_data, f"patients_{get_last_refresh_date().replace(' ', '_')}.csv", "text/csv")

# PAGE 3: FEATURE IMPORTANCE
elif page == "📊 Feature Importance":
    st.header("Feature Importance")
    st.markdown("What drives predictions?")
    
    top_n = st.slider("Show Top N:", 5, 15, 10, 1)
    st.plotly_chart(create_feature_importance_bar(df_importance, top_n=top_n), use_container_width=True)
    
    st.subheader("Definitions")
    df_top = df_importance.head(top_n).copy()
    df_top["definition"] = df_top["feature"].map(lambda x: FEATURE_DEFINITIONS.get(x, "N/A"))
    df_display = df_top[["rank", "feature", "importance", "definition"]].copy()
    df_display.columns = ["Rank", "Feature", "Importance", "Meaning"]
    df_display["Importance"] = df_display["Importance"].apply(format_percentage)
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    st.info("💡 **Key Insight:** Engagement metrics = ~40% of predictive power!")

# PAGE 4: MODEL PERFORMANCE
elif page == "🎯 Model Performance":
    st.header("Model Performance")
    st.markdown("Technical validation.")
    
    metrics = get_model_metrics(df_model_scores) # Uses "rf"
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("AUC", f"{metrics['AUC']:.3f}")
        st.metric("PR-AUC", f"{metrics['PR-AUC']:.3f}")
    with col2:
        st.metric("Precision", format_percentage(metrics['Precision']))
        st.metric("Recall", format_percentage(metrics['Recall']))
    with col3:
        st.metric("F1", f"{metrics['F1 Score']:.3f}")
        st.metric("Brier", f"{metrics['Brier Score']:.3f}")
    
    st.markdown("---")
    st.subheader("Confusion Matrix")
    # This will now use "y_pred" from your data
    st.plotly_chart(create_confusion_matrix(df_merged), use_container_width=True)
    
    st.subheader("Model Comparison")
    df_display = df_model_scores.copy()
    df_display["auc"] = df_display["auc"].apply(lambda x: f"{x:.3f}")
    df_display["precision"] = df_display["precision"].apply(format_percentage)
    df_display["recall"] = df_display["recall"].apply(format_percentage)
    df_display["f1"] = df_display["f1"].apply(lambda x: f"{x:.3f}")
    # Renaming for display
    df_display.columns = ["Model", "AUC", "PR-AUC", "Precision", "Recall", "F1", "Brier"]
    st.dataframe(df_display, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.caption("AplusA Platform | Built with Streamlit")