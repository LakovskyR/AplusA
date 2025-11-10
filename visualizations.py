"""
visualizations.py - Creates the pretty charts
"""

import plotly.express as px
import plotly.graph_objects as go
from config import RISK_COLOR_MAP, COLORS

def create_risk_funnel(df):
    """Funnel chart"""
    risk_counts = df["risk_category"].value_counts().reindex(["High", "Medium", "Low"])
    
    fig = go.Figure(go.Funnel(
        y=risk_counts.index,
        x=risk_counts.values,
        textinfo="value+percent initial",
        marker=dict(color=[RISK_COLOR_MAP[cat] for cat in risk_counts.index])
    ))
    
    fig.update_layout(title="Patient Risk Distribution", height=400, showlegend=False)
    return fig

def create_risk_donut(df):
    """Donut chart"""
    risk_counts = df["risk_category"].value_counts()
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        hole=0.4,
        color=risk_counts.index,
        color_discrete_map=RISK_COLOR_MAP,
    )
    
    fig.update_traces(textinfo="label+percent")
    fig.update_layout(title="Risk Distribution", height=400)
    return fig

def create_feature_importance_bar(df_importance, top_n=10):
    """Bar chart for features"""
    df_top = df_importance.head(top_n)
    
    fig = px.bar(
        df_top,
        y="feature",
        x="importance",
        orientation="h",
        text="importance",
        color_discrete_sequence=[COLORS["accent"]],
    )
    
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(
        title=f"Top {top_n} Most Important Features",
        xaxis_title="Importance",
        yaxis_title="",
        height=500,
        yaxis=dict(autorange="reversed"),
    )
    return fig

def create_probability_histogram(df):
    """Histogram"""
    fig = px.histogram(
        df,
        x="y_proba",
        nbins=20,
        color="risk_category",
        color_discrete_map=RISK_COLOR_MAP,
        barmode="overlay",
    )
    
    fig.update_layout(
        title="Predicted Probabilities",
        xaxis_title="Predicted Probability",
        yaxis_title="Count",
        height=400,
    )
    return fig

def create_confusion_matrix(df):
    """Confusion matrix"""
    
    # --- FIX: Using "y_pred" from your data instead of "pred_label" ---
    # (Indentation is also fixed here)
    tp = len(df[(df["y_true"] == 1) & (df["y_pred"] == 1)])
    tn = len(df[(df["y_true"] == 0) & (df["y_pred"] == 0)])
    fp = len(df[(df["y_true"] == 0) & (df["y_pred"] == 1)])
    fn = len(df[(df["y_true"] == 1) & (df["y_pred"] == 0)])
    
    matrix = [[tn, fp], [fn, tp]]
    labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=["Predicted 0", "Predicted 1"],
        y=["Actual 0", "Actual 1"],
        text=[[f"{labels[i][j]}<br>{matrix[i][j]}" for j in range(2)] for i in range(2)],
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
    ))
    
    fig.update_layout(title="Confusion Matrix", height=400)
    return fig