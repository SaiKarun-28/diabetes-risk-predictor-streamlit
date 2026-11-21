import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import base64
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical aesthetic & dark mode compatibility
st.markdown("""
<style>
    /* Clinical color palette */
    :root {
        --primary-teal: #14b8a6;
        --primary-teal-dark: #0f766e;
        --primary-teal-light: #5eead4;
        --accent-green: #10b981;
        --accent-amber: #fbbf24;
        --accent-red: #ef4444;
        --neutral-white: #ffffff;
        --neutral-gray-50: #f8fafc;
        --neutral-gray-100: #f1f5f9;
        --neutral-gray-200: #e2e8f0;
        --neutral-gray-600: #475569;
        --neutral-gray-800: #1e293b;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
    }

    /* Main container styling */
    .stApp {
        background-color: var(--neutral-gray-50);
        color: var(--neutral-gray-800);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    [data-theme="dark"] .stApp {
        background-color: #0f172a !important;
        color: #f1f5f9 !important;
    }

    [data-theme="dark"] .metric-card,
    [data-theme="dark"] .input-section,
    [data-theme="dark"] .clinical-context-box,
    [data-theme="dark"] .ci-box {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: #334155 !important;
    }

    [data-theme="dark"] .input-section-title {
        color: var(--primary-teal-light) !important;
        border-bottom: 2px solid #334155 !important;
    }

    [data-theme="dark"] p,
    [data-theme="dark"] h1,
    [data-theme="dark"] h2,
    [data-theme="dark"] h3,
    [data-theme="dark"] h4,
    [data-theme="dark"] label,
    [data-theme="dark"] div[data-testid="stMarkdownContainer"] {
        color: #f1f5f9 !important;
    }

    /* Target download buttons and text inside them in sidebar dark mode */
    [data-theme="dark"] .export-button {
        color: #ffffff !important;
    }
    [data-theme="dark"] .export-button-secondary {
        color: var(--primary-teal-light) !important;
        border-color: var(--primary-teal-light) !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    [data-theme="dark"] .export-button-secondary:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
    }

    [data-theme="dark"] .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-color: #334155 !important;
        color: #94a3b8 !important;
    }

    [data-theme="dark"] .stTabs [aria-selected="true"] {
        background-color: var(--primary-teal) !important;
        color: var(--neutral-white) !important;
        border-color: var(--primary-teal) !important;
    }

    [data-theme="dark"] .stMetric [data-testid="metric-value"] {
        color: #f1f5f9 !important;
    }

    /* Professional card styling */
    .metric-card {
        background: var(--neutral-white);
        border-radius: 12px;
        padding: 24px;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--neutral-gray-200);
        transition: all 0.2s ease;
        margin-bottom: 16px;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-color: var(--primary-teal-light);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        background-color: var(--neutral-white);
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        margin-right: 4px;
        border: 1px solid var(--neutral-gray-200);
        border-bottom: none;
        font-weight: 500;
        color: var(--neutral-gray-600);
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--neutral-gray-50);
        color: var(--primary-teal-dark);
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-teal);
        color: var(--neutral-white);
        border-color: var(--primary-teal);
    }

    /* Risk gauge styling */
    .risk-gauge {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--neutral-white);
        margin: 0 auto 20px auto;
        position: relative;
        box-shadow: var(--shadow-lg);
    }

    .risk-low { background: linear-gradient(135deg, var(--accent-green), #059669); }
    .risk-moderate { background: linear-gradient(135deg, var(--accent-amber), #d97706); }
    .risk-high { background: linear-gradient(135deg, var(--accent-red), #dc2626); }

    /* Input section styling */
    .input-section {
        background-color: var(--neutral-white);
        border-radius: 8px;
        padding: 20px;
        border: 1px solid var(--neutral-gray-200);
        margin-bottom: 16px;
    }

    .input-section-title {
        color: var(--primary-teal-dark);
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--neutral-gray-100);
    }

    /* Enhanced metric styling */
    .stMetric {
        background: var(--neutral-white);
        border-radius: 8px;
        padding: 16px;
        border: 1px solid var(--neutral-gray-200);
        box-shadow: var(--shadow-sm);
    }

    .stMetric label {
        color: var(--neutral-gray-600) !important;
        font-weight: 500 !important;
    }

    .stMetric [data-testid="metric-value"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--neutral-gray-800) !important;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: var(--neutral-gray-50);
        padding: 20px;
    }

    /* Export button styling */
    .export-button {
        background-color: var(--primary-teal);
        color: var(--neutral-white);
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
        display: inline-block;
        margin: 8px;
    }

    .export-button:hover {
        background-color: var(--primary-teal-dark);
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
        color: var(--neutral-white);
    }

    .export-button-secondary {
        background-color: var(--neutral-white);
        color: var(--primary-teal-dark);
        border: 2px solid var(--primary-teal-dark);
    }

    .export-button-secondary:hover {
        background-color: var(--neutral-gray-50);
        color: var(--primary-teal-dark);
    }

    /* Clinical Context box styling */
    .clinical-context-box {
        background-color: var(--neutral-gray-100);
        border: 1px solid var(--neutral-gray-200);
        border-radius: 8px;
        padding: 16px;
        margin-top: 20px;
    }

    .ci-box {
        background: var(--neutral-gray-100);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }

    .stSuccess {
        background-color: #f0fdf4;
        border-color: var(--accent-green);
        color: #166534;
    }

    .stError {
        background-color: #fef2f2;
        border-color: var(--accent-red);
        color: #991b1b;
    }

    .stWarning {
        background-color: #fffbeb;
        border-color: var(--accent-amber);
        color: #92400e;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--neutral-gray-800);
        font-weight: 600;
        letter-spacing: -0.025em;
    }

    p {
        color: var(--neutral-gray-600);
        line-height: 1.6;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .risk-gauge {
            width: 150px;
            height: 150px;
            font-size: 1.8rem;
        }

        .metric-card {
            padding: 16px;
        }
    }
</style>

<!-- Google Fonts -->
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
""