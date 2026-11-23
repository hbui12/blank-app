"""Presentation-only styles. Does not change forecasts or ML behavior."""

import streamlit as st


def apply_blue_theme() -> None:
    """Apply coordinated navy/blue/teal accent styling to the dashboard."""
    st.markdown(
        """
        <style>
          /* Soft blue canvas and calm, readable navigation */
          [data-testid="stAppViewContainer"] {
              background: linear-gradient(180deg, #F6FAFF 0%, #FFFFFF 65%);
          }
          [data-testid="stSidebar"] {
              background-color: #EAF3FF;
              border-right: 1px solid #D4E5F5;
          }

          /* Text hierarchy */
          h1 { color: #0B2A50 !important; letter-spacing: -0.025em; }
          h2, h3 { color: #143E70 !important; }
          [data-testid="stSidebar"] h2,
          [data-testid="stSidebar"] h3 { color: #123F75 !important; }

          /* Three easy-to-scan KPI cards */
          [data-testid="stMetric"] {
              background: #EDF5FF;
              border: 1px solid #CEDFF5;
              border-radius: 14px;
              padding: 16px 18px;
              box-shadow: 0 2px 8px rgba(16, 57, 106, 0.04);
          }
          [data-testid="stMetricValue"] {
              color: #1454A1;
          }

          /* Buttons and panels */
          div.stButton > button[kind="primary"] {
              color: #FFFFFF;
              background: #2563EB;
              border-color: #2563EB;
              border-radius: 10px;
          }
          div.stButton > button[kind="primary"]:hover {
              background: #1D4ED8;
              border-color: #1D4ED8;
          }
          [data-testid="stExpander"] {
              border-color: #D4E5F5;
              border-radius: 12px;
          }
          [data-testid="stDataFrame"] {
              border-radius: 12px;
              overflow: hidden;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
