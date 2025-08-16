# -*- coding: utf-8 -*-
# utils/helpers.py
import streamlit as st
import pandas as pd

def set_page_config_once():
    if "page_config_set" not in st.session_state:
        st.set_page_config(page_title="Zero-Day Universal Detector", layout="wide")
        st.session_state["page_config_set"] = True

def app_header(title: str):
    st.markdown(f"## {title}")

def store_uploaded_df(df: pd.DataFrame, filename: str):
    st.session_state["uploaded_df_latest"] = df
    st.session_state["uploaded_filename_latest"] = filename

def get_latest_uploaded_df():
    return st.session_state.get("uploaded_df_latest")

def coerce_datetime_columns(df: pd.DataFrame):
    """Try to convert likely datetime columns (name contains date/time) to datetime."""
    if df is None: 
        return df
    for c in df.columns:
        name = c.lower()
        if any(k in name for k in ["date", "time", "timestamp", "seen"]):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    return df
