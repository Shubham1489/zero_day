# -*- coding: utf-8 -*-
# app.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from utils.helpers import set_page_config_once, store_uploaded_df, get_latest_uploaded_df, coerce_datetime_columns
from utils.preprocess import build_features
from utils.ml import fit_isolation_forest

# ---------- Config ----------
set_page_config_once()
st.title("🛡️ Universal Zero-Day Vulnerability Detection Dashboard")

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    theme = st.radio("Theme", ["Light", "Dark"], index=0, horizontal=True)
    contamination = st.slider("Anomaly contamination", 0.01, 0.20, 0.05, 0.01)
    max_rows = st.number_input("Row cap (keeps CPU/GPU-bound, not RAM-bound)", min_value=1000, max_value=1_000_000, value=100_000, step=1000)
    cat_hash_dims = st.select_slider("Categorical hash dims", options=[64,128,256,512], value=128)
    text_svd_dims = st.select_slider("Text SVD dims", options=[16,32,64,128], value=32)
    use_gpu = st.toggle("Use GPU (cuML) if available", value=False)
    st.caption("Tip: Hashing + SVD keep memory use small; Row cap avoids huge in-RAM matrices.")

# Light/Dark quick CSS
if theme == "Dark":
    st.markdown("<style>body{background:#0f1720;color:#e6eef8} .stButton>button{background:#1f2937;color:#e6eef8}</style>", unsafe_allow_html=True)

# ---------- Upload ----------
st.subheader("📂 Upload Threat Logs (CSV)")
uploaded = st.file_uploader("Drag & drop or click to browse", type=["csv"], accept_multiple_files=False)
if uploaded:
    # try utf-8 then latin-1
    try:
        df = pd.read_csv(uploaded, low_memory=True)
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, encoding="latin-1", low_memory=True)
    df = coerce_datetime_columns(df)
    store_uploaded_df(df, uploaded.name)

df = get_latest_uploaded_df()
if df is None or df.empty:
    st.info("Upload any of your log CSVs (CVE / Incidents / Flows / Detailed). The model will read ALL columns and detect anomalies.")
    st.stop()

st.success(f"Loaded: `{st.session_state.get('uploaded_filename_latest','data.csv')}` — rows: {len(df):,}, cols: {df.shape[1]}")

with st.expander("👀 Data preview", expanded=False):
    st.dataframe(df.head(20))

# ---------- Feature building (ALL columns) ----------
with st.spinner("Building features from numeric + categorical + text..."):
    X, meta, col_types = build_features(
        df, 
        max_rows=int(max_rows),
        cat_hash_dims=int(cat_hash_dims),
        text_svd_dims=int(text_svd_dims)
    )

if X.shape[0] == 0:
    st.warning("Could not build features (no usable columns).")
    st.stop()

st.write(f"Feature matrix shape: **{X.shape[0]} × {X.shape[1]}**")
st.caption(f"Detected columns — Numeric: {len(col_types['numeric'])}, Categorical: {len(col_types['categorical'])}, Text: {len(col_types['text'])}, Datetime: {len(col_types['datetime'])}")

# ---------- Train IF (CPU/GPU) ----------
with st.spinner("Running anomaly detection..."):
    risk, labels, used_gpu = fit_isolation_forest(X, contamination=float(contamination), use_gpu=bool(use_gpu))

st.success(f"Anomaly detection done on {'GPU' if used_gpu else 'CPU'}.")

# Attach scores back to the truncated dataframe used for features
df_work = df.iloc[:X.shape[0]].copy().reset_index(drop=True)
df_work["risk"] = risk
df_work["risk_label"] = labels
df_work["is_anomaly"] = df_work["risk_label"].isin(["Critical", "High"])

# ---------- Interactive filters ----------
st.subheader("🔍 Explore & Filter")
left, right = st.columns([2,1])

with right:
    only_anomalies = st.toggle("Show only anomalies (High/Critical)", value=True)
    # time column for range filter (pick one)
    time_cols = [c for c in col_types["datetime"] if c in df_work.columns]
    time_col = st.selectbox("Time column (optional)", options=["(none)"] + time_cols, index=0)
    # keyword search across text columns
    text_cols = [c for c in col_types["text"] if c in df_work.columns]
    keyword = st.text_input("Keyword contains (search text cols)", value="")
    # quick category filter (prefer common ones)
    quick_cat = None
    for cand in ["Vendor", "Attack Type", "Protocol", "Proto", "Type", "Severity Level"]:
        if cand in df_work.columns:
            quick_cat = cand
            break
    cat_values = []
    if quick_cat:
        uniq = df_work[quick_cat].dropna().astype(str).unique().tolist()
        uniq = uniq[:200]  # guard
        cat_values = st.multiselect(f"Filter by {quick_cat}", uniq, default=uniq[: min(15, len(uniq))])

with left:
    # Apply filters
    filtered = df_work.copy()
    if only_anomalies:
        filtered = filtered[filtered["is_anomaly"]]

    if time_col and time_col != "(none)":
        # ensure datetime
        filtered[time_col] = pd.to_datetime(filtered[time_col], errors="coerce")
        min_d, max_d = filtered[time_col].min(), filtered[time_col].max()
        if pd.notna(min_d) and pd.notna(max_d):
            dr = st.slider("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
            filtered = filtered[(filtered[time_col] >= dr[0]) & (filtered[time_col] <= dr[1])]

    if keyword and text_cols:
        kw = keyword.lower()
        mask = np.zeros(len(filtered), dtype=bool)
        for c in text_cols:
            try:
                mask |= filtered[c].astype(str).str.lower().str.contains(kw, na=False).values
            except Exception:
                pass
        filtered = filtered[mask]

    if quick_cat and cat_values:
        filtered = filtered[filtered[quick_cat].astype(str).isin(cat_values)]

# ---------- Display table + download ----------
st.subheader("📋 Results")
st.write(f"Showing {len(filtered):,} rows (of {len(df_work):,})")
st.dataframe(filtered)

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered results (CSV)", data=csv, file_name="zero_day_anomalies_filtered.csv", mime="text/csv")

# ---------- Visuals (auto) ----------
st.subheader("📊 Visualizations")

# Risk distribution
risk_hist = px.histogram(df_work, x="risk", nbins=40, title="Risk score distribution")
st.plotly_chart(risk_hist, use_container_width=True)

# Top categories automatically (pick a good categorical if present)
cat_for_plot = quick_cat or (col_types["categorical"][0] if col_types["categorical"] else None)
if cat_for_plot:
    topc = (df_work.groupby(cat_for_plot)["risk"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .head(25))
    fig_top = px.bar(topc, x=cat_for_plot, y="risk", title=f"Top {cat_for_plot} by average risk")
    st.plotly_chart(fig_top, use_container_width=True)

# Time trend if any datetime
if col_types["datetime"]:
    dtc = col_types["datetime"][0]
    if dtc in df_work.columns:
        tdf = df_work[[dtc, "risk"]].copy()
        tdf[dtc] = pd.to_datetime(tdf[dtc], errors="coerce")
        tdf = tdf.dropna(subset=[dtc])
        if not tdf.empty:
            tdf["bucket"] = tdf[dtc].dt.to_period("D").dt.to_timestamp()
            tr = tdf.groupby("bucket")["risk"].mean().reset_index()
            fig_t = px.line(tr, x="bucket", y="risk", title=f"Average risk over time ({dtc})")
            st.plotly_chart(fig_t, use_container_width=True)

from utils.schema import detect_schema
# ...
label, scores = detect_schema(df)
color = {"CVE":"#2563eb","Incident":"#16a34a","Flows":"#f59e0b","Detailed":"#8b5cf6","Unknown":"#6b7280"}[label]
st.markdown(
    f'<div style="display:inline-block;padding:6px 12px;border-radius:9999px;'
    f'background:{color};color:white;font-weight:600;">Schema: {label}</div>',
    unsafe_allow_html=True
)

from utils.geo import geocode_and_plot

st.subheader("🌍 Geo Map (IP Geolocation)")
# Try common IP column names in priority order
ip_candidates = [
    "Src IP Addr","Dst IP Addr","Source IP Address","Destination IP Address","Attack Source"
]
geocode_and_plot(df_work, ip_candidates, top_n=200, sleep_ms=200)
st.caption("• Cached for 24h • Gentle per-IP rate-limit • Uses local GeoLite2 if available, else ip-api.com")

from utils.model_io import export_cpu_model, PIPELINE_VERSION

st.subheader("🧳 Export Trained Pipeline")
with st.expander("Export CPU IsolationForest model (portable .pkl)", expanded=False):
    settings = {
        "pipeline_version": PIPELINE_VERSION,
        "contamination": float(contamination),
        "cat_hash_dims": int(cat_hash_dims),
        "text_svd_dims": int(text_svd_dims),
        "detected_columns": col_types  # numeric / categorical / text / datetime
    }
    bytes_obj, fname = export_cpu_model(X, float(contamination), settings)
    st.download_button(
        "⬇️ Download pipeline (.pkl)",
        data=bytes_obj,
        file_name=fname,
        mime="application/octet-stream"
    )
    st.caption("This is a CPU model for maximum portability. Even if you scored on GPU, the exported file uses a CPU IF model.")
