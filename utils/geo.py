# -*- coding: utf-8 -*-
import time
import json
import socket
from typing import List, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Try local GeoLite2 first if present (fast & offline)
try:
    import geoip2.database
    _GEO_DB = geoip2.database.Reader("GeoLite2-City.mmdb")  # put file next to app if you have it
except Exception:
    _GEO_DB = None

# Simple IPv4 check
def _is_ip(ip: str) -> bool:
    try:
        socket.inet_aton(ip)
        return True
    except Exception:
        return False

@st.cache_data(show_spinner=False, ttl=60*60*24, max_entries=5000)
def _geo_ip(ip: str) -> Tuple[float, float, str]:
    """Return (lat, lon, country) using local DB if available, else ip-api.com; cached."""
    if not _is_ip(ip):
        return (np.nan, np.nan, "")
    # Local DB path
    if _GEO_DB is not None:
        try:
            resp = _GEO_DB.city(ip)
            return (resp.location.latitude, resp.location.longitude, resp.country.name or "")
        except Exception:
            pass
    # Fallback: ip-api.com (rate-limit friendly)
    import urllib.request
    import urllib.error
    try:
        url = f"http://ip-api.com/json/{ip}?fields=status,country,lat,lon"
        with urllib.request.urlopen(url, timeout=4) as r:
            data = json.loads(r.read().decode("utf-8"))
        if data.get("status") == "success":
            return (float(data.get("lat")), float(data.get("lon")), data.get("country",""))
    except Exception:
        pass
    return (np.nan, np.nan, "")

def geocode_and_plot(df: pd.DataFrame, candidate_cols: List[str], top_n: int = 200, sleep_ms: int = 200):
    """
    Pick the first existing IP column, geocode top N unique IPs by frequency,
    cache results, and plot a scatter_geo. Sleep between uncached calls to be nice to API.
    """
    ip_col = next((c for c in candidate_cols if c in df.columns), None)
    if not ip_col:
        st.info("🌍 No IP column found for geolocation.")
        return

    ips = df[ip_col].dropna().astype(str)
    ips = ips[ips.map(_is_ip)]
    if ips.empty:
        st.info("🌍 No valid public IPv4 addresses found.")
        return

    top = ips.value_counts().head(top_n).reset_index()
    top.columns = ["ip","count"]

    # Resolve with cache + gentle rate limit
    rows = []
    for ip, cnt in top.values:
        lat, lon, country = _geo_ip(ip)  # cached
        if np.isnan(lat) or np.isnan(lon):
            # brief sleep only when it's likely a network call (uncached)
            time.sleep(sleep_ms / 1000.0)
        rows.append((ip, cnt, lat, lon, country))
    geo = pd.DataFrame(rows, columns=["ip","count","lat","lon","country"]).dropna(subset=["lat","lon"])
    if geo.empty:
        st.info("🌍 Could not geolocate IPs.")
        return

    fig = px.scatter_geo(
        geo,
        lat="lat", lon="lon",
        size="count", hover_name="ip",
        color="country",
        title=f"Top Source/Destination IPs by Count — {ip_col} (geolocated)"
    )
    st.plotly_chart(fig, use_container_width=True)
