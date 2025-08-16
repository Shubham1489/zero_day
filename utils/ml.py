# -*- coding: utf-8 -*-
# utils/ml.py
import numpy as np
import pandas as pd

# CPU models
from sklearn.ensemble import IsolationForest

# Optional GPU
try:
    from cuml.ensemble import IsolationForest as cuIsolationForest  # RAPIDS
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

def fit_isolation_forest(X: np.ndarray, contamination: float = 0.05, use_gpu: bool = False, random_state: int = 42):
    """Train IF on CPU by default; use GPU if requested and available."""
    if X is None or X.shape[0] == 0:
        return np.array([]), None, False

    if use_gpu and _HAS_GPU:
        model = cuIsolationForest(
            n_estimators=200, 
            contamination=contamination,
            random_state_state=random_state  # cuML param name
        )
        model.fit(X)
        # cuML returns scores via decision_function (higher = normal). We invert to "riskish"
        scores = -model.decision_function(X)
        used_gpu = True
    else:
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X)
        scores = -model.decision_function(X)
        used_gpu = False

    # Normalize 0..100 for readability
    if len(scores) == 0:
        risk = np.array([])
    else:
        lo, hi = float(np.min(scores)), float(np.max(scores))
        if hi - lo < 1e-12:
            risk = np.zeros_like(scores, dtype=np.float32)
        else:
            risk = ((scores - lo) / (hi - lo) * 100.0).astype(np.float32)

    def bucket(v):
        if v >= 80: return "Critical"
        if v >= 60: return "High"
        if v >= 40: return "Medium"
        if v >= 20: return "Low"
        return "Info"

    labels = np.array([bucket(r) for r in risk], dtype=object)
    return risk, labels, used_gpu
