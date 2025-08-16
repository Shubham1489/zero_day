# -*- coding: utf-8 -*-
import io
import pickle
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest

PIPELINE_VERSION = "ud-zeroday.v1.2.0"

@dataclass
class PipelineMeta:
    version: str
    timestamp: float
    contamination: float
    feature_dims: int
    settings: Dict[str, Any]  # e.g., cat_hash_dims, text_svd_dims, detected columns

def export_cpu_model(X: np.ndarray, contamination: float, settings: Dict[str, Any]) -> Tuple[bytes, str]:
    """
    Train a CPU IsolationForest and export a portable pickle with:
      - trained model
      - pipeline metadata (version, settings)
    NOTE: If your session used GPU for scoring, we still export a CPU model for portability.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    ).fit(X)

    meta = PipelineMeta(
        version=PIPELINE_VERSION,
        timestamp=time.time(),
        contamination=contamination,
        feature_dims=int(X.shape[1]),
        settings=settings,
    )

    bundle = {
        "model_type": "IsolationForest_CPU",
        "model": model,
        "meta": asdict(meta)
    }
    buf = io.BytesIO()
    pickle.dump(bundle, buf, protocol=pickle.HIGHEST_PROTOCOL)
    buf.seek(0)
    fname = f"zeroday_pipeline_{PIPELINE_VERSION}_{int(meta.timestamp)}.pkl"
    return buf.read(), fname
