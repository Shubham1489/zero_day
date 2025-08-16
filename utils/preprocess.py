# -*- coding: utf-8 -*-
# utils/preprocess.py
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

def _detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Heuristics to split columns into numeric / categorical / text / datetime."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # datetime likely by dtype or name
    dt_cols = []
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            dt_cols.append(c)
        else:
            name = c.lower()
            if any(k in name for k in ["date", "time", "timestamp", "seen"]):
                # try parse on a small sample
                try:
                    pd.to_datetime(df[c], errors="raise")
                    dt_cols.append(c)
                except Exception:
                    pass

    # text vs categorical (object)
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    text_cols, cat_cols = [], []
    for c in obj_cols:
        s = df[c].dropna().astype(str)
        if s.empty:
            cat_cols.append(c)
            continue
        avg_len = s.str.len().mean()
        uniq_ratio = s.nunique() / max(1, len(s))
        # heuristic: long strings or high uniqueness → text; short values → categorical
        if avg_len > 25 or uniq_ratio > 0.7 or any(k in c.lower() for k in ["description", "payload", "analysis", "signature", "url"]):
            text_cols.append(c)
        else:
            cat_cols.append(c)

    # remove overlaps
    cat_cols = [c for c in cat_cols if c not in dt_cols]
    text_cols = [c for c in text_cols if c not in dt_cols]

    return {
        "numeric": [c for c in numeric_cols if c not in dt_cols],
        "categorical": cat_cols,
        "text": text_cols,
        "datetime": dt_cols
    }

def build_features(
    df: pd.DataFrame,
    max_rows: int = 100000,
    cat_hash_dims: int = 128,
    text_hash_dims: int = 2**12,
    text_svd_dims: int = 32
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, List[str]]]:
    """
    Build dense feature matrix from numeric + categorical + text in a memory-conscious way.
    - Numeric -> StandardScaler
    - Categorical -> FeatureHasher (fixed dims)
    - Text -> HashingVectorizer + TruncatedSVD (fixed dims)
    Returns: (X_dense, meta, col_types)
    """
    if df is None or df.empty:
        return np.zeros((0, 1)), {}, {"numeric":[], "categorical":[], "text":[], "datetime":[]}

    # Cap rows to keep memory bounded and ensure CPU/GPU bound
    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    col_types = _detect_column_types(df)

    # Numeric
    X_num = None
    scaler = None
    if col_types["numeric"]:
        num = df[col_types["numeric"]].copy()
        for c in num.columns:
            num[c] = pd.to_numeric(num[c], errors="coerce").fillna(0.0)
        scaler = StandardScaler()
        X_num = scaler.fit_transform(num.values)  # dense

    # Categorical -> FeatureHasher with mappings per row
    X_cat = None
    if col_types["categorical"]:
        hasher = FeatureHasher(n_features=cat_hash_dims, input_type="dict", alternate_sign=False)
        mappings = []
        # use small dict per row: {col_name: value}
        for _, row in df[col_types["categorical"]].astype(str).iterrows():
            m = {}
            for k, v in row.items():
                m[f"{k}={v}"] = 1
            mappings.append(m)
        X_cat_sparse = hasher.transform(mappings)  # sparse
        X_cat = X_cat_sparse.astype(np.float32).toarray()  # small fixed dims → dense OK

    # Text -> HashingVectorizer + SVD per column then concatenate
    X_txt_list = []
    svds = {}
    if col_types["text"]:
        for col in col_types["text"]:
            vec = HashingVectorizer(
                n_features=text_hash_dims,
                alternate_sign=False,
                ngram_range=(1,2),
                norm='l2',
                stop_words='english'
            )
            s = df[col].astype(str).fillna("")
            Xh = vec.transform(s)  # sparse (rows x text_hash_dims)
            # reduce to small dense block
            svd = TruncatedSVD(n_components=min(text_svd_dims, min(Xh.shape)-1) if min(Xh.shape) > 1 else 1, random_state=42)
            Xsvd = svd.fit_transform(Xh)  # dense (rows x d_svd)
            X_txt_list.append(Xsvd.astype(np.float32))
            svds[col] = svd

    # Concatenate blocks
    blocks = [b for b in [X_num, X_cat] + X_txt_list if b is not None]
    if not blocks:
        X = np.zeros((len(df), 1), dtype=np.float32)
    else:
        X = np.concatenate(blocks, axis=1).astype(np.float32)

    meta = {
        "scaler": scaler,
        "col_types": col_types,
        "cat_hash_dims": cat_hash_dims,
        "text_hash_dims": text_hash_dims,
        "text_svd_dims": text_svd_dims,
        "svds": svds
    }
    return X, meta, col_types
