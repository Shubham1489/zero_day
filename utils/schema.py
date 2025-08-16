# -*- coding: utf-8 -*-
import pandas as pd
from typing import Tuple, Dict, List

_SCHEMAS: Dict[str, List[str]] = {
    "CVE": ["CVE","Vendor","Product","Type","Description","Date Discovered","Date Patched"],
    "Incident": ["Country","Year","Attack Type","Target Industry","Financial Loss (in Million $)","Number of Affected Users"],
    "Flows": ["Date first seen","Duration","Proto","Src IP Addr","Dst IP Addr","Packets","Bytes","Flows"],
    "Detailed": ["Timestamp","Source IP Address","Destination IP Address","Source Port","Destination Port","Protocol","Packet Length"]
}

def detect_schema(df: pd.DataFrame) -> Tuple[str, Dict[str, int]]:
    cols = set(df.columns)
    scores = {}
    for name, req in _SCHEMAS.items():
        scores[name] = sum(1 for c in req if c in cols)
    # best match
    label = max(scores, key=lambda k: scores[k])
    if scores[label] == 0:
        label = "Unknown"
    return label, scores
