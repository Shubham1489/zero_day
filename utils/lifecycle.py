# utils/lifecycle.py
import re

import pandas as pd

# Keyword lists for stages (expand as you like)
STAGE_KEYWORDS = {
    "Vulnerability Concealment": ["hide", "conceal", "silent", "cover up", "obfuscate"],
    "Vulnerability Discovery": ["found", "discovered", "discovery", "researcher", "reported"],
    "Vulnerability Exploitation": ["exploit", "exploitation", "payload", "remote code", "rce", "privilege escalation", "sql injection", "xss"],
    "Vulnerability Disclosure": ["disclose", "disclosure", "public", "advisory", "report"],
    "Patch Development": ["patch", "fix development", "fixing", "commit", "pull request"],
    "Patch Deployment": ["deployed", "patched", "rollout", "update applied", "hotfix"],
    "Zero Day Attack Mitigation": ["virtual patch", "workaround", "mitigation", "WAF", "IPS", "block ip", "isolate"]
}

# Flatten keywords for easy matching
def _match_keywords(text, kw_list):
    if not text: return False
    t = str(text).lower()
    for kw in kw_list:
        if kw.lower() in t:
            return True
    return False

def map_record_to_stages(row):
    """
    Heuristic mapping: examine descriptive / advisory / attackDescription / Alerts columns.
    Returns list of stages likely relevant to this record.
    """
    candidates = []
    # combine likely text fields
    combined = ""
    for key in row.index:
        # use typical descriptive columns
        if any(k in key.lower() for k in ["desc", "description", "analysis", "advisory", "attack", "alert", "signature", "root cause"]):
            combined += " " + str(row.get(key,""))
    # also check product/vendor fields
    combined += " " + str(row.get("Vendor","")) + " " + str(row.get("Product",""))
    combined = combined.lower()

    for stage, kws in STAGE_KEYWORDS.items():
        for kw in kws:
            if kw in combined:
                candidates.append(stage)
                break

    # date logic: if Date Discovered exists and Date Patched missing -> discovery or zero-day
    if "Date Discovered" in row.index and pd.notna(row.get("Date Discovered")):
        if "Date Patched" not in row.index or pd.isna(row.get("Date Patched")) or str(row.get("Date Patched")).strip()=="":
            if "Vulnerability Discovery" not in candidates:
                candidates.append("Vulnerability Discovery")
            # if high anomaly score and no patch -> Zero Day Mitigation
            # (app layer can add this rule using anomaly_score)
    # fallback: if no keywords, mark as "Vulnerability Discovery" for CVE-like entries or "Vulnerability Exploitation" if attack keywords exist
    if not candidates:
        comb = combined
        if any(k in comb for k in ["exploit","attack","rce","sql","xss","payload"]):
            candidates.append("Vulnerability Exploitation")
        elif any(k in comb for k in ["cve","vulnerability","vuln","found","discovered","advisory"]):
            candidates.append("Vulnerability Discovery")
    return list(dict.fromkeys(candidates))  # unique
