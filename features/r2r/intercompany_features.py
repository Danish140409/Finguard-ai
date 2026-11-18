"""
FinGuard AI — Intercompany Transaction Features (R2R)
"""

from typing import List
import numpy as np
import pandas as pd

INTERCOMPANY_FEATURE_COLUMNS = [
    "amount_difference_log",
    "currency_mismatch_flag",
    "is_mismatched",
]


def extract_intercompany_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract intercompany matching features."""
    feats = pd.DataFrame(index=df.index)

    src = df.get("source_amount", 0.0).fillna(0.0).astype(float)
    dst = df.get("destination_amount", 0.0).fillna(0.0).astype(float)
    diff = np.abs(src - dst)
    feats["amount_difference_log"] = np.log1p(diff)

    src_curr = df.get("source_currency", "USD").astype(str).str.upper()
    dst_curr = df.get("destination_currency", "USD").astype(str).str.upper()
    feats["currency_mismatch_flag"] = (src_curr != dst_curr).astype(int)

    status = df.get("matching_status", "matched").astype(str).str.lower()
    feats["is_mismatched"] = (status != "matched").astype(int)

    return feats[INTERCOMPANY_FEATURE_COLUMNS]
