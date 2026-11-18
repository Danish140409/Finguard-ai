"""
FinGuard AI — Credit Exposure & Limit Feature Engineering (O2C)
"""

from typing import List
import numpy as np
import pandas as pd

CREDIT_FEATURE_COLUMNS = [
    "credit_utilisation_pct",
    "exposure_to_limit_ratio",
    "available_credit_log",
    "credit_limit_change_pct",
    "is_limit_breached",
    "num_credit_events",
    "num_blocked_orders",
]


def compute_credit_utilisation(exposure: float, limit: float) -> float:
    if limit <= 0:
        return 100.0 if exposure > 0 else 0.0
    return round((exposure / limit) * 100.0, 2)


def compute_available_credit(limit: float, exposure: float) -> float:
    return max(0.0, limit - exposure)


def compute_credit_limit_change_pct(new_limit: float, prev_limit: float) -> float:
    if prev_limit <= 0:
        return 0.0
    return round(((new_limit - prev_limit) / prev_limit) * 100.0, 2)


def extract_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract customer credit features DataFrame."""
    feats = pd.DataFrame(index=df.index)

    exposure = df["total_exposure"].fillna(0.0).astype(float) if "total_exposure" in df.columns else pd.Series(0.0, index=df.index)
    if "credit_limit" in df.columns:
        limit = df["credit_limit"].fillna(0.0).astype(float)
    elif "approved_limit" in df.columns:
        limit = df["approved_limit"].fillna(0.0).astype(float)
    else:
        limit = pd.Series(0.0, index=df.index)

    utilisation = np.where(limit > 0, (exposure / limit) * 100.0, np.where(exposure > 0, 100.0, 0.0))
    feats["credit_utilisation_pct"] = np.clip(utilisation, 0.0, 500.0)

    ratio = np.where(limit > 0, exposure / limit, np.where(exposure > 0, 1.0, 0.0))
    feats["exposure_to_limit_ratio"] = np.clip(ratio, 0.0, 10.0)

    available = np.clip(limit - exposure, 0.0, None)
    feats["available_credit_log"] = np.log1p(available)

    if "previous_limit" in df.columns:
        prev_limit = df["previous_limit"].fillna(limit).astype(float)
    else:
        prev_limit = limit
    change_pct = np.where(prev_limit > 0, ((limit - prev_limit) / prev_limit) * 100.0, 0.0)
    feats["credit_limit_change_pct"] = np.clip(change_pct, -100.0, 1000.0)

    feats["is_limit_breached"] = (exposure > limit).astype(int)
    feats["num_credit_events"] = df["num_credit_events"].fillna(0).astype(int) if "num_credit_events" in df.columns else 0
    feats["num_blocked_orders"] = df["num_blocked_orders"].fillna(0).astype(int) if "num_blocked_orders" in df.columns else 0

    return feats[CREDIT_FEATURE_COLUMNS]
