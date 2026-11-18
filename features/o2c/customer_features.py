"""
FinGuard AI — Customer Master Feature Engineering (O2C)
"""

from datetime import date, datetime
from typing import Any, Dict, List
import numpy as np
import pandas as pd

CUSTOMER_FEATURE_COLUMNS = [
    "customer_age_days",
    "is_new_customer",
    "is_individual",
    "profile_completeness_score",
    "credit_limit_log",
]


def compute_customer_age_days(created_at: Any, reference_date: date | None = None) -> int:
    ref = reference_date or date.today()
    if isinstance(created_at, str):
        try:
            created_at = pd.to_datetime(created_at).date()
        except Exception:
            return 365
    elif isinstance(created_at, datetime):
        created_at = created_at.date()
    if created_at is None or pd.isna(created_at):
        return 365
    return max(0, (ref - created_at).days)


def compute_profile_completeness(row: pd.Series | Dict) -> float:
    expected_fields = ["customer_name", "customer_type", "company_code", "sales_org", "payment_terms", "credit_limit"]
    present = sum(1 for f in expected_fields if row.get(f) is not None and str(row.get(f)).strip() != "")
    return round(present / len(expected_fields), 2)


def extract_customer_features(df: pd.DataFrame, reference_date: date | None = None) -> pd.DataFrame:
    """Extract customer master-level features."""
    ref = reference_date or date.today()
    feats = pd.DataFrame(index=df.index)

    created_col = "created_at" if "created_at" in df.columns else "registration_date"
    if created_col in df.columns:
        c_dates = pd.to_datetime(df[created_col]).dt.date
        feats["customer_age_days"] = [(ref - d).days if pd.notna(d) else 365 for d in c_dates]
    else:
        feats["customer_age_days"] = 365

    feats["is_new_customer"] = (feats["customer_age_days"] < 90).astype(int)
    feats["is_individual"] = (df.get("customer_type", "Organisation") == "Individual").astype(int)
    feats["profile_completeness_score"] = [compute_profile_completeness(row) for _, row in df.iterrows()]

    limits = df.get("credit_limit", df.get("approved_limit", 100000.0)).fillna(0.0).astype(float)
    feats["credit_limit_log"] = np.log1p(limits)

    return feats[CUSTOMER_FEATURE_COLUMNS]
