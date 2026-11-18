"""
FinGuard AI — Accounts Receivable & Payment Behaviour Features (O2C)
"""

from datetime import date
from typing import List
import numpy as np
import pandas as pd

AR_FEATURE_COLUMNS = [
    "avg_days_past_due",
    "max_days_past_due",
    "pct_invoices_late",
    "total_overdue_log",
    "num_overdue_invoices",
    "avg_payment_delay",
    "partial_payment_freq",
    "payment_regularity_score",
]


def compute_days_past_due(due_date: date, reference_date: date | None = None) -> int:
    ref = reference_date or date.today()
    if isinstance(due_date, str):
        try:
            due_date = pd.to_datetime(due_date).date()
        except Exception:
            return 0
    if not due_date or pd.isna(due_date):
        return 0
    return max(0, (ref - due_date).days)


def compute_aging_bucket(dpd: int) -> str:
    if dpd <= 0:
        return "Not Due"
    elif dpd <= 30:
        return "1-30 Days"
    elif dpd <= 60:
        return "31-60 Days"
    elif dpd <= 90:
        return "61-90 Days"
    else:
        return "Above 90 Days"


def extract_ar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract accounts receivable behaviour features."""
    feats = pd.DataFrame(index=df.index)

    feats["avg_days_past_due"] = df["avg_days_past_due"].fillna(0.0).astype(float) if "avg_days_past_due" in df.columns else 0.0
    feats["max_days_past_due"] = df["max_days_past_due"].fillna(0).astype(int) if "max_days_past_due" in df.columns else 0
    feats["pct_invoices_late"] = df["pct_invoices_late"].fillna(0.0).astype(float) if "pct_invoices_late" in df.columns else 0.0

    if "total_overdue_amount" in df.columns:
        overdue_amt = df["total_overdue_amount"].fillna(0.0).astype(float)
    elif "overdue_balance" in df.columns:
        overdue_amt = df["overdue_balance"].fillna(0.0).astype(float)
    else:
        overdue_amt = pd.Series(0.0, index=df.index)
    feats["total_overdue_log"] = np.log1p(np.clip(overdue_amt, 0.0, None))

    if "num_overdue_invoices" in df.columns:
        feats["num_overdue_invoices"] = df["num_overdue_invoices"].fillna(0).astype(int)
    elif "overdue_invoices_count" in df.columns:
        feats["num_overdue_invoices"] = df["overdue_invoices_count"].fillna(0).astype(int)
    else:
        feats["num_overdue_invoices"] = 0

    feats["avg_payment_delay"] = df["avg_payment_delay_days"].fillna(0.0).astype(float) if "avg_payment_delay_days" in df.columns else 0.0
    feats["partial_payment_freq"] = df["partial_payment_freq"].fillna(0.0).astype(float) if "partial_payment_freq" in df.columns else 0.0

    # Regularity score: 1.0 = highly regular/on-time, 0.0 = completely irregular
    delay = feats["avg_days_past_due"]
    late_pct = feats["pct_invoices_late"]
    regularity = 1.0 - np.clip((delay / 60.0) * 0.5 + (late_pct / 100.0) * 0.5, 0.0, 1.0)
    feats["payment_regularity_score"] = np.round(regularity, 2)

    return feats[AR_FEATURE_COLUMNS]
