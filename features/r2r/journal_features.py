"""
FinGuard AI — Journal Entry Feature Engineering (R2R)
"""

from typing import List
import numpy as np
import pandas as pd

JOURNAL_FEATURE_COLUMNS = [
    "journal_amount_log",
    "debit_credit_ratio",
    "num_lines",
    "num_gl_accounts",
    "posting_hour",
    "doc_to_posting_lag_days",
    "is_manual",
    "is_reversal",
    "is_period_end",
    "is_unapproved",
]


def compute_debit_credit_ratio(total_debit: float, total_credit: float) -> float:
    if total_credit <= 0:
        return 1.0 if total_debit == 0 else 2.0
    return round(total_debit / total_credit, 4)


def compute_journal_balance(debit_sum: float, credit_sum: float) -> tuple[float, bool]:
    diff = round(abs(debit_sum - credit_sum), 2)
    return diff, diff == 0.0


def is_round_number_amount(amount: float) -> bool:
    if amount <= 0:
        return False
    return amount >= 1000 and amount % 1000 == 0


def extract_journal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract machine learning features from journal entry records."""
    feats = pd.DataFrame(index=df.index)

    amount = df.get("amount", df.get("total_debit", 0.0)).fillna(0.0).astype(float)
    feats["journal_amount_log"] = np.log1p(amount)

    debit = df.get("total_debit", amount).fillna(amount).astype(float)
    credit = df.get("total_credit", amount).fillna(amount).astype(float)
    ratio = np.where(credit > 0, debit / credit, np.where(debit == 0, 1.0, 2.0))
    feats["debit_credit_ratio"] = np.clip(ratio, 0.0, 5.0)

    if "line_count" in df.columns:
        feats["num_lines"] = df["line_count"].fillna(2).astype(int)
    elif "num_lines" in df.columns:
        feats["num_lines"] = df["num_lines"].fillna(2).astype(int)
    else:
        feats["num_lines"] = 2

    if "num_gl_accounts" in df.columns:
        feats["num_gl_accounts"] = df["num_gl_accounts"].fillna(2).astype(int)
    else:
        feats["num_gl_accounts"] = 2

    if "posting_date" in df.columns:
        p_dates = pd.to_datetime(df["posting_date"])
        feats["posting_hour"] = p_dates.dt.hour.fillna(12).astype(int)
        if "document_date" in df.columns:
            d_dates = pd.to_datetime(df["document_date"])
            lag = (p_dates - d_dates).dt.days.fillna(0)
            feats["doc_to_posting_lag_days"] = np.clip(lag, 0, 365).astype(int)
        else:
            feats["doc_to_posting_lag_days"] = 0
    else:
        feats["posting_hour"] = 12
        feats["doc_to_posting_lag_days"] = 0

    jtype = df.get("journal_type", "Standard").astype(str).str.lower()
    feats["is_manual"] = jtype.isin(["manual", "adjustment"]).astype(int)
    feats["is_reversal"] = (jtype == "reversal").astype(int)

    feats["is_period_end"] = df["is_period_end"].fillna(0).astype(int) if "is_period_end" in df.columns else 0
    status = df.get("approval_status", "approved").astype(str).str.lower()
    feats["is_unapproved"] = (~status.isin(["approved", "authorized"])).astype(int)

    return feats[JOURNAL_FEATURE_COLUMNS]
