"""
FinGuard AI — Feature Engineering Pipeline

Transforms raw AP transaction DataFrames into ML-ready feature matrices.
All features are computed in a single pass and are fully reproducible.

Usage:
    from utils.feature_pipeline import extract_features
    features_df = extract_features(transactions_df, vendors_df, payments_df)
"""

import hashlib
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import get_logger

logger = get_logger(__name__)

# Features expected by the ML ensemble (must match training order)
FEATURE_COLUMNS = [
    "amount_log",
    "amount_zscore_by_vendor",
    "days_since_vendor_registration",
    "is_weekend_payment",
    "duplicate_amount_flag",
    "invoice_to_payment_lag",
    "round_dollar_flag",
    "payment_frequency_spike",
    "po_match_ratio",
    "vendor_payment_count",
    "is_new_vendor",
    "amount_percentile_by_vendor",
]


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def compute_amount_log(df: pd.DataFrame, amount_col: str = "amount") -> pd.Series:
    """Log1p transform of transaction amount — reduces skew from outliers."""
    return np.log1p(df[amount_col].astype(float)).rename("amount_log")


def compute_amount_zscore_by_vendor(df: pd.DataFrame, amount_col: str = "amount") -> pd.Series:
    """
    Per-vendor z-score of transaction amount.
    Flags invoices that are unusually large/small relative to that vendor's history.
    """
    def zscore_safe(x: pd.Series) -> pd.Series:
        if len(x) < 2 or x.std() == 0:
            return pd.Series(0.0, index=x.index)
        return stats.zscore(x, ddof=1)

    result = (
        df.groupby("vendor_id")[amount_col]
        .transform(zscore_safe)
        .fillna(0.0)
        .rename("amount_zscore_by_vendor")
    )
    return result


def compute_days_since_vendor_registration(
    df: pd.DataFrame,
    reference_date_col: str = "transaction_date",
) -> pd.Series:
    """
    Number of days between vendor registration date and transaction date.
    Small values indicate a new / potentially fictitious vendor.
    """
    reg_dates = pd.to_datetime(df["vendor_registration_date"])
    txn_dates = pd.to_datetime(df[reference_date_col])
    days = (txn_dates - reg_dates).dt.days.clip(lower=0).fillna(365).rename(
        "days_since_vendor_registration"
    )
    return days


def compute_is_weekend_payment(df: pd.DataFrame, date_col: str = "transaction_date") -> pd.Series:
    """Boolean: True if the transaction date falls on Saturday (5) or Sunday (6)."""
    return (
        pd.to_datetime(df[date_col]).dt.dayofweek.isin([5, 6])
        .astype(int)
        .rename("is_weekend_payment")
    )


def compute_duplicate_amount_flag(df: pd.DataFrame, window_days: int = 30) -> pd.Series:
    """
    Flag: True if the same (vendor_id, amount) pair appears more than once
    within a rolling 30-day window.
    """
    df = df.copy()
    df["_txn_date"] = pd.to_datetime(df["transaction_date"])
    df = df.sort_values("_txn_date")

    result = pd.Series(0, index=df.index, name="duplicate_amount_flag")

    for (vendor_id, amount), grp in df.groupby(["vendor_id", "amount"]):
        if len(grp) < 2:
            continue
        dates = grp["_txn_date"].values
        for i, idx in enumerate(grp.index):
            current_date = grp.loc[idx, "_txn_date"]
            window_start = current_date - pd.Timedelta(days=window_days)
            count_in_window = ((grp["_txn_date"] >= window_start) &
                               (grp["_txn_date"] <= current_date)).sum()
            if count_in_window > 1:
                result[idx] = 1

    return result


def compute_invoice_to_payment_lag(df: pd.DataFrame) -> pd.Series:
    """
    Days between invoice date and payment date.
    Extreme values (very fast or very delayed) are anomaly signals.
    Returns NaN-filled with 30 (median) where payment data is missing.
    """
    invoice_dates = pd.to_datetime(df.get("invoice_date", pd.NaT))
    payment_dates = pd.to_datetime(df.get("payment_date", pd.NaT))
    lag = (payment_dates - invoice_dates).dt.days.fillna(30).rename("invoice_to_payment_lag")
    return lag


def compute_round_dollar_flag(df: pd.DataFrame, amount_col: str = "amount") -> pd.Series:
    """
    Boolean: 1 if amount is a 'suspiciously round' number (divisible by 100).
    Round-dollar payments are a known AP fraud indicator.
    """
    amounts = df[amount_col].astype(float)
    return ((amounts % 100 == 0) & (amounts >= 100)).astype(int).rename("round_dollar_flag")


def compute_payment_frequency_spike(
    df: pd.DataFrame,
    short_window: int = 7,
    long_window: int = 90,
    spike_ratio: float = 2.0,
) -> pd.Series:
    """
    Boolean: 1 if the rolling 7-day payment count exceeds 2× the 90-day rolling avg.
    Detects burst payment activity that could indicate split-invoice fraud.
    """
    df = df.copy()
    df["_txn_date"] = pd.to_datetime(df["transaction_date"])
    df = df.sort_values("_txn_date")
    result = pd.Series(0, index=df.index, name="payment_frequency_spike")

    for vendor_id, grp in df.groupby("vendor_id"):
        grp = grp.set_index("_txn_date").sort_index()
        if len(grp) < 2:
            continue
        rolling_short = grp["amount"].resample("D").count().rolling(
            f"{short_window}D", min_periods=1
        ).sum()
        rolling_long_avg = grp["amount"].resample("D").count().rolling(
            f"{long_window}D", min_periods=1
        ).mean()
        spike_days = rolling_short.index[rolling_short > spike_ratio * rolling_long_avg]
        for idx in grp.index:
            txn_date = idx.normalize()
            if txn_date in spike_days:
                orig_idx = df[(df["vendor_id"] == vendor_id) &
                              (df["_txn_date"].dt.normalize() == txn_date)].index
                result[orig_idx] = 1

    return result


def compute_po_match_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Ratio of invoice amount to PO amount (capped at 1).
    Values > 1 (before capping) or << 1 indicate PO mismatch.
    """
    invoice_amount = df["amount"].astype(float)
    po_amount = df.get("po_amount", pd.Series(np.nan, index=df.index)).astype(float)
    ratio = (invoice_amount / po_amount.replace(0, np.nan)).clip(upper=2.0).fillna(1.0)
    return ratio.rename("po_match_ratio")


def compute_vendor_payment_count(df: pd.DataFrame) -> pd.Series:
    """Total number of historical transactions per vendor (log-scaled)."""
    counts = df.groupby("vendor_id")["amount"].transform("count").rename("vendor_payment_count")
    return np.log1p(counts)


def compute_is_new_vendor(df: pd.DataFrame, threshold_days: int = 30) -> pd.Series:
    """Boolean: 1 if vendor was registered fewer than 30 days before the transaction."""
    days = compute_days_since_vendor_registration(df)
    return (days < threshold_days).astype(int).rename("is_new_vendor")


def compute_amount_percentile_by_vendor(df: pd.DataFrame, amount_col: str = "amount") -> pd.Series:
    """
    Percentile rank of the transaction amount within that vendor's history [0,1].
    1.0 = highest ever seen for this vendor.
    """
    return (
        df.groupby("vendor_id")[amount_col]
        .transform(lambda x: x.rank(pct=True))
        .fillna(0.5)
        .rename("amount_percentile_by_vendor")
    )


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def extract_features(
    transactions_df: pd.DataFrame,
    reference_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on a transactions DataFrame.

    Required columns in transactions_df:
        vendor_id, amount, transaction_date, vendor_registration_date

    Optional columns (filled with defaults if missing):
        invoice_date, payment_date, po_amount

    Returns a DataFrame with exactly the columns in FEATURE_COLUMNS,
    index aligned with transactions_df.
    """
    if reference_date is None:
        reference_date = date.today()

    df = transactions_df.copy()
    logger.info("Running feature pipeline", extra={"n_rows": len(df)})

    # Ensure required columns exist
    required = ["vendor_id", "amount", "transaction_date", "vendor_registration_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in transactions_df: {missing}")

    features = pd.DataFrame(index=df.index)
    features["amount_log"]                   = compute_amount_log(df)
    features["amount_zscore_by_vendor"]      = compute_amount_zscore_by_vendor(df)
    features["days_since_vendor_registration"] = compute_days_since_vendor_registration(df)
    features["is_weekend_payment"]           = compute_is_weekend_payment(df)
    features["duplicate_amount_flag"]        = compute_duplicate_amount_flag(df)
    features["invoice_to_payment_lag"]       = compute_invoice_to_payment_lag(df)
    features["round_dollar_flag"]            = compute_round_dollar_flag(df)
    features["payment_frequency_spike"]      = compute_payment_frequency_spike(df)
    features["po_match_ratio"]               = compute_po_match_ratio(df)
    features["vendor_payment_count"]         = compute_vendor_payment_count(df)
    features["is_new_vendor"]                = compute_is_new_vendor(df)
    features["amount_percentile_by_vendor"]  = compute_amount_percentile_by_vendor(df)

    # Validate output shape
    assert list(features.columns) == FEATURE_COLUMNS, (
        f"Feature column mismatch: got {list(features.columns)}"
    )

    logger.info("Feature pipeline complete", extra={"n_features": len(FEATURE_COLUMNS)})
    return features.astype(float)
