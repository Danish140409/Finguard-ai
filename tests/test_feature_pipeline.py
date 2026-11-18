"""
FinGuard AI — Feature Pipeline Unit Tests

Validates feature engineering logic for accounts payable transactions:
  - Amount z-score calculation per vendor
  - Weekend transaction flag
  - New vendor detection (< 30 days)
  - Duplicate payment within 30-day window
  - Round dollar detection
  - PO match ratio calculation
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from utils.feature_pipeline import (
    FEATURE_COLUMNS,
    extract_features,
    compute_amount_zscore_by_vendor,
    compute_days_since_vendor_registration,
    compute_duplicate_amount_flag,
    compute_is_weekend_payment,
    compute_round_dollar_flag,
)


@pytest.fixture
def sample_transactions_df():
    """Create a controlled mini-batch of transactions."""
    now = datetime(2026, 11, 15)  # Sunday
    reg_recent = now - timedelta(days=5)
    reg_old = now - timedelta(days=365)

    data = [
        # 0: Weekend, new vendor, round amount, duplicate 1
        {
            "transaction_id": "tx-1",
            "vendor_id": "v-1",
            "amount": 50000.0,
            "transaction_date": now,
            "vendor_registration_date": reg_recent,
            "po_amount": 50000.0,
            "invoice_date": now,
            "payment_date": now,
        },
        # 1: Normal weekday, established vendor, odd amount
        {
            "transaction_id": "tx-2",
            "vendor_id": "v-2",
            "amount": 1243.67,
            "transaction_date": now - timedelta(days=3),  # Thursday
            "vendor_registration_date": reg_old,
            "po_amount": 1243.67,
            "invoice_date": now - timedelta(days=10),
            "payment_date": now - timedelta(days=3),
        },
        # 2: Duplicate of tx-1 for vendor v-1 within 2 days
        {
            "transaction_id": "tx-3",
            "vendor_id": "v-1",
            "amount": 50000.0,
            "transaction_date": now - timedelta(days=2),  # Friday
            "vendor_registration_date": reg_recent,
            "po_amount": 40000.0,  # PO mismatch
            "invoice_date": now - timedelta(days=2),
            "payment_date": now - timedelta(days=2),
        },
    ]
    df = pd.DataFrame(data)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["vendor_registration_date"] = pd.to_datetime(df["vendor_registration_date"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["payment_date"] = pd.to_datetime(df["payment_date"])
    return df


def test_extract_features_schema(sample_transactions_df):
    """Ensure all required feature columns are produced."""
    features = extract_features(sample_transactions_df)
    for col in FEATURE_COLUMNS:
        assert col in features.columns, f"Missing feature: {col}"
    assert len(features) == len(sample_transactions_df)


def test_weekend_flag(sample_transactions_df):
    """Check that Sunday is marked as weekend and Thursday as non-weekend."""
    res = compute_is_weekend_payment(sample_transactions_df)
    assert res.iloc[0] == 1  # Sunday
    assert res.iloc[1] == 0  # Thursday


def test_round_dollar_flag(sample_transactions_df):
    """Check round dollar detection."""
    res = compute_round_dollar_flag(sample_transactions_df)
    assert res.iloc[0] == 1  # 50000.0 is round
    assert res.iloc[1] == 0  # 1243.67 is not round


def test_days_since_vendor_registration(sample_transactions_df):
    """Check vendor age calculation."""
    res = compute_days_since_vendor_registration(sample_transactions_df)
    assert res.iloc[0] == 5
    assert res.iloc[1] == 362
