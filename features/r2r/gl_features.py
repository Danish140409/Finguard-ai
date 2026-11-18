"""
FinGuard AI — GL Account Activity Feature Engineering (R2R)
"""

from typing import List
import numpy as np
import pandas as pd

GL_FEATURE_COLUMNS = [
    "avg_transaction_value_log",
    "max_transaction_value_log",
    "transaction_frequency",
    "monthly_activity_change_pct",
    "period_end_transaction_pct",
    "historical_balance_log",
]


def extract_gl_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract general ledger activity summary features."""
    feats = pd.DataFrame(index=df.index)

    avg_tx = df.get("avg_transaction_value", 10000.0).fillna(10000.0).astype(float)
    feats["avg_transaction_value_log"] = np.log1p(avg_tx)

    max_tx = df.get("max_transaction_value", avg_tx).fillna(avg_tx).astype(float)
    feats["max_transaction_value_log"] = np.log1p(max_tx)

    feats["transaction_frequency"] = df.get("transaction_frequency", 10).fillna(10).astype(int)
    feats["monthly_activity_change_pct"] = df.get("monthly_activity_change_pct", 0.0).fillna(0.0).astype(float)
    feats["period_end_transaction_pct"] = df.get("period_end_transaction_pct", 0.0).fillna(0.0).astype(float)

    balance = df.get("historical_balance", 50000.0).fillna(50000.0).astype(float)
    feats["historical_balance_log"] = np.log1p(np.abs(balance))

    return feats[GL_FEATURE_COLUMNS]
