"""
FinGuard AI — Sales Order Features (O2C)
"""

from typing import List
import numpy as np
import pandas as pd

ORDER_FEATURE_COLUMNS = [
    "avg_order_value_log",
    "max_order_value_log",
    "order_count_period",
    "order_value_increase_pct",
    "credit_check_failure_rate",
    "high_utilisation_order_flag",
]


def extract_order_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract sales order activity features."""
    feats = pd.DataFrame(index=df.index)

    avg_val = df.get("avg_order_value", df.get("order_value", 50000.0)).fillna(0.0).astype(float)
    feats["avg_order_value_log"] = np.log1p(avg_val)

    max_val = df.get("max_order_value", avg_val).fillna(avg_val).astype(float)
    feats["max_order_value_log"] = np.log1p(max_val)

    feats["order_count_period"] = df.get("order_count_period", 1).fillna(1).astype(int)
    feats["order_value_increase_pct"] = df.get("order_value_increase_pct", 0.0).fillna(0.0).astype(float)
    feats["credit_check_failure_rate"] = df.get("credit_check_failure_rate", 0.0).fillna(0.0).astype(float)
    feats["high_utilisation_order_flag"] = df.get("high_utilisation_order_flag", 0).fillna(0).astype(int)

    return feats[ORDER_FEATURE_COLUMNS]
