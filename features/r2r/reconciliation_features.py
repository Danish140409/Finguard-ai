"""
FinGuard AI — Account Reconciliation Features (R2R)
"""

from typing import List
import numpy as np
import pandas as pd

RECON_FEATURE_COLUMNS = [
    "difference_amount_log",
    "difference_pct",
    "num_unresolved_items",
    "avg_item_age_days",
    "has_material_exception",
]


def compute_reconciliation_difference(gl_balance: float, subledger_balance: float) -> float:
    return round(abs(gl_balance - subledger_balance), 2)


def compute_difference_pct(diff_amount: float, gl_balance: float) -> float:
    if gl_balance == 0:
        return 100.0 if diff_amount != 0 else 0.0
    return round((abs(diff_amount) / abs(gl_balance)) * 100.0, 2)


def extract_reconciliation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract account reconciliation features."""
    feats = pd.DataFrame(index=df.index)

    diff = df.get("difference_amount", 0.0).fillna(0.0).astype(float)
    feats["difference_amount_log"] = np.log1p(np.abs(diff))

    gl_bal = df.get("gl_balance", 100000.0).fillna(100000.0).astype(float)
    diff_pct = np.where(gl_bal != 0, (np.abs(diff) / np.abs(gl_bal)) * 100.0, 0.0)
    feats["difference_pct"] = np.clip(diff_pct, 0.0, 100.0)

    feats["num_unresolved_items"] = df.get("unresolved_items_count", df.get("num_unresolved_items", 0)).fillna(0).astype(int)
    feats["avg_item_age_days"] = df.get("avg_item_age_days", 0.0).fillna(0.0).astype(float)
    feats["has_material_exception"] = (np.abs(diff) > 5000.0).astype(int)

    return feats[RECON_FEATURE_COLUMNS]
