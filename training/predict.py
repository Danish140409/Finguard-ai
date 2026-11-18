"""
FinGuard AI — Batch Scoring Pipeline

Loads all unscored transactions from the database, runs the full
feature engineering + ensemble scoring pipeline, and writes results
to the anomaly_results table.

Handles 100K rows in batches of 1,000 for memory efficiency.

Usage:
    python training/predict.py
    python training/predict.py --batch-size 2000 --transaction-type invoice
"""

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import (
    load_production_models, compute_model_scores, ensemble_score,
    score_to_risk_level, FEATURE_COLUMNS,
)
from utils.feature_pipeline import extract_features
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 1_000


# ---------------------------------------------------------------------------
# Data loading from DB
# ---------------------------------------------------------------------------

def fetch_unscored_invoices(engine, limit: int = 0) -> pd.DataFrame:
    """Return invoices not yet present in anomaly_results."""
    query = """
        SELECT
            i.invoice_id        AS transaction_id,
            'invoice'           AS transaction_type,
            i.vendor_id,
            i.amount,
            i.invoice_date      AS transaction_date,
            i.due_date,
            i.payment_terms_days,
            i.po_id,
            v.registration_date AS vendor_registration_date,
            po.po_amount
        FROM invoices i
        JOIN vendors v ON i.vendor_id = v.vendor_id
        LEFT JOIN purchase_orders po ON i.po_id = po.po_id
        WHERE i.invoice_id NOT IN (
            SELECT transaction_id FROM anomaly_results
            WHERE transaction_type = 'invoice'
        )
        {limit_clause}
    """.format(limit_clause=f"LIMIT {limit}" if limit else "")
    return pd.read_sql(query, engine)


def fetch_unscored_payments(engine, limit: int = 0) -> pd.DataFrame:
    """Return payments not yet present in anomaly_results."""
    query = """
        SELECT
            p.payment_id        AS transaction_id,
            'payment'           AS transaction_type,
            p.vendor_id,
            p.amount_paid       AS amount,
            p.payment_date      AS transaction_date,
            i.invoice_date,
            i.due_date,
            COALESCE(po.po_amount, p.amount_paid) AS po_amount,
            v.registration_date AS vendor_registration_date
        FROM payments p
        JOIN vendors v  ON p.vendor_id = v.vendor_id
        LEFT JOIN invoices i ON p.invoice_id = i.invoice_id
        LEFT JOIN purchase_orders po ON i.po_id = po.po_id
        WHERE p.payment_id NOT IN (
            SELECT transaction_id FROM anomaly_results
            WHERE transaction_type = 'payment'
        )
        {limit_clause}
    """.format(limit_clause=f"LIMIT {limit}" if limit else "")
    return pd.read_sql(query, engine)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def chunked(df: pd.DataFrame, size: int) -> Generator[pd.DataFrame, None, None]:
    for start in range(0, len(df), size):
        yield df.iloc[start: start + size]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_batch(
    batch_df: pd.DataFrame,
    models: dict,
    scaler,
    metadata: dict,
) -> list[dict]:
    """Score a batch of transactions. Returns list of anomaly_result dicts."""
    features_df = extract_features(batch_df)
    X_raw       = features_df[FEATURE_COLUMNS].values.astype(np.float32)
    X_scaled    = scaler.transform(X_raw)

    component_scores = compute_model_scores(models, X_scaled)
    ens             = ensemble_score(component_scores)
    risk_scores_100 = ens * 100

    results = []
    for i, (_, row) in enumerate(batch_df.iterrows()):
        risk_score = float(risk_scores_100[i])
        risk_level = score_to_risk_level(risk_score)

        # Build SHAP values dict for this row (from XGBoost feature importances as proxy)
        # (Full SHAP computed on-demand via shap_engine for single transactions)
        feature_contribs = {
            feat: float(component_scores["xgboost"][i] * 
                        models["xgboost"].feature_importances_[j])
            for j, feat in enumerate(FEATURE_COLUMNS)
        }

        # Determine anomaly types from rule-based flags
        anomaly_types = []
        feat_row = features_df.iloc[i]
        if feat_row["duplicate_amount_flag"] > 0:
            anomaly_types.append("duplicate_invoice")
        if feat_row["round_dollar_flag"] > 0:
            anomaly_types.append("round_dollar")
        if feat_row["is_weekend_payment"] > 0 and feat_row["is_new_vendor"] > 0:
            anomaly_types.append("weekend_new_vendor")
        if feat_row["payment_frequency_spike"] > 0:
            anomaly_types.append("split_invoice")
        if feat_row["amount_zscore_by_vendor"] > 3.0:
            anomaly_types.append("inflated_invoice")
        if feat_row["is_new_vendor"] > 0 and risk_score > 65:
            anomaly_types.append("fictitious_vendor")
        if feat_row["po_match_ratio"] > 1.3:
            anomaly_types.append("po_mismatch")

        results.append({
            "score_id":            str(uuid.uuid4()),
            "transaction_id":      str(row["transaction_id"]),
            "transaction_type":    str(row.get("transaction_type", "invoice")),
            "risk_score":          round(risk_score, 2),
            "risk_level":          risk_level,
            "anomaly_types":       anomaly_types,
            "shap_values":         feature_contribs,
            "plain_english_reason": None,  # filled by explainability layer
            "if_score":            round(float(component_scores["isolation_forest"][i]), 4),
            "lof_score":           round(float(component_scores["lof"][i]), 4),
            "ocsvm_score":         round(float(component_scores["ocsvm"][i]), 4),
            "xgb_score":           round(float(component_scores["xgboost"][i]), 4),
            "scored_at":           datetime.now(timezone.utc).isoformat(),
            "model_version":       metadata["version"],
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    batch_size: int = DEFAULT_BATCH_SIZE,
    transaction_type: str = "all",
    limit: int = 0,
) -> dict:
    logger.info("Starting batch scoring", extra={"batch_size": batch_size})
    start_time = datetime.now(timezone.utc)

    models, scaler, metadata = load_production_models()
    logger.info("Models loaded", extra={"version": metadata["version"]})

    from utils.db import get_engine
    from sqlalchemy.orm import sessionmaker
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)

    all_results = []

    if transaction_type in ("invoice", "all"):
        invoices_df = fetch_unscored_invoices(engine, limit=limit)
        logger.info("Unscored invoices fetched", extra={"count": len(invoices_df)})
        for i, batch in enumerate(chunked(invoices_df, batch_size)):
            logger.info(f"Scoring invoice batch {i+1}", extra={"size": len(batch)})
            all_results.extend(score_batch(batch, models, scaler, metadata))

    if transaction_type in ("payment", "all"):
        payments_df = fetch_unscored_payments(engine, limit=limit)
        logger.info("Unscored payments fetched", extra={"count": len(payments_df)})
        for i, batch in enumerate(chunked(payments_df, batch_size)):
            logger.info(f"Scoring payment batch {i+1}", extra={"size": len(batch)})
            all_results.extend(score_batch(batch, models, scaler, metadata))

    # Write to DB in bulk
    if all_results:
        results_df = pd.DataFrame(all_results)
        # Convert JSON cols
        results_df["anomaly_types"] = results_df["anomaly_types"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        results_df.to_sql(
            "anomaly_results", engine,
            if_exists="append", index=False,
            method="multi", chunksize=500,
        )
        logger.info("Results written to anomaly_results table",
                    extra={"count": len(all_results)})

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    summary = {
        "total_scored": len(all_results),
        "elapsed_seconds": round(elapsed, 2),
        "transactions_per_second": round(len(all_results) / max(elapsed, 0.001)),
        "critical_count": sum(1 for r in all_results if r["risk_level"] == "CRITICAL"),
        "high_count":     sum(1 for r in all_results if r["risk_level"] == "HIGH"),
    }
    logger.info("Batch scoring complete", extra=summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinGuard AI batch scorer")
    parser.add_argument("--batch-size",        type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--transaction-type",  default="all", choices=["all", "invoice", "payment"])
    parser.add_argument("--limit",             type=int, default=0, help="Limit rows (0=all)")
    args = parser.parse_args()
    main(args.batch_size, args.transaction_type, args.limit)
