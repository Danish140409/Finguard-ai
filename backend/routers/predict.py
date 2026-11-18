"""
FinGuard AI — Predict Router

Endpoints:
  POST /predict           — on-demand score a single transaction
  POST /predict/batch     — trigger batch scoring (admin only, background task)
  GET  /predict/status    — status of the last batch run
  POST /retrain           — retrain models (admin only, background task)
  GET  /dashboard/stats   — KPI summary for the analyst dashboard
"""

import subprocess
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from backend.deps import get_current_user, get_model_ensemble, invalidate_model_cache, require_admin
from backend.schemas import (
    DashboardStatsSchema,
    ScoreResponse,
    TransactionScoreRequest,
)
from explainability.nlg_templater import generate_explanation
from explainability.shap_engine import SHAPEngine
from training.train import compute_model_scores, ensemble_score, score_to_risk_level
from utils.db import AnomalyResult, Invoice, User, get_db
from utils.feature_pipeline import FEATURE_COLUMNS, extract_features
from utils.logger import get_logger, log_audit_event

logger = get_logger(__name__)
router = APIRouter(tags=["Prediction & Scoring"])

# Batch run state (in-memory; use Redis/DB for production scale)
_batch_status: dict[str, Any] = {
    "status":      "idle",
    "started_at":  None,
    "completed_at": None,
    "result":      None,
    "error":       None,
}


# ---------------------------------------------------------------------------
# On-demand single transaction scoring
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    response_model=ScoreResponse,
    summary="Score a single transaction on-demand",
)
def predict_single(
    body:         TransactionScoreRequest,
    db:           Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> ScoreResponse:
    """
    Score a new transaction immediately.
    Returns risk_score [0–100], risk_level, anomaly types,
    SHAP values, and a plain-English explanation.
    """
    models, scaler, metadata = get_model_ensemble()

    # Build a 1-row DataFrame matching the feature pipeline's expected input
    tx_dict = {
        "vendor_id":               [str(body.vendor_id)],
        "amount":                  [body.amount],
        "transaction_date":        [pd.Timestamp(body.transaction_date)],
        "vendor_registration_date":[pd.Timestamp(body.vendor_registration_date)],
        "invoice_date":            [pd.Timestamp(body.invoice_date or body.transaction_date)],
        "payment_date":            [pd.Timestamp(body.payment_date or body.transaction_date)],
        "po_amount":               [body.po_amount or body.amount],
    }
    df       = pd.DataFrame(tx_dict)
    features = extract_features(df)
    X_raw    = features[FEATURE_COLUMNS].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    component_scores = compute_model_scores(models, X_scaled)
    ens_score        = float(ensemble_score(component_scores)[0])
    risk_score_100   = round(ens_score * 100, 2)
    risk_level       = score_to_risk_level(risk_score_100)

    # SHAP explanation
    shap_engine = SHAPEngine(models["xgboost"], FEATURE_COLUMNS)
    shap_result = shap_engine.explain_local(X_scaled)
    top_feats   = shap_engine.get_top_features(shap_result["shap_values"], n=5)

    tx_data_for_nlg = {
        "amount":                      body.amount,
        "transaction_date":            str(body.transaction_date),
        "risk_score":                  risk_score_100,
        "amount_zscore_by_vendor":     float(features["amount_zscore_by_vendor"].iloc[0]),
        "days_since_vendor_registration": float(features["days_since_vendor_registration"].iloc[0]),
        "invoice_to_payment_lag":      float(features["invoice_to_payment_lag"].iloc[0]),
        "po_match_ratio":              float(features["po_match_ratio"].iloc[0]),
        "amount_percentile_by_vendor": float(features["amount_percentile_by_vendor"].iloc[0]),
    }
    explanation = generate_explanation(top_feats, tx_data_for_nlg)

    # Determine anomaly types from feature flags
    feat_row     = features.iloc[0]
    anomaly_types: list[str] = []
    if feat_row["duplicate_amount_flag"]   > 0: anomaly_types.append("duplicate_invoice")
    if feat_row["round_dollar_flag"]       > 0: anomaly_types.append("round_dollar")
    if feat_row["is_weekend_payment"]      > 0 and feat_row["is_new_vendor"] > 0:
        anomaly_types.append("weekend_new_vendor")
    if feat_row["payment_frequency_spike"] > 0: anomaly_types.append("split_invoice")
    if feat_row["amount_zscore_by_vendor"] > 3: anomaly_types.append("inflated_invoice")
    if feat_row["is_new_vendor"]           > 0 and risk_score_100 > 65:
        anomaly_types.append("fictitious_vendor")
    if feat_row["po_match_ratio"]          > 1.3: anomaly_types.append("po_mismatch")

    score_id = str(uuid.uuid4())

    # Persist to anomaly_results
    ar = AnomalyResult(
        score_id            = uuid.UUID(score_id),
        transaction_id      = uuid.uuid4(),   # ephemeral — not linked to a DB row
        transaction_type    = body.transaction_type,
        risk_score          = risk_score_100,
        risk_level          = risk_level,
        anomaly_types       = anomaly_types,
        shap_values         = shap_result["shap_values"],
        plain_english_reason= explanation,
        if_score            = round(float(component_scores["isolation_forest"][0]), 4),
        lof_score           = round(float(component_scores["lof"][0]), 4),
        ocsvm_score         = round(float(component_scores["ocsvm"][0]), 4),
        xgb_score           = round(float(component_scores["xgboost"][0]), 4),
        model_version       = metadata["version"],
    )
    db.add(ar)
    db.commit()

    log_audit_event(
        action       = "SCORE",
        entity_type  = body.transaction_type,
        entity_id    = ar.transaction_id,
        performed_by = current_user.username,
        details      = {"risk_score": risk_score_100, "risk_level": risk_level},
        db_session   = db,
    )

    return ScoreResponse(
        transaction_id       = str(ar.transaction_id),
        risk_score           = risk_score_100,
        risk_level           = risk_level,
        anomaly_types        = anomaly_types,
        plain_english_reason = explanation,
        shap_values          = shap_result["shap_values"],
        if_score             = float(component_scores["isolation_forest"][0]),
        lof_score            = float(component_scores["lof"][0]),
        ocsvm_score          = float(component_scores["ocsvm"][0]),
        xgb_score            = float(component_scores["xgboost"][0]),
        model_version        = metadata["version"],
    )


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

def _run_batch_scoring() -> None:
    global _batch_status
    _batch_status.update({"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    try:
        import sys
        result = subprocess.run(
            [sys.executable, "training/predict.py"],
            capture_output=True, text=True, timeout=600,
        )
        _batch_status.update({
            "status":       "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result":       result.stdout[-2000:] if result.stdout else "Done",
            "error":        result.stderr[-1000:] if result.returncode != 0 else None,
        })
    except Exception as e:
        _batch_status.update({
            "status": "failed",
            "error":  str(e),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })


@router.post(
    "/predict/batch",
    summary="Trigger batch scoring of all unscored transactions (admin only)",
)
def batch_score(
    background_tasks: BackgroundTasks,
    current_user:     Annotated[User, Depends(require_admin)],
    db:               Annotated[Session, Depends(get_db)],
) -> dict:
    """Launch batch scoring as a background task. Poll /predict/status for progress."""
    if _batch_status["status"] == "running":
        raise HTTPException(status_code=409, detail="A batch scoring run is already in progress")

    background_tasks.add_task(_run_batch_scoring)
    log_audit_event("BATCH_SCORE_TRIGGERED", "system", None, current_user.username, db_session=db)
    return {"message": "Batch scoring started in background", "poll": "/api/v1/predict/status"}


@router.get("/predict/status", summary="Status of the last batch scoring run")
def batch_status(current_user: Annotated[User, Depends(get_current_user)]) -> dict:
    return _batch_status


# ---------------------------------------------------------------------------
# Model retraining
# ---------------------------------------------------------------------------

def _run_retrain() -> None:
    global _batch_status
    try:
        import sys
        subprocess.run(
            [sys.executable, "training/train.py"],
            capture_output=True, text=True, timeout=1200,
        )
        invalidate_model_cache()
    except Exception as e:
        logger.error("Retrain failed", extra={"error": str(e)})


@router.post(
    "/retrain",
    summary="Retrain all ML models (admin only)",
)
def retrain(
    background_tasks: BackgroundTasks,
    current_user:     Annotated[User, Depends(require_admin)],
    db:               Annotated[Session, Depends(get_db)],
) -> dict:
    """Kick off model retraining as a background task."""
    background_tasks.add_task(_run_retrain)
    log_audit_event("RETRAIN_TRIGGERED", "model", None, current_user.username, db_session=db)
    return {"message": "Model retraining started in background. This may take several minutes."}


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------

@router.get(
    "/dashboard/stats",
    response_model=DashboardStatsSchema,
    summary="KPI summary statistics for the analyst dashboard",
)
def dashboard_stats(
    db:           Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> DashboardStatsSchema:
    """Returns total transactions, flagged counts, estimated fraud exposure, and model info."""
    total_invoices = db.query(Invoice).count()
    total_payments = 0  # could add payments table count

    risk_counts = (
        db.query(AnomalyResult.risk_level, func.count().label("cnt"))
        .group_by(AnomalyResult.risk_level)
        .all()
    )
    counts = {row.risk_level: row.cnt for row in risk_counts}

    # Estimated exposure: sum of HIGH+CRITICAL invoice amounts
    exposure_result = db.execute(
        text("""
            SELECT COALESCE(SUM(i.amount), 0) AS exposure
            FROM anomaly_results ar
            JOIN invoices i ON ar.transaction_id = i.invoice_id
            WHERE ar.risk_level IN ('HIGH','CRITICAL')
            AND ar.transaction_type = 'invoice'
        """)
    ).fetchone()
    exposure = float(exposure_result[0]) if exposure_result else 0.0

    # Model metadata
    model_version = None
    model_precision = None
    last_scored = db.query(func.max(AnomalyResult.scored_at)).scalar()
    try:
        _, _, meta = get_model_ensemble()
        model_version   = meta.get("version")
        model_precision = meta.get("metrics", {}).get("precision")
    except Exception:
        pass

    flagged = counts.get("HIGH", 0) + counts.get("CRITICAL", 0)

    return DashboardStatsSchema(
        total_transactions      = total_invoices + total_payments,
        flagged_count           = flagged,
        critical_count          = counts.get("CRITICAL", 0),
        high_count              = counts.get("HIGH", 0),
        medium_count            = counts.get("MEDIUM", 0),
        low_count               = counts.get("LOW", 0),
        estimated_fraud_exposure= round(exposure, 2),
        model_precision         = model_precision,
        model_version           = model_version,
        last_scored_at          = last_scored,
    )
