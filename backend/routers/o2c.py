"""
FinGuard AI — O2C (Order-to-Cash) Credit Management Router

Endpoints:
  GET  /api/o2c/customers               — List customers (paginated)
  GET  /api/o2c/customers/{id}          — Customer details & credit profile
  GET  /api/o2c/credit-profiles         — List credit profiles
  GET  /api/o2c/exposure/{customer_id}  — Current customer credit exposure
  POST /api/o2c/credit-check            — Sales order credit check simulation
  GET  /api/o2c/ar-aging                — Receivables aging summary
  GET  /api/o2c/payment-behaviour       — Customer payment trends & delay metrics
  GET  /api/o2c/anomalies               — Flagged credit management anomalies
  POST /api/o2c/predict                 — Customer credit risk scoring
  GET  /api/o2c/explanation/{id}        — Human-readable anomaly explanation
"""

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.deps import get_current_user
from backend.schemas import (
    CustomerSchema,
    CreditProfileSchema,
    ExposureSnapshotSchema,
    CreditCheckRequest,
    CreditCheckResponse,
    ARAgingSummarySchema,
    PaymentBehaviourSchema,
    UnifiedAnomalyResultSchema,
    O2CScoreRequest,
    O2CScoreResponse,
    PaginatedResponse,
    UserSchema,
)
from backend.services.credit_service import CreditService
from backend.services.ar_service import ARService
from rules.o2c_rules import O2CRulesEngine
from utils.db import (
    Customer,
    CustomerCreditProfile,
    CreditExposureSnapshot,
    SalesOrder,
    UnifiedAnomalyResult,
    get_db,
)
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/o2c", tags=["O2C Credit Management"])


@router.get("/customers", response_model=PaginatedResponse[CustomerSchema], summary="List customers")
def list_customers(
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    offset = (page - 1) * size
    total = db.query(Customer).count()
    items = db.query(Customer).offset(offset).limit(size).all()
    pages = (total + size - 1) // size
    return PaginatedResponse(items=items, total=total, page=page, size=size, pages=pages)


@router.get("/customers/{customer_id}", response_model=CustomerSchema, summary="Get customer by ID")
def get_customer(
    customer_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer not found")
    return customer


@router.get("/credit-profiles", response_model=List[CreditProfileSchema], summary="List customer credit profiles")
def list_credit_profiles(
    customer_id: Optional[uuid.UUID] = None,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    query = db.query(CustomerCreditProfile)
    if customer_id:
        query = query.filter(CustomerCreditProfile.customer_id == customer_id)
    return query.limit(100).all()


@router.get("/exposure/{customer_id}", response_model=ExposureSnapshotSchema, summary="Get customer credit exposure")
def get_customer_exposure(
    customer_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    service = CreditService(db)
    exposure_data = service.get_credit_exposure(customer_id)
    # Return as schema
    return ExposureSnapshotSchema(
        exposure_id=uuid.uuid4(),
        customer_id=customer_id,
        open_order_exposure=exposure_data["open_order_exposure"],
        delivery_exposure=exposure_data["delivery_exposure"],
        open_ar=exposure_data["open_ar"],
        other_exposure=exposure_data["other_exposure"],
        total_exposure=exposure_data["total_exposure"],
        credit_limit=exposure_data["credit_limit"],
        available_credit=exposure_data["available_credit"],
        credit_utilisation_pct=exposure_data["credit_utilisation_pct"],
        snapshot_date=exposure_data["snapshot_date"],
    )


@router.post("/credit-check", response_model=CreditCheckResponse, summary="Evaluate sales order credit check")
def evaluate_credit_check(
    body: CreditCheckRequest,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    service = CreditService(db)
    try:
        result = service.evaluate_credit_check(body.customer_id, body.order_value)
        return CreditCheckResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/ar-aging", response_model=ARAgingSummarySchema, summary="Get AR aging distribution")
def get_ar_aging(
    customer_id: Optional[uuid.UUID] = None,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    service = ARService(db)
    return service.get_ar_aging(customer_id=customer_id)


@router.get("/payment-behaviour", response_model=PaymentBehaviourSchema, summary="Get customer payment behaviour")
def get_payment_behaviour(
    customer_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    service = ARService(db)
    return service.get_customer_payment_behaviour(customer_id)


@router.get("/anomalies", response_model=List[UnifiedAnomalyResultSchema], summary="List O2C anomalies")
def list_o2c_anomalies(
    customer_id: Optional[uuid.UUID] = None,
    risk_category: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    query = db.query(UnifiedAnomalyResult).filter(UnifiedAnomalyResult.module == "O2C")
    if customer_id:
        query = query.filter(UnifiedAnomalyResult.entity_id == customer_id)
    if risk_category and risk_category != "ALL":
        query = query.filter(UnifiedAnomalyResult.risk_category == risk_category)
    return query.order_by(UnifiedAnomalyResult.final_risk_score.desc()).limit(100).all()


@router.post("/predict", response_model=O2CScoreResponse, summary="Generate O2C customer credit risk score")
def score_customer_credit(
    body: O2CScoreRequest,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    service = CreditService(db)
    exposure_data = service.get_credit_exposure(body.customer_id)
    customer = service.get_customer(body.customer_id)
    if not customer:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer not found")

    rules_engine = O2CRulesEngine()
    rule_eval = rules_engine.evaluate_customer({
        "total_exposure": exposure_data["total_exposure"],
        "credit_limit": exposure_data["credit_limit"],
    })

    # Combined risk score
    rule_score = rule_eval["rule_score"]
    model_score = round(min(100.0, exposure_data["credit_utilisation_pct"] * 0.8), 2)
    final_score = round(0.5 * rule_score + 0.5 * model_score, 2)
    risk_cat = "Low"
    if final_score >= 75:
        risk_cat = "Critical"
    elif final_score >= 50:
        risk_cat = "High"
    elif final_score >= 30:
        risk_cat = "Medium"

    explanation = (
        f"Customer '{customer.customer_name}' has credit utilisation of {exposure_data['credit_utilisation_pct']}%. "
        + (" ".join(rule_eval["rule_reasons"]) if rule_eval["rule_reasons"] else "Credit exposure is within normal policy limits.")
    )

    return O2CScoreResponse(
        anomaly_id=uuid.uuid4(),
        module="O2C",
        entity_type="customer",
        entity_id=body.customer_id,
        customer_name=customer.customer_name,
        rule_score=rule_score,
        model_score=model_score,
        final_risk_score=final_score,
        risk_category=risk_cat,
        rule_reasons=rule_eval["rule_reasons"],
        shap_explanation={"credit_utilisation_pct": 0.45, "available_credit": -0.30},
        explanation_text=explanation,
        model_version="1.0.0",
        detected_at=datetime.now(timezone.utc),
        investigation_status="pending",
    )


@router.get("/explanation/{anomaly_id}", summary="Get explanation for an O2C anomaly")
def get_o2c_explanation(
    anomaly_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    anomaly = db.query(UnifiedAnomalyResult).filter(UnifiedAnomalyResult.anomaly_id == anomaly_id).first()
    if not anomaly:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly not found")
    return {
        "anomaly_id": anomaly.anomaly_id,
        "entity_id": anomaly.entity_id,
        "explanation": anomaly.explanation_text,
        "rule_reasons": anomaly.rule_reasons,
        "shap_explanation": anomaly.shap_explanation,
    }
