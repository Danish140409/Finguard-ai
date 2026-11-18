"""
FinGuard AI — Transactions & Vendors Router

Endpoints:
  GET  /transactions                      — paginated transaction list with risk scores
  GET  /transactions/{id}                 — single transaction detail
  POST /transactions/{id}/action          — analyst approve/escalate/flag_audit
  GET  /vendors                           — paginated vendor list with risk summaries
  GET  /vendors/{vendor_id}/risk          — vendor risk history
"""

import uuid
from datetime import date
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.deps import get_current_user
from backend.schemas import (
    AnomalyResultSchema,
    InvoiceSchema,
    PaginatedResponse,
    TransactionActionRequest,
    TransactionWithScore,
    VendorRiskSchema,
    VendorSchema,
)
from utils.db import (
    AnomalyResult,
    AuditLog,
    Invoice,
    Payment,
    User,
    Vendor,
    get_db,
)
from utils.logger import get_logger, log_audit_event

logger = get_logger(__name__)
router = APIRouter(tags=["Transactions & Vendors"])


# ---------------------------------------------------------------------------
# Helper: build a TransactionWithScore from Invoice + AnomalyResult
# ---------------------------------------------------------------------------

def _build_txn_with_score(inv: Invoice, ar: Optional[AnomalyResult]) -> TransactionWithScore:
    return TransactionWithScore(
        transaction_id   = inv.invoice_id,
        transaction_type = "invoice",
        vendor_id        = inv.vendor_id,
        vendor_name      = inv.vendor.vendor_name if inv.vendor else None,
        amount           = float(inv.amount),
        transaction_date = inv.invoice_date,
        risk_score       = float(ar.risk_score) if ar else None,
        risk_level       = ar.risk_level if ar else "UNSCORED",
        anomaly_types    = ar.anomaly_types if ar else [],
        plain_english_reason = ar.plain_english_reason if ar else None,
    )


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

@router.get(
    "/transactions",
    response_model=PaginatedResponse[TransactionWithScore],
    summary="List all transactions with their risk scores",
)
def list_transactions(
    db:           Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    page:         int   = Query(1, ge=1),
    size:         int   = Query(50, ge=1, le=500),
    vendor_id:    Optional[str]  = None,
    date_from:    Optional[date] = None,
    date_to:      Optional[date] = None,
    risk_level:   Optional[str]  = None,
    min_amount:   Optional[float] = None,
    max_amount:   Optional[float] = None,
) -> PaginatedResponse[TransactionWithScore]:
    """
    Paginated list of invoices joined with their latest anomaly score.
    Supports filtering by vendor, date range, risk level, and amount range.
    """
    query = db.query(Invoice).outerjoin(
        AnomalyResult,
        (AnomalyResult.transaction_id == Invoice.invoice_id) &
        (AnomalyResult.transaction_type == "invoice"),
    )

    if vendor_id:
        try:
            vid = uuid.UUID(vendor_id)
            query = query.filter(Invoice.vendor_id == vid)
        except ValueError:
            pass

    if date_from:
        query = query.filter(Invoice.invoice_date >= date_from)
    if date_to:
        query = query.filter(Invoice.invoice_date <= date_to)
    if min_amount is not None:
        query = query.filter(Invoice.amount >= min_amount)
    if max_amount is not None:
        query = query.filter(Invoice.amount <= max_amount)
    if risk_level:
        query = query.filter(AnomalyResult.risk_level == risk_level.upper())

    total  = query.count()
    offset = (page - 1) * size
    invoices = query.order_by(Invoice.invoice_date.desc()).offset(offset).limit(size).all()

    # Fetch corresponding anomaly results in one query
    inv_ids = [str(inv.invoice_id) for inv in invoices]
    ar_map: dict[str, AnomalyResult] = {}
    if inv_ids:
        ars = db.query(AnomalyResult).filter(
            AnomalyResult.transaction_id.in_(
                [uuid.UUID(i) for i in inv_ids]
            ),
            AnomalyResult.transaction_type == "invoice",
        ).all()
        ar_map = {str(ar.transaction_id): ar for ar in ars}

    items = [
        _build_txn_with_score(inv, ar_map.get(str(inv.invoice_id)))
        for inv in invoices
    ]
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        size=size,
        pages=max(1, -(-total // size)),  # ceil division
    )


@router.get(
    "/transactions/{transaction_id}",
    response_model=dict,
    summary="Get a single transaction with full anomaly detail",
)
def get_transaction(
    transaction_id: uuid.UUID,
    db:             Annotated[Session, Depends(get_db)],
    current_user:   Annotated[User, Depends(get_current_user)],
) -> dict:
    """Return full transaction metadata + anomaly result + vendor info."""
    invoice = db.query(Invoice).filter(Invoice.invoice_id == transaction_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Transaction not found")

    ar = db.query(AnomalyResult).filter(
        AnomalyResult.transaction_id == transaction_id,
        AnomalyResult.transaction_type == "invoice",
    ).order_by(AnomalyResult.scored_at.desc()).first()

    vendor = db.query(Vendor).filter(Vendor.vendor_id == invoice.vendor_id).first()

    return {
        "invoice":        InvoiceSchema.model_validate(invoice).model_dump(),
        "anomaly_result": AnomalyResultSchema.model_validate(ar).model_dump() if ar else None,
        "vendor":         VendorSchema.model_validate(vendor).model_dump() if vendor else None,
    }


@router.post(
    "/transactions/{transaction_id}/action",
    summary="Analyst action: approve, escalate, or flag for audit",
)
def transaction_action(
    transaction_id: uuid.UUID,
    body:           TransactionActionRequest,
    db:             Annotated[Session, Depends(get_db)],
    current_user:   Annotated[User, Depends(get_current_user)],
) -> dict:
    """
    Record an analyst decision on a flagged transaction.
    Valid actions: approve | escalate | flag_audit
    All actions are written to the immutable audit_logs table.
    """
    invoice = db.query(Invoice).filter(Invoice.invoice_id == transaction_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Transaction not found")

    # Update invoice status accordingly
    action_status_map = {
        "approve":    "approved",
        "escalate":   "disputed",
        "flag_audit": "disputed",
    }
    if body.action in action_status_map:
        invoice.status = action_status_map[body.action]
        db.commit()

    log_audit_event(
        action        = body.action.upper(),
        entity_type   = "invoice",
        entity_id     = transaction_id,
        performed_by  = current_user.username,
        details       = {"comment": body.comment, "new_status": invoice.status},
        db_session    = db,
    )

    return {
        "message":        f"Action '{body.action}' recorded for transaction {transaction_id}",
        "new_status":     invoice.status,
        "performed_by":   current_user.username,
    }


# ---------------------------------------------------------------------------
# Vendors
# ---------------------------------------------------------------------------

@router.get(
    "/vendors",
    response_model=PaginatedResponse[VendorSchema],
    summary="Paginated vendor list",
)
def list_vendors(
    db:           Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    page:         int  = Query(1, ge=1),
    size:         int  = Query(50, ge=1, le=200),
    risk_tier:    Optional[str] = None,
    country:      Optional[str] = None,
    search:       Optional[str] = None,
) -> PaginatedResponse[VendorSchema]:
    """List vendors with optional filtering by risk tier, country, and name search."""
    query = db.query(Vendor)
    if risk_tier:
        query = query.filter(Vendor.risk_tier == risk_tier)
    if country:
        query = query.filter(Vendor.country == country)
    if search:
        query = query.filter(Vendor.vendor_name.ilike(f"%{search}%"))

    total   = query.count()
    offset  = (page - 1) * size
    vendors = query.order_by(Vendor.vendor_name).offset(offset).limit(size).all()

    return PaginatedResponse(
        items=[VendorSchema.model_validate(v) for v in vendors],
        total=total,
        page=page,
        size=size,
        pages=max(1, -(-total // size)),
    )


@router.get(
    "/vendors/{vendor_id}/risk",
    response_model=dict,
    summary="Vendor risk profile and transaction history",
)
def vendor_risk(
    vendor_id:    uuid.UUID,
    db:           Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Full vendor risk history — all scored invoices and payments for this vendor."""
    vendor = db.query(Vendor).filter(Vendor.vendor_id == vendor_id).first()
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")

    # Fetch all anomaly results for this vendor's invoices
    invoice_ids = [str(inv.invoice_id) for inv in vendor.invoices]
    payment_ids = [str(p.payment_id) for p in vendor.payments]

    all_ids = [uuid.UUID(i) for i in invoice_ids + payment_ids]
    ars     = db.query(AnomalyResult).filter(
        AnomalyResult.transaction_id.in_(all_ids)
    ).order_by(AnomalyResult.scored_at.desc()).limit(100).all()

    risk_scores = [float(ar.risk_score) for ar in ars]
    return {
        "vendor":          VendorSchema.model_validate(vendor).model_dump(),
        "total_scored":    len(ars),
        "avg_risk_score":  round(sum(risk_scores) / len(risk_scores), 2) if risk_scores else 0,
        "max_risk_score":  round(max(risk_scores), 2) if risk_scores else 0,
        "flagged_count":   sum(1 for ar in ars if ar.risk_level in ("HIGH", "CRITICAL")),
        "recent_scores":   [AnomalyResultSchema.model_validate(ar).model_dump() for ar in ars[:20]],
    }
