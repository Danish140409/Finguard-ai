"""
FinGuard AI — Reports Router

Endpoints:
  GET  /reports/anomalies/export  — download flagged transactions as CSV
  POST /reports/pdf               — generate and download PDF risk report
  GET  /reports/audit-log         — paginated audit log (admin only)
"""

import csv
import io
from datetime import date
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session

from backend.deps import get_current_user, require_admin
from backend.schemas import AuditLogSchema, PaginatedResponse, ReportRequestSchema
from utils.db import AnomalyResult, AuditLog, Invoice, User, Vendor, get_db
from utils.logger import get_logger, log_audit_event

logger = get_logger(__name__)
router = APIRouter(prefix="/reports", tags=["Reports & Exports"])


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

@router.get(
    "/anomalies/export",
    summary="Export flagged transactions as CSV download",
)
def export_anomalies_csv(
    db:           Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    date_from:    Optional[date] = None,
    date_to:      Optional[date] = None,
    risk_levels:  str = "HIGH,CRITICAL",
    max_rows:     int = Query(5000, le=50000),
) -> StreamingResponse:
    """
    Export all flagged transactions matching the filters as a downloadable CSV.
    Default: HIGH and CRITICAL risk levels.
    """
    levels = [lvl.strip().upper() for lvl in risk_levels.split(",")]

    query = (
        db.query(AnomalyResult, Invoice, Vendor)
        .outerjoin(Invoice, AnomalyResult.transaction_id == Invoice.invoice_id)
        .outerjoin(Vendor, Invoice.vendor_id == Vendor.vendor_id)
        .filter(AnomalyResult.risk_level.in_(levels))
    )
    if date_from:
        query = query.filter(AnomalyResult.scored_at >= date_from)
    if date_to:
        query = query.filter(AnomalyResult.scored_at <= date_to)

    rows = query.order_by(AnomalyResult.risk_score.desc()).limit(max_rows).all()

    # Build CSV in-memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "score_id", "transaction_id", "transaction_type",
        "risk_score", "risk_level", "anomaly_types",
        "vendor_name", "amount", "invoice_date", "plain_english_reason",
        "model_version", "scored_at",
    ])
    for ar, inv, ven in rows:
        writer.writerow([
            str(ar.score_id),
            str(ar.transaction_id),
            ar.transaction_type,
            float(ar.risk_score),
            ar.risk_level,
            "|".join(ar.anomaly_types or []),
            ven.vendor_name if ven else "",
            float(inv.amount) if inv else "",
            str(inv.invoice_date) if inv else "",
            ar.plain_english_reason or "",
            ar.model_version or "",
            ar.scored_at.isoformat() if ar.scored_at else "",
        ])

    output.seek(0)
    log_audit_event("EXPORT_CSV", "report", None, current_user.username, db_session=db)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=finguard_anomalies.csv"},
    )


# ---------------------------------------------------------------------------
# PDF Report
# ---------------------------------------------------------------------------

@router.post(
    "/pdf",
    summary="Generate a PDF risk report and download it",
)
def generate_pdf_report(
    body:         ReportRequestSchema,
    db:           Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> Response:
    """
    Generate a formatted PDF risk report for the specified date range and risk levels.
    Uses ReportLab to build a professional multi-page document.
    """
    levels = [lvl.upper() for lvl in body.risk_levels]

    query = (
        db.query(AnomalyResult, Invoice, Vendor)
        .outerjoin(Invoice, AnomalyResult.transaction_id == Invoice.invoice_id)
        .outerjoin(Vendor, Invoice.vendor_id == Vendor.vendor_id)
        .filter(AnomalyResult.risk_level.in_(levels))
    )
    if body.date_from:
        query = query.filter(AnomalyResult.scored_at >= body.date_from)
    if body.date_to:
        query = query.filter(AnomalyResult.scored_at <= body.date_to)

    rows = query.order_by(AnomalyResult.risk_score.desc()).limit(body.max_rows).all()

    # Build data list for the PDF generator
    transactions = [
        {
            "score_id":             str(ar.score_id),
            "transaction_id":       str(ar.transaction_id),
            "risk_score":           float(ar.risk_score),
            "risk_level":           ar.risk_level,
            "anomaly_types":        ar.anomaly_types or [],
            "vendor_name":          ven.vendor_name if ven else "Unknown",
            "amount":               float(inv.amount) if inv else 0.0,
            "invoice_date":         str(inv.invoice_date) if inv else "N/A",
            "plain_english_reason": ar.plain_english_reason or "No explanation available",
            "model_version":        ar.model_version or "unknown",
            "scored_at":            ar.scored_at.isoformat() if ar.scored_at else "N/A",
        }
        for ar, inv, ven in rows
    ]

    from reports.pdf_generator import PDFReportGenerator
    generator = PDFReportGenerator()
    pdf_bytes = generator.generate_risk_report(
        transactions=transactions,
        date_range=(str(body.date_from or "All time"), str(body.date_to or "Present")),
    )

    log_audit_event(
        "GENERATE_PDF", "report", None, current_user.username,
        details={"rows": len(transactions)},
        db_session=db,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=finguard_risk_report.pdf"},
    )


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

@router.get(
    "/audit-log",
    response_model=PaginatedResponse[AuditLogSchema],
    summary="Paginated audit log (admin only)",
)
def get_audit_log(
    db:           Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_admin)],
    page:         int = Query(1, ge=1),
    size:         int = Query(100, ge=1, le=1000),
    action:       Optional[str] = None,
    performed_by: Optional[str] = None,
) -> PaginatedResponse[AuditLogSchema]:
    """Return the immutable audit log with optional filters. Admin only."""
    query = db.query(AuditLog)
    if action:
        query = query.filter(AuditLog.action.ilike(f"%{action}%"))
    if performed_by:
        query = query.filter(AuditLog.performed_by == performed_by)

    total  = query.count()
    offset = (page - 1) * size
    logs   = query.order_by(AuditLog.created_at.desc()).offset(offset).limit(size).all()

    return PaginatedResponse(
        items=[AuditLogSchema.model_validate(log) for log in logs],
        total=total,
        page=page,
        size=size,
        pages=max(1, -(-total // size)),
    )
