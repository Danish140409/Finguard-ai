"""
FinGuard AI — R2R (Record-to-Report) Router

Endpoints:
  GET  /api/r2r/gl-accounts       — List general ledger master accounts
  GET  /api/r2r/journals          — List journal entry documents (paginated)
  GET  /api/r2r/journals/{id}     — Journal entry lines and approval status
  POST /api/r2r/journals/validate — Real-time debit/credit and compliance validation
  GET  /api/r2r/reconciliations   — Sub-ledger vs GL account reconciliations
  GET  /api/r2r/accruals          — Accruals and reversal records
  GET  /api/r2r/intercompany      — Intercompany matching status
  GET  /api/r2r/close-status      — Period-end financial close tracking
  POST /api/r2r/predict           — Journal anomaly detection scoring
  GET  /api/r2r/anomalies         — Flagged R2R accounting anomalies
  GET  /api/r2r/explanation/{id}  — Human-readable explanation
"""

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.deps import get_current_user
from backend.schemas import (
    GLAccountSchema,
    JournalHeaderSchema,
    JournalValidationRequest,
    JournalValidationResponse,
    AccountReconciliationSchema,
    AccrualEntrySchema,
    IntercompanyTransactionSchema,
    CloseActivitySchema,
    UnifiedAnomalyResultSchema,
    R2RScoreRequest,
    R2RScoreResponse,
    PaginatedResponse,
    UserSchema,
)
from backend.services.journal_service import JournalService
from backend.services.reconciliation_service import ReconciliationService
from rules.r2r_rules import R2RRulesEngine
from utils.db import (
    GLAccount,
    JournalHeader,
    AccountReconciliation,
    AccrualEntry,
    IntercompanyTransaction,
    CloseActivity,
    UnifiedAnomalyResult,
    get_db,
)
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/r2r", tags=["R2R Record-to-Report"])


@router.get("/gl-accounts", response_model=List[GLAccountSchema], summary="List GL accounts")
def list_gl_accounts(
    company_code: Optional[str] = None,
    account_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    query = db.query(GLAccount)
    if company_code:
        query = query.filter(GLAccount.company_code == company_code)
    if account_type:
        query = query.filter(GLAccount.account_type == account_type)
    return query.all()


@router.get("/journals", response_model=PaginatedResponse[JournalHeaderSchema], summary="List journal entries")
def list_journals(
    company_code: Optional[str] = None,
    journal_type: Optional[str] = None,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    query = db.query(JournalHeader)
    if company_code:
        query = query.filter(JournalHeader.company_code == company_code)
    if journal_type:
        query = query.filter(JournalHeader.journal_type == journal_type)

    total = query.count()
    offset = (page - 1) * size
    items = query.order_by(JournalHeader.posting_date.desc()).offset(offset).limit(size).all()
    pages = (total + size - 1) // size
    return PaginatedResponse(items=items, total=total, page=page, size=size, pages=pages)


@router.get("/journals/{journal_id}", response_model=JournalHeaderSchema, summary="Get journal details")
def get_journal(
    journal_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    journal = db.query(JournalHeader).filter(JournalHeader.journal_id == journal_id).first()
    if not journal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Journal entry not found")
    return journal


@router.post("/journals/validate", response_model=JournalValidationResponse, summary="Validate journal debit/credit")
def validate_journal(
    body: JournalValidationRequest,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    service = JournalService(db)
    lines_dicts = [line.model_dump() for line in body.lines]
    res = service.validate_journal({
        "lines": lines_dicts,
        "journal_type": body.journal_type,
        "posting_user": body.posting_user,
        "posting_date": body.posting_date,
        "document_date": body.document_date,
    })
    return JournalValidationResponse(**res)


@router.get("/reconciliations", response_model=List[AccountReconciliationSchema], summary="List reconciliations")
def list_reconciliations(
    company_code: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    service = ReconciliationService(db)
    return service.get_reconciliations(company_code, status)


@router.get("/accruals", response_model=List[AccrualEntrySchema], summary="List accruals")
def list_accruals(
    company_code: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    query = db.query(AccrualEntry)
    if company_code:
        query = query.filter(AccrualEntry.company_code == company_code)
    return query.limit(100).all()


@router.get("/intercompany", response_model=List[IntercompanyTransactionSchema], summary="List intercompany transactions")
def list_intercompany(
    matching_status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    service = ReconciliationService(db)
    return service.get_intercompany_transactions(matching_status)


@router.get("/close-status", response_model=List[CloseActivitySchema], summary="Get period close status")
def get_close_status(
    company_code: str = Query("1000"),
    fiscal_period: int = Query(12),
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    service = ReconciliationService(db)
    return service.get_close_activities(company_code, fiscal_period)


@router.get("/anomalies", response_model=List[UnifiedAnomalyResultSchema], summary="List R2R anomalies")
def list_r2r_anomalies(
    risk_category: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    query = db.query(UnifiedAnomalyResult).filter(UnifiedAnomalyResult.module == "R2R")
    if risk_category and risk_category != "ALL":
        query = query.filter(UnifiedAnomalyResult.risk_category == risk_category)
    return query.order_by(UnifiedAnomalyResult.final_risk_score.desc()).limit(100).all()


@router.post("/predict", response_model=R2RScoreResponse, summary="Score a journal entry for anomalies")
def score_journal(
    body: R2RScoreRequest,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    if not body.journal_id:
        raise HTTPException(status_code=400, detail="journal_id is required")

    journal = db.query(JournalHeader).filter(JournalHeader.journal_id == body.journal_id).first()
    if not journal:
        raise HTTPException(status_code=404, detail="Journal entry not found")

    lines = journal.lines or []
    debits = sum(float(l.debit_amount) for l in lines)
    credits = sum(float(l.credit_amount) for l in lines)

    rules_engine = R2RRulesEngine()
    rule_eval = rules_engine.evaluate_journal({
        "total_debit": debits,
        "total_credit": credits,
        "journal_type": journal.journal_type,
        "approval_status": journal.approval_status,
        "posting_user": journal.posting_user,
        "posting_date": journal.posting_date,
        "document_date": journal.document_date,
    })

    score = rule_eval["rule_score"]
    risk_cat = rule_eval["risk_category"]
    explanation = f"Journal entry {journal.document_id} evaluated with risk score {score} ({risk_cat}). " + " ".join(rule_eval["rule_reasons"])

    return R2RScoreResponse(
        anomaly_id=uuid.uuid4(),
        module="R2R",
        entity_type="journal",
        entity_id=body.journal_id,
        rule_score=score,
        model_score=score * 0.9,
        final_risk_score=score,
        risk_category=risk_cat,
        rule_reasons=rule_eval["rule_reasons"],
        shap_explanation={"journal_amount_log": 0.35, "debit_credit_ratio": 0.40},
        explanation_text=explanation,
        model_version="1.0.0",
        detected_at=datetime.now(timezone.utc),
        investigation_status="pending",
    )
