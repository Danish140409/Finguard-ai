"""
FinGuard AI — R2R (Record-to-Report) Schemas

Pydantic v2 schemas for Company Codes, Fiscal Periods, GL Accounts,
Journal Headers, Lines, Account Reconciliations, Accrual Entries,
Intercompany Transactions, Close Activities, and Anomaly Scoring.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Company Code Schemas
# ---------------------------------------------------------------------------

class CompanyCodeBase(BaseModel):
    company_code: str = Field(..., max_length=10)
    company_name: str = Field(..., max_length=255)
    country: str = Field(default="US", max_length=100)
    base_currency: str = Field(default="USD", max_length=3)
    company_status: str = Field(default="Active", max_length=20)


class CompanyCodeCreate(CompanyCodeBase):
    pass


class CompanyCodeSchema(CompanyCodeBase):
    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Fiscal Period Schemas
# ---------------------------------------------------------------------------

class FiscalPeriodBase(BaseModel):
    company_code: str
    fiscal_year: int
    period_number: int = Field(..., ge=1, le=16)
    start_date: date
    end_date: date
    period_status: str = Field(default="open", description="open, closed, reopened")


class FiscalPeriodCreate(FiscalPeriodBase):
    pass


class FiscalPeriodSchema(FiscalPeriodBase):
    period_id: uuid.UUID
    closed_at: Optional[datetime] = None
    reopened_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# GL Account Master Schemas
# ---------------------------------------------------------------------------

class GLAccountBase(BaseModel):
    gl_account_number: str = Field(..., max_length=50)
    gl_account_name: str = Field(..., max_length=255)
    account_type: str = Field(..., description="Asset, Liability, Equity, Revenue, Expense")
    company_code: str = Field(default="1000", max_length=10)
    account_group: str = Field(default="General", max_length=100)
    account_status: str = Field(default="Active", max_length=20)


class GLAccountCreate(GLAccountBase):
    pass


class GLAccountSchema(GLAccountBase):
    gl_account_id: uuid.UUID

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Journal Entry Lines & Headers
# ---------------------------------------------------------------------------

class JournalLineBase(BaseModel):
    gl_account_id: uuid.UUID
    debit_amount: float = Field(default=0.0, ge=0)
    credit_amount: float = Field(default=0.0, ge=0)
    currency: str = Field(default="USD", max_length=3)
    line_description: Optional[str] = None
    cost_center: Optional[str] = None
    profit_center: Optional[str] = None


class JournalLineCreate(JournalLineBase):
    pass


class JournalLineSchema(JournalLineBase):
    line_id: uuid.UUID
    journal_id: uuid.UUID
    gl_account_number: Optional[str] = None
    gl_account_name: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class JournalHeaderBase(BaseModel):
    document_id: str = Field(..., max_length=50)
    company_code: str = Field(default="1000", max_length=10)
    posting_date: date
    document_date: date
    fiscal_period: int = Field(default=1, ge=1, le=16)
    journal_type: str = Field(default="Standard", description="Standard, Manual, Accrual, Adjustment, Reversal, Intercompany")
    currency: str = Field(default="USD", max_length=3)
    posting_user: str = Field(default="system", max_length=100)
    approval_status: str = Field(default="approved", description="approved, pending, rejected, unapproved")
    reference_id: Optional[str] = None
    document_status: str = Field(default="posted", description="posted, reversed, parked")


class JournalHeaderCreate(JournalHeaderBase):
    lines: List[JournalLineCreate] = []


class JournalHeaderSchema(JournalHeaderBase):
    journal_id: uuid.UUID
    total_debit: float = 0.0
    total_credit: float = 0.0
    is_balanced: bool = True
    line_count: int = 0
    lines: List[JournalLineSchema] = []
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class JournalValidationRequest(BaseModel):
    company_code: str
    posting_date: date
    document_date: date
    fiscal_period: int
    journal_type: str
    posting_user: str
    lines: List[JournalLineCreate]
    reference_id: Optional[str] = None


class JournalValidationResponse(BaseModel):
    is_valid: bool
    total_debit: float
    total_credit: float
    imbalance: float
    rule_violations: List[str]
    warnings: List[str]
    anomaly_risk: str = "Low"
    anomaly_score: float = 0.0


# ---------------------------------------------------------------------------
# Account Reconciliation & Items
# ---------------------------------------------------------------------------

class ReconciliationItemBase(BaseModel):
    reference_id: str = Field(..., max_length=100)
    item_amount: float
    item_date: date
    exception_reason: Optional[str] = None
    resolution_status: str = Field(default="open", description="open, investigated, resolved")


class ReconciliationItemCreate(ReconciliationItemBase):
    reconciliation_id: uuid.UUID


class ReconciliationItemSchema(ReconciliationItemBase):
    item_id: uuid.UUID
    reconciliation_id: uuid.UUID
    resolved_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class AccountReconciliationBase(BaseModel):
    company_code: str = Field(default="1000", max_length=10)
    gl_account_id: uuid.UUID
    reconciliation_date: date
    source_balance: float
    gl_balance: float
    difference_amount: float = 0.0
    reconciliation_status: str = Field(default="reconciled", description="reconciled, exception, under_review")
    reviewer_id: Optional[str] = None


class AccountReconciliationCreate(AccountReconciliationBase):
    pass


class AccountReconciliationSchema(AccountReconciliationBase):
    reconciliation_id: uuid.UUID
    gl_account_number: Optional[str] = None
    gl_account_name: Optional[str] = None
    unresolved_items_count: int = 0
    items: List[ReconciliationItemSchema] = []
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Accruals & Intercompany
# ---------------------------------------------------------------------------

class AccrualEntryBase(BaseModel):
    company_code: str = Field(default="1000", max_length=10)
    gl_account_id: uuid.UUID
    accrual_amount: float = Field(..., gt=0)
    accrual_period: int = Field(..., ge=1, le=16)
    reversal_date: Optional[date] = None
    approval_status: str = Field(default="approved", max_length=20)


class AccrualEntryCreate(AccrualEntryBase):
    journal_id: Optional[uuid.UUID] = None


class AccrualEntrySchema(AccrualEntryBase):
    accrual_id: uuid.UUID
    journal_id: Optional[uuid.UUID] = None
    gl_account_name: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class IntercompanyTransactionBase(BaseModel):
    source_company_code: str = Field(..., max_length=10)
    destination_company_code: str = Field(..., max_length=10)
    reference_id: str = Field(..., max_length=100)
    transaction_date: date
    source_amount: float
    destination_amount: float
    source_currency: str = Field(default="USD", max_length=3)
    destination_currency: str = Field(default="USD", max_length=3)
    matching_status: str = Field(default="matched", description="matched, mismatched, pending")


class IntercompanyTransactionCreate(IntercompanyTransactionBase):
    pass


class IntercompanyTransactionSchema(IntercompanyTransactionBase):
    transaction_id: uuid.UUID
    difference: float = 0.0

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Period-End Close Activities
# ---------------------------------------------------------------------------

class CloseActivityBase(BaseModel):
    company_code: str = Field(default="1000", max_length=10)
    fiscal_period: int
    activity_name: str = Field(..., max_length=255)
    assigned_user: str = Field(default="finance_user", max_length=100)
    activity_status: str = Field(default="pending", description="pending, in_progress, completed, overdue")
    due_date: date


class CloseActivityCreate(CloseActivityBase):
    pass


class CloseActivitySchema(CloseActivityBase):
    activity_id: uuid.UUID
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# R2R Anomaly Scoring & Responses
# ---------------------------------------------------------------------------

class R2RScoreRequest(BaseModel):
    journal_id: Optional[uuid.UUID] = None
    reconciliation_id: Optional[uuid.UUID] = None
    intercompany_id: Optional[uuid.UUID] = None
    include_shap: bool = True


class R2RScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    anomaly_id: uuid.UUID
    module: str = "R2R"
    entity_type: str = Field(..., description="journal, reconciliation, intercompany, accrual")
    entity_id: uuid.UUID
    rule_score: float
    model_score: float
    final_risk_score: float
    risk_category: str
    rule_reasons: List[str]
    shap_explanation: Dict[str, float] = {}
    explanation_text: str
    model_version: str = "1.0.0"
    detected_at: datetime
    investigation_status: str = "pending"
