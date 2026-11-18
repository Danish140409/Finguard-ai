"""
FinGuard AI — Unified Pydantic v2 Schemas

Defines and re-exports all request/response models for:
  - Accounts Payable (AP)
  - Order-to-Cash (O2C) Credit Management
  - Record-to-Report (R2R)
  - Authentication, Auditing, and Platform Reports
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# Re-export sub-module schemas
from backend.schemas.ap import (
    VendorSchema,
    VendorRiskSchema,
    InvoiceSchema,
    PaymentSchema,
    AnomalyResultSchema,
)
from backend.schemas.o2c import (
    CustomerBase,
    CustomerCreate,
    CustomerSchema,
    CreditProfileBase,
    CreditProfileCreate,
    CreditProfileSchema,
    CreditLimitBase,
    CreditLimitCreate,
    CreditLimitSchema,
    ExposureSnapshotSchema,
    SalesOrderBase,
    SalesOrderCreate,
    SalesOrderSchema,
    CreditCheckRequest,
    CreditCheckResponse,
    ARInvoiceBase,
    ARInvoiceCreate,
    ARInvoiceSchema,
    AgingBucketSchema,
    ARAgingSummarySchema,
    CustomerPaymentBase,
    CustomerPaymentCreate,
    CustomerPaymentSchema,
    PaymentAllocationBase,
    PaymentAllocationCreate,
    PaymentAllocationSchema,
    CreditEventBase,
    CreditEventCreate,
    CreditEventSchema,
    PaymentBehaviourSchema,
    O2CScoreRequest,
    O2CScoreResponse,
)
from backend.schemas.r2r import (
    CompanyCodeBase,
    CompanyCodeCreate,
    CompanyCodeSchema,
    FiscalPeriodBase,
    FiscalPeriodCreate,
    FiscalPeriodSchema,
    GLAccountBase,
    GLAccountCreate,
    GLAccountSchema,
    JournalLineBase,
    JournalLineCreate,
    JournalLineSchema,
    JournalHeaderBase,
    JournalHeaderCreate,
    JournalHeaderSchema,
    JournalValidationRequest,
    JournalValidationResponse,
    ReconciliationItemBase,
    ReconciliationItemCreate,
    ReconciliationItemSchema,
    AccountReconciliationBase,
    AccountReconciliationCreate,
    AccountReconciliationSchema,
    AccrualEntryBase,
    AccrualEntryCreate,
    AccrualEntrySchema,
    IntercompanyTransactionBase,
    IntercompanyTransactionCreate,
    IntercompanyTransactionSchema,
    CloseActivityBase,
    CloseActivityCreate,
    CloseActivitySchema,
    R2RScoreRequest,
    R2RScoreResponse,
)

# ---------------------------------------------------------------------------
# Generic paginated response
# ---------------------------------------------------------------------------

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    size: int
    pages: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class LoginSchema(BaseModel):
    username: str
    password: str


class UserRegisterSchema(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=6, max_length=100)
    role: str = Field(default="analyst", pattern="^(analyst|controller|auditor|admin)$")


class TokenSchema(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

class UserSchema(BaseModel):
    user_id:    uuid.UUID
    username:   str
    email:      str
    role:       str
    is_active:  bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class UserCreateSchema(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    email:    str
    password: str = Field(..., min_length=8)
    role:     str = Field(default="analyst", pattern="^(analyst|admin)$")


# ---------------------------------------------------------------------------
# Unified Anomaly Result & Explorer Models
# ---------------------------------------------------------------------------

class UnifiedAnomalyResultSchema(BaseModel):
    anomaly_id:           uuid.UUID
    module:               str = Field(..., description="AP, O2C, R2R")
    entity_type:          str
    entity_id:            uuid.UUID
    rule_score:           float = 0.0
    model_score:          float = 0.0
    final_risk_score:     float
    risk_category:        str
    rule_reasons:         List[str] = []
    shap_explanation:     dict[str, float] = {}
    explanation_text:     Optional[str] = None
    model_version:        Optional[str] = "1.0.0"
    detected_at:          datetime
    investigation_status: str = "pending"
    reviewed_by:          Optional[str] = None
    resolution_notes:     Optional[str] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class TransactionWithScore(BaseModel):
    """Combined transaction + anomaly result for the transaction explorer."""
    transaction_id:   uuid.UUID
    transaction_type: str
    vendor_id:        Optional[uuid.UUID] = None
    vendor_name:      Optional[str] = None
    amount:           float
    transaction_date: date
    risk_score:       Optional[float] = None
    risk_level:       Optional[str] = "UNSCORED"
    anomaly_types:    List[str] = []
    plain_english_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Scoring Request (on-demand)
# ---------------------------------------------------------------------------

class TransactionScoreRequest(BaseModel):
    """Body for POST /predict — on-demand scoring of a single transaction."""
    vendor_id:                str
    amount:                   float = Field(..., gt=0)
    transaction_date:         date
    vendor_registration_date: date
    transaction_type:         str = Field(default="invoice", pattern="^(invoice|payment)$")
    invoice_date:             Optional[date] = None
    payment_date:             Optional[date] = None
    po_amount:                Optional[float] = None
    payment_method:           Optional[str] = None


class ScoreResponse(BaseModel):
    """Response from POST /predict."""
    model_config = ConfigDict(protected_namespaces=())

    transaction_id:       str
    risk_score:           float
    risk_level:           str
    anomaly_types:        List[str]
    plain_english_reason: str
    shap_values:          dict[str, float]
    if_score:             float
    lof_score:            float
    ocsvm_score:          float
    xgb_score:            float
    model_version:        str


# ---------------------------------------------------------------------------
# Dashboard Stats
# ---------------------------------------------------------------------------

class DashboardStatsSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    total_transactions:       int
    flagged_count:            int
    critical_count:           int
    high_count:               int
    medium_count:             int
    low_count:                int
    estimated_fraud_exposure: float
    model_precision:          Optional[float] = None
    model_version:            Optional[str] = None
    last_scored_at:           Optional[datetime] = None
    # Unified Breakdown
    ap_anomaly_count:         Optional[int] = 0
    o2c_anomaly_count:        Optional[int] = 0
    r2r_anomaly_count:        Optional[int] = 0
    pending_investigations:   Optional[int] = 0
    resolved_exceptions:      Optional[int] = 0


# ---------------------------------------------------------------------------
# Analyst Actions
# ---------------------------------------------------------------------------

class TransactionActionRequest(BaseModel):
    action:  str = Field(..., pattern="^(approve|escalate|flag_audit)$")
    comment: Optional[str] = None


class AnomalyInvestigationUpdate(BaseModel):
    status: str = Field(..., pattern="^(pending|under_review|resolved)$")
    reviewed_by: str
    resolution_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Audit Logs
# ---------------------------------------------------------------------------

class AuditLogSchema(BaseModel):
    log_id:       uuid.UUID
    action:       str
    entity_type:  Optional[str] = None
    entity_id:    Optional[uuid.UUID] = None
    performed_by: str
    details:      dict[str, Any] = {}
    created_at:   datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

class ReportRequestSchema(BaseModel):
    date_from:   Optional[date] = None
    date_to:     Optional[date] = None
    risk_levels: List[str] = ["HIGH", "CRITICAL"]
    modules:     List[str] = ["AP", "O2C", "R2R"]
    max_rows:    int = Field(default=500, le=5000)
