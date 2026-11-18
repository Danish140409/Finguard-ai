"""
FinGuard AI — O2C (Order-to-Cash) Credit Management Schemas

Pydantic v2 schemas for Customer Master, Credit Profiles, Credit Limits,
Exposure Snapshots, Sales Orders, AR Invoices, Payments, and Anomaly Scoring.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Customer Master Schemas
# ---------------------------------------------------------------------------

class CustomerBase(BaseModel):
    customer_name: str = Field(..., min_length=2, max_length=255)
    customer_type: str = Field(default="Organisation", description="Individual or Organisation")
    customer_status: str = Field(default="Active", description="Active or Inactive")
    company_code: str = Field(default="1000", max_length=10)
    sales_org: str = Field(default="SO01", max_length=20)


class CustomerCreate(CustomerBase):
    pass


class CustomerSchema(CustomerBase):
    customer_id: uuid.UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Customer Credit Profile Schemas
# ---------------------------------------------------------------------------

class CreditProfileBase(BaseModel):
    credit_segment: Optional[str] = Field(default="CS_DEFAULT", max_length=50)
    credit_control_area: Optional[str] = Field(default="CCA01", max_length=50)
    risk_category: str = Field(default="Medium", description="Low, Medium, High, Critical")
    payment_terms: str = Field(default="Net 30", max_length=50)
    credit_status: str = Field(default="Active", description="Active, Review, Blocked")
    effective_from: date = Field(default_factory=date.today)


class CreditProfileCreate(CreditProfileBase):
    customer_id: uuid.UUID


class CreditProfileSchema(CreditProfileBase):
    profile_id: uuid.UUID
    customer_id: uuid.UUID

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Credit Limit Schemas
# ---------------------------------------------------------------------------

class CreditLimitBase(BaseModel):
    approved_limit: float = Field(..., ge=0)
    previous_limit: float = Field(default=0.0, ge=0)
    currency: str = Field(default="INR", max_length=3)
    change_reason: Optional[str] = None
    changed_by: str = Field(default="system", max_length=100)
    approval_status: str = Field(default="approved", description="approved, pending, rejected")
    effective_date: date = Field(default_factory=date.today)


class CreditLimitCreate(CreditLimitBase):
    customer_id: uuid.UUID


class CreditLimitSchema(CreditLimitBase):
    credit_limit_id: uuid.UUID
    customer_id: uuid.UUID

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Credit Exposure Snapshot Schemas
# ---------------------------------------------------------------------------

class ExposureSnapshotSchema(BaseModel):
    exposure_id: uuid.UUID
    customer_id: uuid.UUID
    open_order_exposure: float = 0.0
    delivery_exposure: float = 0.0
    open_ar: float = 0.0
    other_exposure: float = 0.0
    total_exposure: float = 0.0
    credit_limit: float = 0.0
    available_credit: float = 0.0
    credit_utilisation_pct: float = 0.0
    snapshot_date: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Sales Order Schemas & Credit Check
# ---------------------------------------------------------------------------

class SalesOrderBase(BaseModel):
    order_number: str = Field(..., max_length=50)
    order_date: date = Field(default_factory=date.today)
    order_value: float = Field(..., gt=0)
    currency: str = Field(default="INR", max_length=3)
    order_status: str = Field(default="created", description="created, processing, delivered, cancelled")
    credit_check_status: str = Field(default="approved", description="approved, approved_with_warning, credit_hold, manual_review")
    credit_block_status: str = Field(default="none", description="none, blocked, released")
    delivery_status: str = Field(default="pending", description="pending, partial, delivered")


class SalesOrderCreate(SalesOrderBase):
    customer_id: uuid.UUID


class SalesOrderSchema(SalesOrderBase):
    sales_order_id: uuid.UUID
    customer_id: uuid.UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CreditCheckRequest(BaseModel):
    customer_id: uuid.UUID
    order_value: float = Field(..., gt=0)
    currency: str = Field(default="INR", max_length=3)
    order_number: Optional[str] = None


class CreditCheckResponse(BaseModel):
    customer_id: uuid.UUID
    customer_name: str
    credit_limit: float
    current_exposure: float
    new_order_value: float
    projected_exposure: float
    available_credit: float
    projected_utilisation_pct: float
    decision: str = Field(..., description="APPROVED, APPROVED_WITH_WARNING, CREDIT_HOLD, MANUAL_REVIEW")
    reason_codes: List[str]
    rule_violations: List[str]
    is_anomaly: bool = False
    risk_score: float = 0.0
    risk_category: str = "Low"


# ---------------------------------------------------------------------------
# AR Invoices & Aging Schemas
# ---------------------------------------------------------------------------

class ARInvoiceBase(BaseModel):
    invoice_number: str = Field(..., max_length=50)
    invoice_date: date
    due_date: date
    invoice_amount: float = Field(..., gt=0)
    outstanding_amount: float = Field(..., ge=0)
    currency: str = Field(default="INR", max_length=3)
    payment_status: str = Field(default="unpaid", description="unpaid, partially_paid, paid, overdue")


class ARInvoiceCreate(ARInvoiceBase):
    customer_id: uuid.UUID
    sales_order_id: Optional[uuid.UUID] = None


class ARInvoiceSchema(ARInvoiceBase):
    invoice_id: uuid.UUID
    customer_id: uuid.UUID
    sales_order_id: Optional[uuid.UUID] = None
    days_past_due: int = 0
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AgingBucketSchema(BaseModel):
    bucket_name: str
    invoice_count: int
    total_amount: float
    percentage_of_total: float


class ARAgingSummarySchema(BaseModel):
    total_outstanding: float
    total_overdue: float
    customer_count: int
    buckets: List[AgingBucketSchema]


# ---------------------------------------------------------------------------
# Customer Payments & Allocations
# ---------------------------------------------------------------------------

class CustomerPaymentBase(BaseModel):
    payment_date: date
    payment_amount: float = Field(..., gt=0)
    currency: str = Field(default="INR", max_length=3)
    payment_method: str = Field(default="bank_transfer", max_length=50)
    payment_reference: Optional[str] = None


class CustomerPaymentCreate(CustomerPaymentBase):
    customer_id: uuid.UUID


class CustomerPaymentSchema(CustomerPaymentBase):
    payment_id: uuid.UUID
    customer_id: uuid.UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PaymentAllocationBase(BaseModel):
    allocated_amount: float = Field(..., gt=0)
    allocation_date: date = Field(default_factory=date.today)


class PaymentAllocationCreate(PaymentAllocationBase):
    payment_id: uuid.UUID
    invoice_id: uuid.UUID


class PaymentAllocationSchema(PaymentAllocationBase):
    allocation_id: uuid.UUID
    payment_id: uuid.UUID
    invoice_id: uuid.UUID

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Credit Events & Payment Behaviour Schemas
# ---------------------------------------------------------------------------

class CreditEventBase(BaseModel):
    event_type: str = Field(..., description="limit_increase, limit_decrease, block_placed, block_released, term_change, risk_category_change")
    previous_value: Optional[str] = None
    new_value: Optional[str] = None
    performed_by: str = Field(default="system", max_length=100)
    approval_status: str = Field(default="approved", max_length=20)
    event_timestamp: datetime = Field(default_factory=datetime.utcnow)


class CreditEventCreate(CreditEventBase):
    customer_id: uuid.UUID


class CreditEventSchema(CreditEventBase):
    event_id: uuid.UUID
    customer_id: uuid.UUID

    model_config = ConfigDict(from_attributes=True)


class PaymentBehaviourSchema(BaseModel):
    customer_id: uuid.UUID
    customer_name: str
    total_invoices: int
    paid_invoices: int
    overdue_invoices: int
    total_billed: float
    total_paid: float
    outstanding_balance: float
    overdue_balance: float
    avg_days_past_due: float
    max_days_past_due: int
    pct_paid_late: float
    avg_payment_delay_days: float
    partial_payment_count: int
    payment_trend: str = Field(default="Stable", description="Improving, Stable, Deteriorating")
    risk_indicator: str = Field(default="Low", description="Low, Medium, High, Critical")


# ---------------------------------------------------------------------------
# O2C Anomaly Scoring & Response Schemas
# ---------------------------------------------------------------------------

class O2CScoreRequest(BaseModel):
    customer_id: uuid.UUID
    order_id: Optional[uuid.UUID] = None
    include_shap: bool = True


class O2CScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    anomaly_id: uuid.UUID
    module: str = "O2C"
    entity_type: str = "customer"
    entity_id: uuid.UUID
    customer_name: str
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
