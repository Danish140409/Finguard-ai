"""
FinGuard AI — AP (Accounts Payable) Schemas
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class VendorSchema(BaseModel):
    vendor_id: uuid.UUID
    vendor_name: str
    registration_date: date
    country: str
    risk_tier: str
    contact_email: Optional[str] = None
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class VendorRiskSchema(BaseModel):
    vendor_id: uuid.UUID
    vendor_name: str
    total_scored: int = 0
    flagged_count: int = 0
    avg_risk_score: float = 0.0
    max_risk_score: float = 0.0
    flagged_invoice_exposure: float = 0.0

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class InvoiceSchema(BaseModel):
    invoice_id: uuid.UUID
    vendor_id: Optional[uuid.UUID] = None
    amount: float
    currency: str
    invoice_date: date
    due_date: Optional[date] = None
    po_number: Optional[str] = None
    payment_terms_days: Optional[int] = 30
    description: Optional[str] = None
    status: str
    is_anomaly: bool = False
    anomaly_type: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class PaymentSchema(BaseModel):
    payment_id: uuid.UUID
    invoice_id: Optional[uuid.UUID] = None
    vendor_id: Optional[uuid.UUID] = None
    amount_paid: float
    payment_date: date
    payment_method: str
    is_manual_override: bool
    is_anomaly: bool = False
    anomaly_type: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class AnomalyResultSchema(BaseModel):
    score_id: uuid.UUID
    transaction_id: uuid.UUID
    transaction_type: str
    risk_score: float
    risk_level: str
    anomaly_types: List[str] = []
    shap_values: Dict[str, float] = {}
    plain_english_reason: Optional[str] = None
    if_score: Optional[float] = None
    lof_score: Optional[float] = None
    ocsvm_score: Optional[float] = None
    xgb_score: Optional[float] = None
    scored_at: datetime
    model_version: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
