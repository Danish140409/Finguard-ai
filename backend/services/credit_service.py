"""
FinGuard AI — Credit Service (O2C)

Business logic for:
- Retrieving customer credit profiles and active limits
- Calculating real-time credit exposure (open orders + open AR + delivery)
- Evaluating sales orders against credit rules
- Managing credit events and exposure snapshots
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import func
from sqlalchemy.orm import Session

from rules.o2c_rules import O2CRulesEngine
from utils.db import (
    Customer,
    CustomerCreditProfile,
    CreditLimit,
    CreditExposureSnapshot,
    SalesOrder,
    ARInvoice,
    CreditEvent,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class CreditService:
    def __init__(self, db: Session, rules_engine: Optional[O2CRulesEngine] = None):
        self.db = db
        self.rules_engine = rules_engine or O2CRulesEngine()

    def get_customer(self, customer_id: uuid.UUID) -> Optional[Customer]:
        return self.db.query(Customer).filter(Customer.customer_id == customer_id).first()

    def get_customer_credit_profile(self, customer_id: uuid.UUID) -> Optional[CustomerCreditProfile]:
        return self.db.query(CustomerCreditProfile).filter(CustomerCreditProfile.customer_id == customer_id).first()

    def get_current_credit_limit(self, customer_id: uuid.UUID) -> float:
        latest = (
            self.db.query(CreditLimit)
            .filter(CreditLimit.customer_id == customer_id, CreditLimit.approval_status == "approved")
            .order_by(CreditLimit.effective_date.desc(), CreditLimit.created_at.desc())
            .first()
        )
        return float(latest.approved_limit) if latest else 0.0

    def get_credit_exposure(self, customer_id: uuid.UUID) -> Dict[str, Any]:
        """Compute real-time credit exposure across open orders, open invoices, and limits."""
        # 1. Open sales orders exposure
        open_orders_val = (
            self.db.query(func.coalesce(func.sum(SalesOrder.order_value), 0.0))
            .filter(
                SalesOrder.customer_id == customer_id,
                SalesOrder.order_status.in_(["created", "processing"]),
            )
            .scalar()
        ) or 0.0

        # 2. Open Accounts Receivable invoices
        open_ar_val = (
            self.db.query(func.coalesce(func.sum(ARInvoice.outstanding_amount), 0.0))
            .filter(
                ARInvoice.customer_id == customer_id,
                ARInvoice.payment_status.in_(["unpaid", "partially_paid", "overdue"]),
            )
            .scalar()
        ) or 0.0

        limit = self.get_current_credit_limit(customer_id)
        total_exposure = float(open_orders_val) + float(open_ar_val)
        available = max(0.0, limit - total_exposure)
        utilisation = (total_exposure / limit * 100.0) if limit > 0 else (100.0 if total_exposure > 0 else 0.0)

        return {
            "customer_id": customer_id,
            "open_order_exposure": float(open_orders_val),
            "delivery_exposure": 0.0,
            "open_ar": float(open_ar_val),
            "other_exposure": 0.0,
            "total_exposure": round(total_exposure, 2),
            "credit_limit": round(limit, 2),
            "available_credit": round(available, 2),
            "credit_utilisation_pct": round(utilisation, 2),
            "snapshot_date": datetime.now(timezone.utc),
        }

    def evaluate_credit_check(self, customer_id: uuid.UUID, order_value: float) -> Dict[str, Any]:
        """Evaluate a sales order against credit rules and current exposure."""
        customer = self.get_customer(customer_id)
        if not customer:
            raise ValueError(f"Customer {customer_id} not found.")

        profile = self.get_customer_credit_profile(customer_id)
        risk_cat = profile.risk_category if profile else "Medium"
        exposure_data = self.get_credit_exposure(customer_id)

        limit = exposure_data["credit_limit"]
        curr_exposure = exposure_data["total_exposure"]
        projected = curr_exposure + order_value
        available = max(0.0, limit - curr_exposure)
        proj_utilisation = (projected / limit * 100.0) if limit > 0 else 100.0

        # Query overdue amount
        overdue_amt = (
            self.db.query(func.coalesce(func.sum(ARInvoice.outstanding_amount), 0.0))
            .filter(
                ARInvoice.customer_id == customer_id,
                ARInvoice.payment_status == "overdue",
            )
            .scalar()
        ) or 0.0

        eval_input = {
            "total_exposure": curr_exposure,
            "credit_limit": limit,
            "new_order_value": order_value,
            "risk_category": risk_cat,
            "total_overdue_amount": float(overdue_amt),
        }
        rule_eval = self.rules_engine.evaluate_customer(eval_input)

        # Determine decision
        if projected > limit or rule_eval["risk_category"] == "Critical":
            decision = "CREDIT_HOLD"
        elif proj_utilisation >= 90.0 or rule_eval["risk_category"] == "High":
            decision = "MANUAL_REVIEW"
        elif proj_utilisation >= 80.0:
            decision = "APPROVED_WITH_WARNING"
        else:
            decision = "APPROVED"

        return {
            "customer_id": customer_id,
            "customer_name": customer.customer_name,
            "credit_limit": limit,
            "current_exposure": curr_exposure,
            "new_order_value": order_value,
            "projected_exposure": round(projected, 2),
            "available_credit": round(available, 2),
            "projected_utilisation_pct": round(proj_utilisation, 2),
            "decision": decision,
            "reason_codes": rule_eval["violation_codes"],
            "rule_violations": rule_eval["rule_reasons"],
            "is_anomaly": decision in ("CREDIT_HOLD", "MANUAL_REVIEW"),
            "risk_score": rule_eval["rule_score"],
            "risk_category": rule_eval["risk_category"],
        }
