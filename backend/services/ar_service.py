"""
FinGuard AI — Accounts Receivable Service (O2C)

Business logic for:
- Accounts Receivable aging analysis (Not Due, 1-30, 31-60, 61-90, Above 90)
- Customer payment behaviour metrics (avg delay, late payment %, regularity)
- Outstanding receivables tracking
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any, Dict, List, Optional
from sqlalchemy import func
from sqlalchemy.orm import Session

from features.o2c.ar_features import compute_aging_bucket, compute_days_past_due
from utils.db import ARInvoice, Customer, CustomerPayment
from utils.logger import get_logger

logger = get_logger(__name__)


class ARService:
    def __init__(self, db: Session):
        self.db = db

    def get_ar_aging(self, customer_id: Optional[uuid.UUID] = None) -> Dict[str, Any]:
        """Aggregate outstanding invoices into aging buckets."""
        today = date.today()
        query = self.db.query(ARInvoice).filter(
            ARInvoice.payment_status.in_(["unpaid", "partially_paid", "overdue"]),
            ARInvoice.outstanding_amount > 0,
        )
        if customer_id:
            query = query.filter(ARInvoice.customer_id == customer_id)

        invoices = query.all()
        buckets = {
            "Not Due": {"count": 0, "amount": 0.0},
            "1-30 Days": {"count": 0, "amount": 0.0},
            "31-60 Days": {"count": 0, "amount": 0.0},
            "61-90 Days": {"count": 0, "amount": 0.0},
            "Above 90 Days": {"count": 0, "amount": 0.0},
        }

        total_outstanding = 0.0
        total_overdue = 0.0

        for inv in invoices:
            dpd = compute_days_past_due(inv.due_date, today)
            bucket_name = compute_aging_bucket(dpd)
            amt = float(inv.outstanding_amount)
            total_outstanding += amt
            if dpd > 0:
                total_overdue += amt

            if bucket_name in buckets:
                buckets[bucket_name]["count"] += 1
                buckets[bucket_name]["amount"] += amt

        # Format bucket response
        bucket_list = []
        for name, data in buckets.items():
            pct = (data["amount"] / total_outstanding * 100.0) if total_outstanding > 0 else 0.0
            bucket_list.append({
                "bucket_name": name,
                "invoice_count": data["count"],
                "total_amount": round(data["amount"], 2),
                "percentage_of_total": round(pct, 2),
            })

        cust_count = (
            self.db.query(func.count(func.distinct(ARInvoice.customer_id)))
            .filter(ARInvoice.payment_status.in_(["unpaid", "partially_paid", "overdue"]))
            .scalar()
        ) or 0

        return {
            "total_outstanding": round(total_outstanding, 2),
            "total_overdue": round(total_overdue, 2),
            "customer_count": int(cust_count),
            "buckets": bucket_list,
        }

    def get_customer_payment_behaviour(self, customer_id: uuid.UUID) -> Dict[str, Any]:
        """Compute customer payment history, delay metrics, and trend classification."""
        customer = self.db.query(Customer).filter(Customer.customer_id == customer_id).first()
        cust_name = customer.customer_name if customer else "Unknown"

        invoices = self.db.query(ARInvoice).filter(ARInvoice.customer_id == customer_id).all()
        today = date.today()

        total_invoices = len(invoices)
        paid_invoices = sum(1 for inv in invoices if inv.payment_status == "paid")
        overdue_invoices = sum(1 for inv in invoices if compute_days_past_due(inv.due_date, today) > 0 and inv.payment_status != "paid")

        total_billed = sum(float(inv.invoice_amount) for inv in invoices)
        outstanding = sum(float(inv.outstanding_amount) for inv in invoices if inv.payment_status != "paid")
        overdue_amt = sum(float(inv.outstanding_amount) for inv in invoices if compute_days_past_due(inv.due_date, today) > 0 and inv.payment_status != "paid")

        dpds = [compute_days_past_due(inv.due_date, today) for inv in invoices if inv.payment_status != "paid"]
        avg_dpd = float(sum(dpds) / len(dpds)) if dpds else 0.0
        max_dpd = max(dpds) if dpds else 0

        pct_late = (overdue_invoices / total_invoices * 100.0) if total_invoices > 0 else 0.0

        # Query payments
        payments = self.db.query(CustomerPayment).filter(CustomerPayment.customer_id == customer_id).all()
        total_paid = sum(float(p.payment_amount) for p in payments)

        # Trend and risk indicator
        if pct_late >= 50.0 or max_dpd > 90:
            trend = "Deteriorating"
            risk = "Critical" if max_dpd > 90 else "High"
        elif pct_late >= 25.0 or max_dpd > 30:
            trend = "Deteriorating"
            risk = "Medium"
        else:
            trend = "Stable" if total_invoices > 0 else "Improving"
            risk = "Low"

        return {
            "customer_id": customer_id,
            "customer_name": cust_name,
            "total_invoices": total_invoices,
            "paid_invoices": paid_invoices,
            "overdue_invoices": overdue_invoices,
            "total_billed": round(total_billed, 2),
            "total_paid": round(total_paid, 2),
            "outstanding_balance": round(outstanding, 2),
            "overdue_balance": round(overdue_amt, 2),
            "avg_days_past_due": round(avg_dpd, 1),
            "max_days_past_due": int(max_dpd),
            "pct_paid_late": round(pct_late, 2),
            "avg_payment_delay_days": round(avg_dpd, 1),
            "partial_payment_count": 0,
            "payment_trend": trend,
            "risk_indicator": risk,
        }
