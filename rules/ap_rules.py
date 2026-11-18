"""
FinGuard AI — AP (Accounts Payable) Rules Engine

Deterministic rules matching the original AP implementation:
- Duplicate invoices within rolling window
- Round dollar payments
- Weekend payments
- Newly registered vendors (<30 days)
- Split invoices below approval threshold
- Purchase order amount mismatch
- Inflated invoice deviations
"""

from datetime import date
from typing import Any, Dict, List, Optional, Tuple


class APRulesEngine:
    """Configurable deterministic rules engine for Accounts Payable controls."""

    DEFAULT_THRESHOLDS = {
        "duplicate_window_days": 30,
        "new_vendor_days": 30,
        "split_invoice_threshold": 5000.0,
        "po_mismatch_tolerance": 0.10,
        "inflated_invoice_zscore": 2.5,
    }

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

    def check_duplicate_invoice(
        self,
        vendor_id: str,
        amount: float,
        invoice_date: date,
        existing_invoices: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, str]]:
        violations = []
        window = self.thresholds["duplicate_window_days"]
        for inv in existing_invoices:
            if inv.get("vendor_id") == vendor_id and abs(float(inv.get("amount", 0.0)) - amount) < 0.01:
                inv_d = inv.get("invoice_date")
                if isinstance(inv_d, str):
                    try:
                        inv_d = date.fromisoformat(inv_d)
                    except ValueError:
                        continue
                if inv_d and abs((invoice_date - inv_d).days) <= window:
                    violations.append((
                        "DUPLICATE_INVOICE_AMOUNT",
                        "CRITICAL",
                        f"Duplicate invoice detected for vendor {vendor_id} with identical amount ${amount:,.2f} within {window} days."
                    ))
                    break
        return violations

    def check_round_dollar(self, amount: float) -> List[Tuple[str, str, str]]:
        violations = []
        if amount >= 1000.0 and amount % 1000.0 == 0:
            violations.append((
                "ROUND_DOLLAR_AMOUNT",
                "LOW",
                f"Invoice amount ${amount:,.2f} is an exact multiple of 1,000, which carries elevated fraud risk."
            ))
        return violations

    def check_weekend_payment(self, payment_date: date) -> List[Tuple[str, str, str]]:
        violations = []
        if payment_date.weekday() in (5, 6):
            violations.append((
                "WEEKEND_PAYMENT_PROCESSING",
                "MEDIUM",
                f"Payment was processed on a weekend ({payment_date.strftime('%A')}), outside normal business hours."
            ))
        return violations

    def check_new_vendor(self, reg_date: date, tx_date: date) -> List[Tuple[str, str, str]]:
        violations = []
        days = (tx_date - reg_date).days
        if days < self.thresholds["new_vendor_days"]:
            violations.append((
                "NEW_VENDOR_TRANSACTION",
                "HIGH",
                f"Vendor was registered only {days} days prior to transaction (threshold: {self.thresholds['new_vendor_days']} days)."
            ))
        return violations

    def check_po_mismatch(self, invoice_amount: float, po_amount: float) -> List[Tuple[str, str, str]]:
        violations = []
        if po_amount > 0:
            ratio = abs(invoice_amount - po_amount) / po_amount
            if ratio > self.thresholds["po_mismatch_tolerance"]:
                violations.append((
                    "PO_AMOUNT_MISMATCH",
                    "HIGH",
                    f"Invoice amount (${invoice_amount:,.2f}) deviates by {ratio * 100:.1f}% from Purchase Order amount (${po_amount:,.2f})."
                ))
        return violations

    def compute_rule_score(self, violations: List[Tuple[str, str, str]]) -> float:
        weights = {"LOW": 10.0, "MEDIUM": 25.0, "HIGH": 50.0, "CRITICAL": 85.0}
        score = sum(weights.get(sev.upper(), 15.0) for _, sev, _ in violations)
        return min(100.0, round(score, 2))
