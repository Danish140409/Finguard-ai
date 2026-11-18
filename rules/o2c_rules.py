"""
FinGuard AI — O2C (Order-to-Cash) Credit Management Rules Engine

Deterministic accounting and credit control rules that evaluate:
1. Credit utilisation thresholds
2. Unusual credit limit increases
3. Exposure breaches
4. Deteriorating overdue receivables & AR aging
5. Abnormal payment behaviour
6. Unauthorized credit block releases
7. High-value orders from high-risk customers
8. Unusual customer master data modifications
"""

from typing import Any, Dict, List, Tuple


class O2CRulesEngine:
    """Configurable deterministic rules engine for O2C Credit Management."""

    DEFAULT_THRESHOLDS = {
        "credit_utilisation_warning": 80.0,    # Warning threshold (%)
        "credit_utilisation_critical": 95.0,   # Critical threshold (%)
        "credit_limit_increase_pct": 50.0,     # Large limit increase flag (%)
        "max_overdue_days": 90,                # Critical overdue aging days
        "high_overdue_ratio": 0.30,            # 30% overdue invoice ratio
        "max_order_vs_available": 1.0,         # Order exceeding available credit
        "require_approval_limit_change": True, # Mandatory approval flag
        "partial_payment_threshold": 0.30,     # Partial payment frequency
    }

    def __init__(self, thresholds: Dict[str, Any] | None = None):
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

    def check_credit_utilisation(self, exposure: float, credit_limit: float) -> List[Tuple[str, str, str]]:
        """Evaluate current credit utilisation percentage."""
        violations = []
        if credit_limit <= 0:
            if exposure > 0:
                violations.append((
                    "ZERO_LIMIT_ACTIVE_EXPOSURE",
                    "CRITICAL",
                    f"Customer has zero credit limit but active exposure of ₹{exposure:,.2f}."
                ))
            return violations

        utilisation = (exposure / credit_limit) * 100.0
        if utilisation >= self.thresholds["credit_utilisation_critical"]:
            violations.append((
                "CRITICAL_CREDIT_UTILISATION",
                "CRITICAL",
                f"Credit utilisation is at {utilisation:.1f}% (critical threshold: {self.thresholds['credit_utilisation_critical']}%)."
            ))
        elif utilisation >= self.thresholds["credit_utilisation_warning"]:
            violations.append((
                "HIGH_CREDIT_UTILISATION",
                "HIGH",
                f"Credit utilisation reached {utilisation:.1f}% (warning threshold: {self.thresholds['credit_utilisation_warning']}%)."
            ))
        return violations

    def check_credit_limit_change(
        self,
        new_limit: float,
        previous_limit: float,
        approval_status: str,
        changed_by: str = ""
    ) -> List[Tuple[str, str, str]]:
        """Flag abnormal credit limit increases and unapproved modifications."""
        violations = []
        if previous_limit > 0:
            increase_pct = ((new_limit - previous_limit) / previous_limit) * 100.0
            if increase_pct >= self.thresholds["credit_limit_increase_pct"]:
                severity = "HIGH" if approval_status.lower() == "approved" else "CRITICAL"
                violations.append((
                    "UNUSUAL_LIMIT_INCREASE",
                    severity,
                    f"Approved credit limit increased by {increase_pct:.1f}% (from ₹{previous_limit:,.2f} to ₹{new_limit:,.2f})."
                ))

        if self.thresholds["require_approval_limit_change"] and approval_status.lower() in ("unapproved", "pending", "rejected"):
            violations.append((
                "UNAPPROVED_LIMIT_CHANGE",
                "CRITICAL",
                f"Credit limit change has non-approved status: '{approval_status}' (changed by: {changed_by or 'unknown'})."
            ))
        return violations

    def check_exposure_breach(
        self,
        total_exposure: float,
        credit_limit: float,
        new_order_value: float = 0.0
    ) -> List[Tuple[str, str, str]]:
        """Check if current exposure or prospective order exceeds approved limit."""
        violations = []
        projected = total_exposure + new_order_value
        available = credit_limit - total_exposure

        if credit_limit > 0 and projected > credit_limit:
            breach_amount = projected - credit_limit
            violations.append((
                "PROJECTED_EXPOSURE_BREACH",
                "CRITICAL",
                f"New order of ₹{new_order_value:,.2f} causes projected exposure (₹{projected:,.2f}) to exceed credit limit (₹{credit_limit:,.2f}) by ₹{breach_amount:,.2f}."
            ))
        elif new_order_value > 0 and available > 0 and new_order_value > available * self.thresholds["max_order_vs_available"]:
            violations.append((
                "ORDER_EXCEEDS_AVAILABLE_CREDIT",
                "HIGH",
                f"Order value of ₹{new_order_value:,.2f} exceeds available credit of ₹{available:,.2f}."
            ))
        return violations

    def check_overdue_receivables(
        self,
        overdue_invoices_count: int,
        total_invoices: int,
        total_overdue_amount: float,
        max_dpd: int
    ) -> List[Tuple[str, str, str]]:
        """Assess risk from overdue accounts receivable."""
        violations = []
        if total_invoices > 0:
            overdue_ratio = overdue_invoices_count / total_invoices
            if overdue_ratio >= self.thresholds["high_overdue_ratio"]:
                violations.append((
                    "HIGH_OVERDUE_INVOICE_RATIO",
                    "HIGH",
                    f"{overdue_ratio * 100:.1f}% of customer invoices are overdue ({overdue_invoices_count}/{total_invoices})."
                ))

        if max_dpd >= self.thresholds["max_overdue_days"]:
            violations.append((
                "SEVERE_AGING_RECEIVABLES",
                "CRITICAL",
                f"Customer has overdue invoices with {max_dpd} Days Past Due (threshold: {self.thresholds['max_overdue_days']} days)."
            ))
        elif max_dpd > 30:
            violations.append((
                "MODERATE_AGING_RECEIVABLES",
                "MEDIUM",
                f"Customer has invoices overdue by {max_dpd} days past due."
            ))
        return violations

    def check_payment_behaviour(
        self,
        avg_dpd: float,
        pct_late: float,
        partial_payment_count: int = 0,
        total_payments: int = 0
    ) -> List[Tuple[str, str, str]]:
        """Check for deteriorating payment trends or irregular partial payments."""
        violations = []
        if pct_late >= 50.0:
            violations.append((
                "REPEATED_LATE_PAYMENTS",
                "HIGH",
                f"Customer pays {pct_late:.1f}% of invoices past the due date (avg delay: {avg_dpd:.1f} days)."
            ))

        if total_payments > 5:
            partial_ratio = partial_payment_count / total_payments
            if partial_ratio >= self.thresholds["partial_payment_threshold"]:
                violations.append((
                    "IRREGULAR_PARTIAL_PAYMENTS",
                    "MEDIUM",
                    f"High frequency of partial payments ({partial_ratio * 100:.1f}% of total payments)."
                ))
        return violations

    def check_credit_block_release(
        self,
        block_released: bool,
        approval_status: str,
        approval_evidence: str = ""
    ) -> List[Tuple[str, str, str]]:
        """Flag sales orders released from credit hold without proper authorization."""
        violations = []
        if block_released:
            if approval_status.lower() not in ("approved", "authorized"):
                violations.append((
                    "UNAUTHORIZED_CREDIT_BLOCK_RELEASE",
                    "CRITICAL",
                    f"Credit block was released with unauthorized status '{approval_status}' and missing business justification."
                ))
            elif not approval_evidence or approval_evidence.strip() == "":
                violations.append((
                    "CREDIT_BLOCK_RELEASE_NO_JUSTIFICATION",
                    "HIGH",
                    "Credit block was released without supporting approval documentation."
                ))
        return violations

    def check_high_value_order_risky_customer(
        self,
        order_value: float,
        risk_category: str,
        overdue_amount: float,
        credit_utilisation_pct: float
    ) -> List[Tuple[str, str, str]]:
        """Flag high-value orders placed by customers with elevated credit risk."""
        violations = []
        if risk_category.upper() in ("HIGH", "CRITICAL"):
            if overdue_amount > 50000.0 or credit_utilisation_pct >= 85.0:
                violations.append((
                    "HIGH_RISK_CUSTOMER_NEW_ORDER",
                    "CRITICAL",
                    f"Customer categorized as {risk_category} with ₹{overdue_amount:,.2f} overdue placed new high-value order of ₹{order_value:,.2f}."
                ))
        return violations

    def check_customer_master_change(
        self,
        field_changed: str,
        old_value: Any,
        new_value: Any,
        changed_by: str = "",
        approval_status: str = "approved"
    ) -> List[Tuple[str, str, str]]:
        """Flag critical master data changes (payment terms, risk classification)."""
        violations = []
        critical_fields = ["payment_terms", "risk_category", "credit_limit", "credit_status"]
        if field_changed.lower() in critical_fields:
            if approval_status.lower() != "approved":
                violations.append((
                    "UNAPPROVED_MASTER_DATA_CHANGE",
                    "HIGH",
                    f"Critical field '{field_changed}' changed from '{old_value}' to '{new_value}' by {changed_by or 'user'} without formal approval."
                ))
        return violations

    def compute_rule_score(self, violations: List[Tuple[str, str, str]]) -> float:
        """Compute normalized 0-100 severity score from rule violations."""
        weights = {"LOW": 10.0, "MEDIUM": 25.0, "HIGH": 50.0, "CRITICAL": 85.0}
        score = 0.0
        for _, severity, _ in violations:
            score += weights.get(severity.upper(), 15.0)
        return min(100.0, round(score, 2))

    def evaluate_customer(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all applicable credit rules for a customer profile."""
        violations = []
        exposure = float(customer_data.get("total_exposure", 0.0))
        limit = float(customer_data.get("credit_limit", 0.0))
        new_order = float(customer_data.get("new_order_value", 0.0))

        violations.extend(self.check_credit_utilisation(exposure, limit))
        violations.extend(self.check_exposure_breach(exposure, limit, new_order))

        if "new_limit" in customer_data and "previous_limit" in customer_data:
            violations.extend(self.check_credit_limit_change(
                float(customer_data["new_limit"]),
                float(customer_data["previous_limit"]),
                customer_data.get("limit_approval_status", "approved"),
                customer_data.get("changed_by", "")
            ))

        violations.extend(self.check_overdue_receivables(
            int(customer_data.get("overdue_invoices_count", 0)),
            int(customer_data.get("total_invoices", 1)),
            float(customer_data.get("total_overdue_amount", 0.0)),
            int(customer_data.get("max_dpd", 0))
        ))

        violations.extend(self.check_payment_behaviour(
            float(customer_data.get("avg_dpd", 0.0)),
            float(customer_data.get("pct_late", 0.0)),
            int(customer_data.get("partial_payment_count", 0)),
            int(customer_data.get("total_payments", 0))
        ))

        if new_order > 0:
            violations.extend(self.check_high_value_order_risky_customer(
                new_order,
                customer_data.get("risk_category", "Low"),
                float(customer_data.get("total_overdue_amount", 0.0)),
                (exposure / limit * 100.0) if limit > 0 else 100.0
            ))

        score = self.compute_rule_score(violations)
        risk_category = "Low"
        if score >= 75.0:
            risk_category = "Critical"
        elif score >= 50.0:
            risk_category = "High"
        elif score >= 30.0:
            risk_category = "Medium"

        return {
            "rule_score": score,
            "risk_category": risk_category,
            "violations": violations,
            "rule_reasons": [desc for _, _, desc in violations],
            "violation_codes": [code for code, _, _ in violations],
        }
