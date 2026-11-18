"""
FinGuard AI — R2R (Record-to-Report) Rules Engine

Deterministic accounting integrity controls that evaluate:
1. Debit vs Credit balancing (Total Debit = Total Credit)
2. Duplicate journal entry detection
3. High-value manual journal risk & approvals
4. Period-end unusual postings
5. Unusual GL account spikes
6. Unexpected journal reversals
7. Account reconciliation differences & aging
8. Intercompany transaction mismatches
9. Abnormal accrual entries
10. Segregation of duties & posting user activity
11. Backdated or post-close postings
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple


class R2RRulesEngine:
    """Configurable deterministic rules engine for Record-to-Report controls."""

    DEFAULT_THRESHOLDS = {
        "max_manual_journal_amount": 500000.0,   # Flag manual journals above $500k
        "debit_credit_tolerance": 0.01,          # Allowable rounding difference ($0.01)
        "duplicate_window_days": 5,              # Duplicate detection window (days)
        "period_end_days": 5,                    # Days before close considered period-end
        "reconciliation_tolerance_pct": 1.0,     # Max allowable reconciliation difference %
        "reconciliation_tolerance_amount": 1000.0, # Max allowable absolute difference
        "intercompany_tolerance": 0.01,          # IC rounding tolerance
        "max_accrual_deviation_pct": 50.0,       # Accrual vs historical avg deviation %
        "require_approval_for_manual": True,     # Manual journals require approval
        "max_backdated_days": 30,                # Backdated threshold (days)
    }

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

    def check_journal_balance(self, total_debit: float, total_credit: float) -> List[Tuple[str, str, str]]:
        """Verify fundamental accounting principle: Total Debit = Total Credit."""
        violations = []
        imbalance = abs(total_debit - total_credit)
        if imbalance > self.thresholds["debit_credit_tolerance"]:
            violations.append((
                "UNBALANCED_JOURNAL_ENTRY",
                "CRITICAL",
                f"Journal entry is out of balance by ${imbalance:,.2f} (Total Debit: ${total_debit:,.2f}, Total Credit: ${total_credit:,.2f})."
            ))
        return violations

    def check_duplicate_journal(
        self,
        document_id: str,
        amount: float,
        posting_date: date,
        existing_journals: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, str]]:
        """Detect potential duplicate journal postings within rolling window."""
        violations = []
        window = self.thresholds["duplicate_window_days"]
        for j in existing_journals:
            if j.get("document_id") != document_id and abs(float(j.get("amount", 0.0)) - amount) < 0.01:
                j_date = j.get("posting_date")
                if isinstance(j_date, str):
                    try:
                        j_date = date.fromisoformat(j_date)
                    except ValueError:
                        continue
                if j_date and abs((posting_date - j_date).days) <= window:
                    violations.append((
                        "POTENTIAL_DUPLICATE_JOURNAL",
                        "HIGH",
                        f"Journal matches existing document {j.get('document_id')} with exact amount ${amount:,.2f} within {window} days."
                    ))
                    break
        return violations

    def check_manual_journal(
        self,
        journal_type: str,
        amount: float,
        approval_status: str,
        posting_user: str = ""
    ) -> List[Tuple[str, str, str]]:
        """Flag high-value or unapproved manual journal entries."""
        violations = []
        is_manual = journal_type.lower() in ("manual", "adjustment")
        if is_manual:
            if amount >= self.thresholds["max_manual_journal_amount"]:
                violations.append((
                    "HIGH_VALUE_MANUAL_JOURNAL",
                    "HIGH",
                    f"Manual journal amount of ${amount:,.2f} exceeds standard audit threshold of ${self.thresholds['max_manual_journal_amount']:,.2f}."
                ))
            if self.thresholds["require_approval_for_manual"] and approval_status.lower() not in ("approved", "authorized"):
                violations.append((
                    "UNAPPROVED_MANUAL_JOURNAL",
                    "CRITICAL",
                    f"Manual journal entry posted without mandatory managerial approval (status: '{approval_status}', user: '{posting_user}')."
                ))
        return violations

    def check_period_end_posting(
        self,
        posting_date: date,
        period_end_date: date,
        amount: float,
        historical_avg_amount: float = 0.0
    ) -> List[Tuple[str, str, str]]:
        """Flag unusually large or rushed postings at fiscal period cut-off."""
        violations = []
        days_to_close = (period_end_date - posting_date).days
        if 0 <= days_to_close <= self.thresholds["period_end_days"]:
            if historical_avg_amount > 0 and amount > historical_avg_amount * 3.0:
                violations.append((
                    "UNUSUAL_PERIOD_END_SPIKE",
                    "HIGH",
                    f"Journal of ${amount:,.2f} posted {days_to_close} days before period close is 3x higher than historical average (${historical_avg_amount:,.2f})."
                ))
            elif amount >= self.thresholds["max_manual_journal_amount"]:
                violations.append((
                    "HIGH_VALUE_PERIOD_END_POSTING",
                    "MEDIUM",
                    f"Large posting of ${amount:,.2f} recorded in final {days_to_close} days of reporting period."
                ))
        return violations

    def check_unusual_gl_activity(
        self,
        account_number: str,
        current_amount: float,
        historical_mean: float,
        historical_std: float
    ) -> List[Tuple[str, str, str]]:
        """Identify statistical anomalies in specific GL account activity."""
        violations = []
        if historical_std > 0:
            zscore = (current_amount - historical_mean) / historical_std
            if zscore >= 3.0:
                violations.append((
                    "GL_ACTIVITY_STATISTICAL_SPIKE",
                    "HIGH",
                    f"GL Account {account_number} recorded activity of ${current_amount:,.2f} ({zscore:.1f} standard deviations above historical mean of ${historical_mean:,.2f})."
                ))
        return violations

    def check_journal_reversal(
        self,
        is_reversal: bool,
        reversal_period: int,
        original_period: int,
        reversal_reason: str = ""
    ) -> List[Tuple[str, str, str]]:
        """Flag suspicious or cross-period accounting reversals."""
        violations = []
        if is_reversal:
            if reversal_period != original_period and abs(reversal_period - original_period) > 1:
                violations.append((
                    "CROSS_PERIOD_REVERSAL",
                    "HIGH",
                    f"Journal entry reversed in non-adjacent period (original: {original_period}, reversal: {reversal_period})."
                ))
            if not reversal_reason or len(reversal_reason.strip()) < 5:
                violations.append((
                    "REVERSAL_MISSING_BUSINESS_JUSTIFICATION",
                    "MEDIUM",
                    "Journal reversal lacks formal audit justification notes."
                ))
        return violations

    def check_reconciliation_difference(
        self,
        difference_amount: float,
        gl_balance: float,
        unresolved_age_days: int = 0
    ) -> List[Tuple[str, str, str]]:
        """Identify material or long-pending sub-ledger to GL reconciliation gaps."""
        violations = []
        abs_diff = abs(difference_amount)
        if abs_diff > self.thresholds["reconciliation_tolerance_amount"]:
            diff_pct = (abs_diff / abs(gl_balance) * 100.0) if gl_balance != 0 else 100.0
            if diff_pct >= self.thresholds["reconciliation_tolerance_pct"]:
                severity = "CRITICAL" if abs_diff > 100000.0 or unresolved_age_days > 60 else "HIGH"
                violations.append((
                    "MATERIAL_RECONCILIATION_EXCEPTION",
                    severity,
                    f"Unresolved reconciliation gap of ${abs_diff:,.2f} ({diff_pct:.2f}% of GL balance; age: {unresolved_age_days} days)."
                ))
        return violations

    def check_intercompany_mismatch(
        self,
        source_amount: float,
        dest_amount: float,
        source_currency: str,
        dest_currency: str
    ) -> List[Tuple[str, str, str]]:
        """Flag elimination and balance mismatches between corporate group entities."""
        violations = []
        if source_currency.upper() != dest_currency.upper():
            violations.append((
                "INTERCOMPANY_CURRENCY_MISMATCH",
                "MEDIUM",
                f"Intercompany entities recorded in conflicting currencies ({source_currency} vs {dest_currency})."
            ))
        diff = abs(source_amount - dest_amount)
        if diff > self.thresholds["intercompany_tolerance"]:
            violations.append((
                "INTERCOMPANY_BALANCE_MISMATCH",
                "HIGH",
                f"Intercompany imbalance of ${diff:,.2f} between source (${source_amount:,.2f}) and destination (${dest_amount:,.2f})."
            ))
        return violations

    def check_accrual_entry(
        self,
        accrual_amount: float,
        historical_avg: float = 0.0,
        approval_status: str = "approved"
    ) -> List[Tuple[str, str, str]]:
        """Evaluate unusual accrual estimates or missing authorization."""
        violations = []
        if historical_avg > 0:
            dev_pct = abs(accrual_amount - historical_avg) / historical_avg * 100.0
            if dev_pct >= self.thresholds["max_accrual_deviation_pct"]:
                violations.append((
                    "ABNORMAL_ACCRUAL_ESTIMATE",
                    "HIGH",
                    f"Accrual of ${accrual_amount:,.2f} deviates by {dev_pct:.1f}% from historical monthly average (${historical_avg:,.2f})."
                ))
        if approval_status.lower() not in ("approved", "authorized"):
            violations.append((
                "UNAPPROVED_ACCRUAL_POSTING",
                "HIGH",
                f"Period-end accrual has unapproved status '{approval_status}'."
            ))
        return violations

    def check_backdated_posting(
        self,
        posting_date: date,
        document_date: date,
        period_status: str = "open"
    ) -> List[Tuple[str, str, str]]:
        """Flag postings to closed periods or significant backdating."""
        violations = []
        if period_status.lower() == "closed":
            violations.append((
                "POSTING_TO_CLOSED_PERIOD",
                "CRITICAL",
                f"Transaction posted with date {posting_date} into an already closed fiscal period."
            ))
        date_lag = (posting_date - document_date).days
        if date_lag >= self.thresholds["max_backdated_days"]:
            violations.append((
                "SEVERE_BACKDATED_POSTING",
                "HIGH",
                f"Transaction document date is {date_lag} days prior to posting date (threshold: {self.thresholds['max_backdated_days']} days)."
            ))
        return violations

    def compute_rule_score(self, violations: List[Tuple[str, str, str]]) -> float:
        """Compute normalized 0-100 severity score from rule violations."""
        weights = {"LOW": 10.0, "MEDIUM": 25.0, "HIGH": 50.0, "CRITICAL": 85.0}
        score = 0.0
        for _, severity, _ in violations:
            score += weights.get(severity.upper(), 15.0)
        return min(100.0, round(score, 2))

    def evaluate_journal(self, journal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all R2R rules on a journal document."""
        violations = []
        total_debit = float(journal_data.get("total_debit", 0.0))
        total_credit = float(journal_data.get("total_credit", 0.0))
        amount = max(total_debit, total_credit)

        violations.extend(self.check_journal_balance(total_debit, total_credit))
        violations.extend(self.check_manual_journal(
            journal_data.get("journal_type", "Standard"),
            amount,
            journal_data.get("approval_status", "approved"),
            journal_data.get("posting_user", "")
        ))

        post_date = journal_data.get("posting_date")
        doc_date = journal_data.get("document_date", post_date)
        if isinstance(post_date, str):
            try:
                post_date = date.fromisoformat(post_date)
            except ValueError:
                post_date = date.today()
        if isinstance(doc_date, str):
            try:
                doc_date = date.fromisoformat(doc_date)
            except ValueError:
                doc_date = post_date

        if post_date and doc_date:
            violations.extend(self.check_backdated_posting(
                post_date,
                doc_date,
                journal_data.get("period_status", "open")
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
