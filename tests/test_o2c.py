"""
FinGuard AI — Unit Tests for O2C Credit Management

Validates:
- Available credit and credit utilisation calculations
- Projected exposure breach detection
- Unusual credit limit increases
- Overdue AR aging and DPD calculations
- Deteriorating payment behaviour
- Full O2C rules engine evaluation
"""

import sys
from datetime import date, timedelta
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.o2c.credit_features import (
    compute_available_credit,
    compute_credit_limit_change_pct,
    compute_credit_utilisation,
)
from features.o2c.ar_features import (
    compute_aging_bucket,
    compute_days_past_due,
)
from rules.o2c_rules import O2CRulesEngine


def test_credit_utilisation_calculation():
    """Verify credit utilisation formula: (exposure / limit) * 100."""
    util = compute_credit_utilisation(exposure=750000.0, limit=1000000.0)
    assert util == 75.0

    util_zero = compute_credit_utilisation(exposure=50000.0, limit=0.0)
    assert util_zero == 100.0


def test_available_credit_calculation():
    """Verify available credit formula: max(0, limit - exposure)."""
    avail = compute_available_credit(limit=1000000.0, exposure=750000.0)
    assert avail == 250000.0

    avail_excess = compute_available_credit(limit=1000000.0, exposure=1200000.0)
    assert avail_excess == 0.0


def test_credit_limit_change_percentage():
    """Verify credit limit percentage change computation."""
    pct = compute_credit_limit_change_pct(new_limit=1500000.0, prev_limit=1000000.0)
    assert pct == 50.0


def test_days_past_due_calculation():
    """Verify DPD is calculated correctly relative to reference date."""
    ref_date = date(2026, 11, 15)
    due_date = date(2026, 10, 15)  # 31 days overdue
    future_date = date(2026, 11, 25) # Not yet due

    assert compute_days_past_due(due_date, ref_date) == 31
    assert compute_days_past_due(future_date, ref_date) == 0


def test_aging_bucket_classification():
    """Verify aging bucket assignments."""
    assert compute_aging_bucket(0) == "Not Due"
    assert compute_aging_bucket(15) == "1-30 Days"
    assert compute_aging_bucket(45) == "31-60 Days"
    assert compute_aging_bucket(75) == "61-90 Days"
    assert compute_aging_bucket(120) == "Above 90 Days"


def test_rules_engine_exposure_breach():
    """Verify rule engine flags prospective order breaching credit limit."""
    engine = O2CRulesEngine()
    violations = engine.check_exposure_breach(
        total_exposure=800000.0,
        credit_limit=1000000.0,
        new_order_value=300000.0,  # Projected = 1.1M (> 1.0M)
    )
    assert len(violations) >= 1
    code, severity, desc = violations[0]
    assert code == "PROJECTED_EXPOSURE_BREACH"
    assert severity == "CRITICAL"


def test_rules_engine_unusual_limit_increase():
    """Verify rule engine flags large unapproved credit limit increases."""
    engine = O2CRulesEngine()
    violations = engine.check_credit_limit_change(
        new_limit=2000000.0,
        previous_limit=1000000.0,  # 100% increase
        approval_status="pending",
    )
    codes = [v[0] for v in violations]
    assert "UNUSUAL_LIMIT_INCREASE" in codes
    assert "UNAPPROVED_LIMIT_CHANGE" in codes


def test_rules_engine_full_customer_evaluation():
    """Verify complete customer evaluation pipeline returns risk score and category."""
    engine = O2CRulesEngine()
    eval_res = engine.evaluate_customer({
        "total_exposure": 980000.0,
        "credit_limit": 1000000.0,
        "new_order_value": 150000.0,
        "overdue_invoices_count": 5,
        "total_invoices": 10,
        "total_overdue_amount": 350000.0,
        "max_dpd": 95,
        "pct_late": 60.0,
        "avg_dpd": 45.0,
    })
    assert eval_res["risk_category"] in ("High", "Critical")
    assert eval_res["rule_score"] >= 50.0
    assert len(eval_res["violations"]) > 0
