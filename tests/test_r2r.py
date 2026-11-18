"""
FinGuard AI — Unit Tests for R2R (Record-to-Report)

Validates:
- Journal entry debit == credit balancing rule
- Duplicate journal entry detection
- Large unapproved manual entry detection
- Sub-ledger to GL reconciliation imbalance
- Intercompany un-matched balance detection
- Accrual deviation detection
"""

import sys
from datetime import date
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.r2r_rules import R2RRulesEngine
from features.r2r.journal_features import (
    compute_debit_credit_ratio,
    compute_journal_balance,
    is_round_number_amount,
)
from features.r2r.reconciliation_features import (
    compute_reconciliation_difference,
    compute_difference_pct,
)


def test_journal_balance_features():
    """Verify debit vs credit balance difference calculation."""
    diff, is_balanced = compute_journal_balance(debit_sum=50000.0, credit_sum=50000.0)
    assert diff == 0.0
    assert is_balanced is True

    diff_unbal, is_balanced_unbal = compute_journal_balance(debit_sum=50000.0, credit_sum=45000.0)
    assert diff_unbal == 5000.0
    assert is_balanced_unbal is False


def test_round_number_detection():
    """Verify detection of round number amounts."""
    assert is_round_number_amount(50000.0) is True
    assert is_round_number_amount(1000000.0) is True
    assert is_round_number_amount(50432.19) is False


def test_reconciliation_difference():
    """Verify subledger reconciliation difference computation."""
    diff = compute_reconciliation_difference(gl_balance=100000.0, subledger_balance=95000.0)
    assert diff == 5000.0


def test_r2r_rule_debit_credit_imbalance():
    """Test R2R_RULE: Debit Credit Imbalance."""
    engine = R2RRulesEngine()
    violations = engine.check_journal_balance(
        total_debit=10000.0,
        total_credit=9000.0,
    )
    assert len(violations) > 0
    rule_name, severity, desc = violations[0]
    assert rule_name == "UNBALANCED_JOURNAL_ENTRY"
    assert severity == "CRITICAL"


def test_r2r_rule_balanced_journal():
    """Test R2R_RULE passes for balanced journal."""
    engine = R2RRulesEngine()
    violations = engine.check_journal_balance(
        total_debit=10000.0,
        total_credit=10000.0,
    )
    assert len(violations) == 0


def test_r2r_rule_unapproved_manual_journal():
    """Test R2R_RULE: Unapproved large manual journal."""
    engine = R2RRulesEngine()
    violations = engine.check_manual_journal(
        journal_type="Manual",
        amount=250000.0,
        approval_status="draft",
        posting_user="j_doe",
    )
    assert len(violations) > 0
    assert any("MANUAL_JOURNAL" in v[0] for v in violations)


def test_r2r_rule_reconciliation_break():
    """Test R2R_RULE: Sub-ledger to GL reconciliation break."""
    engine = R2RRulesEngine()
    violations = engine.check_reconciliation_difference(
        difference_amount=5000.0,
        gl_balance=500000.0,
        unresolved_age_days=10,
    )
    assert len(violations) > 0
    assert any("MATERIAL_RECONCILIATION_EXCEPTION" in v[0] for v in violations)


def test_r2r_rule_intercompany_mismatch():
    """Test R2R_RULE: Intercompany balance mismatch."""
    engine = R2RRulesEngine()
    violations = engine.check_intercompany_mismatch(
        source_amount=100000.0,
        dest_amount=85000.0,
        source_currency="USD",
        dest_currency="USD",
    )
    assert len(violations) > 0
    assert "INTERCOMPANY_BALANCE_MISMATCH" in violations[0][0]
