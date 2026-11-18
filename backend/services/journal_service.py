"""
FinGuard AI — Journal Service (R2R)

Business logic for:
- Retrieving and validating journal entries
- Checking debit-credit balance constraints
- Detecting duplicate and unusual manual journals
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from rules.r2r_rules import R2RRulesEngine
from utils.db import JournalHeader, JournalEntryLine, GLAccount
from utils.logger import get_logger

logger = get_logger(__name__)


class JournalService:
    def __init__(self, db: Session, rules_engine: Optional[R2RRulesEngine] = None):
        self.db = db
        self.rules_engine = rules_engine or R2RRulesEngine()

    def get_journal(self, journal_id: uuid.UUID) -> Optional[JournalHeader]:
        return self.db.query(JournalHeader).filter(JournalHeader.journal_id == journal_id).first()

    def validate_journal(self, journal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate journal lines for debit-credit equality and compliance rules."""
        lines = journal_data.get("lines", [])
        total_debit = sum(float(line.get("debit_amount", 0.0)) for line in lines)
        total_credit = sum(float(line.get("credit_amount", 0.0)) for line in lines)
        imbalance = round(abs(total_debit - total_credit), 2)

        eval_input = {
            "total_debit": total_debit,
            "total_credit": total_credit,
            "journal_type": journal_data.get("journal_type", "Standard"),
            "approval_status": journal_data.get("approval_status", "approved"),
            "posting_user": journal_data.get("posting_user", "system"),
            "posting_date": journal_data.get("posting_date"),
            "document_date": journal_data.get("document_date"),
        }
        rule_eval = self.rules_engine.evaluate_journal(eval_input)

        is_valid = imbalance <= 0.01 and len(lines) >= 2

        return {
            "is_valid": is_valid,
            "total_debit": round(total_debit, 2),
            "total_credit": round(total_credit, 2),
            "imbalance": imbalance,
            "rule_violations": rule_eval["rule_reasons"],
            "warnings": [
                f"Journal contains {len(lines)} lines."
            ] if len(lines) < 2 else [],
            "anomaly_risk": rule_eval["risk_category"],
            "anomaly_score": rule_eval["rule_score"],
        }
