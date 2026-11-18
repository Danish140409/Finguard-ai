"""
FinGuard AI — Reconciliation & Period Close Service (R2R)

Business logic for:
- Account reconciliation exceptions and items
- Intercompany transaction mismatch checks
- Accrual entries analysis and close activities
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from utils.db import (
    AccountReconciliation,
    ReconciliationItem,
    IntercompanyTransaction,
    AccrualEntry,
    CloseActivity,
    GLAccount,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class ReconciliationService:
    def __init__(self, db: Session):
        self.db = db

    def get_reconciliations(self, company_code: Optional[str] = None, status: Optional[str] = None) -> List[AccountReconciliation]:
        query = self.db.query(AccountReconciliation)
        if company_code:
            query = query.filter(AccountReconciliation.company_code == company_code)
        if status:
            query = query.filter(AccountReconciliation.reconciliation_status == status)
        return query.order_by(AccountReconciliation.reconciliation_date.desc()).all()

    def get_reconciliation(self, reconciliation_id: uuid.UUID) -> Optional[AccountReconciliation]:
        return self.db.query(AccountReconciliation).filter(AccountReconciliation.reconciliation_id == reconciliation_id).first()

    def get_intercompany_transactions(self, matching_status: Optional[str] = None) -> List[IntercompanyTransaction]:
        query = self.db.query(IntercompanyTransaction)
        if matching_status:
            query = query.filter(IntercompanyTransaction.matching_status == matching_status)
        return query.all()

    def get_close_activities(self, company_code: str, fiscal_period: int) -> List[CloseActivity]:
        return (
            self.db.query(CloseActivity)
            .filter(
                CloseActivity.company_code == company_code,
                CloseActivity.fiscal_period == fiscal_period,
            )
            .order_by(CloseActivity.due_date.asc())
            .all()
        )
