"""
FinGuard AI — Structured Logger
Provides JSON-formatted logging and audit event helpers.
"""

import logging
import json
import sys
import uuid
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            log_obj.update(record.extra)
        return json.dumps(log_obj, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger with JSON formatting.

    Usage:
        logger = get_logger(__name__)
        logger.info("Scoring complete", extra={"transaction_id": str(txn_id)})
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_audit_event(
    action: str,
    entity_type: str,
    entity_id: str | uuid.UUID | None,
    performed_by: str,
    details: dict[str, Any] | None = None,
    db_session=None,
) -> None:
    """
    Write a structured audit event to both the logger and (optionally) the
    audit_logs database table.

    Args:
        action:        Action identifier, e.g. 'SCORE', 'APPROVE', 'LOGIN'.
        entity_type:   Entity type, e.g. 'invoice', 'payment', 'user'.
        entity_id:     UUID of the entity being acted upon.
        performed_by:  Username or system identifier performing the action.
        details:       Arbitrary JSON-serialisable metadata.
        db_session:    SQLAlchemy Session. If provided, also writes to audit_logs.
    """
    logger = get_logger("finguard.audit")
    payload: dict[str, Any] = {
        "action": action,
        "entity_type": entity_type,
        "entity_id": str(entity_id) if entity_id else None,
        "performed_by": performed_by,
        "details": details or {},
    }
    logger.info("AUDIT EVENT", extra=payload)

    if db_session is not None:
        try:
            from utils.db import AuditLog  # local import to avoid circular deps
            log_entry = AuditLog(
                log_id=uuid.uuid4(),
                action=action,
                entity_type=entity_type,
                entity_id=uuid.UUID(str(entity_id)) if entity_id else None,
                performed_by=performed_by,
                details=details or {},
            )
            db_session.add(log_entry)
            db_session.commit()
        except Exception as exc:
            logger.error("Failed to write audit log to DB", extra={"error": str(exc)})
