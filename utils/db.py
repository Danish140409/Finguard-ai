"""
FinGuard AI — SQLAlchemy Database Engine, Session, and ORM Models.

Supports:
  - Accounts Payable (AP)
  - Order-to-Cash (O2C) Credit Management
  - Record-to-Report (R2R)
  - Unified Anomaly Results & Audit Trails
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Any, List, Optional

from sqlalchemy import (
    Boolean, Column, Date, DateTime, ForeignKey,
    Index, Integer, Numeric, String, Text, create_engine, JSON,
)
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker, relationship

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cross-Database UUID & JSON Types (Postgres & SQLite Compatible)
# ---------------------------------------------------------------------------

class GUID(TypeDecorator):
    """Platform-independent GUID/UUID type.
    Uses PostgreSQL's native UUID type if available, otherwise CHAR(36).
    """
    impl = CHAR(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


# ---------------------------------------------------------------------------
# Engine & Session
# ---------------------------------------------------------------------------

_engine = None
_SessionLocal = None


def get_engine():
    """Return the singleton SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is None:
        from backend.config import settings
        from pathlib import Path
        db_url = settings.DATABASE_URL
        
        # SQLite vs Postgres connection pooling
        if db_url.startswith("sqlite"):
            # Resolve relative sqlite path or fallback to project root finguard_local.db
            sqlite_path = db_url.replace("sqlite:///", "")
            if not Path(sqlite_path).exists():
                db_file = Path(__file__).parent.parent / "finguard_local.db"
                db_url = f"sqlite:///{db_file.as_posix()}"
            _engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False},
                echo=(settings.ENVIRONMENT == "development"),
            )
        else:
            try:
                test_engine = create_engine(
                    db_url,
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                    connect_args={"connect_timeout": 2},
                )
                with test_engine.connect() as conn:
                    pass
                _engine = test_engine
                logger.info("Connected to PostgreSQL successfully")
            except Exception as e:
                logger.warning(f"Could not connect to PostgreSQL ({e}), falling back to SQLite file.")
                db_file = Path(__file__).parent.parent / "finguard_local.db"
                _engine = create_engine(
                    f"sqlite:///{db_file.as_posix()}",
                    connect_args={"check_same_thread": False},
                )

        logger.info("Database engine ready")
    return _engine


def get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    SessionLocal = get_session_local()
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all_tables():
    """Create all tables defined below and seed initial users/data if empty."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("All tables created / verified")

    # Seed initial users and sample data
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        from backend.deps import hash_password
        
        # 1. Users
        if db.query(User).count() == 0:
            user_jane = User(
                user_id=uuid.uuid4(),
                username="analyst_jane",
                email="jane@finguard.ai",
                hashed_password=hash_password("secure_password_123"),
                role="analyst",
                is_active=True,
            )
            user_admin = User(
                user_id=uuid.uuid4(),
                username="admin",
                email="admin@finguard.ai",
                hashed_password=hash_password("admin_password_123"),
                role="admin",
                is_active=True,
            )
            user_controller = User(
                user_id=uuid.uuid4(),
                username="controller_danish",
                email="danish@finguard.ai",
                hashed_password=hash_password("controller_pass_123"),
                role="controller",
                is_active=True,
            )
            user_auditor = User(
                user_id=uuid.uuid4(),
                username="auditor_rachel",
                email="rachel@finguard.ai",
                hashed_password=hash_password("auditor_pass_123"),
                role="auditor",
                is_active=True,
            )
            db.add_all([user_jane, user_admin, user_controller, user_auditor])
            db.commit()
            logger.info("Default users seeded (analyst_jane, admin, controller_danish, auditor_rachel)")

        # 2. Vendors & Invoices (from data/ if available)
        from pathlib import Path
        import pandas as pd
        data_dir = Path(__file__).parent.parent / "data"

        if db.query(Vendor).count() == 0 and (data_dir / "vendors.csv").exists():
            v_df = pd.read_csv(data_dir / "vendors.csv").head(50)
            for _, r in v_df.iterrows():
                v = Vendor(
                    vendor_id=uuid.UUID(str(r["vendor_id"])),
                    vendor_name=str(r["vendor_name"]),
                    registration_date=pd.to_datetime(r["registration_date"]).date(),
                    country=str(r.get("country", "US")),
                    risk_tier=str(r.get("risk_tier", "standard")),
                    tax_id=str(r.get("tax_id", "")),
                    contact_email=str(r.get("contact_email", "")),
                    is_active=True,
                )
                db.add(v)
            db.commit()

        if db.query(Invoice).count() == 0 and (data_dir / "invoices.csv").exists():
            inv_df = pd.read_csv(data_dir / "invoices.csv").head(150)
            for _, r in inv_df.iterrows():
                inv = Invoice(
                    invoice_id=uuid.UUID(str(r["invoice_id"])),
                    vendor_id=uuid.UUID(str(r["vendor_id"])) if pd.notna(r.get("vendor_id")) else None,
                    amount=float(r["amount"]),
                    currency=str(r.get("currency", "USD")),
                    invoice_date=pd.to_datetime(r["invoice_date"]).date(),
                    due_date=pd.to_datetime(r["due_date"]).date() if pd.notna(r.get("due_date")) else None,
                    po_number=str(r.get("po_number", "")),
                    status=str(r.get("status", "pending")),
                    is_anomaly=bool(r.get("is_anomaly", False)),
                    anomaly_type=str(r.get("anomaly_type", "")),
                )
                db.add(inv)
            db.commit()

        # 3. O2C Customers (from data/o2c/ if available)
        o2c_dir = data_dir / "o2c"
        if db.query(Customer).count() == 0 and (o2c_dir / "customers.csv").exists():
            c_df = pd.read_csv(o2c_dir / "customers.csv").head(50)
            for _, r in c_df.iterrows():
                cid = uuid.UUID(str(r["customer_id"]))
                cust = Customer(
                    customer_id=cid,
                    customer_name=str(r["customer_name"]),
                    customer_type=str(r.get("customer_type", "Organisation")),
                    customer_status=str(r.get("customer_status", "Active")),
                    company_code=str(r.get("company_code", "1000")),
                    sales_org=str(r.get("sales_org", "SO01")),
                )
                db.add(cust)
                # Add default credit profile
                prof = CustomerCreditProfile(
                    profile_id=uuid.uuid4(),
                    customer_id=cid,
                    risk_category=str(r.get("risk_category", "Medium")),
                    credit_status="Active",
                )
                db.add(prof)
            db.commit()

        # 4. R2R GL Accounts & Journals (from data/r2r/ if available)
        r2r_dir = data_dir / "r2r"
        if db.query(CompanyCode).count() == 0 and (r2r_dir / "company_codes.csv").exists():
            comp_df = pd.read_csv(r2r_dir / "company_codes.csv")
            for _, r in comp_df.iterrows():
                db.add(CompanyCode(
                    company_code=str(r["company_code"]),
                    company_name=str(r["company_name"]),
                    country=str(r.get("country", "US")),
                    base_currency=str(r.get("base_currency", "USD")),
                    company_status=str(r.get("company_status", "Active")),
                ))
            db.commit()

        if db.query(GLAccount).count() == 0 and (r2r_dir / "gl_accounts.csv").exists():
            gl_df = pd.read_csv(r2r_dir / "gl_accounts.csv")
            for _, r in gl_df.iterrows():
                db.add(GLAccount(
                    gl_account_id=uuid.UUID(str(r["gl_account_id"])),
                    gl_account_number=str(r["gl_account_number"]),
                    gl_account_name=str(r["gl_account_name"]),
                    account_type=str(r["account_type"]),
                    company_code=str(r.get("company_code", "1000")),
                    account_group=str(r.get("account_group", "General")),
                ))
            db.commit()

        if db.query(JournalHeader).count() == 0 and (r2r_dir / "journal_headers.csv").exists():
            j_df = pd.read_csv(r2r_dir / "journal_headers.csv").head(100)
            for _, r in j_df.iterrows():
                db.add(JournalHeader(
                    journal_id=uuid.UUID(str(r["journal_id"])),
                    document_id=str(r["document_id"]),
                    company_code=str(r.get("company_code", "1000")),
                    posting_date=pd.to_datetime(r["posting_date"]).date(),
                    document_date=pd.to_datetime(r.get("document_date", r["posting_date"])).date(),
                    fiscal_period=int(r.get("fiscal_period", 1)),
                    journal_type=str(r.get("journal_type", "Standard")),
                    currency=str(r.get("currency", "USD")),
                    posting_user=str(r.get("posting_user", "system")),
                    approval_status=str(r.get("approval_status", "approved")),
                ))
            db.commit()

        logger.info("Database seeding complete.")
    except Exception as e:
        logger.warning(f"Error during seeding: {e}")
        db.rollback()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# ORM Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ===========================================================================
# 1. AUTH & AP ORM MODELS
# ===========================================================================

class User(Base):
    __tablename__ = "users"

    user_id         = Column(GUID, primary_key=True, default=uuid.uuid4)
    username        = Column(String(100), unique=True, nullable=False, index=True)
    email           = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role            = Column(String(20), nullable=False, default="analyst")
    is_active       = Column(Boolean, nullable=False, default=True)
    created_at      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Vendor(Base):
    __tablename__ = "vendors"

    vendor_id           = Column(GUID, primary_key=True, default=uuid.uuid4)
    vendor_name         = Column(String(255), nullable=False)
    registration_date   = Column(Date, nullable=False)
    country             = Column(String(100), default="US")
    risk_tier           = Column(String(20), default="standard")
    tax_id              = Column(String(50))
    bank_account_hash   = Column(String(255))
    contact_email       = Column(String(255))
    is_active           = Column(Boolean, default=True)
    created_at          = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    invoices            = relationship("Invoice", back_populates="vendor")
    payments            = relationship("Payment", back_populates="vendor")
    purchase_orders     = relationship("PurchaseOrder", back_populates="vendor")


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"

    po_id       = Column(GUID, primary_key=True, default=uuid.uuid4)
    vendor_id   = Column(GUID, ForeignKey("vendors.vendor_id"), index=True)
    po_number   = Column(String(100), unique=True, nullable=False)
    po_amount   = Column(Numeric(15, 2), nullable=False)
    po_date     = Column(Date, nullable=False)
    approved_by = Column(String(100))
    status      = Column(String(20), default="open")
    created_at  = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    vendor      = relationship("Vendor", back_populates="purchase_orders")
    invoices    = relationship("Invoice", back_populates="purchase_order")


class Invoice(Base):
    __tablename__ = "invoices"

    invoice_id          = Column(GUID, primary_key=True, default=uuid.uuid4)
    vendor_id           = Column(GUID, ForeignKey("vendors.vendor_id"), index=True)
    po_id               = Column(GUID, ForeignKey("purchase_orders.po_id"))
    amount              = Column(Numeric(15, 2), nullable=False)
    currency            = Column(String(3), default="USD")
    invoice_date        = Column(Date, nullable=False, index=True)
    due_date            = Column(Date)
    po_number           = Column(String(100))
    payment_terms_days  = Column(Numeric, default=30)
    description         = Column(Text)
    status              = Column(String(20), default="pending")
    is_anomaly          = Column(Boolean, default=False, index=True)
    anomaly_type        = Column(String(100))
    created_at          = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    vendor          = relationship("Vendor", back_populates="invoices")
    purchase_order  = relationship("PurchaseOrder", back_populates="invoices")
    payments        = relationship("Payment", back_populates="invoice")


class Payment(Base):
    __tablename__ = "payments"

    payment_id          = Column(GUID, primary_key=True, default=uuid.uuid4)
    invoice_id          = Column(GUID, ForeignKey("invoices.invoice_id"), index=True)
    vendor_id           = Column(GUID, ForeignKey("vendors.vendor_id"), index=True)
    amount_paid         = Column(Numeric(15, 2), nullable=False)
    payment_date        = Column(Date, nullable=False, index=True)
    payment_method      = Column(String(50), default="ACH")
    bank_account_hash   = Column(String(255))
    is_manual_override  = Column(Boolean, default=False)
    is_anomaly          = Column(Boolean, default=False)
    anomaly_type        = Column(String(100))
    created_at          = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    vendor  = relationship("Vendor", back_populates="payments")
    invoice = relationship("Invoice", back_populates="payments")


class JournalEntry(Base):
    """Legacy journal entries model for AP table."""
    __tablename__ = "journal_entries"

    entry_id        = Column(GUID, primary_key=True, default=uuid.uuid4)
    posting_date    = Column(Date, nullable=False, index=True)
    debit_account   = Column(String(50), nullable=False)
    credit_account  = Column(String(50), nullable=False)
    amount          = Column(Numeric(15, 2), nullable=False)
    description     = Column(Text)
    posted_by       = Column(String(100))
    is_reversed     = Column(Boolean, default=False)
    is_anomaly      = Column(Boolean, default=False)
    anomaly_type    = Column(String(100))
    created_at      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AnomalyResult(Base):
    __tablename__ = "anomaly_results"

    score_id            = Column(GUID, primary_key=True, default=uuid.uuid4)
    transaction_id      = Column(GUID, nullable=False, index=True)
    transaction_type    = Column(String(50), nullable=False)
    module              = Column(String(10), default="AP", index=True)
    risk_score          = Column(Numeric(5, 2), nullable=False)
    risk_level          = Column(String(20), nullable=False, index=True)
    anomaly_types       = Column(JSON, default=list)
    shap_values         = Column(JSON, default=dict)
    plain_english_reason = Column(Text)
    if_score            = Column(Numeric(5, 4))
    lof_score           = Column(Numeric(5, 4))
    ocsvm_score         = Column(Numeric(5, 4))
    xgb_score           = Column(Numeric(5, 4))
    scored_at           = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    model_version       = Column(String(50))


class AuditLog(Base):
    __tablename__ = "audit_logs"

    log_id          = Column(GUID, primary_key=True, default=uuid.uuid4)
    action          = Column(String(100), nullable=False)
    entity_type     = Column(String(50))
    entity_id       = Column(GUID, index=True)
    performed_by    = Column(String(100), nullable=False)
    details         = Column(JSON, default=dict)
    created_at      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


# ===========================================================================
# 2. ORDER-TO-CASH (O2C) ORM MODELS
# ===========================================================================

class Customer(Base):
    __tablename__ = "customers"

    customer_id     = Column(GUID, primary_key=True, default=uuid.uuid4)
    customer_name   = Column(String(255), nullable=False, index=True)
    customer_type   = Column(String(50), default="Organisation")
    customer_status = Column(String(20), default="Active")
    company_code    = Column(String(10), default="1000")
    sales_org       = Column(String(20), default="SO01")
    created_at      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    credit_profile  = relationship("CustomerCreditProfile", back_populates="customer", uselist=False)
    credit_limits   = relationship("CreditLimit", back_populates="customer")
    sales_orders    = relationship("SalesOrder", back_populates="customer")
    ar_invoices     = relationship("ARInvoice", back_populates="customer")
    payments        = relationship("CustomerPayment", back_populates="customer")
    credit_events   = relationship("CreditEvent", back_populates="customer")


class CustomerCreditProfile(Base):
    __tablename__ = "customer_credit_profiles"

    profile_id          = Column(GUID, primary_key=True, default=uuid.uuid4)
    customer_id         = Column(GUID, ForeignKey("customers.customer_id"), unique=True, nullable=False)
    credit_segment      = Column(String(50), default="CS_DEFAULT")
    credit_control_area = Column(String(50), default="CCA01")
    risk_category       = Column(String(20), default="Medium")
    payment_terms       = Column(String(50), default="Net 30")
    credit_status       = Column(String(20), default="Active")
    effective_from      = Column(Date, default=date.today)

    customer            = relationship("Customer", back_populates="credit_profile")


class CreditLimit(Base):
    __tablename__ = "credit_limits"

    credit_limit_id = Column(GUID, primary_key=True, default=uuid.uuid4)
    customer_id     = Column(GUID, ForeignKey("customers.customer_id"), nullable=False, index=True)
    approved_limit  = Column(Numeric(15, 2), nullable=False)
    previous_limit  = Column(Numeric(15, 2), default=0.0)
    currency        = Column(String(3), default="INR")
    change_reason   = Column(Text)
    changed_by      = Column(String(100), default="system")
    approval_status = Column(String(20), default="approved")
    effective_date  = Column(Date, default=date.today)
    created_at      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    customer        = relationship("Customer", back_populates="credit_limits")


class CreditExposureSnapshot(Base):
    __tablename__ = "credit_exposure_snapshots"

    exposure_id             = Column(GUID, primary_key=True, default=uuid.uuid4)
    customer_id             = Column(GUID, ForeignKey("customers.customer_id"), nullable=False, index=True)
    open_order_exposure     = Column(Numeric(15, 2), default=0.0)
    delivery_exposure       = Column(Numeric(15, 2), default=0.0)
    open_ar                 = Column(Numeric(15, 2), default=0.0)
    other_exposure          = Column(Numeric(15, 2), default=0.0)
    total_exposure          = Column(Numeric(15, 2), default=0.0)
    credit_limit            = Column(Numeric(15, 2), default=0.0)
    available_credit        = Column(Numeric(15, 2), default=0.0)
    credit_utilisation_pct  = Column(Numeric(5, 2), default=0.0)
    snapshot_date           = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class SalesOrder(Base):
    __tablename__ = "sales_orders"

    sales_order_id      = Column(GUID, primary_key=True, default=uuid.uuid4)
    customer_id         = Column(GUID, ForeignKey("customers.customer_id"), nullable=False, index=True)
    order_number        = Column(String(50), nullable=False, unique=True)
    order_date          = Column(Date, nullable=False, index=True)
    order_value         = Column(Numeric(15, 2), nullable=False)
    currency            = Column(String(3), default="INR")
    order_status        = Column(String(20), default="created")
    credit_check_status = Column(String(30), default="approved")
    credit_block_status = Column(String(20), default="none")
    delivery_status     = Column(String(20), default="pending")
    created_at          = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    customer            = relationship("Customer", back_populates="sales_orders")
    invoices            = relationship("ARInvoice", back_populates="sales_order")


class ARInvoice(Base):
    __tablename__ = "ar_invoices"

    invoice_id          = Column(GUID, primary_key=True, default=uuid.uuid4)
    customer_id         = Column(GUID, ForeignKey("customers.customer_id"), nullable=False, index=True)
    sales_order_id      = Column(GUID, ForeignKey("sales_orders.sales_order_id"), nullable=True)
    invoice_number      = Column(String(50), nullable=False, unique=True)
    invoice_date        = Column(Date, nullable=False, index=True)
    due_date            = Column(Date, nullable=False, index=True)
    invoice_amount      = Column(Numeric(15, 2), nullable=False)
    outstanding_amount  = Column(Numeric(15, 2), nullable=False)
    currency            = Column(String(3), default="INR")
    payment_status      = Column(String(20), default="unpaid")
    created_at          = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    customer            = relationship("Customer", back_populates="ar_invoices")
    sales_order         = relationship("SalesOrder", back_populates="invoices")
    allocations         = relationship("PaymentAllocation", back_populates="invoice")


class CustomerPayment(Base):
    __tablename__ = "customer_payments"

    payment_id          = Column(GUID, primary_key=True, default=uuid.uuid4)
    customer_id         = Column(GUID, ForeignKey("customers.customer_id"), nullable=False, index=True)
    payment_date        = Column(Date, nullable=False, index=True)
    payment_amount      = Column(Numeric(15, 2), nullable=False)
    currency            = Column(String(3), default="INR")
    payment_method      = Column(String(50), default="bank_transfer")
    payment_reference   = Column(String(100))
    created_at          = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    customer            = relationship("Customer", back_populates="payments")
    allocations         = relationship("PaymentAllocation", back_populates="payment")


class PaymentAllocation(Base):
    __tablename__ = "payment_allocations"

    allocation_id   = Column(GUID, primary_key=True, default=uuid.uuid4)
    payment_id      = Column(GUID, ForeignKey("customer_payments.payment_id"), nullable=False)
    invoice_id      = Column(GUID, ForeignKey("ar_invoices.invoice_id"), nullable=False)
    allocated_amount = Column(Numeric(15, 2), nullable=False)
    allocation_date = Column(Date, default=date.today)

    payment         = relationship("CustomerPayment", back_populates="allocations")
    invoice         = relationship("ARInvoice", back_populates="allocations")


class CreditEvent(Base):
    __tablename__ = "credit_events"

    event_id        = Column(GUID, primary_key=True, default=uuid.uuid4)
    customer_id     = Column(GUID, ForeignKey("customers.customer_id"), nullable=False, index=True)
    event_type      = Column(String(50), nullable=False)
    previous_value  = Column(Text)
    new_value       = Column(Text)
    performed_by    = Column(String(100), default="system")
    approval_status = Column(String(20), default="approved")
    event_timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    customer        = relationship("Customer", back_populates="credit_events")


# ===========================================================================
# 3. RECORD-TO-REPORT (R2R) ORM MODELS
# ===========================================================================

class CompanyCode(Base):
    __tablename__ = "company_codes"

    company_code    = Column(String(10), primary_key=True)
    company_name    = Column(String(255), nullable=False)
    country         = Column(String(100), default="US")
    base_currency   = Column(String(3), default="USD")
    company_status  = Column(String(20), default="Active")


class FiscalPeriod(Base):
    __tablename__ = "fiscal_periods"

    period_id       = Column(GUID, primary_key=True, default=uuid.uuid4)
    company_code    = Column(String(10), ForeignKey("company_codes.company_code"), nullable=False)
    fiscal_year     = Column(Integer, nullable=False)
    period_number   = Column(Integer, nullable=False)
    start_date      = Column(Date, nullable=False)
    end_date        = Column(Date, nullable=False)
    period_status   = Column(String(20), default="open")
    closed_at       = Column(DateTime(timezone=True))
    reopened_at     = Column(DateTime(timezone=True))


class GLAccount(Base):
    __tablename__ = "gl_accounts"

    gl_account_id       = Column(GUID, primary_key=True, default=uuid.uuid4)
    gl_account_number   = Column(String(50), nullable=False, index=True)
    gl_account_name     = Column(String(255), nullable=False)
    account_type        = Column(String(50), nullable=False)  # Asset, Liability, Equity, Revenue, Expense
    company_code        = Column(String(10), ForeignKey("company_codes.company_code"), nullable=False)
    account_group       = Column(String(100), default="General")
    account_status      = Column(String(20), default="Active")


class JournalHeader(Base):
    __tablename__ = "journal_headers"

    journal_id      = Column(GUID, primary_key=True, default=uuid.uuid4)
    document_id     = Column(String(50), nullable=False, unique=True, index=True)
    company_code    = Column(String(10), ForeignKey("company_codes.company_code"), nullable=False)
    posting_date    = Column(Date, nullable=False, index=True)
    document_date   = Column(Date, nullable=False)
    fiscal_period   = Column(Integer, default=1)
    journal_type    = Column(String(30), default="Standard")
    currency        = Column(String(3), default="USD")
    posting_user    = Column(String(100), default="system")
    approval_status = Column(String(20), default="approved")
    reference_id    = Column(String(100))
    document_status = Column(String(20), default="posted")
    created_at      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    lines           = relationship("JournalEntryLine", back_populates="journal")


class JournalEntryLine(Base):
    __tablename__ = "journal_entry_lines"

    line_id             = Column(GUID, primary_key=True, default=uuid.uuid4)
    journal_id          = Column(GUID, ForeignKey("journal_headers.journal_id"), nullable=False, index=True)
    gl_account_id       = Column(GUID, ForeignKey("gl_accounts.gl_account_id"), nullable=False)
    debit_amount        = Column(Numeric(15, 2), default=0.0)
    credit_amount       = Column(Numeric(15, 2), default=0.0)
    currency            = Column(String(3), default="USD")
    line_description    = Column(Text)
    cost_center         = Column(String(50))
    profit_center       = Column(String(50))

    journal             = relationship("JournalHeader", back_populates="lines")
    gl_account          = relationship("GLAccount")


class AccountReconciliation(Base):
    __tablename__ = "account_reconciliations"

    reconciliation_id   = Column(GUID, primary_key=True, default=uuid.uuid4)
    company_code        = Column(String(10), ForeignKey("company_codes.company_code"), nullable=False)
    gl_account_id       = Column(GUID, ForeignKey("gl_accounts.gl_account_id"), nullable=False)
    reconciliation_date = Column(Date, nullable=False, index=True)
    source_balance      = Column(Numeric(15, 2), nullable=False)
    gl_balance          = Column(Numeric(15, 2), nullable=False)
    difference_amount   = Column(Numeric(15, 2), default=0.0)
    reconciliation_status = Column(String(20), default="reconciled")
    reviewer_id         = Column(String(100))
    created_at          = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    gl_account          = relationship("GLAccount")
    items               = relationship("ReconciliationItem", back_populates="reconciliation")


class ReconciliationItem(Base):
    __tablename__ = "reconciliation_items"

    item_id             = Column(GUID, primary_key=True, default=uuid.uuid4)
    reconciliation_id   = Column(GUID, ForeignKey("account_reconciliations.reconciliation_id"), nullable=False)
    reference_id        = Column(String(100), nullable=False)
    item_amount         = Column(Numeric(15, 2), nullable=False)
    item_date           = Column(Date, nullable=False)
    exception_reason    = Column(Text)
    resolution_status   = Column(String(20), default="open")
    resolved_at         = Column(DateTime(timezone=True))

    reconciliation      = relationship("AccountReconciliation", back_populates="items")


class AccrualEntry(Base):
    __tablename__ = "accrual_entries"

    accrual_id      = Column(GUID, primary_key=True, default=uuid.uuid4)
    journal_id      = Column(GUID, ForeignKey("journal_headers.journal_id"), nullable=True)
    company_code    = Column(String(10), ForeignKey("company_codes.company_code"), nullable=False)
    gl_account_id   = Column(GUID, ForeignKey("gl_accounts.gl_account_id"), nullable=False)
    accrual_amount  = Column(Numeric(15, 2), nullable=False)
    accrual_period  = Column(Integer, nullable=False)
    reversal_date   = Column(Date)
    approval_status = Column(String(20), default="approved")

    gl_account      = relationship("GLAccount")


class IntercompanyTransaction(Base):
    __tablename__ = "intercompany_transactions"

    transaction_id              = Column(GUID, primary_key=True, default=uuid.uuid4)
    source_company_code         = Column(String(10), ForeignKey("company_codes.company_code"), nullable=False)
    destination_company_code    = Column(String(10), ForeignKey("company_codes.company_code"), nullable=False)
    reference_id                = Column(String(100), nullable=False)
    transaction_date            = Column(Date, nullable=False)
    source_amount               = Column(Numeric(15, 2), nullable=False)
    destination_amount          = Column(Numeric(15, 2), nullable=False)
    source_currency             = Column(String(3), default="USD")
    destination_currency        = Column(String(3), default="USD")
    matching_status             = Column(String(20), default="matched")


class CloseActivity(Base):
    __tablename__ = "close_activities"

    activity_id     = Column(GUID, primary_key=True, default=uuid.uuid4)
    company_code    = Column(String(10), ForeignKey("company_codes.company_code"), nullable=False)
    fiscal_period   = Column(Integer, nullable=False)
    activity_name   = Column(String(255), nullable=False)
    assigned_user   = Column(String(100), default="finance_user")
    activity_status = Column(String(20), default="pending")
    due_date        = Column(Date, nullable=False)
    completed_at    = Column(DateTime(timezone=True))


# ===========================================================================
# 4. UNIFIED ANOMALY RESULTS
# ===========================================================================

class UnifiedAnomalyResult(Base):
    __tablename__ = "unified_anomaly_results"

    anomaly_id          = Column(GUID, primary_key=True, default=uuid.uuid4)
    module              = Column(String(10), nullable=False, index=True)  # AP, O2C, R2R
    entity_type         = Column(String(50), nullable=False)              # invoice, customer, order, journal, reconciliation
    entity_id           = Column(GUID, nullable=False, index=True)
    rule_score          = Column(Numeric(5, 2), default=0.0)
    model_score         = Column(Numeric(5, 2), default=0.0)
    final_risk_score    = Column(Numeric(5, 2), nullable=False)
    risk_category       = Column(String(20), nullable=False, index=True)  # LOW, MEDIUM, HIGH, CRITICAL
    rule_reasons        = Column(JSON, default=list)
    shap_explanation    = Column(JSON, default=dict)
    explanation_text    = Column(Text)
    model_version       = Column(String(50), default="1.0.0")
    detected_at         = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    investigation_status = Column(String(20), default="pending", index=True)
    reviewed_by         = Column(String(100))
    resolution_notes    = Column(Text)
