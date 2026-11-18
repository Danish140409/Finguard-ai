-- ============================================================
-- FinGuard AI — PostgreSQL Database Schema
-- Run: psql -U finguard -d finguard_db -f database/schema.sql
-- ============================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- USERS — authentication and role management
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    user_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username      VARCHAR(100) UNIQUE NOT NULL,
    email         VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role          VARCHAR(20) NOT NULL DEFAULT 'analyst'
                  CHECK (role IN ('analyst', 'admin')),
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================
-- VENDORS — vendor master data
-- ============================================================
CREATE TABLE IF NOT EXISTS vendors (
    vendor_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name          VARCHAR(255) NOT NULL,
    registration_date    DATE NOT NULL,
    country              VARCHAR(100) DEFAULT 'US',
    risk_tier            VARCHAR(20) DEFAULT 'standard'
                         CHECK (risk_tier IN ('standard', 'high_risk', 'fictitious')),
    tax_id               VARCHAR(50),
    bank_account_hash    VARCHAR(255),
    contact_email        VARCHAR(255),
    is_active            BOOLEAN NOT NULL DEFAULT TRUE,
    created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vendors_risk_tier ON vendors(risk_tier);
CREATE INDEX IF NOT EXISTS idx_vendors_registration_date ON vendors(registration_date);

-- ============================================================
-- PURCHASE ORDERS — PO master for 3-way matching
-- ============================================================
CREATE TABLE IF NOT EXISTS purchase_orders (
    po_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_id     UUID REFERENCES vendors(vendor_id) ON DELETE SET NULL,
    po_number     VARCHAR(100) UNIQUE NOT NULL,
    po_amount     NUMERIC(15, 2) NOT NULL CHECK (po_amount > 0),
    po_date       DATE NOT NULL,
    approved_by   VARCHAR(100),
    status        VARCHAR(20) DEFAULT 'open'
                  CHECK (status IN ('open', 'closed', 'cancelled')),
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_po_vendor_id ON purchase_orders(vendor_id);

-- ============================================================
-- INVOICES — AP invoice data
-- ============================================================
CREATE TABLE IF NOT EXISTS invoices (
    invoice_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_id           UUID REFERENCES vendors(vendor_id) ON DELETE SET NULL,
    po_id               UUID REFERENCES purchase_orders(po_id) ON DELETE SET NULL,
    amount              NUMERIC(15, 2) NOT NULL CHECK (amount > 0),
    currency            VARCHAR(3) DEFAULT 'USD',
    invoice_date        DATE NOT NULL,
    due_date            DATE,
    po_number           VARCHAR(100),
    payment_terms_days  INTEGER DEFAULT 30,
    description         TEXT,
    status              VARCHAR(20) DEFAULT 'pending'
                        CHECK (status IN ('pending', 'approved', 'paid', 'disputed', 'cancelled')),
    -- Anomaly ground truth (for supervised training)
    is_anomaly          BOOLEAN DEFAULT FALSE,
    anomaly_type        VARCHAR(100),
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_invoices_vendor_id ON invoices(vendor_id);
CREATE INDEX IF NOT EXISTS idx_invoices_invoice_date ON invoices(invoice_date);
CREATE INDEX IF NOT EXISTS idx_invoices_amount ON invoices(amount);
CREATE INDEX IF NOT EXISTS idx_invoices_is_anomaly ON invoices(is_anomaly);

-- ============================================================
-- PAYMENTS — payment records
-- ============================================================
CREATE TABLE IF NOT EXISTS payments (
    payment_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    invoice_id          UUID REFERENCES invoices(invoice_id) ON DELETE SET NULL,
    vendor_id           UUID REFERENCES vendors(vendor_id) ON DELETE SET NULL,
    amount_paid         NUMERIC(15, 2) NOT NULL CHECK (amount_paid > 0),
    payment_date        DATE NOT NULL,
    payment_method      VARCHAR(50) DEFAULT 'ACH'
                        CHECK (payment_method IN ('ACH', 'wire', 'check', 'card', 'manual')),
    bank_account_hash   VARCHAR(255),
    is_manual_override  BOOLEAN DEFAULT FALSE,
    -- Anomaly ground truth
    is_anomaly          BOOLEAN DEFAULT FALSE,
    anomaly_type        VARCHAR(100),
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_payments_vendor_id ON payments(vendor_id);
CREATE INDEX IF NOT EXISTS idx_payments_payment_date ON payments(payment_date);
CREATE INDEX IF NOT EXISTS idx_payments_invoice_id ON payments(invoice_id);

-- ============================================================
-- JOURNAL ENTRIES — GL postings
-- ============================================================
CREATE TABLE IF NOT EXISTS journal_entries (
    entry_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    posting_date    DATE NOT NULL,
    debit_account   VARCHAR(50) NOT NULL,
    credit_account  VARCHAR(50) NOT NULL,
    amount          NUMERIC(15, 2) NOT NULL CHECK (amount > 0),
    description     TEXT,
    posted_by       VARCHAR(100),
    is_reversed     BOOLEAN DEFAULT FALSE,
    -- Anomaly ground truth
    is_anomaly      BOOLEAN DEFAULT FALSE,
    anomaly_type    VARCHAR(100),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_je_posting_date ON journal_entries(posting_date);

-- ============================================================
-- ANOMALY RESULTS — ML scoring output
-- ============================================================
CREATE TABLE IF NOT EXISTS anomaly_results (
    score_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id      UUID NOT NULL,
    transaction_type    VARCHAR(50) NOT NULL
                        CHECK (transaction_type IN (
                            'invoice', 'payment', 'journal_entry',
                            'customer', 'sales_order', 'credit_event',
                            'reconciliation', 'journal', 'intercompany', 'accrual'
                        )),
    module              VARCHAR(10) DEFAULT 'AP',
    risk_score          NUMERIC(5, 2) NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    risk_level          VARCHAR(20) NOT NULL
                        CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    anomaly_types       JSONB DEFAULT '[]',   -- list of detected anomaly type strings
    shap_values         JSONB DEFAULT '{}',   -- {feature: shap_value}
    plain_english_reason TEXT,
    if_score            NUMERIC(5, 4),        -- Isolation Forest component
    lof_score           NUMERIC(5, 4),        -- LOF component
    ocsvm_score         NUMERIC(5, 4),        -- One-Class SVM component
    xgb_score           NUMERIC(5, 4),        -- XGBoost component
    scored_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_version       VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_ar_transaction_id ON anomaly_results(transaction_id);
CREATE INDEX IF NOT EXISTS idx_ar_risk_level ON anomaly_results(risk_level);
CREATE INDEX IF NOT EXISTS idx_ar_scored_at ON anomaly_results(scored_at);
CREATE INDEX IF NOT EXISTS idx_ar_risk_score ON anomaly_results(risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_ar_module ON anomaly_results(module);

-- ============================================================
--  O2C (Order-to-Cash) CREDIT MANAGEMENT TABLES
-- ============================================================

-- ============================================================
-- CUSTOMERS — customer master data
-- ============================================================
CREATE TABLE IF NOT EXISTS customers (
    customer_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_name   VARCHAR(255) NOT NULL,
    customer_type   VARCHAR(50) DEFAULT 'Organisation',
    customer_status VARCHAR(20) DEFAULT 'Active',
    company_code    VARCHAR(10) DEFAULT '1000',
    sales_org       VARCHAR(20),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_customers_status ON customers(customer_status);
CREATE INDEX IF NOT EXISTS idx_customers_company_code ON customers(company_code);

-- ============================================================
-- CUSTOMER CREDIT PROFILES — credit risk segmentation
-- ============================================================
CREATE TABLE IF NOT EXISTS customer_credit_profiles (
    profile_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id         UUID REFERENCES customers(customer_id) ON DELETE CASCADE,
    credit_segment      VARCHAR(50),
    credit_control_area VARCHAR(50),
    risk_category       VARCHAR(20),
    payment_terms       VARCHAR(50),
    credit_status       VARCHAR(20),
    effective_from      DATE
);

CREATE INDEX IF NOT EXISTS idx_ccp_customer_id ON customer_credit_profiles(customer_id);
CREATE INDEX IF NOT EXISTS idx_ccp_risk_category ON customer_credit_profiles(risk_category);
CREATE INDEX IF NOT EXISTS idx_ccp_credit_status ON customer_credit_profiles(credit_status);

-- ============================================================
-- CREDIT LIMITS — approved credit limits and history
-- ============================================================
CREATE TABLE IF NOT EXISTS credit_limits (
    credit_limit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id     UUID REFERENCES customers(customer_id) ON DELETE CASCADE,
    approved_limit  NUMERIC(15, 2),
    previous_limit  NUMERIC(15, 2) DEFAULT 0,
    currency        VARCHAR(3) DEFAULT 'INR',
    change_reason   TEXT,
    changed_by      VARCHAR(100),
    approval_status VARCHAR(20),
    effective_date  DATE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cl_customer_id ON credit_limits(customer_id);
CREATE INDEX IF NOT EXISTS idx_cl_approval_status ON credit_limits(approval_status);

-- ============================================================
-- CREDIT EXPOSURE SNAPSHOTS — point-in-time exposure data
-- ============================================================
CREATE TABLE IF NOT EXISTS credit_exposure_snapshots (
    exposure_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id             UUID REFERENCES customers(customer_id) ON DELETE CASCADE,
    open_order_exposure     NUMERIC(15, 2) DEFAULT 0,
    delivery_exposure       NUMERIC(15, 2) DEFAULT 0,
    open_ar                 NUMERIC(15, 2) DEFAULT 0,
    other_exposure          NUMERIC(15, 2) DEFAULT 0,
    total_exposure          NUMERIC(15, 2) DEFAULT 0,
    credit_limit            NUMERIC(15, 2) DEFAULT 0,
    available_credit        NUMERIC(15, 2) DEFAULT 0,
    credit_utilisation_pct  NUMERIC(5, 2) DEFAULT 0,
    snapshot_date           TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ces_customer_id ON credit_exposure_snapshots(customer_id);
CREATE INDEX IF NOT EXISTS idx_ces_snapshot_date ON credit_exposure_snapshots(snapshot_date);

-- ============================================================
-- SALES ORDERS — sales order master
-- ============================================================
CREATE TABLE IF NOT EXISTS sales_orders (
    sales_order_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id         UUID REFERENCES customers(customer_id) ON DELETE SET NULL,
    order_number        VARCHAR(50),
    order_date          DATE,
    order_value         NUMERIC(15, 2),
    currency            VARCHAR(3) DEFAULT 'INR',
    order_status        VARCHAR(20),
    credit_check_status VARCHAR(30),
    credit_block_status VARCHAR(20) DEFAULT 'none',
    delivery_status     VARCHAR(20) DEFAULT 'pending',
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_so_customer_id ON sales_orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_so_order_status ON sales_orders(order_status);
CREATE INDEX IF NOT EXISTS idx_so_credit_check_status ON sales_orders(credit_check_status);
CREATE INDEX IF NOT EXISTS idx_so_order_date ON sales_orders(order_date);

-- ============================================================
-- AR INVOICES — accounts receivable invoices
-- ============================================================
CREATE TABLE IF NOT EXISTS ar_invoices (
    invoice_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id         UUID REFERENCES customers(customer_id) ON DELETE SET NULL,
    sales_order_id      UUID REFERENCES sales_orders(sales_order_id) ON DELETE SET NULL,
    invoice_number      VARCHAR(50),
    invoice_date        DATE,
    due_date            DATE,
    invoice_amount      NUMERIC(15, 2),
    outstanding_amount  NUMERIC(15, 2),
    currency            VARCHAR(3) DEFAULT 'INR',
    payment_status      VARCHAR(20) DEFAULT 'unpaid',
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ari_customer_id ON ar_invoices(customer_id);
CREATE INDEX IF NOT EXISTS idx_ari_sales_order_id ON ar_invoices(sales_order_id);
CREATE INDEX IF NOT EXISTS idx_ari_payment_status ON ar_invoices(payment_status);
CREATE INDEX IF NOT EXISTS idx_ari_due_date ON ar_invoices(due_date);

-- ============================================================
-- CUSTOMER PAYMENTS — incoming customer payments
-- ============================================================
CREATE TABLE IF NOT EXISTS customer_payments (
    payment_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id         UUID REFERENCES customers(customer_id) ON DELETE SET NULL,
    payment_date        DATE,
    payment_amount      NUMERIC(15, 2),
    currency            VARCHAR(3) DEFAULT 'INR',
    payment_method      VARCHAR(50),
    payment_reference   VARCHAR(100),
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cp_customer_id ON customer_payments(customer_id);
CREATE INDEX IF NOT EXISTS idx_cp_payment_date ON customer_payments(payment_date);

-- ============================================================
-- PAYMENT ALLOCATIONS — link payments to AR invoices
-- ============================================================
CREATE TABLE IF NOT EXISTS payment_allocations (
    allocation_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    payment_id      UUID REFERENCES customer_payments(payment_id) ON DELETE CASCADE,
    invoice_id      UUID REFERENCES ar_invoices(invoice_id) ON DELETE CASCADE,
    allocated_amount NUMERIC(15, 2),
    allocation_date DATE
);

CREATE INDEX IF NOT EXISTS idx_pa_payment_id ON payment_allocations(payment_id);
CREATE INDEX IF NOT EXISTS idx_pa_invoice_id ON payment_allocations(invoice_id);

-- ============================================================
-- CREDIT EVENTS — credit-related event log
-- ============================================================
CREATE TABLE IF NOT EXISTS credit_events (
    event_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id     UUID REFERENCES customers(customer_id) ON DELETE CASCADE,
    event_type      VARCHAR(50),
    previous_value  TEXT,
    new_value       TEXT,
    performed_by    VARCHAR(100),
    approval_status VARCHAR(20),
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ce_customer_id ON credit_events(customer_id);
CREATE INDEX IF NOT EXISTS idx_ce_event_type ON credit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_ce_event_timestamp ON credit_events(event_timestamp);

-- ============================================================
--  R2R (Record-to-Report) TABLES
-- ============================================================

-- ============================================================
-- COMPANY CODES — legal entity master
-- ============================================================
CREATE TABLE IF NOT EXISTS company_codes (
    company_code    VARCHAR(10) PRIMARY KEY,
    company_name    VARCHAR(255),
    country         VARCHAR(100),
    base_currency   VARCHAR(3) DEFAULT 'USD',
    company_status  VARCHAR(20) DEFAULT 'Active'
);

-- ============================================================
-- FISCAL PERIODS — fiscal calendar periods
-- ============================================================
CREATE TABLE IF NOT EXISTS fiscal_periods (
    period_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_code    VARCHAR(10) REFERENCES company_codes(company_code) ON DELETE CASCADE,
    fiscal_year     INT,
    period_number   INT,
    start_date      DATE,
    end_date        DATE,
    period_status   VARCHAR(20) DEFAULT 'open',
    closed_at       TIMESTAMP WITH TIME ZONE,
    reopened_at     TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_fp_company_code ON fiscal_periods(company_code);
CREATE INDEX IF NOT EXISTS idx_fp_period_status ON fiscal_periods(period_status);

-- ============================================================
-- GL ACCOUNTS — general ledger chart of accounts
-- ============================================================
CREATE TABLE IF NOT EXISTS gl_accounts (
    gl_account_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gl_account_number   VARCHAR(50),
    gl_account_name     VARCHAR(255),
    account_type        VARCHAR(50),
    company_code        VARCHAR(10) REFERENCES company_codes(company_code) ON DELETE CASCADE,
    account_group       VARCHAR(100),
    account_status      VARCHAR(20) DEFAULT 'Active'
);

CREATE INDEX IF NOT EXISTS idx_gla_company_code ON gl_accounts(company_code);
CREATE INDEX IF NOT EXISTS idx_gla_account_type ON gl_accounts(account_type);

-- ============================================================
-- JOURNAL HEADERS — journal document headers
-- ============================================================
CREATE TABLE IF NOT EXISTS journal_headers (
    journal_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     VARCHAR(50),
    company_code    VARCHAR(10) REFERENCES company_codes(company_code) ON DELETE CASCADE,
    posting_date    DATE,
    document_date   DATE,
    fiscal_period   INT,
    journal_type    VARCHAR(30),
    currency        VARCHAR(3) DEFAULT 'USD',
    posting_user    VARCHAR(100),
    approval_status VARCHAR(20),
    reference_id    VARCHAR(100),
    document_status VARCHAR(20) DEFAULT 'posted',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jh_company_code ON journal_headers(company_code);
CREATE INDEX IF NOT EXISTS idx_jh_posting_date ON journal_headers(posting_date);
CREATE INDEX IF NOT EXISTS idx_jh_journal_type ON journal_headers(journal_type);
CREATE INDEX IF NOT EXISTS idx_jh_document_status ON journal_headers(document_status);

-- ============================================================
-- JOURNAL ENTRY LINES — journal line items
-- ============================================================
CREATE TABLE IF NOT EXISTS journal_entry_lines (
    line_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    journal_id          UUID REFERENCES journal_headers(journal_id) ON DELETE CASCADE,
    gl_account_id       UUID REFERENCES gl_accounts(gl_account_id) ON DELETE SET NULL,
    debit_amount        NUMERIC(15, 2) DEFAULT 0,
    credit_amount       NUMERIC(15, 2) DEFAULT 0,
    currency            VARCHAR(3) DEFAULT 'USD',
    line_description    TEXT,
    cost_center         VARCHAR(50),
    profit_center       VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_jel_journal_id ON journal_entry_lines(journal_id);
CREATE INDEX IF NOT EXISTS idx_jel_gl_account_id ON journal_entry_lines(gl_account_id);

-- ============================================================
-- ACCOUNT RECONCILIATIONS — GL account reconciliation
-- ============================================================
CREATE TABLE IF NOT EXISTS account_reconciliations (
    reconciliation_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_code            VARCHAR(10) REFERENCES company_codes(company_code) ON DELETE CASCADE,
    gl_account_id           UUID REFERENCES gl_accounts(gl_account_id) ON DELETE SET NULL,
    reconciliation_date     DATE,
    source_balance          NUMERIC(15, 2),
    gl_balance              NUMERIC(15, 2),
    difference_amount       NUMERIC(15, 2) DEFAULT 0,
    reconciliation_status   VARCHAR(20),
    reviewer_id             VARCHAR(100),
    created_at              TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_arec_company_code ON account_reconciliations(company_code);
CREATE INDEX IF NOT EXISTS idx_arec_gl_account_id ON account_reconciliations(gl_account_id);
CREATE INDEX IF NOT EXISTS idx_arec_status ON account_reconciliations(reconciliation_status);

-- ============================================================
-- RECONCILIATION ITEMS — line-level reconciliation details
-- ============================================================
CREATE TABLE IF NOT EXISTS reconciliation_items (
    item_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reconciliation_id   UUID REFERENCES account_reconciliations(reconciliation_id) ON DELETE CASCADE,
    reference_id        VARCHAR(100),
    item_amount         NUMERIC(15, 2),
    item_date           DATE,
    exception_reason    TEXT,
    resolution_status   VARCHAR(20) DEFAULT 'open',
    resolved_at         TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_ri_reconciliation_id ON reconciliation_items(reconciliation_id);
CREATE INDEX IF NOT EXISTS idx_ri_resolution_status ON reconciliation_items(resolution_status);

-- ============================================================
-- ACCRUAL ENTRIES — period-end accruals
-- ============================================================
CREATE TABLE IF NOT EXISTS accrual_entries (
    accrual_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    journal_id      UUID REFERENCES journal_headers(journal_id) ON DELETE SET NULL,
    company_code    VARCHAR(10) REFERENCES company_codes(company_code) ON DELETE CASCADE,
    gl_account_id   UUID REFERENCES gl_accounts(gl_account_id) ON DELETE SET NULL,
    accrual_amount  NUMERIC(15, 2),
    accrual_period  INT,
    reversal_date   DATE,
    approval_status VARCHAR(20)
);

CREATE INDEX IF NOT EXISTS idx_ae_journal_id ON accrual_entries(journal_id);
CREATE INDEX IF NOT EXISTS idx_ae_company_code ON accrual_entries(company_code);
CREATE INDEX IF NOT EXISTS idx_ae_gl_account_id ON accrual_entries(gl_account_id);

-- ============================================================
-- INTERCOMPANY TRANSACTIONS — cross-entity transactions
-- ============================================================
CREATE TABLE IF NOT EXISTS intercompany_transactions (
    transaction_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_company_code     VARCHAR(10) REFERENCES company_codes(company_code) ON DELETE CASCADE,
    destination_company_code VARCHAR(10) REFERENCES company_codes(company_code) ON DELETE CASCADE,
    reference_id            VARCHAR(100),
    transaction_date        DATE,
    source_amount           NUMERIC(15, 2),
    destination_amount      NUMERIC(15, 2),
    source_currency         VARCHAR(3),
    destination_currency    VARCHAR(3),
    matching_status         VARCHAR(20) DEFAULT 'matched'
);

CREATE INDEX IF NOT EXISTS idx_ict_source_cc ON intercompany_transactions(source_company_code);
CREATE INDEX IF NOT EXISTS idx_ict_dest_cc ON intercompany_transactions(destination_company_code);
CREATE INDEX IF NOT EXISTS idx_ict_matching_status ON intercompany_transactions(matching_status);

-- ============================================================
-- CLOSE ACTIVITIES — period-end close task tracking
-- ============================================================
CREATE TABLE IF NOT EXISTS close_activities (
    activity_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_code    VARCHAR(10) REFERENCES company_codes(company_code) ON DELETE CASCADE,
    fiscal_period   INT,
    activity_name   VARCHAR(255),
    assigned_user   VARCHAR(100),
    activity_status VARCHAR(20) DEFAULT 'pending',
    due_date        DATE,
    completed_at    TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_ca_company_code ON close_activities(company_code);
CREATE INDEX IF NOT EXISTS idx_ca_activity_status ON close_activities(activity_status);

-- ============================================================
-- AUDIT LOGS — immutable audit trail (SOX compliant)
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    action          VARCHAR(100) NOT NULL,  -- e.g. 'SCORE', 'APPROVE', 'ESCALATE', 'LOGIN'
    entity_type     VARCHAR(50),            -- 'invoice', 'payment', 'user', 'model'
    entity_id       UUID,
    performed_by    VARCHAR(100) NOT NULL,
    details         JSONB DEFAULT '{}',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_entity_id ON audit_logs(entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_performed_by ON audit_logs(performed_by);

-- Immutability: prevent UPDATE and DELETE on audit_logs
CREATE OR REPLACE FUNCTION prevent_audit_modification()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'audit_logs is immutable: UPDATE and DELETE are not permitted (SOX compliance)';
END;
$$;

DROP TRIGGER IF EXISTS trg_audit_no_update ON audit_logs;
CREATE TRIGGER trg_audit_no_update
    BEFORE UPDATE ON audit_logs
    FOR EACH ROW EXECUTE FUNCTION prevent_audit_modification();

DROP TRIGGER IF EXISTS trg_audit_no_delete ON audit_logs;
CREATE TRIGGER trg_audit_no_delete
    BEFORE DELETE ON audit_logs
    FOR EACH ROW EXECUTE FUNCTION prevent_audit_modification();
