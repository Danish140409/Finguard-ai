-- ============================================================
-- FinGuard AI — Analytical Views
-- Run after schema.sql
-- ============================================================

-- ============================================================
-- v_flagged_transactions
-- All HIGH and CRITICAL transactions with vendor info
-- ============================================================
CREATE OR REPLACE VIEW v_flagged_transactions AS
SELECT
    ar.score_id,
    ar.transaction_id,
    ar.transaction_type,
    ar.risk_score,
    ar.risk_level,
    ar.anomaly_types,
    ar.plain_english_reason,
    ar.scored_at,
    ar.model_version,
    -- Invoice fields (when transaction_type = 'invoice')
    i.amount          AS invoice_amount,
    i.currency,
    i.invoice_date,
    i.due_date,
    i.po_number,
    i.description     AS invoice_description,
    i.status          AS invoice_status,
    -- Payment fields (when transaction_type = 'payment')
    p.amount_paid,
    p.payment_date,
    p.payment_method,
    p.is_manual_override,
    -- Vendor info
    v.vendor_name,
    v.country         AS vendor_country,
    v.risk_tier       AS vendor_risk_tier,
    v.registration_date AS vendor_registration_date
FROM anomaly_results ar
LEFT JOIN invoices  i ON ar.transaction_id = i.invoice_id  AND ar.transaction_type = 'invoice'
LEFT JOIN payments  p ON ar.transaction_id = p.payment_id  AND ar.transaction_type = 'payment'
LEFT JOIN vendors   v ON COALESCE(i.vendor_id, p.vendor_id) = v.vendor_id
WHERE ar.risk_level IN ('HIGH', 'CRITICAL')
ORDER BY ar.risk_score DESC, ar.scored_at DESC;


-- ============================================================
-- v_vendor_risk_summary
-- Per-vendor risk aggregation for the vendor risk heatmap
-- ============================================================
CREATE OR REPLACE VIEW v_vendor_risk_summary AS
SELECT
    v.vendor_id,
    v.vendor_name,
    v.country,
    v.risk_tier,
    v.registration_date,
    COUNT(ar.score_id)                          AS total_scored,
    COUNT(ar.score_id) FILTER (
        WHERE ar.risk_level IN ('HIGH','CRITICAL')
    )                                           AS flagged_count,
    ROUND(AVG(ar.risk_score)::NUMERIC, 2)       AS avg_risk_score,
    ROUND(MAX(ar.risk_score)::NUMERIC, 2)       AS max_risk_score,
    COALESCE(SUM(i.amount) FILTER (
        WHERE ar.risk_level IN ('HIGH','CRITICAL')
        AND ar.transaction_type = 'invoice'
    ), 0)                                       AS flagged_invoice_exposure,
    COALESCE(SUM(p.amount_paid) FILTER (
        WHERE ar.risk_level IN ('HIGH','CRITICAL')
        AND ar.transaction_type = 'payment'
    ), 0)                                       AS flagged_payment_exposure
FROM vendors v
LEFT JOIN invoices  i  ON v.vendor_id = i.vendor_id
LEFT JOIN payments  p  ON v.vendor_id = p.vendor_id
LEFT JOIN anomaly_results ar ON (
    (ar.transaction_id = i.invoice_id AND ar.transaction_type = 'invoice')
    OR
    (ar.transaction_id = p.payment_id AND ar.transaction_type = 'payment')
)
GROUP BY v.vendor_id, v.vendor_name, v.country, v.risk_tier, v.registration_date
ORDER BY avg_risk_score DESC NULLS LAST;


-- ============================================================
-- v_daily_anomaly_trend
-- Daily flagged transaction counts for the last 90 days
-- ============================================================
CREATE OR REPLACE VIEW v_daily_anomaly_trend AS
SELECT
    DATE(ar.scored_at)                          AS score_date,
    COUNT(*)                                    AS total_scored,
    COUNT(*) FILTER (WHERE ar.risk_level = 'LOW')      AS low_count,
    COUNT(*) FILTER (WHERE ar.risk_level = 'MEDIUM')   AS medium_count,
    COUNT(*) FILTER (WHERE ar.risk_level = 'HIGH')     AS high_count,
    COUNT(*) FILTER (WHERE ar.risk_level = 'CRITICAL') AS critical_count,
    ROUND(AVG(ar.risk_score)::NUMERIC, 2)       AS avg_risk_score,
    COALESCE(SUM(i.amount) FILTER (
        WHERE ar.risk_level IN ('HIGH','CRITICAL')
        AND ar.transaction_type = 'invoice'
    ), 0)                                       AS daily_exposure
FROM anomaly_results ar
LEFT JOIN invoices i ON ar.transaction_id = i.invoice_id AND ar.transaction_type = 'invoice'
WHERE ar.scored_at >= NOW() - INTERVAL '90 days'
GROUP BY DATE(ar.scored_at)
ORDER BY score_date DESC;
