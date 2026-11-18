"""
FinGuard AI — NLG Explanation Templater

Converts top SHAP features into deterministic, auditor-friendly English sentences.

Design principle: NO LLM calls — explanations are fully deterministic and
reproducible. The same input always produces the same output, satisfying
SOX audit requirements where explanation consistency is critical.

Usage:
    from explainability.nlg_templater import generate_explanation

    top_features = [
        ("amount_zscore_by_vendor", 0.42, "increases_risk"),
        ("is_weekend_payment",      0.31, "increases_risk"),
        ("duplicate_amount_flag",   0.28, "increases_risk"),
    ]
    text = generate_explanation(top_features, transaction_data)
"""

from datetime import date


# ---------------------------------------------------------------------------
# Feature → human-readable label map
# ---------------------------------------------------------------------------

FEATURE_LABELS: dict[str, str] = {
    "amount_log":                    "Transaction Amount (log-scaled)",
    "amount_zscore_by_vendor":       "Amount Deviation from Vendor Average",
    "days_since_vendor_registration":"Vendor Age at Transaction Date",
    "is_weekend_payment":            "Weekend/Holiday Transaction",
    "duplicate_amount_flag":         "Duplicate Amount Detected",
    "invoice_to_payment_lag":        "Invoice-to-Payment Timing",
    "round_dollar_flag":             "Round Dollar Amount",
    "payment_frequency_spike":       "Payment Frequency Spike",
    "po_match_ratio":                "Purchase Order Match Ratio",
    "vendor_payment_count":          "Vendor Transaction History",
    "is_new_vendor":                 "New Vendor",
    "amount_percentile_by_vendor":   "Amount Percentile for Vendor",
}

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Per-feature sentence generators
# ---------------------------------------------------------------------------

def _fmt_amount_zscore(shap_val: float, tx: dict) -> str:
    zscore = tx.get("amount_zscore_by_vendor", abs(shap_val) * 3)
    if abs(zscore) < 1:
        return "Invoice amount is within the vendor's normal range."
    direction = "higher" if zscore > 0 else "lower"
    return (
        f"Invoice amount of ${tx.get('amount', 0):,.2f} is "
        f"{abs(zscore):.1f} standard deviations {direction} than "
        f"this vendor's historical average — a significant deviation."
    )


def _fmt_weekend_payment(shap_val: float, tx: dict) -> str:
    txn_date = tx.get("transaction_date")
    day_name = "weekend"
    if txn_date:
        try:
            d = date.fromisoformat(str(txn_date))
            day_name = WEEKDAY_NAMES[d.weekday()]
        except (ValueError, AttributeError):
            pass
    return (
        f"Payment was processed on a {day_name} — transactions outside "
        f"normal business hours (Mon–Fri) carry elevated fraud risk."
    )


def _fmt_vendor_age(shap_val: float, tx: dict) -> str:
    days = int(tx.get("days_since_vendor_registration", abs(shap_val) * 100))
    if days < 7:
        return (
            f"Vendor was registered only {days} day(s) before this transaction. "
            f"Payments to vendors registered within the past week are a critical fraud indicator."
        )
    elif days < 30:
        return (
            f"Vendor was registered {days} days before this transaction. "
            f"New vendors (< 30 days) have elevated fictitious-vendor risk."
        )
    return f"Vendor has been registered for {days} days — relatively new supplier."


def _fmt_duplicate(shap_val: float, tx: dict) -> str:
    amount = tx.get("amount", 0)
    return (
        f"An identical payment of ${amount:,.2f} was made to this vendor "
        f"within the past 30 days. Duplicate invoices are the most common "
        f"AP fraud scheme (ACFE)."
    )


def _fmt_round_dollar(shap_val: float, tx: dict) -> str:
    amount = tx.get("amount", 0)
    return (
        f"Invoice amount is an exact round number (${amount:,.0f}). "
        f"Round-dollar payments are statistically associated with fictitious "
        f"or manually-keyed fraudulent invoices."
    )


def _fmt_payment_spike(shap_val: float, tx: dict) -> str:
    return (
        "The number of payments to this vendor in the past 7 days is more than "
        "2× the 90-day rolling average. This pattern is consistent with "
        "split-invoice fraud designed to stay below approval thresholds."
    )


def _fmt_new_vendor(shap_val: float, tx: dict) -> str:
    return (
        "This vendor has been active for less than 30 days. "
        "Payments to recently-onboarded vendors with no transaction history "
        "require additional verification."
    )


def _fmt_lag(shap_val: float, tx: dict) -> str:
    lag = tx.get("invoice_to_payment_lag", 30)
    if lag < 0:
        return (
            f"Payment was made {abs(int(lag))} days before the invoice date — "
            f"paying before invoice receipt indicates a process control failure."
        )
    elif lag > 90:
        return (
            f"Payment was made {int(lag)} days after the invoice date, "
            f"which is {int(lag) - 30} days past standard 30-day terms."
        )
    return f"Invoice-to-payment lag of {int(lag)} days is within acceptable range."


def _fmt_po_ratio(shap_val: float, tx: dict) -> str:
    ratio = tx.get("po_match_ratio", 1.0)
    if ratio > 1.1:
        pct_over = (ratio - 1) * 100
        return (
            f"Invoice amount is {pct_over:.0f}% above the matching purchase order value. "
            f"Three-way PO matching failed — manual review required."
        )
    return f"Invoice amount is {ratio:.0%} of the matching purchase order."


def _fmt_amount_percentile(shap_val: float, tx: dict) -> str:
    pct = tx.get("amount_percentile_by_vendor", 0.9)
    return (
        f"This invoice is in the top {(1 - pct) * 100:.0f}% of amounts "
        f"ever processed for this vendor — unusually large relative to history."
    )


def _fmt_generic(feature: str, shap_val: float, tx: dict) -> str:
    label     = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
    direction = "contributed to" if shap_val > 0 else "mitigated"
    return f"{label} {direction} the elevated risk score (SHAP contribution: {shap_val:+.4f})."


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_FORMATTERS = {
    "amount_zscore_by_vendor":       _fmt_amount_zscore,
    "is_weekend_payment":            _fmt_weekend_payment,
    "days_since_vendor_registration":_fmt_vendor_age,
    "duplicate_amount_flag":         _fmt_duplicate,
    "round_dollar_flag":             _fmt_round_dollar,
    "payment_frequency_spike":       _fmt_payment_spike,
    "is_new_vendor":                 _fmt_new_vendor,
    "invoice_to_payment_lag":        _fmt_lag,
    "po_match_ratio":                _fmt_po_ratio,
    "amount_percentile_by_vendor":   _fmt_amount_percentile,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_explanation(
    shap_features: list[tuple[str, float, str]],
    transaction_data: dict,
    max_sentences: int = 3,
) -> str:
    """
    Convert the top SHAP features into a plain-English explanation paragraph.

    Args:
        shap_features:    List of (feature_name, shap_value, direction) tuples,
                          sorted by absolute SHAP value descending.
                          Typically from SHAPEngine.get_top_features().
        transaction_data: Raw transaction dict (amount, dates, vendor info, etc.)
        max_sentences:    Maximum number of sentences to include.

    Returns:
        A single human-readable string explaining why this transaction was flagged.
        Always deterministic — same input produces same output.
    """
    sentences: list[str] = []

    for feature, shap_val, direction in shap_features[:max_sentences]:
        if direction == "decreases_risk":
            continue  # Only explain risk-increasing factors for brevity
        formatter = _FORMATTERS.get(feature, None)
        if formatter:
            sentence = formatter(shap_val, transaction_data)
        else:
            sentence = _fmt_generic(feature, shap_val, transaction_data)
        if sentence:
            sentences.append(sentence)

    if not sentences:
        # Fallback if all top features decrease risk (unusual)
        risk_score = transaction_data.get("risk_score", 0)
        return (
            f"This transaction received a composite risk score of {risk_score:.1f}/100. "
            f"Multiple low-weight signals in combination triggered this flag. "
            f"Manual review is recommended."
        )

    # Join sentences into a coherent paragraph
    return " ".join(sentences)


def get_feature_description(feature_name: str) -> str:
    """Return the human-readable label for a feature name."""
    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").title())
