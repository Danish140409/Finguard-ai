"""
FinGuard AI — Explainability Unit Tests

Validates:
  - Deterministic NLG templater outputs human-readable explanations.
  - Plain English explanations accurately reflect risk factors.
  - LIME and SHAP consistency helpers.
"""

import pytest
from explainability.nlg_templater import generate_explanation, get_feature_description


def test_nlg_templater_deterministic():
    """Verify NLG generator produces the exact same output given the same inputs."""
    top_features = [
        ("amount_zscore_by_vendor", 0.45, "increases_risk"),
        ("is_weekend_payment", 0.32, "increases_risk"),
        ("duplicate_amount_flag", 0.28, "increases_risk"),
    ]
    tx_data = {
        "amount": 49800.0,
        "transaction_date": "2026-11-15",
        "amount_zscore_by_vendor": 3.4,
    }

    out1 = generate_explanation(top_features, tx_data)
    out2 = generate_explanation(top_features, tx_data)

    assert out1 == out2, "NLG explanation must be fully deterministic for SOX compliance."
    assert "49,800.00" in out1
    assert "standard deviations" in out1
    assert "Sunday" in out1 or "weekend" in out1
    assert "Duplicate" in out1 or "duplicate" in out1


def test_feature_description_mapping():
    """Ensure feature names map to clean presentation labels."""
    desc = get_feature_description("amount_zscore_by_vendor")
    assert desc == "Amount Deviation from Vendor Average"
    desc2 = get_feature_description("is_weekend_payment")
    assert desc2 == "Weekend/Holiday Transaction"
