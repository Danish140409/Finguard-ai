"""
FinGuard AI — SHAP Explainability Engine

Wraps SHAP TreeExplainer for the XGBoost model to produce:
  - Local (per-transaction) SHAP explanations
  - Global (dataset-level) feature importances

Usage:
    engine = SHAPEngine(xgb_model, feature_names=FEATURE_COLUMNS)
    local_shap = engine.explain_local(X_single_row)
    top_features = engine.get_top_features(local_shap, n=5)
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import shap

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class SHAPEngine:
    """
    SHAP explanation engine backed by TreeExplainer for XGBoost.
    Provides deterministic, game-theoretically grounded explanations —
    suitable for SOX audit sign-off.
    """

    def __init__(self, xgb_model, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(
            xgb_model,
            feature_perturbation="tree_path_dependent",
        )
        logger.info("SHAPEngine initialized", extra={"n_features": len(feature_names)})

    def explain_local(self, X: np.ndarray) -> dict[str, Any]:
        """
        Compute SHAP values for a single transaction (or small batch).

        Args:
            X: Feature array of shape (1, n_features) or (n, n_features).

        Returns:
            dict with keys:
                'shap_values'  — {feature_name: shap_value} for each feature
                'base_value'   — model's expected value (baseline prediction)
                'prediction'   — model output for this instance
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        shap_values  = self.explainer.shap_values(X)
        base_value   = float(self.explainer.expected_value)

        if isinstance(shap_values, list):
            # Binary classification returns [neg_class, pos_class]
            shap_values = shap_values[1]

        # Take first row if batch
        row_shap = shap_values[0] if shap_values.ndim == 2 else shap_values

        shap_dict = {
            feat: round(float(val), 6)
            for feat, val in zip(self.feature_names, row_shap)
        }

        return {
            "shap_values": shap_dict,
            "base_value":  round(base_value, 6),
            "prediction":  round(float(base_value + sum(row_shap)), 6),
        }

    def explain_global(self, X: np.ndarray) -> dict[str, float]:
        """
        Compute mean absolute SHAP values across a dataset — global feature importance.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            {feature_name: mean_abs_shap_value} sorted descending by importance.
        """
        logger.info("Computing global SHAP values", extra={"n_samples": len(X)})
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs = np.abs(shap_values).mean(axis=0)
        result = {
            feat: round(float(val), 6)
            for feat, val in zip(self.feature_names, mean_abs)
        }
        return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))

    def get_top_features(
        self,
        shap_values: dict[str, float],
        n: int = 5,
    ) -> list[tuple[str, float, str]]:
        """
        Return the top-N features ranked by absolute SHAP value.

        Returns:
            List of (feature_name, shap_value, direction)
            where direction is 'increases_risk' or 'decreases_risk'.
        """
        sorted_features = sorted(
            shap_values.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:n]

        return [
            (
                feat,
                val,
                "increases_risk" if val > 0 else "decreases_risk",
            )
            for feat, val in sorted_features
        ]
