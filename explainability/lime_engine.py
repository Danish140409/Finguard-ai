"""
FinGuard AI — LIME Explainability Engine

Wraps lime.lime_tabular.LimeTabularExplainer to provide model-agnostic,
alternative explanations for individual transaction scores.

Auditors appreciate seeing two independent explanation methods (SHAP + LIME)
agree on the top contributing features — it validates the ML decision.

Usage:
    lime_engine = LIMEEngine(predict_fn, feature_names, training_data)
    explanation = lime_engine.explain(instance_array)
"""

import sys
from pathlib import Path
from typing import Callable

import numpy as np
from lime import lime_tabular

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class LIMEEngine:
    """
    LIME tabular explainer for FinGuard AI transaction scoring.
    Model-agnostic: works with any predict function that returns [0,1] probability.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: list[str],
        training_data: np.ndarray,
        mode: str = "classification",
    ) -> None:
        """
        Args:
            predict_fn:     Function that takes X (n, p) → probabilities (n, 2).
                            For regression scoring, wrap to return [[1-s, s]] per row.
            feature_names:  List of feature names (must match training_data columns).
            training_data:  Training feature matrix (n_samples, n_features).
                            Used to estimate feature distributions.
            mode:           'classification' (default) or 'regression'.
        """
        self.predict_fn   = predict_fn
        self.feature_names = feature_names

        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=["Normal", "Anomaly"],
            mode=mode,
            discretize_continuous=True,
            random_state=42,
        )
        logger.info("LIMEEngine initialized", extra={"n_features": len(feature_names)})

    def explain(
        self,
        instance: np.ndarray,
        num_features: int = 8,
        num_samples: int = 1000,
    ) -> dict:
        """
        Generate a LIME explanation for a single transaction.

        Args:
            instance:     1-D feature array for the transaction.
            num_features: Number of features to include in the explanation.
            num_samples:  Number of perturbation samples for LIME.

        Returns:
            dict with keys:
                'feature_weights'  — {feature_name: lime_weight}
                'explanation_text' — list of human-readable strings per feature
                'predicted_class'  — 0 (Normal) or 1 (Anomaly)
                'prediction_score' — model's probability for predicted class
        """
        if instance.ndim == 2:
            instance = instance.flatten()

        exp = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self.predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=1,
        )

        # Extract weights for the Anomaly class (index 1)
        label_idx = 1  # Anomaly class
        feature_weights: dict[str, float] = {}
        explanation_text: list[str] = []

        try:
            lime_list = exp.as_list(label=label_idx)
        except KeyError:
            lime_list = exp.as_list()

        for feat_description, weight in lime_list:
            feature_weights[feat_description] = round(float(weight), 6)
            direction = "increases" if weight > 0 else "decreases"
            explanation_text.append(
                f"{feat_description} {direction} anomaly likelihood "
                f"by {abs(weight):.4f}"
            )

        predicted_class  = int(np.argmax(exp.predict_proba))
        prediction_score = round(float(exp.predict_proba[predicted_class]), 4)

        return {
            "feature_weights":  feature_weights,
            "explanation_text": explanation_text,
            "predicted_class":  predicted_class,
            "prediction_score": prediction_score,
        }
