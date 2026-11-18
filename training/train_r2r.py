"""
FinGuard AI — R2R Model Training Pipeline

Trains ensemble anomaly detection models for Record-to-Report accounting controls:
1. Isolation Forest
2. Local Outlier Factor (LOF)
3. One-Class SVM (OCSVM)
4. XGBoost Classifier

Saves production artifacts to models/production/r2r/.
"""

import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.r2r.journal_features import extract_journal_features
from training.generate_r2r_data import generate_r2r_data
from utils.logger import get_logger

logger = get_logger(__name__)

PROD_DIR = Path(__file__).parent.parent / "models" / "production" / "r2r"
PROD_DIR.mkdir(parents=True, exist_ok=True)


def train_r2r_ensemble():
    logger.info("Starting R2R Model Training Pipeline...")
    data_dir = Path(__file__).parent.parent / "data" / "r2r"
    if not (data_dir / "journal_headers.csv").exists():
        generate_r2r_data(str(data_dir))

    journals = pd.read_csv(data_dir / "journal_headers.csv")

    # Extract features
    X_df = extract_journal_features(journals)
    feature_names = list(X_df.columns)
    X = X_df.values

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Ground truth labels
    y = journals.get("is_anomaly", (np.abs(journals["total_debit"] - journals["total_credit"]) > 0.01)).astype(int).values

    logger.info(f"Training R2R models on {len(X)} records across {len(feature_names)} features...")

    # 1. Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.10, random_state=42)
    iso_forest.fit(X_scaled)

    # 2. Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=15, contamination=0.10, novelty=True)
    lof.fit(X_scaled)

    # 3. One-Class SVM
    ocsvm = OneClassSVM(kernel="rbf", nu=0.10, gamma="scale")
    ocsvm.fit(X_scaled)

    # 4. XGBoost Classifier
    xgb = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="logloss")
    xgb.fit(X_scaled, y)

    # Save artifacts
    joblib.dump(iso_forest, PROD_DIR / "isolation_forest.joblib")
    joblib.dump(lof, PROD_DIR / "lof.joblib")
    joblib.dump(ocsvm, PROD_DIR / "ocsvm.joblib")
    joblib.dump(xgb, PROD_DIR / "xgboost.joblib")
    joblib.dump(scaler, PROD_DIR / "scaler.joblib")

    metadata = {
        "module": "R2R",
        "version": "1.0.0",
        "n_samples": len(X),
        "feature_names": feature_names,
        "models": ["isolation_forest", "lof", "ocsvm", "xgboost"],
        "anomaly_rate": float(y.mean()),
    }
    with open(PROD_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"R2R model artifacts successfully saved to {PROD_DIR}")
    print("R2R models trained and saved successfully.")


if __name__ == "__main__":
    train_r2r_ensemble()
