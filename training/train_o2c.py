"""
FinGuard AI — O2C Model Training Pipeline

Trains ensemble anomaly detection models for O2C Credit Management:
1. Isolation Forest
2. Local Outlier Factor (LOF)
3. One-Class SVM (OCSVM)
4. XGBoost Classifier

Saves production artifacts to models/production/o2c/.
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

from features.o2c.customer_features import extract_customer_features
from features.o2c.credit_features import extract_credit_features
from features.o2c.ar_features import extract_ar_features
from training.generate_o2c_data import generate_o2c_data
from utils.logger import get_logger

logger = get_logger(__name__)

PROD_DIR = Path(__file__).parent.parent / "models" / "production" / "o2c"
PROD_DIR.mkdir(parents=True, exist_ok=True)


def train_o2c_ensemble():
    logger.info("Starting O2C Model Training Pipeline...")
    data_dir = Path(__file__).parent.parent / "data" / "o2c"
    if not (data_dir / "customers.csv").exists():
        generate_o2c_data(str(data_dir))

    customers = pd.read_csv(data_dir / "customers.csv")
    limits = pd.read_csv(data_dir / "credit_limits.csv")

    # Combine basic customer profile features
    df = customers.merge(limits[["customer_id", "approved_limit", "previous_limit"]], on="customer_id", how="left")
    df["total_exposure"] = df["approved_limit"] * np.random.uniform(0.1, 1.1, len(df))
    df["avg_days_past_due"] = np.random.exponential(15.0, len(df))
    df["pct_invoices_late"] = np.clip(np.random.normal(20.0, 15.0, len(df)), 0.0, 100.0)

    # Extract features
    cust_feats = extract_customer_features(df)
    credit_feats = extract_credit_features(df)
    ar_feats = extract_ar_features(df)

    X_df = pd.concat([cust_feats, credit_feats, ar_feats], axis=1)
    feature_names = list(X_df.columns)
    X = X_df.values

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Ground truth / pseudo labels for XGBoost
    # Anomalous if high utilisation, large increase, or late payments
    y = ((X_df["credit_utilisation_pct"] > 95.0) | (X_df["credit_limit_change_pct"] > 50.0) | (X_df["avg_days_past_due"] > 60.0)).astype(int).values

    logger.info(f"Training models on {len(X)} records across {len(feature_names)} features...")

    # 1. Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
    iso_forest.fit(X_scaled)

    # 2. Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=15, contamination=0.15, novelty=True)
    lof.fit(X_scaled)

    # 3. One-Class SVM
    ocsvm = OneClassSVM(kernel="rbf", nu=0.15, gamma="scale")
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
        "module": "O2C",
        "version": "1.0.0",
        "n_samples": len(X),
        "feature_names": feature_names,
        "models": ["isolation_forest", "lof", "ocsvm", "xgboost"],
        "anomaly_rate": float(y.mean()),
    }
    with open(PROD_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"O2C model artifacts successfully saved to {PROD_DIR}")
    print("O2C models trained and saved successfully.")


if __name__ == "__main__":
    train_o2c_ensemble()
