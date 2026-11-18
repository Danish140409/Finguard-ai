"""
FinGuard AI — Synthetic ERP Data Generator

Generates realistic Accounts Payable data with injected, labeled anomalies.
Saves data to PostgreSQL and a CSV backup in data/.

Usage:
    python training/generate_data.py
    python training/generate_data.py --rows 50000 --output-only-csv
"""

import argparse
import hashlib
import os
import random
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)
fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_VENDORS       = 200
N_INVOICES      = 10_000
N_PAYMENTS      = 9_500   # ~95% of invoices get paid
N_JOURNAL_ENTRIES = 500

ANOMALY_COUNTS = {
    "duplicate_invoice":    300,
    "round_dollar":         200,
    "weekend_new_vendor":   150,
    "split_invoice":        100,
    "inflated_invoice":     100,
    "fictitious_vendor":    100,
    "po_mismatch":           50,
}

APPROVAL_THRESHOLD = 5_000.0  # split invoices stay below this

START_DATE = date(2023, 1, 1)
END_DATE   = date(2024, 12, 31)


def random_date(start: date = START_DATE, end: date = END_DATE) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def hash_account(account_number: str) -> str:
    return hashlib.sha256(account_number.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1. Vendor Generation
# ---------------------------------------------------------------------------

def generate_vendors() -> pd.DataFrame:
    logger.info("Generating vendors", extra={"count": N_VENDORS})
    rows = []
    for i in range(N_VENDORS):
        if i < 20:                        # 10% fictitious
            risk_tier = "fictitious"
            reg_date  = random_date(date(2023, 10, 1), END_DATE)  # recently registered
        elif i < 60:                      # 20% high-risk
            risk_tier = "high_risk"
            reg_date  = random_date()
        else:
            risk_tier = "standard"
            reg_date  = random_date(date(2018, 1, 1), date(2022, 12, 31))

        rows.append({
            "vendor_id":          str(uuid.uuid4()),
            "vendor_name":        fake.company(),
            "registration_date":  reg_date,
            "country":            random.choice(["US"] * 8 + ["CA", "GB", "MX", "CN", "IN"]),
            "risk_tier":          risk_tier,
            "tax_id":             fake.ssn(),
            "bank_account_hash":  hash_account(fake.bban()),
            "contact_email":      fake.company_email(),
            "is_active":          True,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Purchase Order Generation
# ---------------------------------------------------------------------------

def generate_purchase_orders(vendors_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating purchase orders")
    rows = []
    standard_vendors = vendors_df[vendors_df["risk_tier"] == "standard"]["vendor_id"].tolist()
    n_pos = int(N_INVOICES * 0.7)  # 70% of invoices have a matching PO
    for _ in range(n_pos):
        vendor_id = random.choice(standard_vendors)
        po_amount = round(random.uniform(500, 50_000), 2)
        rows.append({
            "po_id":       str(uuid.uuid4()),
            "vendor_id":   vendor_id,
            "po_number":   f"PO-{fake.numerify('######')}",
            "po_amount":   po_amount,
            "po_date":     random_date(START_DATE, date(2024, 6, 1)),
            "approved_by": fake.name(),
            "status":      "open",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Invoice Generation (normal)
# ---------------------------------------------------------------------------

def _vendor_typical_amounts(vendors_df: pd.DataFrame) -> dict[str, float]:
    """Assign a 'typical invoice amount' per vendor for z-score calculations."""
    typical = {}
    for _, v in vendors_df.iterrows():
        if v["risk_tier"] == "fictitious":
            typical[v["vendor_id"]] = random.uniform(2_000, 10_000)
        elif v["risk_tier"] == "high_risk":
            typical[v["vendor_id"]] = random.uniform(1_000, 25_000)
        else:
            typical[v["vendor_id"]] = random.uniform(500, 15_000)
    return typical


def generate_invoices(
    vendors_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    typical_amounts: dict[str, float],
) -> pd.DataFrame:
    logger.info("Generating normal invoices", extra={"count": N_INVOICES})
    rows = []
    vendor_ids = vendors_df["vendor_id"].tolist()
    po_map: dict[str, list] = {}  # vendor_id -> list of po dicts
    for _, po in pos_df.iterrows():
        po_map.setdefault(po["vendor_id"], []).append(po)

    for _ in range(N_INVOICES):
        vendor_id = random.choice(vendor_ids)
        typical   = typical_amounts.get(vendor_id, 5_000)
        amount    = round(abs(np.random.normal(typical, typical * 0.3)), 2)
        amount    = max(50.0, amount)
        inv_date  = random_date()
        due_days  = random.choice([15, 30, 45, 60])

        # Try to link a PO
        po_id     = None
        po_number = None
        if vendor_id in po_map and random.random() < 0.7:
            po        = random.choice(po_map[vendor_id])
            po_id     = po["po_id"]
            po_number = po["po_number"]

        rows.append({
            "invoice_id":         str(uuid.uuid4()),
            "vendor_id":          vendor_id,
            "po_id":              po_id,
            "amount":             amount,
            "currency":           "USD",
            "invoice_date":       inv_date,
            "due_date":           inv_date + timedelta(days=due_days),
            "po_number":          po_number,
            "payment_terms_days": due_days,
            "description":        fake.bs(),
            "status":             "pending",
            "is_anomaly":         False,
            "anomaly_type":       None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Inject Anomalies
# ---------------------------------------------------------------------------

def inject_anomalies(
    invoices_df: pd.DataFrame,
    vendors_df: pd.DataFrame,
    typical_amounts: dict[str, float],
) -> pd.DataFrame:
    logger.info("Injecting anomalies", extra={"types": list(ANOMALY_COUNTS.keys())})
    df = invoices_df.copy()

    fictitious_vendors = vendors_df[vendors_df["risk_tier"] == "fictitious"]["vendor_id"].tolist()
    new_vendors = vendors_df[
        vendors_df["registration_date"] >= date(2024, 6, 1)
    ]["vendor_id"].tolist()
    all_vendor_ids = vendors_df["vendor_id"].tolist()

    # --- Duplicate Invoices ---
    normal_rows = df[~df["is_anomaly"]].sample(ANOMALY_COUNTS["duplicate_invoice"], random_state=1)
    new_rows = []
    for _, row in normal_rows.iterrows():
        dup = row.copy()
        dup["invoice_id"]   = str(uuid.uuid4())
        dup["invoice_date"] = row["invoice_date"] + timedelta(days=random.randint(1, 20))
        dup["due_date"]     = dup["invoice_date"] + timedelta(days=int(row["payment_terms_days"] or 30))
        dup["is_anomaly"]   = True
        dup["anomaly_type"] = "duplicate_invoice"
        new_rows.append(dup)
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # --- Round Dollar Payments ---
    round_amounts = [500, 1000, 2000, 5000, 10000, 25000, 50000]
    for _ in range(ANOMALY_COUNTS["round_dollar"]):
        vendor_id = random.choice(all_vendor_ids)
        inv_date  = random_date()
        new_rows.append({
            "invoice_id":    str(uuid.uuid4()),
            "vendor_id":     vendor_id,
            "po_id":         None,
            "amount":        float(random.choice(round_amounts)),
            "currency":      "USD",
            "invoice_date":  inv_date,
            "due_date":      inv_date + timedelta(days=30),
            "po_number":     None,
            "payment_terms_days": 30,
            "description":   fake.bs(),
            "status":        "pending",
            "is_anomaly":    True,
            "anomaly_type":  "round_dollar",
        })
    df = pd.concat([df, pd.DataFrame(new_rows[-ANOMALY_COUNTS["round_dollar"]:])], ignore_index=True)

    # --- Weekend Payments to New Vendors ---
    new_rows = []
    weekend_dates = [d for d in (START_DATE + timedelta(n) for n in range((END_DATE - START_DATE).days))
                     if d.weekday() in (5, 6)]
    vendors_for_weekend = new_vendors or fictitious_vendors or all_vendor_ids
    for _ in range(ANOMALY_COUNTS["weekend_new_vendor"]):
        vendor_id = random.choice(vendors_for_weekend)
        inv_date  = random.choice(weekend_dates)
        new_rows.append({
            "invoice_id":    str(uuid.uuid4()),
            "vendor_id":     vendor_id,
            "po_id":         None,
            "amount":        round(random.uniform(200, 8_000), 2),
            "currency":      "USD",
            "invoice_date":  inv_date,
            "due_date":      inv_date + timedelta(days=30),
            "po_number":     None,
            "payment_terms_days": 30,
            "description":   fake.bs(),
            "status":        "pending",
            "is_anomaly":    True,
            "anomaly_type":  "weekend_new_vendor",
        })
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # --- Split Invoices (just below approval threshold) ---
    new_rows = []
    for _ in range(ANOMALY_COUNTS["split_invoice"]):
        vendor_id  = random.choice(all_vendor_ids)
        base_date  = random_date()
        n_splits   = random.randint(3, 5)
        total_intended = random.uniform(APPROVAL_THRESHOLD * n_splits * 0.8,
                                        APPROVAL_THRESHOLD * n_splits * 1.2)
        for j in range(n_splits):
            split_amount = round(APPROVAL_THRESHOLD - random.uniform(50, 400), 2)
            new_rows.append({
                "invoice_id":    str(uuid.uuid4()),
                "vendor_id":     vendor_id,
                "po_id":         None,
                "amount":        split_amount,
                "currency":      "USD",
                "invoice_date":  base_date + timedelta(days=j * 2),
                "due_date":      base_date + timedelta(days=30 + j * 2),
                "po_number":     None,
                "payment_terms_days": 30,
                "description":   fake.bs(),
                "status":        "pending",
                "is_anomaly":    True,
                "anomaly_type":  "split_invoice",
            })
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # --- Inflated Invoices ---
    new_rows = []
    for _ in range(ANOMALY_COUNTS["inflated_invoice"]):
        vendor_id  = random.choice(all_vendor_ids)
        typical    = typical_amounts.get(vendor_id, 5_000)
        inv_date   = random_date()
        new_rows.append({
            "invoice_id":    str(uuid.uuid4()),
            "vendor_id":     vendor_id,
            "po_id":         None,
            "amount":        round(typical * random.uniform(3.5, 8.0), 2),
            "currency":      "USD",
            "invoice_date":  inv_date,
            "due_date":      inv_date + timedelta(days=30),
            "po_number":     None,
            "payment_terms_days": 30,
            "description":   fake.bs(),
            "status":        "pending",
            "is_anomaly":    True,
            "anomaly_type":  "inflated_invoice",
        })
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # --- Fictitious Vendor Payments ---
    new_rows = []
    for _ in range(ANOMALY_COUNTS["fictitious_vendor"]):
        vendor_id = random.choice(fictitious_vendors) if fictitious_vendors else random.choice(all_vendor_ids)
        inv_date  = random_date()
        new_rows.append({
            "invoice_id":    str(uuid.uuid4()),
            "vendor_id":     vendor_id,
            "po_id":         None,
            "amount":        round(random.uniform(1_000, 20_000), 2),
            "currency":      "USD",
            "invoice_date":  inv_date,
            "due_date":      inv_date + timedelta(days=30),
            "po_number":     None,
            "payment_terms_days": 30,
            "description":   fake.bs(),
            "status":        "pending",
            "is_anomaly":    True,
            "anomaly_type":  "fictitious_vendor",
        })
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # --- PO Mismatches ---
    new_rows = []
    for _ in range(ANOMALY_COUNTS["po_mismatch"]):
        vendor_id  = random.choice(all_vendor_ids)
        inv_date   = random_date()
        po_amount  = round(random.uniform(500, 10_000), 2)
        inv_amount = round(po_amount * random.uniform(1.3, 2.5), 2)  # significantly over PO
        new_rows.append({
            "invoice_id":    str(uuid.uuid4()),
            "vendor_id":     vendor_id,
            "po_id":         None,
            "amount":        inv_amount,
            "currency":      "USD",
            "invoice_date":  inv_date,
            "due_date":      inv_date + timedelta(days=30),
            "po_number":     f"PO-MISMATCH-{fake.numerify('###')}",
            "payment_terms_days": 30,
            "description":   fake.bs(),
            "status":        "pending",
            "is_anomaly":    True,
            "anomaly_type":  "po_mismatch",
        })
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    df = df.reset_index(drop=True)
    total_anomalies = df["is_anomaly"].sum()
    logger.info("Anomaly injection complete",
                extra={"total_anomalies": int(total_anomalies), "total_records": len(df)})
    return df


# ---------------------------------------------------------------------------
# 5. Payment Generation
# ---------------------------------------------------------------------------

def generate_payments(invoices_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating payments", extra={"count": N_PAYMENTS})
    paid_invoices = invoices_df.sample(min(N_PAYMENTS, len(invoices_df)), random_state=42)
    rows = []
    for _, inv in paid_invoices.iterrows():
        inv_date     = pd.to_datetime(inv["invoice_date"]).date()
        payment_lag  = random.randint(-5, int(inv["payment_terms_days"] or 30) + 15)
        payment_date = inv_date + timedelta(days=max(0, payment_lag))
        rows.append({
            "payment_id":         str(uuid.uuid4()),
            "invoice_id":         inv["invoice_id"],
            "vendor_id":          inv["vendor_id"],
            "amount_paid":        float(inv["amount"]),
            "payment_date":       payment_date,
            "payment_method":     random.choice(["ACH"] * 6 + ["wire", "check", "card", "manual"]),
            "bank_account_hash":  hash_account(fake.bban()),
            "is_manual_override": random.random() < 0.05,
            "is_anomaly":         bool(inv["is_anomaly"]),
            "anomaly_type":       inv["anomaly_type"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Journal Entries
# ---------------------------------------------------------------------------

def generate_journal_entries() -> pd.DataFrame:
    logger.info("Generating journal entries", extra={"count": N_JOURNAL_ENTRIES})
    accounts = ["1000", "2000", "3000", "5100", "5200", "6000", "7000"]
    rows = []
    for _ in range(N_JOURNAL_ENTRIES):
        rows.append({
            "entry_id":      str(uuid.uuid4()),
            "posting_date":  random_date(),
            "debit_account": random.choice(accounts),
            "credit_account": random.choice(accounts),
            "amount":        round(random.uniform(100, 50_000), 2),
            "description":   fake.bs(),
            "posted_by":     fake.name(),
            "is_reversed":   random.random() < 0.05,
            "is_anomaly":    False,
            "anomaly_type":  None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Save Data
# ---------------------------------------------------------------------------

def save_to_csv(
    vendors_df: pd.DataFrame,
    invoices_df: pd.DataFrame,
    payments_df: pd.DataFrame,
    output_dir: str = "data",
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    vendors_df.to_csv(f"{output_dir}/vendors.csv", index=False)
    invoices_df.to_csv(f"{output_dir}/invoices.csv", index=False)
    payments_df.to_csv(f"{output_dir}/payments.csv", index=False)

    # Combined transactions CSV for ML pipeline
    invoices_flat = invoices_df.copy()
    invoices_flat["transaction_date"] = invoices_flat["invoice_date"]
    invoices_flat["transaction_type"] = "invoice"

    payments_flat = payments_df.copy()
    payments_flat["transaction_date"] = payments_flat["payment_date"]
    payments_flat["transaction_type"] = "payment"
    payments_flat["amount"]           = payments_flat["amount_paid"]

    combined = pd.concat([invoices_flat, payments_flat], ignore_index=True)
    combined.to_csv(f"{output_dir}/synthetic_transactions.csv", index=False)
    logger.info("CSVs saved", extra={"output_dir": output_dir})


def save_to_db(
    vendors_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    invoices_df: pd.DataFrame,
    payments_df: pd.DataFrame,
    journal_entries_df: pd.DataFrame,
) -> None:
    from utils.db import get_engine
    engine = get_engine()
    logger.info("Saving data to PostgreSQL")

    vendors_df.to_sql("vendors",         engine, if_exists="append", index=False, method="multi", chunksize=500)
    pos_df.to_sql("purchase_orders",     engine, if_exists="append", index=False, method="multi", chunksize=500)
    invoices_df.to_sql("invoices",       engine, if_exists="append", index=False, method="multi", chunksize=500)
    payments_df.to_sql("payments",       engine, if_exists="append", index=False, method="multi", chunksize=500)
    journal_entries_df.to_sql("journal_entries", engine, if_exists="append", index=False, method="multi", chunksize=500)
    logger.info("Data saved to PostgreSQL successfully")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(csv_only: bool = False) -> None:
    logger.info("Starting synthetic data generation")

    vendors_df          = generate_vendors()
    typical_amounts     = _vendor_typical_amounts(vendors_df)
    pos_df              = generate_purchase_orders(vendors_df)
    invoices_df         = generate_invoices(vendors_df, pos_df, typical_amounts)
    invoices_df         = inject_anomalies(invoices_df, vendors_df, typical_amounts)
    payments_df         = generate_payments(invoices_df)
    journal_entries_df  = generate_journal_entries()

    save_to_csv(vendors_df, invoices_df, payments_df)

    if not csv_only:
        try:
            save_to_db(vendors_df, pos_df, invoices_df, payments_df, journal_entries_df)
        except Exception as e:
            logger.error("DB save failed (CSV backup available)", extra={"error": str(e)})

    logger.info("Data generation complete", extra={
        "vendors":         len(vendors_df),
        "purchase_orders": len(pos_df),
        "invoices":        len(invoices_df),
        "anomalies":       int(invoices_df["is_anomaly"].sum()),
        "payments":        len(payments_df),
        "journal_entries": len(journal_entries_df),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinGuard AI synthetic data generator")
    parser.add_argument("--csv-only", action="store_true",
                        help="Skip DB write, save CSV only")
    args = parser.parse_args()
    main(csv_only=args.csv_only)
