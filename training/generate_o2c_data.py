"""
FinGuard AI — Synthetic O2C (Order-to-Cash) Data Generator

Generates realistic customer master records, credit limits, sales orders,
AR invoices, customer payments, and credit events with injected anomalies.
"""

import argparse
import random
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from faker import Faker

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)
fake = Faker("en_IN")
Faker.seed(42)
random.seed(42)
np.random.seed(42)

N_CUSTOMERS = 150
START_DATE = date(2023, 1, 1)
END_DATE = date(2024, 12, 31)


def random_date(start: date = START_DATE, end: date = END_DATE) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, max(1, delta)))


def generate_o2c_data(output_dir: str = "data/o2c") -> Dict[str, pd.DataFrame]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Customers
    cust_rows = []
    risk_cats = ["Low"] * 60 + ["Medium"] * 50 + ["High"] * 30 + ["Critical"] * 10
    random.shuffle(risk_cats)

    for i in range(N_CUSTOMERS):
        cid = str(uuid.uuid4())
        cname = fake.company()
        ctype = "Organisation" if random.random() > 0.15 else "Individual"
        reg_date = random_date(date(2020, 1, 1), date(2023, 6, 1))

        cust_rows.append({
            "customer_id": cid,
            "customer_name": cname,
            "customer_type": ctype,
            "customer_status": "Active",
            "company_code": "1000",
            "sales_org": "SO01",
            "created_at": reg_date,
            "risk_category": risk_cats[i % len(risk_cats)],
        })
    customers_df = pd.DataFrame(cust_rows)

    # 2. Credit Limits
    limit_rows = []
    customer_limits = {}
    for _, c in customers_df.iterrows():
        base_limit = random.choice([500000.0, 1000000.0, 2000000.0, 5000000.0])
        customer_limits[c["customer_id"]] = base_limit

        limit_rows.append({
            "credit_limit_id": str(uuid.uuid4()),
            "customer_id": c["customer_id"],
            "approved_limit": base_limit,
            "previous_limit": base_limit * 0.8,
            "currency": "INR",
            "change_reason": "Annual credit review",
            "changed_by": "credit_mgr",
            "approval_status": "approved",
            "effective_date": c["created_at"],
        })
    credit_limits_df = pd.DataFrame(limit_rows)

    # 3. Sales Orders
    order_rows = []
    for i in range(1200):
        c = customers_df.iloc[i % N_CUSTOMERS]
        cid = c["customer_id"]
        limit = customer_limits[cid]
        val = round(random.uniform(25000.0, limit * 0.4), 2)

        # Inject anomalies: ~5% exceed limit or are blocked
        status = "created"
        block = "none"
        chk_status = "approved"
        if random.random() < 0.08:
            val = round(limit * random.uniform(1.1, 1.8), 2)
            block = "blocked"
            chk_status = "credit_hold"

        order_rows.append({
            "sales_order_id": str(uuid.uuid4()),
            "customer_id": cid,
            "order_number": f"SO-{10000 + i}",
            "order_date": random_date(),
            "order_value": val,
            "currency": "INR",
            "order_status": status,
            "credit_check_status": chk_status,
            "credit_block_status": block,
            "delivery_status": "delivered" if block == "none" else "pending",
        })
    orders_df = pd.DataFrame(order_rows)

    # 4. AR Invoices
    inv_rows = []
    for i in range(1000):
        c = customers_df.iloc[i % N_CUSTOMERS]
        cid = c["customer_id"]
        amt = round(random.uniform(30000.0, 500000.0), 2)
        inv_date = random_date(date(2023, 1, 1), date(2024, 11, 1))
        due_date = inv_date + timedelta(days=30)

        # Inject ~15% overdue
        is_overdue = random.random() < 0.15
        pay_status = "overdue" if (is_overdue and due_date < date(2024, 11, 18)) else "paid"
        out_amt = amt if pay_status == "overdue" else 0.0

        inv_rows.append({
            "invoice_id": str(uuid.uuid4()),
            "customer_id": cid,
            "invoice_number": f"INV-{20000 + i}",
            "invoice_date": inv_date,
            "due_date": due_date,
            "invoice_amount": amt,
            "outstanding_amount": out_amt,
            "currency": "INR",
            "payment_status": pay_status,
        })
    invoices_df = pd.DataFrame(inv_rows)

    # Save CSVs
    customers_df.to_csv(out_path / "customers.csv", index=False)
    credit_limits_df.to_csv(out_path / "credit_limits.csv", index=False)
    orders_df.to_csv(out_path / "sales_orders.csv", index=False)
    invoices_df.to_csv(out_path / "ar_invoices.csv", index=False)

    logger.info(f"O2C synthetic data generated in {output_dir}")
    return {
        "customers": customers_df,
        "credit_limits": credit_limits_df,
        "sales_orders": orders_df,
        "ar_invoices": invoices_df,
    }


if __name__ == "__main__":
    generate_o2c_data()
    print("O2C synthetic data generated successfully.")
