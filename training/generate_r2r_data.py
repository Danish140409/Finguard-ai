"""
FinGuard AI — Synthetic R2R (Record-to-Report) Data Generator

Generates GL accounts, journal headers, journal lines, account reconciliations,
and intercompany transactions with injected accounting anomalies.
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
fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

N_JOURNALS = 1000


def generate_r2r_data(output_dir: str = "data/r2r") -> Dict[str, pd.DataFrame]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Company Codes
    companies_df = pd.DataFrame([
        {"company_code": "1000", "company_name": "FinGuard US Corp", "country": "US", "base_currency": "USD", "company_status": "Active"},
        {"company_code": "2000", "company_name": "FinGuard UK Ltd", "country": "GB", "base_currency": "GBP", "company_status": "Active"},
        {"company_code": "3000", "company_name": "FinGuard India Pvt", "country": "IN", "base_currency": "INR", "company_status": "Active"},
    ])

    # 2. GL Accounts
    gl_rows = [
        {"gl_account_id": str(uuid.uuid4()), "gl_account_number": "101000", "gl_account_name": "Cash & Bank", "account_type": "Asset", "company_code": "1000", "account_group": "Cash"},
        {"gl_account_id": str(uuid.uuid4()), "gl_account_number": "120000", "gl_account_name": "Accounts Receivable", "account_type": "Asset", "company_code": "1000", "account_group": "Receivables"},
        {"gl_account_id": str(uuid.uuid4()), "gl_account_number": "200000", "gl_account_name": "Accounts Payable", "account_type": "Liability", "company_code": "1000", "account_group": "Payables"},
        {"gl_account_id": str(uuid.uuid4()), "gl_account_number": "400000", "gl_account_name": "Operating Revenue", "account_type": "Revenue", "company_code": "1000", "account_group": "Revenue"},
        {"gl_account_id": str(uuid.uuid4()), "gl_account_number": "500000", "gl_account_name": "Cost of Goods Sold", "account_type": "Expense", "company_code": "1000", "account_group": "COGS"},
        {"gl_account_id": str(uuid.uuid4()), "gl_account_number": "600000", "gl_account_name": "Operating Expenses", "account_type": "Expense", "company_code": "1000", "account_group": "OpEx"},
    ]
    gl_df = pd.DataFrame(gl_rows)

    # 3. Journal Headers & Lines
    j_rows = []
    line_rows = []

    j_types = ["Standard"] * 60 + ["Manual"] * 25 + ["Accrual"] * 10 + ["Reversal"] * 5
    for i in range(N_JOURNALS):
        jid = str(uuid.uuid4())
        doc_id = f"DOC-{100000 + i}"
        jtype = random.choice(j_types)
        post_date = date(2024, 1, 1) + timedelta(days=random.randint(0, 320))
        amt = round(random.uniform(5000.0, 250000.0), 2)

        # Inject anomalies: ~5% unbalanced, ~5% huge unapproved manual
        is_balanced = True
        status = "approved"
        if random.random() < 0.05:
            is_balanced = False  # Out of balance anomaly
        if jtype == "Manual" and random.random() < 0.15:
            amt = round(random.uniform(600000.0, 2000000.0), 2)
            status = "unapproved"

        j_rows.append({
            "journal_id": jid,
            "document_id": doc_id,
            "company_code": "1000",
            "posting_date": post_date,
            "document_date": post_date,
            "fiscal_period": (post_date.month),
            "journal_type": jtype,
            "currency": "USD",
            "posting_user": random.choice(["j_smith", "a_kumar", "system", "c_finance"]),
            "approval_status": status,
            "amount": amt,
            "total_debit": amt,
            "total_credit": amt if is_balanced else amt * 0.9,
            "is_anomaly": not is_balanced or status == "unapproved",
        })

        # Lines (Debit & Credit)
        line_rows.append({
            "line_id": str(uuid.uuid4()),
            "journal_id": jid,
            "gl_account_id": gl_rows[0]["gl_account_id"],
            "debit_amount": amt,
            "credit_amount": 0.0,
            "currency": "USD",
        })
        line_rows.append({
            "line_id": str(uuid.uuid4()),
            "journal_id": jid,
            "gl_account_id": gl_rows[1]["gl_account_id"],
            "debit_amount": 0.0,
            "credit_amount": amt if is_balanced else amt * 0.9,
            "currency": "USD",
        })

    journals_df = pd.DataFrame(j_rows)
    lines_df = pd.DataFrame(line_rows)

    # 4. Account Reconciliations
    recon_rows = []
    for i in range(50):
        src_bal = round(random.uniform(100000.0, 1000000.0), 2)
        diff = 0.0
        r_status = "reconciled"
        if random.random() < 0.20:
            diff = round(random.uniform(500.0, 25000.0), 2)
            r_status = "exception"

        recon_rows.append({
            "reconciliation_id": str(uuid.uuid4()),
            "company_code": "1000",
            "gl_account_id": gl_rows[0]["gl_account_id"],
            "reconciliation_date": date(2024, 10, 31),
            "source_balance": src_bal,
            "gl_balance": src_bal - diff,
            "difference_amount": diff,
            "reconciliation_status": r_status,
            "reviewer_id": "auditor_1",
        })
    recon_df = pd.DataFrame(recon_rows)

    # Save CSVs
    companies_df.to_csv(out_path / "company_codes.csv", index=False)
    gl_df.to_csv(out_path / "gl_accounts.csv", index=False)
    journals_df.to_csv(out_path / "journal_headers.csv", index=False)
    lines_df.to_csv(out_path / "journal_lines.csv", index=False)
    recon_df.to_csv(out_path / "account_reconciliations.csv", index=False)

    logger.info(f"R2R synthetic data generated in {output_dir}")
    return {
        "company_codes": companies_df,
        "gl_accounts": gl_df,
        "journal_headers": journals_df,
        "journal_lines": lines_df,
        "account_reconciliations": recon_df,
    }


if __name__ == "__main__":
    generate_r2r_data()
    print("R2R synthetic data generated successfully.")
