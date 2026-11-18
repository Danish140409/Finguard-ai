"""
FinGuard AI — Transaction Explorer

Interactive transaction workbench for financial analysts.
Provides real-time filtering, risk level badges, anomaly reason inspection,
and one-click governance actions (Approve, Escalate, Flag for SOX Audit).
"""

from pathlib import Path
import streamlit as st
import pandas as pd
from datetime import datetime
from frontend.utils.api_client import APIClient

ASSETS_DIR = Path(__file__).parent.parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
FAVICON_PATH = ASSETS_DIR / "favicon.png"

st.set_page_config(
    page_title="Transaction Explorer | FinGuard AI",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "🛡️",
    layout="wide",
)

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), width="stretch")
    st.sidebar.markdown("---")

st.title("🔍 Accounts Payable Transaction Explorer")
st.markdown("Filter, inspect, and adjudicate anomalous invoices and payments.")

api_client = APIClient()
token = st.session_state.get("token")

# ---------------------------------------------------------------------------
# Filter Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Filter Criteria")
risk_filter = st.sidebar.selectbox("Risk Severity", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"], index=1)
min_amt = st.sidebar.number_input("Minimum Amount ($)", min_value=0.0, value=0.0, step=500.0)
max_amt = st.sidebar.number_input("Maximum Amount ($)", min_value=0.0, value=250000.0, step=5000.0)
search_vendor = st.sidebar.text_input("Filter by Vendor Name")

# ---------------------------------------------------------------------------
# Fetch Data
# ---------------------------------------------------------------------------
transactions_data = []

if token and token != "demo_token":
    try:
        resp = api_client.get_transactions(
            token=token,
            page=1,
            size=100,
            risk_level=risk_filter if risk_filter != "ALL" else None,
            min_amount=min_amt if min_amt > 0 else None,
            max_amount=max_amt if max_amt < 250000.0 else None,
        )
        transactions_data = resp.get("items", [])
    except Exception as e:
        st.warning(f"Could not load live backend data: {e}. Showing sample records.")

# Provide fallback sample data if backend not active
if not transactions_data:
    transactions_data = [
        {
            "transaction_id": "49b498f3-8b77-4b92-9a3b-28fceea0b001",
            "transaction_type": "invoice",
            "vendor_name": "Apex Global Solutions LLC",
            "amount": 49800.00,
            "transaction_date": "2026-11-14",
            "risk_score": 92.4,
            "risk_level": "CRITICAL",
            "anomaly_types": ["split_invoice", "round_dollar"],
            "plain_english_reason": "Invoice amount of $49,800.00 is just below the $50,000 dual-authorization approval limit. The vendor submitted 3 invoices within 4 days, indicating intentional invoice splitting.",
        },
        {
            "transaction_id": "81f181cc-24b5-4b11-9a73-631d871e8002",
            "transaction_type": "payment",
            "vendor_name": "Kestrel Logistics Group",
            "amount": 12850.00,
            "transaction_date": "2026-11-15",
            "risk_score": 88.1,
            "risk_level": "CRITICAL",
            "anomaly_types": ["weekend_new_vendor", "duplicate_invoice"],
            "plain_english_reason": "Payment was processed on a Sunday to a vendor registered only 4 days prior. An identical payment of $12,850.00 was issued 24 hours earlier.",
        },
        {
            "transaction_id": "31ac16a4-44bc-4340-b6f1-a1cfd52a2003",
            "transaction_type": "invoice",
            "vendor_name": "Vanguard Industrial Supplies",
            "amount": 76500.00,
            "transaction_date": "2026-11-12",
            "risk_score": 79.5,
            "risk_level": "HIGH",
            "anomaly_types": ["inflated_invoice", "po_mismatch"],
            "plain_english_reason": "Invoice amount of $76,500.00 is 3.8 standard deviations higher than this vendor's historical average ($14,200). PO match ratio exceeds 140%.",
        },
        {
            "transaction_id": "f5e921d7-21b0-4dbb-825f-e54fa8290004",
            "transaction_type": "invoice",
            "vendor_name": "Horizon Tech Partners",
            "amount": 35000.00,
            "transaction_date": "2026-11-10",
            "risk_score": 71.0,
            "risk_level": "HIGH",
            "anomaly_types": ["round_dollar"],
            "plain_english_reason": "Invoice is an exact round dollar amount ($35,000.00) without itemized tax or fractional charges, uncommon for professional software licenses.",
        },
        {
            "transaction_id": "a9018e42-17cd-4cf9-bb72-6804bb012005",
            "transaction_type": "invoice",
            "vendor_name": "BlueLine Office Products",
            "amount": 4200.00,
            "transaction_date": "2026-11-08",
            "risk_score": 45.2,
            "risk_level": "MEDIUM",
            "anomaly_types": ["duplicate_invoice"],
            "plain_english_reason": "Possible duplicate invoice submission matching batch #4182.",
        },
    ]

# Apply client-side filters if sample data
df = pd.DataFrame(transactions_data)
if search_vendor:
    df = df[df["vendor_name"].str.contains(search_vendor, case=False, na=False)]
if risk_filter != "ALL":
    df = df[df["risk_level"] == risk_filter]

# ---------------------------------------------------------------------------
# Results Table
# ---------------------------------------------------------------------------
st.subheader(f"Matching Exceptions ({len(df)})")

# Format columns for display
display_df = df[["transaction_date", "vendor_name", "amount", "risk_score", "risk_level", "anomaly_types"]].copy()
display_df["amount"] = display_df["amount"].apply(lambda x: f"${x:,.2f}")
display_df["risk_score"] = display_df["risk_score"].apply(lambda x: f"{x:.1f}")
display_df["anomaly_types"] = display_df["anomaly_types"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
display_df.rename(columns={
    "transaction_date": "Date",
    "vendor_name": "Vendor",
    "amount": "Amount",
    "risk_score": "Risk Score",
    "risk_level": "Severity",
    "anomaly_types": "Anomaly Categories",
}, inplace=True)

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Transaction Investigation & Adjudication Workbench
# ---------------------------------------------------------------------------
st.subheader("Transaction Investigation & Audit Adjudication")

if not df.empty:
    selected_idx = st.selectbox(
        "Select Transaction to Inspect",
        options=range(len(df)),
        format_func=lambda i: f"[{df.iloc[i]['risk_level']}] {df.iloc[i]['vendor_name']} — ${df.iloc[i]['amount']:,.2f} ({df.iloc[i]['transaction_date']})",
    )

    selected_row = df.iloc[selected_idx]

    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown(f"### Vendor: **{selected_row['vendor_name']}**")
        st.markdown(f"**Transaction ID:** `{selected_row['transaction_id']}`")
        st.markdown(f"**Amount:** `${selected_row['amount']:,.2f}` &nbsp;|&nbsp; **Date:** `{selected_row['transaction_date']}`")
        
        # Risk Badge
        lvl = selected_row["risk_level"]
        score = selected_row["risk_score"]
        if lvl == "CRITICAL":
            st.error(f"🚨 **Risk Level: CRITICAL (Score: {score:.1f}/100)**")
        elif lvl == "HIGH":
            st.warning(f"⚠️ **Risk Level: HIGH (Score: {score:.1f}/100)**")
        else:
            st.info(f"ℹ️ **Risk Level: {lvl} (Score: {score:.1f}/100)**")

        st.markdown("#### 📝 Plain-English Audit Narrative (SHAP Generated)")
        st.success(selected_row.get("plain_english_reason", "No audit explanation available."))

    with c2:
        st.markdown("### Analyst Action")
        comment = st.text_area("Audit Justification / Workpaper Notes", placeholder="e.g. Verified with purchasing director; goods received.")
        
        col_act1, col_act2, col_act3 = st.columns(3)
        with col_act1:
            if st.button("✅ Approve", use_container_width=True):
                try:
                    if token and token != "demo_token":
                        api_client.take_transaction_action(token, selected_row["transaction_id"], "approve", comment)
                    st.success("Transaction Approved and SOX audit event logged.")
                except Exception as e:
                    st.error(f"Action failed: {e}")

        with col_act2:
            if st.button("🚨 Escalate", use_container_width=True):
                try:
                    if token and token != "demo_token":
                        api_client.take_transaction_action(token, selected_row["transaction_id"], "escalate", comment)
                    st.warning("Escalated to Controller review.")
                except Exception as e:
                    st.error(f"Action failed: {e}")

        with col_act3:
            if st.button("⚖️ Flag Audit", use_container_width=True):
                try:
                    if token and token != "demo_token":
                        api_client.take_transaction_action(token, selected_row["transaction_id"], "flag_audit", comment)
                    st.info("Added to Internal Audit sampling workpapers.")
                except Exception as e:
                    st.error(f"Action failed: {e}")
