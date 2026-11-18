"""
FinGuard AI — Record-to-Report (R2R) Accounting Controls Dashboard

Provides accounting governance, period-end monitoring, and anomaly detection for:
  - General Ledger transaction activity
  - Journal entry debit-credit balancing and audit controls
  - Sub-ledger to GL account reconciliations
  - Accrual estimates, adjustments, and cross-period reversals
  - Intercompany matching and elimination differences
  - Period-end financial close workflow tracking
  - Interactive Journal Entry balance validator
"""

from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Paths & Branding
ASSETS_DIR = Path(__file__).parent.parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
FAVICON_PATH = ASSETS_DIR / "favicon.png"

st.set_page_config(
    page_title="Record-to-Report (R2R) | FinGuard AI",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    :root {
        --primary-blue: #0b4f8a;
        --electric-cyan: #00c6ff;
        --dark-navy: #08162b;
    }
    div[data-testid="stMetric"] {
        background: #111d2e !important;
        border: 1px solid #1e3a5f !important;
        border-top: 3px solid #0099ff !important;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width="stretch")
    else:
        st.title("📑 FinGuard AI")
        st.caption("Record-to-Report Module")
    st.markdown("---")
    st.markdown("### 🏢 Accounting Entity")
    selected_company = st.selectbox("Company Code", ["1000 — FinGuard US Corp", "2000 — FinGuard UK Ltd", "3000 — FinGuard India Pvt"])
    selected_period = st.selectbox("Fiscal Period", [f"Period {i:02d} / 2024" for i in range(1, 13)], index=11)
    st.markdown("---")
    st.caption("FinGuard AI v2.0.0 • Record-to-Report (R2R)")


@st.cache_data(ttl=60)
def load_r2r_demo_data():
    """Generate demo GL and journal data for visualization."""
    np.random.seed(42)
    j_types = ["Standard", "Manual", "Accrual", "Adjustment", "Reversal", "Intercompany"]
    journals = []
    for i in range(150):
        jtype = np.random.choice(j_types, p=[0.55, 0.20, 0.10, 0.08, 0.04, 0.03])
        amt = float(round(np.random.uniform(5000, 250000), 2))
        is_manual = jtype in ("Manual", "Adjustment")
        if is_manual and np.random.random() < 0.15:
            amt = float(round(np.random.uniform(500000, 2000000), 2))

        is_balanced = True
        status = "approved"
        if np.random.random() < 0.04:
            is_balanced = False
        if is_manual and np.random.random() < 0.10:
            status = "unapproved"

        journals.append({
            "document_id": f"DOC-{10000 + i}",
            "posting_date": f"2024-11-{np.random.randint(1, 28):02d}",
            "journal_type": jtype,
            "posting_user": np.random.choice(["j_smith", "a_kumar", "system_auto", "m_davis"]),
            "debit_amount": amt,
            "credit_amount": amt if is_balanced else amt * 0.95,
            "approval_status": status,
            "is_balanced": is_balanced,
            "risk_score": 92.0 if not is_balanced else (85.0 if status == "unapproved" else np.random.uniform(5, 45)),
        })
    return pd.DataFrame(journals)


journals_df = load_r2r_demo_data()

st.title("📑 Record-to-Report — Financial Accounting & Close Governance")
st.markdown("General ledger anomaly detection, debit-credit consistency validation, and period-end close assurance.")

# Top KPIs
c1, c2, c3, c4, c5 = st.columns(5)
total_journals = len(journals_df)
total_debits = journals_df["debit_amount"].sum()
total_credits = journals_df["credit_amount"].sum()
unbalanced_count = len(journals_df[~journals_df["is_balanced"]])
manual_count = len(journals_df[journals_df["journal_type"] == "Manual"])

with c1:
    st.metric("Total Journal Postings", f"{total_journals}")
with c2:
    st.metric("Total Debit Volume", f"${total_debits / 1e6:.2f} M")
with c3:
    st.metric("Total Credit Volume", f"${total_credits / 1e6:.2f} M")
with c4:
    st.metric("Unbalanced Journals", f"{unbalanced_count}", delta="Requires Correction" if unbalanced_count > 0 else "0", delta_color="inverse")
with c5:
    st.metric("High-Risk Manual Journals", f"{manual_count}", delta=f"{manual_count / total_journals * 100:.0f}% of total")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 General Ledger Overview",
    "📝 Journal Entry Explorer",
    "🔄 Account Reconciliation",
    "📈 Accruals & Adjustments",
    "🏢 Intercompany Matching",
    "⚖️ Interactive Journal Validator",
])

# ---------------------------------------------------------------------------
# TAB 1: General Ledger Overview
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("GL Account Transaction Volume & Distribution")
    col_g1, col_g2 = st.columns([3, 2])

    with col_g1:
        type_summary = journals_df.groupby("journal_type")["debit_amount"].sum().reset_index()
        fig_type = px.bar(
            type_summary,
            x="journal_type",
            y="debit_amount",
            title="Total Transaction Volume by Journal Type ($)",
            color="journal_type",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            text_auto=".2s",
        )
        fig_type.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_type, use_container_width=True)

    with col_g2:
        user_summary = journals_df.groupby("posting_user")["document_id"].count().reset_index()
        fig_pie = px.pie(
            user_summary,
            names="posting_user",
            values="document_id",
            title="Posting User Distribution",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig_pie.update_layout(height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: Journal Entry Explorer
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Journal Header & Line-Item Explorer")
    filter_type = st.multiselect("Filter Journal Types", ["Standard", "Manual", "Accrual", "Adjustment", "Reversal", "Intercompany"], default=["Manual", "Standard", "Adjustment"])
    display_journals = journals_df[journals_df["journal_type"].isin(filter_type)].copy()

    display_journals["debit_amount"] = display_journals["debit_amount"].apply(lambda x: f"${x:,.2f}")
    display_journals["credit_amount"] = display_journals["credit_amount"].apply(lambda x: f"${x:,.2f}")
    display_journals["risk_score"] = display_journals["risk_score"].apply(lambda x: f"{x:.1f}")

    st.dataframe(display_journals, use_container_width=True, height=350)

# ---------------------------------------------------------------------------
# TAB 3: Account Reconciliation
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Sub-Ledger to General Ledger Reconciliations")
    recon_data = [
        {"GL Account": "101000 — Cash at Bank", "GL Balance": "$1,450,200.00", "Bank Statement Balance": "$1,450,200.00", "Variance": "$0.00", "Status": "Reconciled"},
        {"GL Account": "120000 — Accounts Receivable", "GL Balance": "$4,820,400.00", "Sub-ledger Balance": "$4,815,100.00", "Variance": "$5,300.00", "Status": "Exception (Unmatched)"},
        {"GL Account": "200000 — Accounts Payable", "GL Balance": "$3,210,000.00", "AP Sub-ledger Balance": "$3,210,000.00", "Variance": "$0.00", "Status": "Reconciled"},
        {"GL Account": "140000 — Inventory GL", "GL Balance": "$2,105,000.00", "Perpetual Inventory": "$2,080,000.00", "Variance": "$25,000.00", "Status": "Material Discrepancy"},
        {"GL Account": "190000 — Intercompany Clearing", "GL Balance": "$850,000.00", "Entity 2000 Mirror": "$842,500.00", "Variance": "$7,500.00", "Status": "Under Review"},
    ]
    st.dataframe(pd.DataFrame(recon_data), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4: Accruals & Adjustments
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Period-End Accruals & Adjustment Monitoring")
    st.info("FinGuard AI monitors accrual sizing against historical averages and detects unusual cross-period reversal patterns.")
    accruals_data = [
        {"Accrual ID": "ACC-501", "Account": "610000 — Legal & Professional", "Period": "11/2024", "Accrued Amount": "$185,000.00", "Historical Avg": "$45,000.00", "Deviation": "+311%", "Status": "HIGH RISK (Spike)"},
        {"Accrual ID": "ACC-502", "Account": "620000 — Marketing Services", "Period": "11/2024", "Accrued Amount": "$50,000.00", "Historical Avg": "$48,000.00", "Deviation": "+4%", "Status": "Normal"},
        {"Accrual ID": "ACC-503", "Account": "630000 — IT Infrastructure", "Period": "11/2024", "Accrued Amount": "$95,000.00", "Historical Avg": "$90,000.00", "Deviation": "+5%", "Status": "Normal"},
        {"Accrual ID": "ACC-504", "Account": "640000 — Travel & Entertainment", "Period": "11/2024", "Accrued Amount": "$12,000.00", "Historical Avg": "$11,500.00", "Deviation": "+4%", "Status": "Normal"},
    ]
    st.dataframe(pd.DataFrame(accruals_data), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 5: Intercompany Matching
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Intercompany Balance Matching & Elimination")
    ic_data = [
        {"Transaction Ref": "IC-TXN-101", "From Entity": "1000 (US)", "To Entity": "2000 (UK)", "Source Amount": "$125,000.00", "Mirror Amount": "$125,000.00", "Variance": "$0.00", "Match": "MATCHED"},
        {"Transaction Ref": "IC-TXN-102", "From Entity": "1000 (US)", "To Entity": "3000 (IN)", "Source Amount": "$80,000.00", "Mirror Amount": "$78,200.00", "Variance": "$1,800.00", "Match": "MISMATCHED (FX)"},
        {"Transaction Ref": "IC-TXN-103", "From Entity": "2000 (UK)", "To Entity": "3000 (IN)", "Source Amount": "$45,000.00", "Mirror Amount": "$45,000.00", "Variance": "$0.00", "Match": "MATCHED"},
    ]
    st.dataframe(pd.DataFrame(ic_data), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 6: Interactive Journal Entry Validator
# ---------------------------------------------------------------------------
with tab6:
    st.subheader("Interactive Journal Balance & Controls Validator")
    st.markdown("Test the deterministic accounting control engine by simulating debit and credit lines.")

    col_v1, col_v2 = st.columns([1, 1])

    with col_v1:
        st.markdown("#### Journal Header")
        v_jtype = st.selectbox("Journal Type", ["Standard", "Manual", "Accrual", "Adjustment", "Reversal"])
        v_user = st.text_input("Posting User", value="finance_analyst_1")
        v_approval = st.selectbox("Approval Status", ["approved", "unapproved", "pending"])

        st.markdown("#### Posting Lines")
        debit_line = st.number_input("Debit Amount ($)", min_value=0.0, value=50000.0, step=1000.0)
        credit_line = st.number_input("Credit Amount ($)", min_value=0.0, value=50000.0, step=1000.0)

        validate_btn = st.button("⚖️ Run Journal Compliance Validation", type="primary")

    with col_v2:
        if validate_btn:
            imbalance = abs(debit_line - credit_line)
            st.markdown("### 📋 Validation Output")

            if imbalance > 0.01:
                st.error(f"""
                ### 🛑 UNBALANCED JOURNAL ENTRY (CRITICAL)
                - **Total Debit:** ${debit_line:,.2f}
                - **Total Credit:** ${credit_line:,.2f}
                - **Imbalance Variance:** ${imbalance:,.2f}
                - **Rule Triggered:** `UNBALANCED_JOURNAL_ENTRY`
                - **SOX Control Violation:** Total Debits must exactly equal Total Credits under general accounting rules.
                """)
            elif v_jtype == "Manual" and v_approval != "approved":
                st.error(f"""
                ### 🛑 UNAPPROVED MANUAL JOURNAL (CRITICAL)
                - **Status:** '{v_approval}'
                - **Rule Triggered:** `UNAPPROVED_MANUAL_JOURNAL`
                - **Action:** Manual entries require recorded supervisory sign-off before GL posting.
                """)
            elif debit_line >= 500000.0 and v_jtype == "Manual":
                st.warning(f"""
                ### ⚠️ HIGH VALUE MANUAL JOURNAL (WARNING)
                - **Amount:** ${debit_line:,.2f}
                - **Rule Triggered:** `HIGH_VALUE_MANUAL_JOURNAL`
                - **Note:** Journal is balanced but exceeds $500k standard threshold, triggering secondary controller review.
                """)
            else:
                st.success(f"""
                ### ✅ JOURNAL COMPLIANCE CONFIRMED
                - **Total Debit:** ${debit_line:,.2f}
                - **Total Credit:** ${credit_line:,.2f}
                - **Status:** Balanced and compliant with all core R2R accounting controls.
                """)
