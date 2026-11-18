"""
FinGuard AI — Order-to-Cash (O2C) Credit Management Dashboard

Provides real-time visibility and AI-powered anomaly detection for:
  - Customer credit profiles and limits
  - Real-time credit exposure and utilisation monitoring
  - Accounts Receivable (AR) aging analysis
  - Customer payment trends and behaviour tracking
  - Credit hold and order block releases
  - Interactive Sales Order credit check simulator
"""

import sys
from datetime import date
from pathlib import Path

# Ensure project root is in sys.path
for _p in Path(__file__).resolve().parents:
    if (_p / "frontend").is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
        break

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.utils.api_client import APIClient

# Paths & Branding
ASSETS_DIR = Path(__file__).parent.parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
FAVICON_PATH = ASSETS_DIR / "favicon.png"

st.set_page_config(
    page_title="Credit Management (O2C) | FinGuard AI",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "🏦",
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
    .status-badge {
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width="stretch")
    else:
        st.title("🏦 FinGuard AI")
        st.caption("Credit Management Module")
    st.markdown("---")
    st.markdown("### 🏢 Filters")
    selected_risk = st.selectbox("Risk Category", ["ALL", "Low", "Medium", "High", "Critical"])
    selected_segment = st.selectbox("Credit Segment", ["ALL", "CS_DEFAULT", "CS_ENTERPRISE", "CS_RETAIL"])
    st.markdown("---")
    st.caption("FinGuard AI v2.0.0 • Order-to-Cash (O2C)")


@st.cache_data(ttl=60)
def load_o2c_demo_data():
    """Generate or fetch rich O2C data for demonstration."""
    np.random.seed(42)
    names = [
        "Apex Global Industries", "Bharati Enterprises", "Zenith Logistics Ltd",
        "Orion Infrastructure", "Tata Sterling Trading", "Nova Chemicals Corp",
        "Sunlight Pharma Ltd", "Kaveri Engineering", "Vanguard Steelworks",
        "Blue Horizon Retail", "Metro Distribution Co", "Imperial Textiles",
        "Falcon Technologies", "Matrix Automotives", "Crown Food Products"
    ]
    rows = []
    for i, name in enumerate(names):
        limit = float(np.random.choice([500000, 1000000, 2000000, 5000000, 10000000]))
        exposure = float(round(limit * np.random.uniform(0.15, 1.15), 2))
        available = max(0.0, limit - exposure)
        utilisation = round((exposure / limit) * 100.0, 1)

        overdue = round(exposure * np.random.uniform(0.0, 0.4), 2) if utilisation > 50 else 0.0
        avg_dpd = int(np.random.exponential(12)) if overdue > 0 else 0
        max_dpd = int(avg_dpd * np.random.uniform(1.2, 2.5))

        if utilisation > 95 or max_dpd > 60:
            risk = "Critical"
        elif utilisation > 80 or max_dpd > 30:
            risk = "High"
        elif utilisation > 60 or max_dpd > 10:
            risk = "Medium"
        else:
            risk = "Low"

        rows.append({
            "customer_id": f"CUST-{1000 + i}",
            "customer_name": name,
            "credit_segment": np.random.choice(["CS_DEFAULT", "CS_ENTERPRISE", "CS_RETAIL"]),
            "credit_limit": limit,
            "credit_exposure": exposure,
            "available_credit": available,
            "credit_utilisation": utilisation,
            "overdue_receivables": overdue,
            "avg_dpd": avg_dpd,
            "max_dpd": max_dpd,
            "risk_category": risk,
            "blocked_orders": np.random.choice([0, 0, 0, 1, 2]) if risk in ("High", "Critical") else 0,
            "payment_trend": "Deteriorating" if risk in ("High", "Critical") else ("Stable" if risk == "Medium" else "Improving"),
        })
    return pd.DataFrame(rows)


df = load_o2c_demo_data()

# Apply Sidebar Filters
filtered_df = df.copy()
if selected_risk != "ALL":
    filtered_df = filtered_df[filtered_df["risk_category"] == selected_risk]
if selected_segment != "ALL":
    filtered_df = filtered_df[filtered_df["credit_segment"] == selected_segment]

# Header
st.title("💳 Order-to-Cash — Credit Management & AR Analytics")
st.markdown("Intelligent credit limit monitoring, real-time exposure calculations, and predictive payment behaviour analytics.")

# Top KPIs
col1, col2, col3, col4, col5 = st.columns(5)
total_limit = df["credit_limit"].sum()
total_exposure = df["credit_exposure"].sum()
avg_util = df["credit_utilisation"].mean()
total_overdue = df["overdue_receivables"].sum()
high_risk_count = len(df[df["risk_category"].isin(["High", "Critical"])])

with col1:
    st.metric("Total Approved Credit", f"₹{total_limit / 1e7:.2f} Cr")
with col2:
    st.metric("Total Exposure", f"₹{total_exposure / 1e7:.2f} Cr", delta=f"{avg_util:.1f}% Avg Util")
with col3:
    st.metric("Available Credit", f"₹{(total_limit - total_exposure) / 1e7:.2f} Cr")
with col4:
    st.metric("Total Overdue AR", f"₹{total_overdue / 1e5:.1f} Lakh", delta_color="inverse")
with col5:
    st.metric("High/Critical Risk Accts", f"{high_risk_count}", delta=f"{high_risk_count / len(df) * 100:.0f}% of portfolio", delta_color="inverse")

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Customer Credit Overview",
    "📈 Credit Exposure Analysis",
    "📋 AR Aging Analysis",
    "🕒 Payment Behaviour",
    "⚠️ Credit Anomaly Explorer",
    "🔍 Credit Check Simulator & 360° Profile",
])

# ---------------------------------------------------------------------------
# TAB 1: Customer Credit Overview
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Customer Master & Credit Profile Directory")
    st.caption("Real-time snapshot of customer approved limits, current commitments, and assigned risk ratings.")

    display_df = filtered_df[[
        "customer_id", "customer_name", "credit_segment", "credit_limit",
        "credit_exposure", "available_credit", "credit_utilisation",
        "overdue_receivables", "risk_category", "payment_trend"
    ]].copy()

    # Format numbers for display
    display_df["credit_limit"] = display_df["credit_limit"].apply(lambda x: f"₹{x:,.0f}")
    display_df["credit_exposure"] = display_df["credit_exposure"].apply(lambda x: f"₹{x:,.0f}")
    display_df["available_credit"] = display_df["available_credit"].apply(lambda x: f"₹{x:,.0f}")
    display_df["credit_utilisation"] = display_df["credit_utilisation"].apply(lambda x: f"{x:.1f}%")
    display_df["overdue_receivables"] = display_df["overdue_receivables"].apply(lambda x: f"₹{x:,.0f}")

    st.dataframe(display_df, use_container_width=True, height=350)

# ---------------------------------------------------------------------------
# TAB 2: Credit Exposure Analysis
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Credit Exposure vs. Approved Credit Limits")
    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Approved Limit",
            x=filtered_df["customer_name"],
            y=filtered_df["credit_limit"],
            marker_color="#94a3b8",
        ))
        fig_bar.add_trace(go.Bar(
            name="Current Exposure",
            x=filtered_df["customer_name"],
            y=filtered_df["credit_exposure"],
            marker_color="#0284c7",
        ))
        fig_bar.update_layout(
            barmode="group",
            title="Customer Exposure Comparison (Limit vs Exposure)",
            xaxis_tickangle=-45,
            height=400,
            margin=dict(t=40, b=80, l=40, r=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        fig_hist = px.histogram(
            df,
            x="credit_utilisation",
            nbins=10,
            title="Portfolio Credit Utilisation Distribution (%)",
            color_discrete_sequence=["#0284c7"],
        )
        fig_hist.add_vline(x=80, line_dash="dash", line_color="orange", annotation_text="Warning (80%)")
        fig_hist.add_vline(x=95, line_dash="dash", line_color="red", annotation_text="Critical (95%)")
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: Accounts Receivable Aging
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Accounts Receivable (AR) Aging Structure")
    col_ar1, col_ar2 = st.columns([2, 3])

    with col_ar1:
        aging_summary = pd.DataFrame([
            {"Bucket": "Not Due", "Amount": total_exposure * 0.55, "Color": "#10b981"},
            {"Bucket": "1–30 Days", "Amount": total_exposure * 0.20, "Color": "#38bdf8"},
            {"Bucket": "31–60 Days", "Amount": total_exposure * 0.12, "Color": "#f59e0b"},
            {"Bucket": "61–90 Days", "Amount": total_exposure * 0.08, "Color": "#f97316"},
            {"Bucket": "Above 90 Days", "Amount": total_exposure * 0.05, "Color": "#ef4444"},
        ])
        fig_donut = px.pie(
            aging_summary,
            names="Bucket",
            values="Amount",
            title="Aging Breakdown (% of Total AR)",
            hole=0.45,
            color="Bucket",
            color_discrete_map={row["Bucket"]: row["Color"] for _, row in aging_summary.iterrows()},
        )
        fig_donut.update_layout(height=380)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_ar2:
        fig_aging_bar = px.bar(
            aging_summary,
            x="Bucket",
            y="Amount",
            color="Bucket",
            title="Total Receivables by Aging Bucket (₹)",
            text_auto=".2s",
            color_discrete_map={row["Bucket"]: row["Color"] for _, row in aging_summary.iterrows()},
        )
        fig_aging_bar.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_aging_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4: Customer Payment Behaviour
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Customer Payment Behaviour & Deterioration Watchlist")
    col_p1, col_p2 = st.columns([3, 2])

    with col_p1:
        fig_scatter = px.scatter(
            df,
            x="avg_dpd",
            y="credit_utilisation",
            size="credit_limit",
            color="risk_category",
            hover_name="customer_name",
            title="Utilisation vs Average Days Past Due (DPD)",
            labels={"avg_dpd": "Average Days Past Due (Days)", "credit_utilisation": "Credit Utilisation (%)"},
            color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#f97316", "Critical": "#ef4444"},
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_p2:
        st.markdown("#### 🚨 Deteriorating Payment Accounts")
        deteriorating = df[df["payment_trend"] == "Deteriorating"][["customer_name", "avg_dpd", "overdue_receivables", "risk_category"]]
        deteriorating["overdue_receivables"] = deteriorating["overdue_receivables"].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(deteriorating, use_container_width=True, height=330)

# ---------------------------------------------------------------------------
# TAB 5: Credit Anomaly Explorer
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("AI-Detected Credit & Payment Anomalies")
    anomalies_data = [
        {"Customer": "Orion Infrastructure", "Anomaly Scenario": "Unusual Credit Limit Increase", "Risk Score": 88.5, "Severity": "CRITICAL", "Reason": "Approved limit increased by 150% without documented managerial approval."},
        {"Customer": "Vanguard Steelworks", "Anomaly Scenario": "High Exposure Breach", "Risk Score": 92.0, "Severity": "CRITICAL", "Reason": "Credit utilisation reached 112.5% with ₹18.4L in 60+ DPD overdue invoices."},
        {"Customer": "Tata Sterling Trading", "Anomaly Scenario": "Deteriorating Payment Pattern", "Risk Score": 74.0, "Severity": "HIGH", "Reason": "Consistent 35-day payment lag observed over the past 3 consecutive invoice cycles."},
        {"Customer": "Nova Chemicals Corp", "Anomaly Scenario": "High-Value Order from Risky Account", "Risk Score": 81.0, "Severity": "CRITICAL", "Reason": "Customer with ₹12L overdue balance placed new order of ₹35L exceeding available credit."},
        {"Customer": "Apex Global Industries", "Anomaly Scenario": "Unauthorized Block Release", "Risk Score": 78.0, "Severity": "HIGH", "Reason": "Sales order SO-10492 released from credit hold without required authorization audit trail."},
    ]
    anom_df = pd.DataFrame(anomalies_data)
    st.dataframe(anom_df, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 6: Interactive Credit Check Simulator
# ---------------------------------------------------------------------------
with tab6:
    st.subheader("Interactive Sales Order Credit Check Simulator")
    st.markdown("Simulate how a new incoming sales order is evaluated against real-time customer exposure and credit policies.")

    sim_col1, sim_col2 = st.columns([1, 1])

    with sim_col1:
        selected_cust_name = st.selectbox("Select Customer", df["customer_name"].tolist())
        selected_row = df[df["customer_name"] == selected_cust_name].iloc[0]

        st.info(f"""
        **Customer:** {selected_row['customer_name']}  
        **Approved Credit Limit:** ₹{selected_row['credit_limit']:,.2f}  
        **Current Exposure:** ₹{selected_row['credit_exposure']:,.2f}  
        **Current Available Credit:** ₹{selected_row['available_credit']:,.2f}  
        **Utilisation:** {selected_row['credit_utilisation']:.1f}%  
        **Current Risk Tier:** {selected_row['risk_category']}
        """)

        order_val = st.number_input("New Sales Order Value (₹)", min_value=10000.0, max_value=20000000.0, value=500000.0, step=50000.0)
        simulate_btn = st.button("🚀 Run Real-time Credit Evaluation", type="primary")

    with sim_col2:
        if simulate_btn:
            limit = selected_row["credit_limit"]
            curr_exp = selected_row["credit_exposure"]
            proj_exp = curr_exp + order_val
            proj_util = (proj_exp / limit * 100.0) if limit > 0 else 100.0
            avail_credit = limit - curr_exp

            st.markdown("### 📋 Evaluation Result")
            if proj_exp > limit:
                st.error(f"""
                ### 🛑 DECISION: CREDIT HOLD
                - **Projected Exposure:** ₹{proj_exp:,.2f} exceeds Credit Limit (₹{limit:,.2f}) by ₹{proj_exp - limit:,.2f}.
                - **Projected Utilisation:** {proj_util:.1f}% (Policy Limit: 100%).
                - **Action Required:** Order placed on credit hold pending managerial limit adjustment or customer payment.
                """)
            elif proj_util >= 90:
                st.warning(f"""
                ### ⚠️ DECISION: MANUAL CREDIT REVIEW
                - **Projected Exposure:** ₹{proj_exp:,.2f} (Utilisation: {proj_util:.1f}%).
                - **Available Headroom:** ₹{limit - proj_exp:,.2f}.
                - **Action Required:** Forwarded to Senior Credit Analyst for secondary review.
                """)
            elif proj_util >= 80:
                st.info(f"""
                ### ⚡ DECISION: APPROVED WITH WARNING
                - **Projected Exposure:** ₹{proj_exp:,.2f} (Utilisation: {proj_util:.1f}%).
                - **Available Headroom:** ₹{limit - proj_exp:,.2f}.
                - **Action:** Order accepted with automatic warning sent to account manager.
                """)
            else:
                st.success(f"""
                ### ✅ DECISION: APPROVED
                - **Projected Exposure:** ₹{proj_exp:,.2f} (Utilisation: {proj_util:.1f}%).
                - **Available Headroom:** ₹{limit - proj_exp:,.2f}.
                - **Action:** Sales order cleared for immediate delivery and fulfilment.
                """)
