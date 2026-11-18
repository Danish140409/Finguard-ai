"""
FinGuard AI — Reports & Workpaper Export

Enables analysts and controllers to generate executive-ready PDF audit workpapers
and export filtered anomaly datasets to CSV for external audit submissions.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
for _p in Path(__file__).resolve().parents:
    if (_p / "frontend").is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
        break

import streamlit as st
import pandas as pd
from datetime import date, timedelta
from frontend.utils.api_client import APIClient

ASSETS_DIR = Path(__file__).parent.parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
FAVICON_PATH = ASSETS_DIR / "favicon.png"

st.set_page_config(
    page_title="Reports & Export | FinGuard AI",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "📑",
    layout="wide",
)

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), width="stretch")
    st.sidebar.markdown("---")

st.title("📑 Audit Reports & Workpaper Export")
st.markdown(
    "Generate formal compliance reports formatted with executive summaries, "
    "risk metrics, flagged transaction details, and controller sign-off blocks."
)

api_client = APIClient()
token = st.session_state.get("token")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Configure Export Scope")
    d_start = st.date_input("Audit Period Start Date", value=date.today() - timedelta(days=90))
    d_end = st.date_input("Audit Period End Date", value=date.today())
    
    selected_risks = st.multiselect(
        "Included Risk Severity Levels",
        options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH"],
    )

with col2:
    st.subheader("2. Workpaper Generation")
    st.markdown("#### Formal PDF Audit Workpaper")
    st.caption("Includes executive summary KPI box, SHAP narrative breakdown, and sign-off block.")
    
    if st.button("📄 Generate & Download Audit PDF", use_container_width=True):
        with st.spinner("Compiling ReportLab PDF workpaper..."):
            try:
                if token and token != "demo_token":
                    pdf_bytes = api_client.download_pdf_report(
                        token=token,
                        date_from=str(d_start),
                        date_to=str(d_end),
                        risk_levels=selected_risks,
                    )
                else:
                    # Generate directly via PDF generator
                    from reports.pdf_generator import PDFReportGenerator
                    gen = PDFReportGenerator()
                    demo_tx = [
                        {
                            "invoice_date": "2026-11-14",
                            "vendor_name": "Apex Global Solutions LLC",
                            "amount": 49800.00,
                            "risk_score": 92.4,
                            "risk_level": "CRITICAL",
                            "anomaly_types": ["split_invoice", "round_dollar"],
                            "plain_english_reason": "Invoice amount of $49,800.00 is just below the $50,000 threshold. 3 invoices submitted within 4 days.",
                        },
                        {
                            "invoice_date": "2026-11-15",
                            "vendor_name": "Kestrel Logistics Group",
                            "amount": 12850.00,
                            "risk_score": 88.1,
                            "risk_level": "CRITICAL",
                            "anomaly_types": ["weekend_new_vendor"],
                            "plain_english_reason": "Payment processed on Sunday to vendor registered 4 days prior.",
                        },
                        {
                            "invoice_date": "2026-11-12",
                            "vendor_name": "Vanguard Industrial Supplies",
                            "amount": 76500.00,
                            "risk_score": 79.5,
                            "risk_level": "HIGH",
                            "anomaly_types": ["inflated_invoice"],
                            "plain_english_reason": "Invoice amount is 3.8 standard deviations above vendor historical mean.",
                        },
                    ]
                    pdf_bytes = gen.generate_risk_report(demo_tx, (str(d_start), str(d_end)))

                st.download_button(
                    label="⬇️ Click Here to Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"FinGuard_Audit_Report_{d_start}_{d_end}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success("PDF generated successfully!")
            except Exception as e:
                st.error(f"Failed to compile PDF: {e}")

    st.markdown("---")
    st.markdown("#### CSV Data Extract")
    st.caption("Raw tabular extract for ingestion into Excel, Alteryx, or PowerBI.")
    
    if st.button("📊 Export Filtered Exceptions to CSV", use_container_width=True):
        with st.spinner("Extracting records to CSV..."):
            try:
                if token and token != "demo_token":
                    csv_bytes = api_client.download_csv_export(token, str(d_start), str(d_end))
                else:
                    sample_csv = (
                        "score_id,transaction_id,risk_score,risk_level,vendor_name,amount,plain_english_reason\n"
                        "1,49b498f3,92.4,CRITICAL,Apex Global Solutions LLC,49800.00,Invoice amount is just below $50k threshold.\n"
                        "2,81f181cc,88.1,CRITICAL,Kestrel Logistics Group,12850.00,Sunday payment to vendor registered 4 days prior.\n"
                    ).encode("utf-8")
                    csv_bytes = sample_csv

                st.download_button(
                    label="⬇️ Click Here to Download CSV",
                    data=csv_bytes,
                    file_name=f"finguard_exceptions_{d_start}_{d_end}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.success("CSV export ready!")
            except Exception as e:
                st.error(f"Failed to export CSV: {e}")
