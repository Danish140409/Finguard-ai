"""
FinGuard AI — ReportLab PDF Generator

Generates executive-level, audit-ready PDF reports for financial anomalies,
featuring executive summaries, risk distribution breakdowns, flagged transaction tables,
and plain-English explanation excerpts formatted for CFO/Controller/Internal Audit review.
"""

from datetime import datetime, timezone
import io
from typing import Any, List, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class NumberedCanvas:
    """Canvas wrapper to draw running footers with page numbers."""

    def __init__(self, *args, **kwargs):
        pass


class PDFReportGenerator:
    """Enterprise PDF report generator for FinGuard AI anomaly audits."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.title_style = ParagraphStyle(
            "DocTitle",
            parent=self.styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1e293b"),
            fontName="Helvetica-Bold",
        )
        self.subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=self.styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64748b"),
            fontName="Helvetica",
        )
        self.section_heading = ParagraphStyle(
            "SectionHeading",
            parent=self.styles["Heading2"],
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#0f172a"),
            fontName="Helvetica-Bold",
            spaceBefore=12,
            spaceAfter=6,
        )
        self.body_style = ParagraphStyle(
            "CustomBody",
            parent=self.styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#334155"),
            fontName="Helvetica",
        )
        self.table_header_style = ParagraphStyle(
            "TableHeader",
            parent=self.styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.white,
            fontName="Helvetica-Bold",
        )
        self.table_cell_style = ParagraphStyle(
            "TableCell",
            parent=self.styles["Normal"],
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#1e293b"),
            fontName="Helvetica",
        )
        self.callout_style = ParagraphStyle(
            "Callout",
            parent=self.styles["Normal"],
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#0369a1"),
            fontName="Helvetica-Oblique",
        )

    def generate_risk_report(
        self,
        transactions: List[dict[str, Any]],
        date_range: Tuple[str, str] = ("All Time", "Present"),
    ) -> bytes:
        """
        Generate a multi-page PDF risk audit document.

        Args:
            transactions: list of transaction dictionaries containing scores and details.
            date_range: tuple of (start_date_str, end_date_str).

        Returns:
            Raw bytes of the compiled PDF.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
        )

        elements = []

        # 1. Header Banner
        elements.append(Paragraph("FinGuard AI — Exception & Anomaly Audit Report", self.title_style))
        gen_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        elements.append(
            Paragraph(
                f"Generated on: <b>{gen_time}</b> | Period: <b>{date_range[0]} to {date_range[1]}</b> | Classification: <b>Confidential / Internal Audit</b>",
                self.subtitle_style,
            )
        )
        elements.append(Spacer(1, 10))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#6366f1"), spaceAfter=14))

        # 2. Executive Metrics Summary Box
        total_tx = len(transactions)
        critical_tx = [t for t in transactions if t.get("risk_level") == "CRITICAL"]
        high_tx = [t for t in transactions if t.get("risk_level") == "HIGH"]
        total_exposure = sum(t.get("amount", 0.0) for t in transactions)

        summary_data = [
            [
                Paragraph("<b>Total Flagged Exceptions</b>", self.body_style),
                Paragraph("<b>Critical Risk Items</b>", self.body_style),
                Paragraph("<b>High Risk Items</b>", self.body_style),
                Paragraph("<b>Total Potential Exposure</b>", self.body_style),
            ],
            [
                Paragraph(f"<font size=13 color='#0f172a'><b>{total_tx:,}</b></font>", self.body_style),
                Paragraph(f"<font size=13 color='#dc2626'><b>{len(critical_tx):,}</b></font>", self.body_style),
                Paragraph(f"<font size=13 color='#ea580c'><b>{len(high_tx):,}</b></font>", self.body_style),
                Paragraph(f"<font size=13 color='#0f172a'><b>${total_exposure:,.2f}</b></font>", self.body_style),
            ],
        ]
        summary_table = Table(summary_data, colWidths=[1.8 * inch, 1.8 * inch, 1.8 * inch, 2.1 * inch])
        summary_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ])
        )
        elements.append(summary_table)
        elements.append(Spacer(1, 15))

        # 3. Purpose & Compliance Scope
        scope_p = (
            "This report lists accounts payable (AP) transactions flagged by the FinGuard AI multi-model ensemble "
            "(Isolation Forest, Local Outlier Factor, One-Class SVM, and XGBoost). Transactions are scored from "
            "0 to 100 based on vendor history, timing, amount deviations, and structural inconsistencies. Explanations "
            "are rendered deterministically using SHAP/LIME attribution for Sarbanes-Oxley (SOX) Section 404 auditability."
        )
        elements.append(Paragraph(scope_p, self.body_style))
        elements.append(Spacer(1, 12))

        # 4. Detailed Exceptions Table
        elements.append(Paragraph("Flagged Transactions Detail", self.section_heading))

        table_rows = [
            [
                Paragraph("Date", self.table_header_style),
                Paragraph("Vendor", self.table_header_style),
                Paragraph("Amount ($)", self.table_header_style),
                Paragraph("Score", self.table_header_style),
                Paragraph("Level", self.table_header_style),
                Paragraph("Anomaly Category", self.table_header_style),
                Paragraph("Audit Narrative & Explainability Reason", self.table_header_style),
            ]
        ]

        for item in transactions[:200]:  # Limit rows in PDF for performance
            risk_lvl = item.get("risk_level", "UNKNOWN")
            lvl_color = "#dc2626" if risk_lvl == "CRITICAL" else ("#ea580c" if risk_lvl == "HIGH" else "#475569")

            cats = ", ".join(item.get("anomaly_types", [])) or "Outlier"
            reason_txt = item.get("plain_english_reason", "No narrative available.")

            row = [
                Paragraph(str(item.get("invoice_date", "N/A")), self.table_cell_style),
                Paragraph(str(item.get("vendor_name", "N/A"))[:22], self.table_cell_style),
                Paragraph(f"${float(item.get('amount', 0)):,.2f}", self.table_cell_style),
                Paragraph(f"{float(item.get('risk_score', 0)):.1f}", self.table_cell_style),
                Paragraph(f"<font color='{lvl_color}'><b>{risk_lvl}</b></font>", self.table_cell_style),
                Paragraph(cats[:30], self.table_cell_style),
                Paragraph(reason_txt[:180] + ("..." if len(reason_txt) > 180 else ""), self.table_cell_style),
            ]
            table_rows.append(row)

        col_widths = [0.8 * inch, 1.4 * inch, 0.9 * inch, 0.5 * inch, 0.7 * inch, 1.1 * inch, 2.1 * inch]
        detail_table = Table(table_rows, colWidths=col_widths, repeatRows=1)
        detail_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        elements.append(detail_table)
        elements.append(Spacer(1, 15))

        # 5. Sign-off Footer
        elements.append(
            Paragraph(
                "<b>Audit Sign-off:</b><br/>"
                "Controller Reviewer: ________________________ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Date: _______________<br/>"
                "Internal Audit Lead: ________________________ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Date: _______________",
                self.body_style,
            )
        )

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
