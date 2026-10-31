"""
RetainIQ — Project Overview & Screenshots PDF Generator.
Generates an elegant, publication-quality PDF containing:
1. Clear, simple explanation of the project in accessible language.
2. Actual high-resolution screenshots of the live system.
3. Descriptive headings, 'what is this showing', and key highlights for each view:
   - Input Interface (Customer Profile Setup)
   - Prediction Result (Exit Risk & Probability Gauge)
   - Reason / Explainability (SHAP Factors)
   - Batch Account Scoring (Portfolio Upload & Prioritized Table)
   - Model Evaluation (Scorecard & Verification Artifacts)
   - Backend Serving (FastAPI Swagger UI)
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RetainIQ_Project_Walkthrough.pdf")
SCREENSHOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "screenshots")

# Color palette
PRIMARY = colors.HexColor("#0f2942")     # Deep Navy
SECONDARY = colors.HexColor("#2563eb")   # Royal Blue
ACCENT_GREEN = colors.HexColor("#059669")# Emerald Green
ACCENT_AMBER = colors.HexColor("#d97706")# Amber
BG_LIGHT = colors.HexColor("#f8fafc")    # Slate Light
TEXT_MAIN = colors.HexColor("#1e293b")   # Slate Dark
TEXT_MUTED = colors.HexColor("#64748b")  # Muted Slate
BORDER_COL = colors.HexColor("#cbd5e1")  # Border


class NumberedCanvas(canvas.Canvas):
    """Canvas that performs a two-pass calculation for total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, total_pages):
        self.saveState()
        # Suppress header on cover/title page
        if self._pageNumber > 1:
            # Header
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(TEXT_MUTED)
            self.drawString(40, 810, "RetainIQ — Customer Exit & Retention Prediction System")
            self.setFont("Helvetica", 8)
            self.drawRightString(555, 810, "Application Walkthrough & Screenshots Guide")
            self.setStrokeColor(BORDER_COL)
            self.setLineWidth(0.5)
            self.line(40, 804, 555, 804)

        # Footer on all pages
        self.setFont("Helvetica", 8)
        self.setFillColor(TEXT_MUTED)
        self.drawString(40, 30, "Confidential — RetainIQ Project Demonstration Documentation")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(555, 30, page_str)
        self.setStrokeColor(BORDER_COL)
        self.setLineWidth(0.5)
        self.line(40, 42, 555, 42)
        self.restoreState()


def create_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=45,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=PRIMARY,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=SECONDARY,
        spaceAfter=15
    )

    heading1_style = ParagraphStyle(
        "SectionHeading",
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=PRIMARY,
        spaceBefore=4,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_MAIN,
        spaceAfter=6
    )

    bold_label = ParagraphStyle(
        "BoldLabel",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13.5,
        textColor=PRIMARY
    )

    card_text = ParagraphStyle(
        "CardText",
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=TEXT_MAIN
    )

    callout_style = ParagraphStyle(
        "CalloutStyle",
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12.5,
        textColor=SECONDARY
    )

    story = []

    # =========================================================================
    # PAGE 1: PROJECT OVERVIEW (IN SIMPLE / "LITTLE" LANGUAGE)
    # =========================================================================
    story.append(Paragraph("RetainIQ: Customer Exit & Churn Prediction System", title_style))
    story.append(Paragraph("System Walkthrough, Real Application Screenshots & Explainability Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceAfter=14))

    # Section: What is this project? (Simple words)
    story.append(Paragraph("1. What is this project in simple words?", heading1_style))
    p1 = (
        "<b>Customer Churn</b> is when a bank's customer decides to close their savings account, stop using their debit card, "
        "or transfer their money to another bank. In retail banking, <b>losing existing customers is very expensive</b>—it costs "
        "5 to 7 times more money to find and sign up a brand-new customer than it does to keep an existing customer happy."
    )
    story.append(Paragraph(p1, body_style))

    p2 = (
        "<b>RetainIQ</b> is an intelligent software platform built for banks. It acts as an early-warning alarm system. "
        "By looking at a customer's basic information (such as their age, account balance, how many bank products they use, "
        "and whether they regularly use online banking), RetainIQ's Machine Learning model instantly calculates the <b>exact risk percentage</b> "
        "that this customer might leave the bank in the near future."
    )
    story.append(Paragraph(p2, body_style))

    story.append(Spacer(1, 8))

    # Section: The Three Key Capabilities
    story.append(Paragraph("2. The Three Core Capabilities", heading1_style))

    features_table_data = [
        [
            Paragraph("<b>🎯 1. Instant Risk Prediction</b>", bold_label),
            Paragraph("Bank staff can enter a customer's details and immediately get a calibrated probability score (e.g. <b>95.8% Exit Risk</b> vs. <b>2.1% Low Risk</b>) with color-coded risk tiers (Low, Moderate, High).", card_text)
        ],
        [
            Paragraph("<b>🔍 2. The Reason (Explainability)</b>", bold_label),
            Paragraph("Instead of just giving a number, RetainIQ explains <b>WHY</b>. Powered by <b>SHAP (Explainable AI)</b>, it lists the exact factors pushing the customer to leave (such as single-product vulnerability or inactivity) and factors anchoring them to stay.", card_text)
        ],
        [
            Paragraph("<b>📁 3. Bulk Portfolio Scoring</b>", bold_label),
            Paragraph("Branch managers can upload an entire spreadsheet containing hundreds or thousands of customer records and receive a prioritized, scored list highlighting which accounts need urgent retention calls today.", card_text)
        ],
        [
            Paragraph("<b>⚡ 4. Enterprise Architecture</b>", bold_label),
            Paragraph("Built with a high-speed <b>FastAPI</b> backend (sub-100 millisecond response time) and an intuitive, interactive <b>Streamlit</b> analyst dashboard accessible from any web browser.", card_text)
        ]
    ]

    t_features = Table(features_table_data, colWidths=[150, 365])
    t_features.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 1, BORDER_COL),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COL),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_features)

    story.append(Spacer(1, 12))

    # Key Performance Summary Callout Box
    story.append(Paragraph("3. Verified Model Performance on Held-Out Test Data", heading1_style))
    p_perf = (
        "The winning machine learning model (<b>XGBoost</b> trained on 27 domain-engineered features) was tested on "
        "3,000 previously unseen customer accounts. It achieved an outstanding <b>93.35% ROC-AUC</b>, <b>87.37% Accuracy</b>, "
        "and successfully caught <b>81.70% of all customers who actually exited</b> (500 out of 612 churners)."
    )
    story.append(Paragraph(p_perf, body_style))

    story.append(PageBreak())

    # =========================================================================
    # SCREENSHOT 1: DASHBOARD INPUT INTERFACE
    # =========================================================================
    story.append(Paragraph("Screenshot 1: Main Dashboard — Customer Profile Input Interface", heading1_style))
    
    desc_1 = (
        "<b>What is this showing?</b><br/>"
        "This is the primary user interface of the RetainIQ web application (built with Streamlit). "
        "A banking relationship manager or loan officer enters a customer's demographic and financial attributes into the interactive form on the left. "
        "Inputs include <i>Credit Score, Country (France, Germany, Spain), Gender, Age, Relationship Tenure, Account Balance, Number of Products, "
        "Credit Card ownership, Active Membership status, and Estimated Annual Salary</i>. "
        "At the top, the system displays real-time connection badges confirming the server is online with the production XGBoost model."
    )
    story.append(Paragraph(desc_1, body_style))

    img_1_path = os.path.join(SCREENSHOTS_DIR, "01_dashboard_inputs.png")
    if os.path.exists(img_1_path):
        story.append(RLImage(img_1_path, width=500, height=303))
    
    note_1 = (
        "<b>Key Takeaway:</b> Clean, intuitive web controls allow non-technical banking employees to simulate customer profiles "
        "and test 'what-if' retention scenarios without writing a single line of code."
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph(note_1, callout_style))

    story.append(PageBreak())

    # =========================================================================
    # SCREENSHOT 2: PREDICTION RESULT & CHURN RISK GAUGE
    # =========================================================================
    story.append(Paragraph("Screenshot 2: Prediction Result — Customer Exit Probability & Risk Assessment", heading1_style))

    desc_2 = (
        "<b>What is this showing?</b><br/>"
        "This is the immediate result displayed after clicking the <i>'Evaluate Customer Exit Risk'</i> button. "
        "The system runs the customer data through the trained XGBoost model in under 100 milliseconds and outputs:<br/>"
        "• <b>Exit Probability Gauge:</b> A calibrated risk meter showing the exact calculated likelihood (e.g., <b>1.96% Churn Probability</b>).<br/>"
        "• <b>Status Banner:</b> A clear, color-coded verdict banner (e.g. <b>STATUS: LIKELY TO STAY</b> in green, or <b>LIKELY TO EXIT</b> in red).<br/>"
        "• <b>Risk Tier Badge:</b> Categorizes the customer into <i>Low Risk (&lt;40%)</i>, <i>Moderate Risk (40–70%)</i>, or <i>High Risk (&ge;70%)</i>.<br/>"
        "• <b>Customer Profile Snapshot:</b> Quick-reference financial summary cards showing demographics, balances, and relationship depth."
    )
    story.append(Paragraph(desc_2, body_style))

    img_2_path = os.path.join(SCREENSHOTS_DIR, "02_prediction_result.png")
    if os.path.exists(img_2_path):
        story.append(RLImage(img_2_path, width=500, height=303))

    note_2 = (
        "<b>Key Takeaway:</b> Provides immediate decision support. Relationship managers do not have to guess; they receive an objective, "
        "probabilistic risk rating backed by mathematical machine learning."
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph(note_2, callout_style))

    story.append(PageBreak())

    # =========================================================================
    # SCREENSHOT 3: THE REASON (SHAP EXPLAINABILITY)
    # =========================================================================
    story.append(Paragraph("Screenshot 3: The Reason (Explainability) — Why Did the AI Make This Decision?", heading1_style))

    desc_3 = (
        "<b>What is this showing?</b><br/>"
        "This is the <b>Explainable AI (XAI)</b> section powered by <b>SHAP (SHapley Additive exPlanations)</b>. "
        "Traditional AI models are often criticized as 'black boxes' because banks cannot tell customers or regulators why an automated decision was made. "
        "RetainIQ solves this problem completely by breaking down the decision into individual contributing reasons:<br/>"
        "• <b>Risk Escalators (Red Factors):</b> Attributes that increased the customer's likelihood of leaving (e.g., Older age cohort, resident in Germany, single product).<br/>"
        "• <b>Retention Anchors (Green Factors):</b> Attributes that kept the customer loyal (e.g., Active membership status, holding 2 products, healthy credit score)."
    )
    story.append(Paragraph(desc_3, body_style))

    img_3_path = os.path.join(SCREENSHOTS_DIR, "03_shap_explainability.png")
    if os.path.exists(img_3_path):
        story.append(RLImage(img_3_path, width=500, height=303))

    note_3 = (
        "<b>Key Takeaway:</b> Gives bank staff the exact talking points for a phone call or meeting. If 'Single Product' is the primary risk driver, "
        "the relationship manager can proactively offer a savings bundle or fee waiver to anchor the relationship."
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph(note_3, callout_style))

    story.append(PageBreak())

    # =========================================================================
    # SCREENSHOT 4: BATCH ACCOUNT SCORING
    # =========================================================================
    story.append(Paragraph("Screenshot 4: Batch Portfolio Scoring — Evaluating Multiple Customers at Once", heading1_style))

    desc_4 = (
        "<b>What is this showing?</b><br/>"
        "This is <b>Tab 2: Batch Account Scoring</b>. Banking teams often manage thousands of accounts and cannot enter them one by one. "
        "Here, an analyst can drag-and-drop a CSV spreadsheet containing customer accounts (or click <i>'Load Benchmark Sample'</i>). "
        "The system processes the entire batch in a single vectorized pass and presents:<br/>"
        "• <b>Portfolio KPI Cards:</b> Total Customers Evaluated, Number of Flagged Churners, High Risk Count, and Mean Portfolio Churn Rate.<br/>"
        "• <b>Prioritized Action Table:</b> A sorted, searchable table listing customers by their Exit Probability so the bank can prioritize who to contact first.<br/>"
        "• <b>One-Click CRM Export:</b> A button to download the enriched spreadsheet with predictions and risk tiers for direct import into core banking CRMs."
    )
    story.append(Paragraph(desc_4, body_style))

    img_4_path = os.path.join(SCREENSHOTS_DIR, "04_batch_scoring.png")
    if os.path.exists(img_4_path):
        story.append(RLImage(img_4_path, width=500, height=303))

    note_4 = (
        "<b>Key Takeaway:</b> Transforms RetainIQ from a single-customer desktop tool into an enterprise-scale portfolio surveillance system "
        "capable of auditing an entire branch's customer base in seconds."
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph(note_4, callout_style))

    story.append(PageBreak())

    # =========================================================================
    # SCREENSHOT 5: MODEL SCORECARD & EVALUATION ARTIFACTS
    # =========================================================================
    story.append(Paragraph("Screenshot 5: Model Evaluation — Performance Scorecard & Scientific Benchmarks", heading1_style))

    desc_5 = (
        "<b>What is this showing?</b><br/>"
        "This is <b>Tab 3: Model Scorecard & Evaluation Artifacts</b>. This view guarantees transparency and scientific rigor for auditors and academic evaluators. "
        "It pulls live verified metrics measured against the 3,000 held-out test customers:<br/>"
        "• <b>Verified Metric Cards:</b> <b>87.37% Accuracy</b>, <b>65.19% Precision</b>, <b>81.70% Recall</b> (500 of 612 actual exiters captured), "
        "<b>72.52% F1 Score</b>, and <b>93.35% ROC-AUC</b>.<br/>"
        "• <b>Confusion Matrix:</b> Displays exact counts of True Retained (2,121), False Positives (267), False Negatives (112), and True Exiters (500).<br/>"
        "• <b>Interactive Artifact Viewer:</b> Directly embeds Phase 5 publication-grade figures (ROC curve, PR curve, Feature Importance bar charts)."
    )
    story.append(Paragraph(desc_5, body_style))

    img_5_path = os.path.join(SCREENSHOTS_DIR, "05_model_metrics_scorecard.png")
    if os.path.exists(img_5_path):
        story.append(RLImage(img_5_path, width=500, height=303))

    note_5 = (
        "<b>Key Takeaway:</b> Proves that the model is genuinely evaluated on an untouched test split with no data leakage, "
        "meeting university academic standards and enterprise banking model risk governance requirements."
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph(note_5, callout_style))

    story.append(PageBreak())

    # =========================================================================
    # SCREENSHOT 6: FASTAPI BACKEND & SWAGGER DOCUMENTATION
    # =========================================================================
    story.append(Paragraph("Screenshot 6: Backend Serving — FastAPI Interactive OpenAPI Documentation (/docs)", heading1_style))

    desc_6 = (
        "<b>What is this showing?</b><br/>"
        "This is the automated interactive <b>Swagger / OpenAPI</b> documentation interface generated by the <b>FastAPI backend</b> at <i>http://127.0.0.1:8000/docs</i>. "
        "It provides software engineers and enterprise integration teams with live, testable API endpoints:<br/>"
        "• <b>GET /health:</b> System health check verifying model memory residency and version details.<br/>"
        "• <b>GET /model/info:</b> Model metadata, training timestamp, and test set performance scorecard.<br/>"
        "• <b>POST /predict:</b> Real-time single customer inference with strict Pydantic V2 input validation and SHAP explainability.<br/>"
        "• <b>POST /predict/batch:</b> High-throughput vectorized portfolio scoring for batch payloads.<br/>"
        "• <b>GET /monitoring/telemetry:</b> Real-time inference request counts, latency percentiles, and audit logs.<br/>"
        "• <b>POST /monitoring/drift:</b> Statistical covariate data drift monitoring comparing serving data against training baselines."
    )
    story.append(Paragraph(desc_6, body_style))

    img_6_path = os.path.join(SCREENSHOTS_DIR, "06_fastapi_swagger_docs.png")
    if os.path.exists(img_6_path):
        story.append(RLImage(img_6_path, width=500, height=303))

    note_6 = (
        "<b>Key Takeaway:</b> Demonstrates clean decoupling between the frontend UI and the backend API. "
        "Any external core-banking CRM or mobile app can easily connect to RetainIQ through standard RESTful JSON requests."
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph(note_6, callout_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {PDF_PATH}")


if __name__ == "__main__":
    create_pdf()

