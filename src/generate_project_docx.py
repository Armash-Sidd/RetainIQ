"""
RetainIQ — Editable Word Document (.docx) Generator.
Generates an elegant, publication-quality editable Microsoft Word document (.docx) containing:
1. Clear, plain-language ("little language") explanation of the project.
2. Dedicated sections with descriptive headings ("evaluation", "prediction", "reason", etc.).
3. Embedded genuine, high-resolution screenshots of the live system.
4. "What is this showing?" breakdown and "Key Takeaways" for every view.
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DOCX = os.path.join(BASE_DIR, "RetainIQ_Project_Walkthrough.docx")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

# Theme Palette
COLOR_NAVY = RGBColor(15, 41, 66)      # #0F2942
COLOR_BLUE = RGBColor(37, 99, 235)     # #2563EB
COLOR_DARK = RGBColor(30, 41, 59)      # #1E293B
COLOR_MUTED = RGBColor(100, 116, 139)  # #64748B
COLOR_GREEN = RGBColor(5, 150, 105)    # #059669
HEX_LIGHT_BG = "F8FAFC"
HEX_BORDER = "CBD5E1"
HEX_PRIMARY_BG = "EFF6FF"

def set_cell_background(cell, hex_color):
    """Sets background shading of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    """Sets inner margins (padding) of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'  <w:top w:w="{top}" w:type="dxa"/>'
        f'  <w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'  <w:left w:w="{left}" w:type="dxa"/>'
        f'  <w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)

def add_callout_box(doc, title, text, border_color="2563EB", bg_color="EFF6FF"):
    """Adds a stylish callout box using a 1x1 table."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)

    tc_pr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="none"/>'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
        f'  <w:bottom w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tc_pr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    run_t = p.add_run(title)
    run_t.bold = True
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(10)
    run_t.font.color.rgb = COLOR_BLUE

    p2 = cell.add_paragraph()
    p2.paragraph_format.space_before = Pt(2)
    p2.paragraph_format.space_after = Pt(0)
    run_b = p2.add_run(text)
    run_b.font.name = "Calibri"
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = COLOR_DARK

    # Add spacing after table
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)

def generate_docx():
    doc = docx.Document()

    # Configure Margins (0.8 inches all around)
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        
        # Header & Footer setup
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("RetainIQ — Customer Churn Prediction & Decision System | Project Walkthrough")
        hrun.font.name = "Calibri"
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = COLOR_MUTED

        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.LEFT
        frun = fp.add_run("Confidential — Academic & Technical Evaluation Document")
        frun.font.name = "Calibri"
        frun.font.size = Pt(8.5)
        frun.font.color.rgb = COLOR_MUTED

    # =========================================================================
    # COVER / SECTION 1: OVERVIEW IN SIMPLE LANGUAGE
    # =========================================================================
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    r_title = title_p.add_run("RetainIQ: Customer Exit & Churn Prediction System")
    r_title.bold = True
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(24)
    r_title.font.color.rgb = COLOR_NAVY

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(0)
    sub_p.paragraph_format.space_after = Pt(14)
    r_sub = sub_p.add_run("Complete System Walkthrough, Real Application Screenshots & Explainability Guide")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = COLOR_BLUE

    # 1. What is this project in simple words?
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(10)
    h1.paragraph_format.space_after = Pt(4)
    r_h1 = h1.add_run("1. What is this project in simple words?")
    r_h1.bold = True
    r_h1.font.name = "Calibri"
    r_h1.font.size = Pt(15)
    r_h1.font.color.rgb = COLOR_NAVY

    p_what1 = doc.add_paragraph()
    p_what1.paragraph_format.space_after = Pt(6)
    p_what1.paragraph_format.line_spacing = 1.15
    r = p_what1.add_run(
        "Customer Churn happens when a bank's customer decides to close their savings account, stop using their debit card, "
        "or transfer their money to another bank. In retail banking, losing existing customers is very expensive—it typically costs "
        "5 to 7 times more money to advertise and sign up a brand-new customer than it does to keep an existing customer happy."
    )
    r.font.name = "Calibri"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_DARK

    p_what2 = doc.add_paragraph()
    p_what2.paragraph_format.space_after = Pt(10)
    p_what2.paragraph_format.line_spacing = 1.15
    r = p_what2.add_run(
        "RetainIQ is an intelligent software platform designed specifically for banks. It acts as an early-warning alarm system. "
        "By looking at a customer's basic profile (such as their age, account balance, how many bank products they use, "
        "and whether they regularly use online banking), RetainIQ's Machine Learning model instantly calculates the exact risk percentage "
        "that this customer might leave the bank in the near future."
    )
    r.font.name = "Calibri"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_DARK

    # 2. Four Core Capabilities
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(6)
    h2.paragraph_format.space_after = Pt(4)
    r_h2 = h2.add_run("2. The Four Core Capabilities")
    r_h2.bold = True
    r_h2.font.name = "Calibri"
    r_h2.font.size = Pt(15)
    r_h2.font.color.rgb = COLOR_NAVY

    table_caps = doc.add_table(rows=4, cols=2)
    table_caps.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_caps.autofit = False

    caps_data = [
        (
            "🎯 1. Instant Risk Prediction",
            "Bank staff can enter a customer's details and immediately get a calibrated probability score "
            "(e.g. 1.96% Low Risk vs. 95.8% High Risk) with color-coded risk tiers (Low, Moderate, High)."
        ),
        (
            "🔍 2. The Reason (Explainability)",
            "Instead of just giving a number, RetainIQ explains WHY. Powered by SHAP (Explainable AI), "
            "it highlights the exact factors pushing the customer to leave (risk escalators) and factors keeping them loyal (retention anchors)."
        ),
        (
            "📁 3. Bulk Portfolio Scoring",
            "Branch managers can upload an entire spreadsheet containing hundreds or thousands of customer accounts "
            "and receive a prioritized, sorted action table highlighting which accounts need urgent retention calls today."
        ),
        (
            "⚡ 4. Enterprise Architecture",
            "Built with a high-speed FastAPI backend (sub-100ms response time) and an intuitive, interactive "
            "Streamlit analyst dashboard accessible from any web browser."
        )
    ]

    for row_idx, (col1_text, col2_text) in enumerate(caps_data):
        row = table_caps.rows[row_idx]
        
        # Col 1
        c1 = row.cells[0]
        c1.width = Inches(2.2)
        set_cell_background(c1, HEX_LIGHT_BG)
        set_cell_margins(c1, top=80, bottom=80, left=100, right=100)
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(col1_text)
        r1.bold = True
        r1.font.name = "Calibri"
        r1.font.size = Pt(9.5)
        r1.font.color.rgb = COLOR_NAVY

        # Col 2
        c2 = row.cells[1]
        c2.width = Inches(4.3)
        set_cell_background(c2, HEX_LIGHT_BG)
        set_cell_margins(c2, top=80, bottom=80, left=100, right=100)
        p2 = c2.paragraphs[0]
        p2.paragraph_format.space_after = Pt(0)
        r2 = p2.add_run(col2_text)
        r2.font.name = "Calibri"
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = COLOR_DARK

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 3. Model Performance Summary
    h3 = doc.add_paragraph()
    h3.paragraph_format.space_before = Pt(6)
    h3.paragraph_format.space_after = Pt(4)
    r_h3 = h3.add_run("3. Verified Model Performance on Held-Out Test Data")
    r_h3.bold = True
    r_h3.font.name = "Calibri"
    r_h3.font.size = Pt(15)
    r_h3.font.color.rgb = COLOR_NAVY

    p_perf = doc.add_paragraph()
    p_perf.paragraph_format.space_after = Pt(10)
    p_perf.paragraph_format.line_spacing = 1.15
    r = p_perf.add_run(
        "The winning machine learning model (XGBoost trained on 27 domain-engineered features) was tested on "
        "3,000 previously unseen customer accounts. It achieved an outstanding 93.35% ROC-AUC, 87.37% Accuracy, "
        "and successfully caught 81.70% of all customers who actually exited (500 out of 612 churners)."
    )
    r.font.name = "Calibri"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_DARK

    # Page Break to Screenshot 1
    doc.add_page_break()

    # =========================================================================
    # SCREENSHOT 1: DASHBOARD INPUT INTERFACE
    # =========================================================================
    h_s1 = doc.add_paragraph()
    h_s1.paragraph_format.space_before = Pt(0)
    h_s1.paragraph_format.space_after = Pt(4)
    r_s1 = h_s1.add_run("Screenshot 1: Main Dashboard — Customer Profile Input Interface")
    r_s1.bold = True
    r_s1.font.name = "Calibri"
    r_s1.font.size = Pt(14)
    r_s1.font.color.rgb = COLOR_NAVY

    p_d1 = doc.add_paragraph()
    p_d1.paragraph_format.space_after = Pt(6)
    p_d1.paragraph_format.line_spacing = 1.15
    r = p_d1.add_run("What is this showing?\n")
    r.bold = True
    r.font.name = "Calibri"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_BLUE

    r_desc1 = p_d1.add_run(
        "This is the primary user interface of the RetainIQ web application (built with Streamlit). "
        "A banking relationship manager or loan officer enters a customer's demographic and financial attributes into the interactive form on the left. "
        "Inputs include Credit Score, Country (France, Germany, Spain), Gender, Age, Relationship Tenure, Account Balance, Number of Products, "
        "Credit Card ownership, Active Membership status, and Estimated Annual Salary. "
        "At the top, the system displays real-time connection badges confirming the server is online with the production XGBoost model."
    )
    r_desc1.font.name = "Calibri"
    r_desc1.font.size = Pt(9.5)
    r_desc1.font.color.rgb = COLOR_DARK

    img1_path = os.path.join(SCREENSHOTS_DIR, "01_dashboard_inputs.png")
    if os.path.exists(img1_path):
        p_img1 = doc.add_paragraph()
        p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img1.paragraph_format.space_before = Pt(4)
        p_img1.paragraph_format.space_after = Pt(6)
        doc.add_picture(img1_path, width=Inches(6.2))

    add_callout_box(
        doc,
        "💡 Key Takeaway: Intuitive Customer Simulation",
        "Clean, intuitive web controls allow non-technical banking employees to simulate customer profiles "
        "and test 'what-if' retention scenarios without writing a single line of code."
    )

    doc.add_page_break()

    # =========================================================================
    # SCREENSHOT 2: PREDICTION RESULT & EXIT PROBABILITY GAUGE
    # =========================================================================
    h_s2 = doc.add_paragraph()
    h_s2.paragraph_format.space_before = Pt(0)
    h_s2.paragraph_format.space_after = Pt(4)
    r_s2 = h_s2.add_run("Screenshot 2: Prediction Result — Customer Exit Probability & Risk Assessment")
    r_s2.bold = True
    r_s2.font.name = "Calibri"
    r_s2.font.size = Pt(14)
    r_s2.font.color.rgb = COLOR_NAVY

    p_d2 = doc.add_paragraph()
    p_d2.paragraph_format.space_after = Pt(6)
    p_d2.paragraph_format.line_spacing = 1.15
    r = p_d2.add_run("What is this showing?\n")
    r.bold = True
    r.font.name = "Calibri"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_BLUE

    r_desc2 = p_d2.add_run(
        "This is the immediate result displayed after clicking the 'Evaluate Customer Exit Risk' button. "
        "The system runs the customer data through the trained XGBoost model in under 100 milliseconds and outputs:\n"
        "• Exit Probability Gauge: A calibrated risk meter showing the exact calculated likelihood (e.g. 1.96% Churn Probability).\n"
        "• Status Banner: A clear, color-coded verdict banner (e.g. STATUS: LIKELY TO STAY in green, or LIKELY TO EXIT in red).\n"
        "• Risk Tier Badge: Categorizes the customer into Low Risk (<40%), Moderate Risk (40–70%), or High Risk (>=70%).\n"
        "• Customer Profile Snapshot: Quick-reference financial summary cards showing demographics, balances, and relationship depth."
    )
    r_desc2.font.name = "Calibri"
    r_desc2.font.size = Pt(9.5)
    r_desc2.font.color.rgb = COLOR_DARK

    img2_path = os.path.join(SCREENSHOTS_DIR, "02_prediction_result.png")
    if os.path.exists(img2_path):
        p_img2 = doc.add_paragraph()
        p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img2.paragraph_format.space_before = Pt(4)
        p_img2.paragraph_format.space_after = Pt(6)
        doc.add_picture(img2_path, width=Inches(6.2))

    add_callout_box(
        doc,
        "💡 Key Takeaway: Immediate Decision Support",
        "Provides immediate decision support. Relationship managers do not have to guess; they receive an objective, "
        "probabilistic risk rating backed by mathematical machine learning."
    )

    doc.add_page_break()

    # =========================================================================
    # SCREENSHOT 3: THE REASON (SHAP EXPLAINABILITY)
    # =========================================================================
    h_s3 = doc.add_paragraph()
    h_s3.paragraph_format.space_before = Pt(0)
    h_s3.paragraph_format.space_after = Pt(4)
    r_s3 = h_s3.add_run("Screenshot 3: The Reason (Explainability) — Why Did the AI Make This Decision?")
    r_s3.bold = True
    r_s3.font.name = "Calibri"
    r_s3.font.size = Pt(14)
    r_s3.font.color.rgb = COLOR_NAVY

    p_d3 = doc.add_paragraph()
    p_d3.paragraph_format.space_after = Pt(6)
    p_d3.paragraph_format.line_spacing = 1.15
    r = p_d3.add_run("What is this showing?\n")
    r.bold = True
    r.font.name = "Calibri"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_BLUE

    r_desc3 = p_d3.add_run(
        "This is the Explainable AI (XAI) section powered by SHAP (SHapley Additive exPlanations). "
        "Traditional AI models are often criticized as 'black boxes' because banks cannot explain why an automated decision was made. "
        "RetainIQ solves this problem completely by breaking down the decision into individual contributing reasons:\n"
        "• Risk Escalators (Red Factors): Attributes that increased the customer's likelihood of leaving (e.g., Older age cohort, resident in Germany, single product).\n"
        "• Retention Anchors (Green Factors): Attributes that kept the customer loyal (e.g., Active membership status, holding 2 products, healthy credit score)."
    )
    r_desc3.font.name = "Calibri"
    r_desc3.font.size = Pt(9.5)
    r_desc3.font.color.rgb = COLOR_DARK

    img3_path = os.path.join(SCREENSHOTS_DIR, "03_shap_explainability.png")
    if os.path.exists(img3_path):
        p_img3 = doc.add_paragraph()
        p_img3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img3.paragraph_format.space_before = Pt(4)
        p_img3.paragraph_format.space_after = Pt(6)
        doc.add_picture(img3_path, width=Inches(6.2))

    add_callout_box(
        doc,
        "💡 Key Takeaway: Actionable Retention Talking Points",
        "Gives bank staff the exact talking points for a phone call or meeting. If 'Single Product' is the primary risk driver, "
        "the relationship manager can proactively offer a savings bundle or fee waiver to anchor the relationship."
    )

    doc.add_page_break()

    # =========================================================================
    # SCREENSHOT 4: BATCH ACCOUNT SCORING
    # =========================================================================
    h_s4 = doc.add_paragraph()
    h_s4.paragraph_format.space_before = Pt(0)
    h_s4.paragraph_format.space_after = Pt(4)
    r_s4 = h_s4.add_run("Screenshot 4: Batch Portfolio Scoring — Evaluating Multiple Customers at Once")
    r_s4.bold = True
    r_s4.font.name = "Calibri"
    r_s4.font.size = Pt(14)
    r_s4.font.color.rgb = COLOR_NAVY

    p_d4 = doc.add_paragraph()
    p_d4.paragraph_format.space_after = Pt(6)
    p_d4.paragraph_format.line_spacing = 1.15
    r = p_d4.add_run("What is this showing?\n")
    r.bold = True
    r.font.name = "Calibri"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_BLUE

    r_desc4 = p_d4.add_run(
        "This is Tab 2: Batch Account Scoring. Banking teams often manage thousands of accounts and cannot enter them one by one. "
        "Here, an analyst can drag-and-drop a CSV spreadsheet containing customer accounts (or click 'Load Benchmark Sample'). "
        "The system processes the entire batch in a single vectorized pass and presents:\n"
        "• Portfolio KPI Cards: Total Customers Evaluated, Number of Flagged Churners, High Risk Count, and Mean Portfolio Churn Rate.\n"
        "• Prioritized Action Table: A sorted, searchable table listing customers by their Exit Probability so the bank can prioritize who to contact first.\n"
        "• One-Click CRM Export: A button to download the enriched spreadsheet with predictions and risk tiers for direct import into core banking CRMs."
    )
    r_desc4.font.name = "Calibri"
    r_desc4.font.size = Pt(9.5)
    r_desc4.font.color.rgb = COLOR_DARK

    img4_path = os.path.join(SCREENSHOTS_DIR, "04_batch_scoring.png")
    if os.path.exists(img4_path):
        p_img4 = doc.add_paragraph()
        p_img4.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img4.paragraph_format.space_before = Pt(4)
        p_img4.paragraph_format.space_after = Pt(6)
        doc.add_picture(img4_path, width=Inches(6.2))

    add_callout_box(
        doc,
        "💡 Key Takeaway: Enterprise Portfolio Risk Surveillance",
        "Transforms RetainIQ from a single-customer desktop tool into an enterprise-scale portfolio surveillance system "
        "capable of auditing an entire branch's customer base in seconds."
    )

    doc.add_page_break()

    # =========================================================================
    # SCREENSHOT 5: MODEL SCORECARD & EVALUATION ARTIFACTS
    # =========================================================================
    h_s5 = doc.add_paragraph()
    h_s5.paragraph_format.space_before = Pt(0)
    h_s5.paragraph_format.space_after = Pt(4)
    r_s5 = h_s5.add_run("Screenshot 5: Model Evaluation — Performance Scorecard & Scientific Benchmarks")
    r_s5.bold = True
    r_s5.font.name = "Calibri"
    r_s5.font.size = Pt(14)
    r_s5.font.color.rgb = COLOR_NAVY

    p_d5 = doc.add_paragraph()
    p_d5.paragraph_format.space_after = Pt(6)
    p_d5.paragraph_format.line_spacing = 1.15
    r = p_d5.add_run("What is this showing?\n")
    r.bold = True
    r.font.name = "Calibri"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_BLUE

    r_desc5 = p_d5.add_run(
        "This is Tab 3: Model Scorecard & Evaluation Artifacts. This view guarantees transparency and scientific rigor for auditors and academic evaluators. "
        "It pulls live verified metrics measured against the 3,000 held-out test customers:\n"
        "• Verified Metric Cards: 87.37% Accuracy, 65.19% Precision, 81.70% Recall (500 of 612 actual exiters captured), "
        "72.52% F1 Score, and 93.35% ROC-AUC.\n"
        "• Confusion Matrix: Displays exact counts of True Retained (2,121), False Positives (267), False Negatives (112), and True Exiters (500).\n"
        "• Interactive Artifact Viewer: Directly embeds Phase 5 publication-grade figures (ROC curve, PR curve, Feature Importance bar charts)."
    )
    r_desc5.font.name = "Calibri"
    r_desc5.font.size = Pt(9.5)
    r_desc5.font.color.rgb = COLOR_DARK

    img5_path = os.path.join(SCREENSHOTS_DIR, "05_model_metrics_scorecard.png")
    if os.path.exists(img5_path):
        p_img5 = doc.add_paragraph()
        p_img5.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img5.paragraph_format.space_before = Pt(4)
        p_img5.paragraph_format.space_after = Pt(6)
        doc.add_picture(img5_path, width=Inches(6.2))

    add_callout_box(
        doc,
        "💡 Key Takeaway: Academic & Governance Rigor",
        "Proves that the model is genuinely evaluated on an untouched test split with no data leakage, "
        "meeting university academic standards and enterprise banking model risk governance requirements."
    )

    doc.add_page_break()

    # =========================================================================
    # SCREENSHOT 6: FASTAPI BACKEND & SWAGGER DOCUMENTATION
    # =========================================================================
    h_s6 = doc.add_paragraph()
    h_s6.paragraph_format.space_before = Pt(0)
    h_s6.paragraph_format.space_after = Pt(4)
    r_s6 = h_s6.add_run("Screenshot 6: Backend Serving — FastAPI Interactive OpenAPI Documentation (/docs)")
    r_s6.bold = True
    r_s6.font.name = "Calibri"
    r_s6.font.size = Pt(14)
    r_s6.font.color.rgb = COLOR_NAVY

    p_d6 = doc.add_paragraph()
    p_d6.paragraph_format.space_after = Pt(6)
    p_d6.paragraph_format.line_spacing = 1.15
    r = p_d6.add_run("What is this showing?\n")
    r.bold = True
    r.font.name = "Calibri"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_BLUE

    r_desc6 = p_d6.add_run(
        "This is the automated interactive Swagger / OpenAPI documentation interface generated by the FastAPI backend at http://127.0.0.1:8000/docs. "
        "It provides software engineers and enterprise integration teams with live, testable API endpoints:\n"
        "• GET /health: System health check verifying model memory residency and version details.\n"
        "• GET /model/info: Model metadata, training timestamp, and test set performance scorecard.\n"
        "• POST /predict: Real-time single customer inference with strict Pydantic V2 input validation and SHAP explainability.\n"
        "• POST /predict/batch: High-throughput vectorized portfolio scoring for batch payloads.\n"
        "• GET /monitoring/telemetry: Real-time inference request counts, latency percentiles, and audit logs.\n"
        "• POST /monitoring/drift: Statistical covariate data drift monitoring comparing serving data against training baselines."
    )
    r_desc6.font.name = "Calibri"
    r_desc6.font.size = Pt(9.5)
    r_desc6.font.color.rgb = COLOR_DARK

    img6_path = os.path.join(SCREENSHOTS_DIR, "06_fastapi_swagger_docs.png")
    if os.path.exists(img6_path):
        p_img6 = doc.add_paragraph()
        p_img6.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img6.paragraph_format.space_before = Pt(4)
        p_img6.paragraph_format.space_after = Pt(6)
        doc.add_picture(img6_path, width=Inches(6.2))

    add_callout_box(
        doc,
        "💡 Key Takeaway: Decoupled Enterprise Microservice",
        "Demonstrates clean decoupling between the frontend UI and the backend API. "
        "Any external core-banking CRM or mobile app can easily connect to RetainIQ through standard RESTful JSON requests."
    )

    # Save Word document
    doc.save(OUTPUT_DOCX)
    print(f"Successfully generated editable Word document: {OUTPUT_DOCX}")

if __name__ == "__main__":
    generate_docx()
