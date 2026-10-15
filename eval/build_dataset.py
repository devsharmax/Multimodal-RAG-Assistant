"""Builds a labelled multimodal evaluation set.

Three synthetic PDFs are generated in which every fact lives in exactly one modality
(body text, a table, a raster chart/diagram, or a vector drawing), so retrieval and answer
quality can be measured per modality. The project synopsis PDF is added as a real-world
document. Output: eval/data/*.pdf and eval/data/qa.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).parent
OUT = HERE / "data"
SYNOPSIS = HERE.parent / "samples" / "Multimodal_RAG_Synopsis_final.pdf"
ss = getSampleStyleSheet()
H1, H2, P, CAP = ss["Title"], ss["Heading2"], ss["BodyText"], ss["Italic"]


def _png(fig, name) -> str:
    p = OUT / "_img" / f"{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def _table(rows):
    t = Table(rows, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde4ee")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return t


def _build(name, title, story):
    doc = SimpleDocTemplate(str(OUT / name), pagesize=A4, title=title, topMargin=2 * cm, bottomMargin=2 * cm)

    def footer(c, d):
        c.setFont("Helvetica", 8)
        c.drawString(2 * cm, 1.2 * cm, f"{title} - confidential draft")
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(d.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# ---------------------------------------------------------------------------------------------
def aurora():
    # bar chart (raster): regional capacity - numbers appear ONLY in the chart
    fig, ax = plt.subplots(figsize=(6, 3.4))
    regions, mw = ["Nordics", "Iberia", "Germany", "Italy"], [620, 540, 410, 237]
    bars = ax.bar(regions, mw, color=["#2a6f97", "#61a5c2", "#89c2d9", "#a9d6e5"])
    ax.bar_label(bars, labels=[f"{v} MW" for v in mw])
    ax.set_title("Installed capacity by region, 2025 (MW)")
    ax.set_ylabel("MW")
    bar_png = _png(fig, "aurora_bar")

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.pie([46, 32, 22], labels=["Residential", "Commercial", "Utility-scale"], autopct="%d%%", startangle=90,
           colors=["#f4a261", "#2a9d8f", "#264653"])
    ax.set_title("Revenue mix by segment, 2025")
    pie_png = _png(fig, "aurora_pie")

    # vector line chart drawn with reportlab paths
    d = Drawing(420, 220)
    lp = LinePlot()
    lp.x, lp.y, lp.width, lp.height = 50, 40, 340, 150
    years = list(range(2019, 2026))
    eff = [19.2, 19.9, 20.6, 21.4, 22.1, 22.9, 23.6]
    lp.data = [list(zip(years, eff))]
    lp.lines[0].strokeColor = colors.HexColor("#2a6f97")
    lp.lineLabelFormat = "%2.1f%%"
    lp.xValueAxis.valueMin, lp.xValueAxis.valueMax, lp.xValueAxis.valueStep = 2019, 2025, 1
    lp.yValueAxis.valueMin, lp.yValueAxis.valueMax = 18, 25
    d.add(lp)
    d.add(String(210, 205, "Average module efficiency (%)", textAnchor="middle", fontSize=11))

    story = [
        Paragraph("Aurora Solar ASA - Annual Report 2025", H1),
        Paragraph("1. Letter from the Chief Executive", H2),
        Paragraph(
            "Aurora Solar ASA is headquartered in Bergen, Norway, and employed 4,850 people at the end of the "
            "year. Our Chief Executive Officer, Priya Raghunathan, joined the company in 2021 and has led its "
            "expansion from a regional installer into a pan-European solar developer. This report covers the "
            "financial year ending 31 December 2025.", P),
        Paragraph(
            "During 2025 we strengthened our installation partner network, opened two new training academies, "
            "and continued to invest in high-efficiency bifacial modules. Safety performance improved, with a "
            "lost-time injury rate of 0.8 per million hours worked.", P),
        Paragraph("2. Financial performance", H2),
        Paragraph("Quarterly results improved steadily throughout the year, as summarised below.", P),
        Spacer(1, 6),
        _table([
            ["Quarter", "Revenue (USD M)", "Net margin", "New installations (MW)"],
            ["Q1 2025", "182.4", "9.1%", "410"],
            ["Q2 2025", "205.7", "10.4%", "455"],
            ["Q3 2025", "231.9", "12.2%", "502"],
            ["Q4 2025", "248.3", "12.8%", "540"],
        ]),
        Paragraph("Table 1: Quarterly financial results, 2025", CAP),
        PageBreak(),
        Paragraph("3. Regional development", H2),
        Paragraph(
            "Regional expansion continued across all four operating markets. Growth was strongest where "
            "permitting reforms shortened project lead times. The distribution of installed capacity across "
            "regions is shown in Figure 1.", P),
        Image(bar_png, width=14 * cm, height=8 * cm),
        Paragraph("Figure 1: Installed capacity by region (MW), 2025", CAP),
        Spacer(1, 10),
        Paragraph("4. Revenue by segment", H2),
        Paragraph("Our customer base spans households, businesses and grid-scale buyers (see Figure 2).", P),
        Image(pie_png, width=10 * cm, height=7.8 * cm),
        Paragraph("Figure 2: Revenue mix by customer segment", CAP),
        PageBreak(),
        Paragraph("5. Technology", H2),
        Paragraph(
            "Module efficiency is a key driver of the levelised cost of energy. Figure 3 shows the average "
            "efficiency of modules deployed in our projects since 2019.", P),
        d,
        Paragraph("Figure 3: Average deployed module efficiency, 2019-2025", CAP),
        Paragraph("6. Outlook", H2),
        Paragraph(
            "For 2026, the board has set a target of 3.1 GW of cumulative installed capacity and plans to enter "
            "the Polish market in the second half of the year.", P),
    ]
    _build("aurora_annual_report_2025.pdf", "Aurora Solar Annual Report 2025", story)


# ---------------------------------------------------------------------------------------------
def helix():
    fig, ax = plt.subplots(figsize=(7.5, 1.9))
    steps = ["Power On", "Self-Test", "Home Axes", "Load Program", "Run Cycle"]
    for i, s_ in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i * 1.6, 0.3), 1.25, 0.6, boxstyle="round,pad=0.05", fc="#e9f5db", ec="#386641"))
        ax.text(i * 1.6 + 0.625, 0.6, s_, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((i * 1.6 + 1.3, 0.6), (i * 1.6 + 1.55, 0.6), arrowstyle="->", mutation_scale=12))
    ax.set_xlim(-0.2, 8.1)
    ax.set_ylim(0, 1.2)
    ax.axis("off")
    ax.set_title("Controller startup sequence")
    flow_png = _png(fig, "helix_flow")

    # vector diagram: joint layout with labels drawn as PDF text/paths
    d = Drawing(430, 170)
    joints = [("J1", "Base", "120°/s"), ("J2", "Shoulder", "120°/s"), ("J3", "Elbow", "180°/s"),
              ("J4", "Wrist roll", "250°/s"), ("J5", "Wrist pitch", "250°/s"), ("J6", "Flange", "320°/s")]
    for i, (j, name, speed) in enumerate(joints):
        x = 10 + i * 70
        d.add(Rect(x, 60, 60, 50, fillColor=colors.HexColor("#cfe1f2"), strokeColor=colors.HexColor("#1d3557")))
        d.add(String(x + 30, 92, j, textAnchor="middle", fontSize=11, fontName="Helvetica-Bold"))
        d.add(String(x + 30, 78, name, textAnchor="middle", fontSize=7))
        d.add(String(x + 30, 64, speed, textAnchor="middle", fontSize=7))
        if i:
            d.add(Line(x - 10, 85, x, 85, strokeColor=colors.HexColor("#1d3557")))
    d.add(String(215, 140, "Helix R7 joint layout and maximum joint speeds", textAnchor="middle", fontSize=10))
    d.add(String(215, 30, "Joint speeds measured at rated payload", textAnchor="middle", fontSize=7))

    story = [
        Paragraph("Helix R7 Collaborative Robot Arm - Technical Manual", H1),
        Paragraph("1. Introduction", H2),
        Paragraph(
            "The Helix R7 is a six-axis collaborative robot arm designed for machine tending, screw driving and "
            "light assembly. Units ship with controller firmware HX-OS 4.2.1. Read the safety chapter before "
            "operating the arm.", P),
        Paragraph("2. Safety", H2),
        Paragraph(
            "The emergency stop circuit must be functionally tested every 250 operating hours and after any "
            "collision. Never bypass the protective stop inputs. Operators must complete the Level 2 Helix "
            "certification course before changing safety parameters.", P),
        Paragraph("3. Specifications", H2),
        _table([
            ["Parameter", "Value"],
            ["Maximum payload", "7 kg"],
            ["Reach", "912 mm"],
            ["Pose repeatability", "±0.02 mm"],
            ["Arm weight", "24.5 kg"],
            ["Ingress protection", "IP54"],
            ["Supply voltage", "48 V DC"],
        ]),
        Paragraph("Table 1: Helix R7 specifications", CAP),
        PageBreak(),
        Paragraph("4. Startup", H2),
        Paragraph("When the controller boots it executes a fixed sequence of steps before any program can run.", P),
        Image(flow_png, width=16 * cm, height=4.1 * cm),
        Paragraph("Figure 1: Controller startup sequence", CAP),
        Spacer(1, 10),
        Paragraph("5. Mechanical layout", H2),
        Paragraph("The arm consists of six revolute joints arranged as shown in Figure 2.", P),
        d,
        Paragraph("Figure 2: Joint layout of the Helix R7", CAP),
        PageBreak(),
        Paragraph("6. Troubleshooting", H2),
        Paragraph("The controller reports faults with a three-digit error code.", P),
        _table([
            ["Error code", "Meaning", "Corrective action"],
            ["E101", "Overcurrent on joint J2", "Reduce payload or speed"],
            ["E204", "Encoder fault", "Recalibrate the joint encoder"],
            ["E305", "Communication timeout", "Check the EtherCAT cable"],
            ["E412", "Over-temperature", "Allow the arm to cool for 30 minutes"],
        ]),
        Paragraph("Table 2: Controller error codes", CAP),
    ]
    _build("helix_r7_manual.pdf", "Helix R7 Technical Manual", story)


# ---------------------------------------------------------------------------------------------
def river():
    fig, ax = plt.subplots(figsize=(6, 3.4))
    st, val = ["S1", "S2", "S3", "S4", "S5", "S6"], [4.2, 6.8, 9.1, 11.8, 7.5, 5.3]
    bars = ax.bar(st, val, color="#457b9d")
    ax.bar_label(bars, labels=[f"{v}" for v in val])
    ax.axhline(10, color="#e63946", ls="--")
    ax.text(5.4, 10.3, "Limit 10 mg/L", color="#e63946", ha="right")
    ax.set_title("Mean nitrate concentration by station (mg/L)")
    ax.set_ylabel("mg/L")
    nit_png = _png(fig, "river_nitrate")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    xs = [0, 1, 2, 3, 4, 5, 6, 7]
    ax.plot(xs, [0, 0.6, 0.4, 1.1, 1.5, 1.2, 1.9, 2.3], color="#4a90d9", lw=6, alpha=0.6)
    for i, (x, y) in enumerate([(0.5, 0.3), (1.8, 0.6), (3.0, 1.0), (4.2, 1.45), (5.4, 1.4), (6.6, 2.1)]):
        ax.scatter([x], [y], s=60, color="#d62828", zorder=3)
        ax.annotate(f"S{i + 1}", (x, y + 0.18), ha="center")
    ax.text(3.6, 2.2, "Millbrook WTP", fontsize=9)
    ax.set_title("Map of monitoring stations along the Tamsin River")
    ax.set_xticks([])
    ax.set_yticks([])
    map_png = _png(fig, "river_map")

    story = [
        Paragraph("Tamsin River Water Quality Study 2024", H1),
        Paragraph("1. Background", H2),
        Paragraph(
            "Samples were collected from 14 monitoring stations along the Tamsin River between March and "
            "October 2024, giving 1,176 laboratory analyses in total. The study was commissioned by the "
            "Riverside District Council following algal blooms in 2023.", P),
        Paragraph("2. Results", H2),
        Paragraph("Nitrate concentrations varied substantially between upstream and downstream stations.", P),
        Image(nit_png, width=14 * cm, height=8 * cm),
        Paragraph("Figure 1: Mean nitrate concentration by station", CAP),
        Spacer(1, 8),
        _table([
            ["Parameter", "Mean value", "Unit"],
            ["Lead", "3.2", "µg/L"],
            ["Phosphate", "0.41", "mg/L"],
            ["Dissolved oxygen", "7.9", "mg/L"],
            ["Turbidity", "4.6", "NTU"],
        ]),
        Paragraph("Table 1: Summary of other measured parameters", CAP),
        PageBreak(),
        Paragraph("3. Station locations", H2),
        Image(map_png, width=14 * cm, height=8.4 * cm),
        Paragraph("Figure 2: Station map", CAP),
        Paragraph("4. Conclusions", H2),
        Paragraph(
            "Nutrient loading downstream of the agricultural belt is the main pressure on the river. The study "
            "recommends upgrading the Millbrook treatment plant with tertiary nutrient removal by 2027.", P),
    ]
    _build("tamsin_river_water_quality_2024.pdf", "Tamsin River Water Quality Study 2024", story)


# ---------------------------------------------------------------------------------------------
A, H, R, S = (
    "aurora_annual_report_2025.pdf",
    "helix_r7_manual.pdf",
    "tamsin_river_water_quality_2024.pdf",
    "Multimodal_RAG_Synopsis_final.pdf",
)


def q(question, answers, evidence, modality):
    return {"question": question, "answers": answers, "evidence": evidence, "modality": modality}


def E(doc, page, type_=None):
    return {"source": doc, "page": page, "type": type_}


QA = [
    # ---- Aurora -------------------------------------------------------------------------
    q("Who is the CEO of Aurora Solar?", ["Raghunathan"], [E(A, 1)], "text"),
    q("Where is Aurora Solar headquartered?", ["Bergen"], [E(A, 1)], "text"),
    q("How many employees did Aurora Solar have at the end of 2025?", ["4,850", "4850"], [E(A, 1)], "text"),
    q("What was Aurora Solar's revenue in Q3 2025?", ["231.9"], [E(A, 1, "table")], "table"),
    q("What net margin did Aurora report for Q4 2025?", ["12.8"], [E(A, 1, "table")], "table"),
    q("How many MW of new installations were added in Q2 2025?", ["455"], [E(A, 1, "table")], "table"),
    q("Which region had the highest installed capacity in 2025?", ["Nordics"], [E(A, 2, "image")], "image"),
    q("What installed capacity did Iberia reach in 2025?", ["540"], [E(A, 2, "image")], "image"),
    q("What share of revenue came from the residential segment?", ["46"], [E(A, 2, "image")], "image"),
    q("Show me the pie chart of the revenue mix", [], [E(A, 2, "image")], "image-visual"),
    q("What was the average module efficiency in 2025?", ["23.6"], [E(A, 3, "image")], "image"),
    q("What cumulative capacity target has the board set for 2026?", ["3.1"], [E(A, 3)], "text"),
    q("Which new market does Aurora plan to enter?", ["Poland", "Polish"], [E(A, 3)], "text"),
    # ---- Helix --------------------------------------------------------------------------
    q("Which firmware version ships with the Helix R7?", ["4.2.1"], [E(H, 1)], "text"),
    q("How often must the emergency stop circuit be tested?", ["250"], [E(H, 1)], "text"),
    q("What is the maximum payload of the Helix R7?", ["7 kg"], [E(H, 1, "table")], "table"),
    q("What is the pose repeatability of the arm?", ["0.02"], [E(H, 1, "table")], "table"),
    q("What IP rating does the Helix R7 have?", ["IP54"], [E(H, 1, "table")], "table"),
    q("Which step comes right after the self-test in the controller startup sequence?", ["Home Axes", "home the axes", "homing"], [E(H, 2, "image")], "image"),
    q("Show the startup sequence flowchart", [], [E(H, 2, "image")], "image-visual"),
    q("What is the maximum speed of joint J3?", ["180"], [E(H, 2, "image")], "image-vector"),
    q("Which joint is the shoulder?", ["J2"], [E(H, 2, "image")], "image-vector"),
    q("What does error code E204 mean?", ["Encoder"], [E(H, 3, "table")], "table"),
    q("How do I fix error E305?", ["EtherCAT"], [E(H, 3, "table")], "table"),
    # ---- River --------------------------------------------------------------------------
    q("How many monitoring stations were sampled on the Tamsin River?", ["14"], [E(R, 1)], "text"),
    q("Which station had the highest mean nitrate concentration?", ["S4"], [E(R, 1, "image")], "image"),
    q("What nitrate limit is marked on the nitrate chart?", ["10"], [E(R, 1, "image")], "image"),
    q("What was the mean turbidity measured in the study?", ["4.6"], [E(R, 1, "table")], "table"),
    q("What was the mean dissolved oxygen level?", ["7.9"], [E(R, 1, "table")], "table"),
    q("Show the map of the monitoring stations", [], [E(R, 2, "image")], "image-visual"),
    q("Which treatment plant does the study recommend upgrading?", ["Millbrook"], [E(R, 2)], "text"),
    # ---- Synopsis (real document) -------------------------------------------------------
    q("Which vector database does the multimodal RAG system use?", ["Pinecone"],
      [E(S, p) for p in (3, 4, 6, 8, 10, 15)], "text"),
    q("Which text embedding model is specified in the technology stack?", ["MiniLM"], [E(S, 15, "table")], "table"),
    q("What backend framework is used?", ["FastAPI"], [E(S, 15, "table"), E(S, 12)], "table"),
    q("What is the minimum RAM requirement?", ["8 GB", "8GB"], [E(S, 16)], "text"),
    q("Over how many weeks will the project be completed?", ["12"], [E(S, 19)], "text"),
    q("In the Gantt chart, which phase is scheduled for weeks 7-8?", ["Frontend"], [E(S, 19, "image")], "image"),
    q("In the system architecture diagram, which layer sits between the Backend API and the document parser?", ["Orchestration", "LangChain"], [E(S, 12, "image")], "image"),
    q("Who is the faculty mentor of the project?", ["Manisha Jangid"], [E(S, 1)], "text"),
    q("Which modalities are out of scope for the project?", ["audio", "video"], [E(S, 8)], "text"),
    q("What is the title of the paper that introduced CLIP?", ["Transferable Visual Models"], [E(S, 21)], "text"),
    q("Which PDF parsing library is used?", ["PyMuPDF"], [E(S, 15, "table"), E(S, 9), E(S, 12)], "table"),
    # ---- unanswerable -------------------------------------------------------------------
    q("What dividend per share did Aurora Solar pay in 2025?", [], [], "unanswerable"),
    q("What is the battery life of the Helix R7 teach pendant?", [], [], "unanswerable"),
    q("What was the mercury concentration in the Tamsin River?", [], [], "unanswerable"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    aurora()
    helix()
    river()
    if SYNOPSIS.exists():
        shutil.copy(SYNOPSIS, OUT / S)
    (OUT / "qa.json").write_text(json.dumps(QA, indent=1, ensure_ascii=False))
    shutil.rmtree(OUT / "_img", ignore_errors=True)
    print(f"wrote {len(QA)} questions and {len(list(OUT.glob('*.pdf')))} PDFs to {OUT}")


if __name__ == "__main__":
    main()
