"""Render the Co-MoE manuscript from recorded evidence without running experiments."""

# Run from the repository root:
# uv run --with reportlab --with matplotlib --with pypdf python paper/build.py

import html
import io
import json
from pathlib import Path
import re
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
OUTPUT = ROOT / "output" / "pdf" / "co-moe-technical-report.pdf"
EVIDENCE_TAG = "v0.1.0"


def receipt(name):
    """Read chart inputs from the frozen evidence snapshot, not the working tree."""
    return json.loads(subprocess.check_output(
        ["git", "show", f"{EVIDENCE_TAG}:results/{name}/receipt.json"], cwd=ROOT
    ))


def make_figures():
    """Draw the ordered exchange and the recorded fixed-budget imitation results."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
    })
    blue, green, ink = "#255f8b", "#27705d", "#263641"
    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.03)
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis("off")

    def box(x, y, label, color):
        ax.add_patch(FancyBboxPatch(
            (x, y), 3.15, 0.72, boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor="white", edgecolor=color, linewidth=1.1,
        ))
        ax.text(x + 1.575, y + 0.36, label, ha="center", va="center", color=ink, fontsize=9)

    def arrow(start, end, color):
        ax.annotate("", end, start, arrowprops={"arrowstyle": "->", "lw": 1.3, "color": color})

    ax.text(2, 5.7, "A · OLMoE · width 2048", ha="center", color=blue, weight="bold")
    ax.text(8, 5.7, "B · Granite · width 1024", ha="center", color=green, weight="bold")
    box(0.4, 4.5, "Layer-7 input → expert (7, 0)", blue)
    box(6.4, 4.5, "Layer-7 residual + message", green)
    arrow((3.65, 4.86), (6.3, 4.86), blue)
    ax.text(4.98, 5.12, "truncate × 0.01", ha="center", fontsize=8)
    ax.text(4.98, 4.43, "cut this delivery", ha="center", fontsize=8, color="#a03c37")
    arrow((1.98, 4.35), (1.98, 2.85), blue)
    arrow((7.98, 4.35), (7.98, 2.85), green)
    ax.text(1.98, 3.55, "native continuation\nno first-round input", ha="center", va="center",
            fontsize=8.5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    ax.text(7.98, 3.55, "native continuation\nfirst message propagates", ha="center", va="center",
            fontsize=8.5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    box(0.4, 2.0, "Layer-9 residual + return", blue)
    box(6.4, 2.0, "Layer-9 input → expert (7, 0)", green)
    arrow((6.3, 2.36), (3.65, 2.36), green)
    ax.text(4.98, 2.72, "zero-pad × 0.01", ha="center", fontsize=8)
    ax.text(4.98, 1.92, "return remains active", ha="center", fontsize=8)
    arrow((1.98, 1.85), (1.98, 0.8), blue)
    arrow((7.98, 1.85), (7.98, 0.8), green)
    ax.text(1.98, 0.4, "Layer 10 → logits → generation", ha="center", color=blue, fontsize=9)
    ax.text(7.98, 0.4, "Layer 10 → logits → generation", ha="center", color=green, fontsize=9)
    fig.savefig(FIGURES / "feedback.svg")
    fig.savefig(FIGURES / "feedback.png", dpi=300)
    plt.close(fig)

    data = receipt("native-expert-fit")
    conditions = ["pretrained", "bypass", "random", "scrambled"]
    values = [data["results"][key]["history"][-1]["heldout"]["relative_mse"] for key in conditions]
    fig, ax = plt.subplots(figsize=(7.1, 1.9))
    fig.subplots_adjust(left=0.19, right=0.96, top=0.94, bottom=0.27)
    for i, val in enumerate(values):
        ax.hlines(i, 0, val, color="#d5dce1", lw=2)
        ax.plot(val, i, "o", color=blue, markersize=5)
        ax.text(val - 0.035, i - 0.14, f"{val:.4f}", ha="right", fontsize=9, color=ink)
    ax.axvline(1, color="#6b7280", lw=1, ls="--")
    ax.set(yticks=range(4), yticklabels=["Pretrained", "Bypass", "Random", "Scrambled"],
           xlim=(0, 1.055), ylim=(3.5, -0.7), xlabel="Held-out relative MSE (lower is better)")
    ax.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.50", "0.75", "1.00"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9)
    fig.savefig(FIGURES / "imitation.svg")
    fig.savefig(FIGURES / "imitation.png", dpi=300)
    plt.close(fig)


def inline(text):
    """Convert the limited inline Markdown used in this manuscript to Paragraph XML."""
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#255f8b">\1</link>', text)
    text = re.sub(r"`([^`]+)`", r'<font name="Mono" size="8.3">\1</font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)


def build_pdf():
    """Typeset the editable manuscript with explicit section breaks and live links."""
    for name, family in [
        ("Body", "DejaVu Serif"), ("Body-Bold", "DejaVu Serif:style=Bold"),
        ("Body-Italic", "DejaVu Serif:style=Italic"),
        ("Body-BoldItalic", "DejaVu Serif:style=Bold Italic"),
        ("Sans", "DejaVu Sans"), ("Sans-Bold", "DejaVu Sans:style=Bold"),
        ("Mono", "DejaVu Sans Mono"),
    ]:
        font = subprocess.check_output(["fc-match", "-f", "%{file}", family], text=True)
        pdfmetrics.registerFont(TTFont(name, font))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold",
                                  italic="Body-Italic", boldItalic="Body-BoldItalic")
    styles = {
        "body": ParagraphStyle("body", fontName="Body", fontSize=10, leading=13.5,
                               spaceAfter=7, textColor=colors.HexColor("#19232a")),
        "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=19, leading=23,
                                spaceAfter=11, keepWithNext=True),
        "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=12.5, leading=17,
                             spaceBefore=10, spaceAfter=8, keepWithNext=True),
        "h3": ParagraphStyle("h3", fontName="Sans-Bold", fontSize=10.5, leading=14,
                             spaceBefore=8, spaceAfter=6, keepWithNext=True),
        "caption": ParagraphStyle("caption", fontName="Body", fontSize=8.5, leading=11.5,
                                  spaceBefore=5, spaceAfter=10),
        "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.5, leading=11),
        "head": ParagraphStyle("head", fontName="Sans-Bold", fontSize=8.2, leading=10.6),
        "code": ParagraphStyle("code", fontName="Mono", fontSize=8, leading=12,
                               spaceAfter=4),
    }
    width = A4[0] - 104
    story, buffers = [], []
    lines = (PAPER / "manuscript.md").read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["title"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["h3"]))
        elif line.startswith("## "):
            if line.startswith("## Appendix"):
                styles["body"] = ParagraphStyle("appendix", parent=styles["body"],
                                                fontSize=9, leading=12, spaceAfter=6)
            elif line == "## References":
                styles["body"] = ParagraphStyle("references", parent=styles["body"],
                                                fontSize=9.3, leading=13, spaceAfter=8)
            story.append(Paragraph(inline(line[3:]), styles["h2"]))
        elif line == "$$":
            formula = []
            while lines[i].strip() != "$$":
                formula.append(lines[i].strip())
                i += 1
            i += 1
            buf = io.BytesIO()
            mathtext.math_to_image("$" + " ".join(formula) + "$", buf, dpi=360, format="png",
                                   prop=FontProperties(size=12, math_fontfamily="dejavuserif"))
            buffers.append(buf)
            buf.seek(0)
            w, h = PILImage.open(buf).size
            scale = min(72 / 360, width / w)
            buf.seek(0)
            story.extend([Image(buf, width=w * scale, height=h * scale), Spacer(1, 9)])
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[(.+)\]\((.+)\)", line)
            caption, path = match.groups()
            source = PAPER / path
            w, h = PILImage.open(source).size
            story.append(KeepTogether([
                Image(str(source), width=width, height=h * width / w),
                Paragraph(inline(caption), styles["caption"]),
            ]))
        elif line.startswith("|"):
            rows = [line]
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            rows = [[c.strip() for c in row.strip("|").split("|")] for row in rows]
            rows.pop(1)
            count = len(rows[0])
            if count == 2:
                widths = [width * .29, width * .71] if rows[0][0] == "Arrangement" else [width * .76, width * .24]
            elif count == 3:
                widths = [width * .46, width * .27, width * .27]
                if rows[0][0] == "Evidence":
                    widths = [width * .31, width * .34, width * .35]
            else:
                widths = [width * .27] + [width * .73 / (count - 1)] * (count - 1)
            cells = [[Paragraph(inline(cell), styles["head" if r == 0 else "cell"])
                      for cell in row] for r, row in enumerate(rows)]
            table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("LINEABOVE", (0, 0), (-1, 0), .8, colors.HexColor("#38434c")),
                ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#9aa4ad")),
                ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#9aa4ad")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
            ]))
            # Keep each small evidence table with its immediately following caption.
            after = i
            while after < len(lines) and not lines[after].strip():
                after += 1
            if after < len(lines) and lines[after].startswith("*Table"):
                story.append(KeepTogether([
                    table, Paragraph(inline(lines[after]), styles["caption"]),
                ]))
                i = after + 1
            else:
                story.append(table)
        elif line.startswith("```"):
            code = []
            while not lines[i].startswith("```"):
                code.append(Paragraph(html.escape(lines[i]), styles["code"]))
                i += 1
            i += 1
            story.append(KeepTogether(code + [Spacer(1, 5)]))
        else:
            paragraph = [line]
            while i < len(lines) and lines[i].strip():
                paragraph.append(lines[i].strip())
                i += 1
            text = " ".join(paragraph)
            style = "caption" if text.startswith("*Table") else "body"
            story.append(Paragraph(inline(text), styles[style]))

    def furniture(canvas, doc):
        canvas.saveState()
        canvas.setFont("Sans", 7.5)
        canvas.setFillColor(colors.HexColor("#66717a"))
        canvas.drawString(52, A4[1] - 29, "CO-MOE  /  TECHNICAL REPORT")
        canvas.drawRightString(A4[0] - 52, A4[1] - 29, "First draft · 10 November 2026")
        canvas.drawString(52, 26, "Evidence snapshot: v0.2.0 (historical receipts preserved; Appendix A)")
        canvas.drawRightString(A4[0] - 52, 26, str(doc.page))
        canvas.restoreState()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=52, rightMargin=52,
                            topMargin=49, bottomMargin=46, title="Co-MoE: Causal Computational Coupling",
                            author="Ztripez von Matérn",
                            subject="A bounded feasibility study of heterogeneous MoE inference")
    doc.build(story, onFirstPage=furniture, onLaterPages=furniture)
    print(OUTPUT)


if __name__ == "__main__":
    make_figures()
    build_pdf()
