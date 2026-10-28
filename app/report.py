"""
report.py
=========
In-memory PDF report generator using ReportLab.
Zero disk writes: generated purely in-memory and returned as bytes.
Complies with faculty and privacy rules:
- No patient name or personal identifiers.
- Includes Case ID, Date/Time, and Anatomical Site (labeled 'recorded only').
- Includes predicted tier, action, all 4 probabilities, and low-confidence banner if applicable.
- Includes preprocessed image, Grad-CAM, attention rollout, and LIME (if computed).
- Includes one-line computational definitions for each XAI method.
- Includes model version and mandatory screening-aid disclaimer.
"""

import io
from datetime import datetime, timezone
import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)

from app.explainability import format_confidence


def _ndarray_to_png_io(img_rgb: np.ndarray) -> io.BytesIO:
    """Encode an RGB numpy array to in-memory PNG bytes."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Could not encode image for PDF generation.")
    return io.BytesIO(buf.tobytes())


def generate_screening_report_pdf(case_data: dict) -> bytes:
    """
    Generate a complete, self-contained PDF screening report in memory.

    Parameters
    ----------
    case_data : dict containing:
        - case_id: str
        - timestamp: datetime or str
        - site: str or None
        - result: dict with tier_index, tier_name, action, color, probabilities, low_confidence
        - img_224_rgb: np.ndarray (224, 224, 3)
        - gradcam_overlay: np.ndarray (224, 224, 3)
        - rollout_overlay: np.ndarray (224, 224, 3)
        - lime_overlay: np.ndarray (224, 224, 3) or None
        - model_version: str

    Returns
    -------
    bytes : Raw PDF bytes (ready for HTTP response)
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=10,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=4,
    )
    meta_key_style = ParagraphStyle(
        "MetaKey",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#334155"),
    )
    meta_val_style = ParagraphStyle(
        "MetaVal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
    )
    tier_title_style = ParagraphStyle(
        "TierTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=16,
        textColor=colors.HexColor("#ffffff"),
    )
    action_style = ParagraphStyle(
        "ActionText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#ffffff"),
    )
    prob_header_style = ParagraphStyle(
        "ProbHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#475569"),
    )
    prob_cell_style = ParagraphStyle(
        "ProbCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1e293b"),
    )
    prob_val_style = ParagraphStyle(
        "ProbVal",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        alignment=2,  # Right-aligned
        textColor=colors.HexColor("#0f172a"),
    )
    caption_title = ParagraphStyle(
        "CaptionTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,  # Center
    )
    caption_desc = ParagraphStyle(
        "CaptionDesc",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor("#475569"),
        alignment=1,  # Center
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#b91c1c"),
        alignment=1,  # Center
    )
    footnote_style = ParagraphStyle(
        "Footnote",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.8,
        leading=8.5,
        textColor=colors.HexColor("#64748b"),
        alignment=1,  # Center
    )

    story = []

    # ── Header ───────────────────────────────────────────────────────────────
    story.append(Paragraph("OralScan AI — Triage Screening Report", title_style))
    story.append(
        Paragraph(
            "Research Prototype · On-Device ViT-B/16 Inference · Zero Cloud Dependency · No Patient Identifiers",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=8))

    # ── Case & Screening Metadata Table ───────────────────────────────────────
    case_id = case_data.get("case_id", "SCR-UNKNOWN")
    ts = case_data.get("timestamp")
    if isinstance(ts, datetime):
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    elif isinstance(ts, str):
        ts_str = ts
    else:
        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    site_raw = case_data.get("site") or "Not Specified / General"
    site_display = f"{site_raw} <font color='#64748b'>(Recorded for report only – not used by model)</font>"

    model_label = case_data.get(
        "model_version",
        "ViT-B/16 (vit_base_patch16_224) trained on SMART-OM (2,469 smartphone intraoral images)",
    )

    meta_table_data = [
        [
            Paragraph("Case Identifier:", meta_key_style),
            Paragraph(f"<b>{case_id}</b>", meta_val_style),
            Paragraph("Screening Timestamp:", meta_key_style),
            Paragraph(ts_str, meta_val_style),
        ],
        [
            Paragraph("Anatomical Site:", meta_key_style),
            Paragraph(site_display, meta_val_style),
            Paragraph("Model Pipeline:", meta_key_style),
            Paragraph(model_label, meta_val_style),
        ],
    ]

    meta_table = Table(meta_table_data, colWidths=[1.3 * inch, 2.3 * inch, 1.3 * inch, 2.3 * inch])
    meta_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # ── Triage Classification Banner ──────────────────────────────────────────
    result = case_data.get("result", {})
    tier_idx = result.get("tier_index", 0)
    tier_name = result.get("tier_name", "T0 Normal")
    action_text = result.get("action", "Routine recall")
    color_hex = result.get("color", "#22c55e")
    low_confidence = result.get("low_confidence", False)

    bg_color = colors.HexColor(color_hex)

    banner_data = [
        [
            Paragraph(f"PREDICTED TRIAGE: {tier_name.upper()}", tier_title_style),
            Paragraph(f"RECOMMENDED ACTION: {action_text}", action_style),
        ]
    ]
    banner_table = Table(banner_data, colWidths=[3.6 * inch, 3.6 * inch])
    banner_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg_color),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(banner_table)

    # Low-Confidence Warning if applicable
    if low_confidence:
        story.append(Spacer(1, 4))
        warn_data = [[
            Paragraph(
                "<b>LOW CONFIDENCE WARNING:</b> Top probability is below the 60.0% threshold. "
                "Specialist clinical examination is strongly advised regardless of triage tier.",
                ParagraphStyle(
                    "LowConf",
                    parent=styles["Normal"],
                    fontName="Helvetica-Bold",
                    fontSize=8,
                    leading=10,
                    textColor=colors.HexColor("#92400e"),
                ),
            )
        ]]
        warn_table = Table(warn_data, colWidths=[7.2 * inch])
        warn_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#f59e0b")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(warn_table)

    story.append(Spacer(1, 8))

    # ── Probability Distribution Table ────────────────────────────────────────
    story.append(Paragraph("Class Probability Distribution", section_heading))
    probs = result.get("probabilities", [0.0, 0.0, 0.0, 0.0])
    tiers_info = [
        ("T0 Normal", "Routine recall"),
        ("T1 Variation from normal", "Short-interval review"),
        ("T2 OPMD (incl. OSMF)", "Refer to oral-medicine specialist"),
        ("T3 Oral cancer", "URGENT referral"),
    ]

    prob_rows = [
        [
            Paragraph("Classification Tier", prob_header_style),
            Paragraph("Clinical Referral Action", prob_header_style),
            Paragraph("Calculated Probability", ParagraphStyle("RPH", parent=prob_header_style, alignment=2)),
        ]
    ]

    for i, (t_name, act) in enumerate(tiers_info):
        p_val = probs[i] if i < len(probs) else 0.0
        p_str = format_confidence(p_val)
        is_top = (i == tier_idx)
        weight_font = "Helvetica-Bold" if is_top else "Helvetica"
        prefix = "<b>" if is_top else ""
        suffix = "</b>" if is_top else ""

        prob_rows.append([
            Paragraph(f"{prefix}{t_name}{suffix}", prob_cell_style),
            Paragraph(f"{prefix}{act}{suffix}", prob_cell_style),
            Paragraph(f"{prefix}{p_str}{suffix}", prob_val_style),
        ])

    prob_table = Table(prob_rows, colWidths=[2.2 * inch, 3.6 * inch, 1.4 * inch])
    prob_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ])
    )
    story.append(prob_table)
    story.append(Spacer(1, 10))

    # ── Explainable AI (XAI) Visual Evidence ──────────────────────────────────
    story.append(Paragraph("Explainable AI (XAI) Visual Evidence", section_heading))

    img_224 = case_data.get("img_224_rgb")
    gradcam_img = case_data.get("gradcam_overlay")
    rollout_img = case_data.get("rollout_overlay")
    lime_img = case_data.get("lime_overlay")

    target_img_dim = 1.62 * inch

    def make_rl_image(arr: np.ndarray | None):
        if arr is None:
            return None
        png_io = _ndarray_to_png_io(arr)
        return RLImage(png_io, width=target_img_dim, height=target_img_dim)

    rl_img_prep = make_rl_image(img_224)
    rl_img_gc = make_rl_image(gradcam_img)
    rl_img_ro = make_rl_image(rollout_img)
    rl_img_lime = make_rl_image(lime_img)

    # Four columns: Preprocessed, Grad-CAM, Rollout, LIME
    col_w = 1.80 * inch

    cell_prep = rl_img_prep or Paragraph("No image", caption_desc)
    cell_gc = rl_img_gc or Paragraph("Pending calculation", caption_desc)
    cell_ro = rl_img_ro or Paragraph("Pending calculation", caption_desc)
    cell_lime = (
        rl_img_lime
        if rl_img_lime is not None
        else Paragraph(
            "<i>(LIME was not executed for this case. Can be generated on-demand in the screening interface.)</i>",
            caption_desc,
        )
    )

    xai_image_row = [cell_prep, cell_gc, cell_ro, cell_lime]

    xai_caption_row = [
        Paragraph("<b>1. Preprocessed Input</b>", caption_title),
        Paragraph("<b>2. Grad-CAM</b>", caption_title),
        Paragraph("<b>3. Attention Rollout</b>", caption_title),
        Paragraph("<b>4. LIME Superpixels</b>", caption_title),
    ]

    xai_desc_row = [
        Paragraph("Exact 224×224 RGB input fed to model after specular inpainting, colour constancy & crop.", caption_desc),
        Paragraph("Target-class channel gradients weighting final layer token activations (blocks[-1].norm1).", caption_desc),
        Paragraph("Recursive matrix product of attention weights across all 12 transformer layers (0.5·A + 0.5·I).", caption_desc),
        Paragraph("Top 5 positive superpixels (gold outlines) fitted on 300 perturbed samples with black padding excluded.", caption_desc),
    ]

    xai_table_data = [xai_caption_row, xai_image_row, xai_desc_row]
    xai_table = Table(xai_table_data, colWidths=[col_w] * 4)
    xai_table.setStyle(
        TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ])
    )
    story.append(xai_table)
    story.append(Spacer(1, 10))

    # ── Screening Aid Disclaimer (Bordered Box at bottom) ─────────────────────
    disclaimer_data = [
        [
            Paragraph(
                "MANDATORY CLINICAL DISCLAIMER & BOUNDARY CONDITIONS<br/>"
                "Highlighted regions influenced this prediction according to each method. Not proof of disease. Screening aid only.<br/>"
                "This report is generated by a research prototype for educational and screening triage evaluation. "
                "It is NOT a certified medical diagnostic device. It does not replace comprehensive clinical oral examination, "
                "tissue biopsy, or histopathological confirmation by qualified healthcare professionals.",
                disclaimer_style,
            )
        ]
    ]
    disclaimer_table = Table(disclaimer_data, colWidths=[7.2 * inch])
    disclaimer_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#f87171")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(KeepTogether(disclaimer_table))

    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "OralScan AI · Zero third-party cloud APIs · Privacy-preserving local processing · Confidential screening aid",
            footnote_style,
        )
    )

    doc.build(story)
    return buffer.getvalue()
