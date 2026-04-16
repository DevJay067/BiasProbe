"""
pdf_exporter.py
===============
BiasProbe — Professional PDF audit report generator.

Layout
------
  Page 1:  Cover — logo, audit title, score gauge, risk badge, date
  Page 2:  Executive Summary + Methodology
  Page 3…: Key Findings (one section per attribute)
  Last:    Remediation Checklist + Regulatory Flags + Footer

Uses ReportLab (open-source, no external dependencies beyond pip install).
Uploads to GCS and returns a signed URL valid for 1 hour.
"""

from __future__ import annotations

import datetime
import io
import logging
import math
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from google.cloud import storage as _gcs

load_dotenv()

log = logging.getLogger(__name__)

GCS_BUCKET      = os.getenv("GCS_BUCKET_NAME", "biasprobeaudit-batteries")
SIGNED_URL_TTL  = 3600   # seconds (1 hour)

if TYPE_CHECKING:
    from services.report_generator import AuditReport

# ---------------------------------------------------------------------------
# Colour palette (RGB 0-1)
# ---------------------------------------------------------------------------
C_DARK_BG    = (0.07, 0.07, 0.12)      # #12121f — cover background
C_ACCENT     = (0.40, 0.22, 0.90)      # #6638e5 — brand purple
C_ACCENT2    = (0.18, 0.76, 0.83)      # #2dc2d4 — teal
C_WHITE      = (1.00, 1.00, 1.00)
C_LIGHT_GREY = (0.93, 0.93, 0.96)
C_TEXT       = (0.15, 0.15, 0.20)
C_MID_GREY   = (0.55, 0.55, 0.60)

SEVERITY_COLORS: dict[str, tuple] = {
    "compliant":     (0.13, 0.76, 0.37),   # green
    "at_risk":       (0.96, 0.62, 0.04),   # amber
    "non_compliant": (0.93, 0.27, 0.27),   # red
    "critical":      (0.50, 0.05, 0.05),   # dark red
}


class PdfExporter:
    """
    Generates a professional multi-page PDF for an AuditReport and
    uploads it to GCS.

    Usage
    -----
    >>> exporter = PdfExporter()
    >>> gcs_uri, signed_url = exporter.export(audit_id, report_id, report)
    """

    def __init__(self) -> None:
        self._gcs: _gcs.Client | None = None

    @property
    def gcs_client(self) -> _gcs.Client:
        if self._gcs is None:
            self._gcs = _gcs.Client()
        return self._gcs

    # ==================================================================
    # PUBLIC
    # ==================================================================

    def export(
        self,
        audit_id: str,
        report_id: str,
        report: "AuditReport",
    ) -> tuple[str, str]:
        """
        Build PDF, upload to GCS, return (gs:// URI, signed URL).

        Parameters
        ----------
        audit_id  : str
        report_id : str
        report    : AuditReport

        Returns
        -------
        (gcs_uri: str, signed_url: str)
        """
        pdf_bytes = self._build_pdf(report)
        gcs_path  = f"reports/{audit_id}/{report_id}/report.pdf"
        gcs_uri   = self._upload(gcs_path, pdf_bytes)
        signed    = self._sign_url(gcs_path)
        log.info("PDF exported: %s (%.1f KB)", gcs_uri, len(pdf_bytes) / 1024)
        return gcs_uri, signed

    # ==================================================================
    # INTERNAL — PDF construction
    # ==================================================================

    def _build_pdf(self, report: "AuditReport") -> bytes:
        """Build the full PDF and return raw bytes."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Spacer, Paragraph, Table, TableStyle,
            HRFlowable, PageBreak,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.graphics.shapes import Drawing, Circle, Wedge, String, Rect
        from reportlab.graphics import renderPDF

        buf  = io.BytesIO()
        doc  = SimpleDocTemplate(
            buf,
            pagesize=A4,
            rightMargin=20 * mm,
            leftMargin=20 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=f"BiasProbe Audit Report — {report.label}",
            author="BiasProbe",
        )

        W, H = A4
        styles = getSampleStyleSheet()

        # ------------------------------------------------------------------
        # Custom paragraph styles
        # ------------------------------------------------------------------
        def _rgb(t: tuple) -> colors.Color:
            return colors.Color(*t)

        S_COVER_TITLE = ParagraphStyle(
            "CoverTitle",
            fontName="Helvetica-Bold",
            fontSize=28,
            textColor=_rgb(C_WHITE),
            leading=34,
            spaceAfter=6,
        )
        S_COVER_SUB = ParagraphStyle(
            "CoverSub",
            fontName="Helvetica",
            fontSize=13,
            textColor=_rgb(C_ACCENT2),
            leading=18,
            spaceAfter=4,
        )
        S_COVER_META = ParagraphStyle(
            "CoverMeta",
            fontName="Helvetica",
            fontSize=10,
            textColor=_rgb(C_MID_GREY),
            leading=14,
        )
        S_H1 = ParagraphStyle(
            "H1",
            fontName="Helvetica-Bold",
            fontSize=16,
            textColor=_rgb(C_ACCENT),
            spaceBefore=14,
            spaceAfter=6,
        )
        S_H2 = ParagraphStyle(
            "H2",
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=_rgb(C_TEXT),
            spaceBefore=10,
            spaceAfter=4,
        )
        S_BODY = ParagraphStyle(
            "Body",
            fontName="Helvetica",
            fontSize=10,
            textColor=_rgb(C_TEXT),
            leading=15,
            spaceAfter=4,
        )
        S_LABEL = ParagraphStyle(
            "Label",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=_rgb(C_MID_GREY),
            spaceAfter=2,
        )
        S_SMALL = ParagraphStyle(
            "Small",
            fontName="Helvetica",
            fontSize=8,
            textColor=_rgb(C_MID_GREY),
            leading=11,
        )
        S_EVIDENCE = ParagraphStyle(
            "Evidence",
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor=_rgb(C_TEXT),
            leading=13,
            leftIndent=12,
            borderPad=4,
        )

        story: list = []

        # ==============================================================
        # PAGE 1 — Cover
        # ==============================================================
        risk_color = SEVERITY_COLORS.get(report.risk_level, (0.5, 0.5, 0.5))

        # Dark background rectangle spanning full page — drawn via canvas callback
        # We achieve this with a Table cell with dark background.
        cover_content = [
            [
                Paragraph("BiasProbe", S_COVER_SUB),
            ],
            [Spacer(1, 8 * mm)],
            [Paragraph("AI Bias Audit Report", S_COVER_TITLE)],
            [Paragraph(report.label or report.scenario, S_COVER_SUB)],
            [Spacer(1, 6 * mm)],
            [Paragraph(f"Scenario: {report.scenario.replace('_', ' ').title()}", S_COVER_META)],
            [Paragraph(f"Tested: {report.tested_at[:10]}", S_COVER_META)],
            [Paragraph(f"Report ID: {report.report_id}", S_COVER_META)],
            [Spacer(1, 10 * mm)],
        ]

        cover_table = Table(
            cover_content,
            colWidths=[W - 40 * mm],
        )
        cover_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _rgb(C_DARK_BG)),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ]))
        story.append(cover_table)
        story.append(Spacer(1, 8 * mm))

        # Score gauge — drawn as a simple arc using Drawing
        story.append(self._score_gauge(
            report.fairness_score, report.risk_level, W - 40 * mm
        ))

        story.append(Spacer(1, 6 * mm))

        # Risk badge
        risk_label = report.risk_level.replace("_", " ").upper()
        badge_data  = [[Paragraph(f"  RISK LEVEL: {risk_label}  ", ParagraphStyle(
            "Badge",
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=colors.white,
        ))]]
        badge_table = Table(badge_data, colWidths=[80 * mm])
        badge_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), _rgb(risk_color)),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("ROUNDEDCORNERS", [4]),
        ]))
        story.append(badge_table)
        story.append(Spacer(1, 4 * mm))

        # Certification badge
        cert_text = "✓ CERTIFICATION ELIGIBLE" if report.certification_eligible else "✗ NOT CERTIFICATION ELIGIBLE"
        cert_color = _rgb((0.13, 0.76, 0.37)) if report.certification_eligible else _rgb((0.93, 0.27, 0.27))
        story.append(Paragraph(cert_text, ParagraphStyle(
            "Cert", fontName="Helvetica-Bold", fontSize=10,
            textColor=cert_color, spaceAfter=6,
        )))

        story.append(PageBreak())

        # ==============================================================
        # PAGE 2 — Executive Summary + Methodology
        # ==============================================================
        story.append(Paragraph("Executive Summary", S_H1))
        story.append(HRFlowable(width="100%", thickness=1, color=_rgb(C_ACCENT), spaceAfter=8))
        story.append(Paragraph(report.executive_summary, S_BODY))
        story.append(Spacer(1, 6 * mm))

        # Key metrics table
        metrics = [
            ["Metric", "Value"],
            ["Overall Fairness Score", f"{report.fairness_score:.1f} / 100"],
            ["Risk Level", report.risk_level.replace("_", " ").title()],
            ["Probes Tested", str(report.probe_count)],
            ["Matched Pairs", str(report.pair_count)],
            ["Certification Eligible", "Yes" if report.certification_eligible else "No"],
            ["Report Date", report.tested_at[:10]],
        ]
        metrics_table = Table(metrics, colWidths=[80 * mm, 80 * mm])
        metrics_table.setStyle(self._table_style())
        story.append(metrics_table)
        story.append(Spacer(1, 8 * mm))

        story.append(Paragraph("Methodology", S_H1))
        story.append(HRFlowable(width="100%", thickness=1, color=_rgb(C_ACCENT), spaceAfter=8))
        story.append(Paragraph(report.methodology, S_BODY))
        story.append(PageBreak())

        # ==============================================================
        # PAGES 3+ — Key Findings
        # ==============================================================
        story.append(Paragraph("Key Findings", S_H1))
        story.append(HRFlowable(width="100%", thickness=1, color=_rgb(C_ACCENT), spaceAfter=8))

        if not report.key_findings:
            story.append(Paragraph("No statistically significant bias findings were detected.", S_BODY))
        else:
            for i, finding in enumerate(report.key_findings, 1):
                sev_color = {
                    "high": _rgb((0.93, 0.27, 0.27)),
                    "medium": _rgb((0.96, 0.62, 0.04)),
                    "low": _rgb((0.13, 0.76, 0.37)),
                }.get(finding.severity, _rgb(C_MID_GREY))

                story.append(Paragraph(
                    f"{i}. {finding.attribute.replace('_', ' ').title()} Bias",
                    S_H2,
                ))

                sev_data = [[Paragraph(
                    f"  {finding.severity.upper()} SEVERITY  ",
                    ParagraphStyle("SevBadge", fontName="Helvetica-Bold",
                                   fontSize=8, textColor=colors.white),
                )]]
                sev_table = Table(sev_data, colWidths=[40 * mm])
                sev_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), sev_color),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(sev_table)
                story.append(Spacer(1, 3 * mm))

                story.append(Paragraph("<b>Finding:</b>", S_LABEL))
                story.append(Paragraph(finding.finding, S_BODY))

                story.append(Paragraph("<b>Evidence:</b>", S_LABEL))
                story.append(Paragraph(finding.evidence, S_EVIDENCE))

                story.append(Paragraph("<b>Statistical Basis:</b>", S_LABEL))
                story.append(Paragraph(finding.statistical_basis, S_SMALL))

                if finding.regulatory_flags:
                    story.append(Paragraph("<b>Regulatory Exposure:</b>", S_LABEL))
                    story.append(Paragraph(
                        " | ".join(finding.regulatory_flags),
                        ParagraphStyle("Regs", fontName="Helvetica", fontSize=9,
                                       textColor=_rgb(C_ACCENT), leading=13),
                    ))

                story.append(HRFlowable(
                    width="100%", thickness=0.5,
                    color=_rgb(C_LIGHT_GREY), spaceAfter=6, spaceBefore=6,
                ))

        story.append(PageBreak())

        # ==============================================================
        # Final page — Remediation Checklist + Regulatory Flags
        # ==============================================================
        story.append(Paragraph("Remediation Checklist", S_H1))
        story.append(HRFlowable(width="100%", thickness=1, color=_rgb(C_ACCENT), spaceAfter=8))

        if report.remediation_steps:
            rem_data = [["#", "Action", "Approach", "Effort", "Expected Impact"]]
            for step in sorted(report.remediation_steps, key=lambda s: s.priority):
                rem_data.append([
                    str(step.priority),
                    Paragraph(step.action, S_SMALL),
                    Paragraph(step.technical_approach, S_SMALL),
                    step.effort.title(),
                    Paragraph(step.expected_impact, S_SMALL),
                ])
            rem_table = Table(
                rem_data,
                colWidths=[8 * mm, 52 * mm, 40 * mm, 18 * mm, 42 * mm],
            )
            rem_table.setStyle(self._table_style())
            story.append(rem_table)
        else:
            story.append(Paragraph("No remediation steps recommended — system appears fair.", S_BODY))

        story.append(Spacer(1, 8 * mm))
        story.append(Paragraph("Unique Regulatory Frameworks Triggered", S_H1))
        story.append(HRFlowable(width="100%", thickness=1, color=_rgb(C_ACCENT), spaceAfter=8))

        all_flags: list[str] = []
        for finding in report.key_findings:
            all_flags.extend(finding.regulatory_flags)
        unique_flags = sorted(set(all_flags))

        if unique_flags:
            flags_data = [["Regulation"]] + [[f] for f in unique_flags]
            flags_table = Table(flags_data, colWidths=[W - 40 * mm])
            flags_table.setStyle(self._table_style())
            story.append(flags_table)
        else:
            story.append(Paragraph("No regulatory frameworks triggered.", S_BODY))

        story.append(Spacer(1, 10 * mm))
        story.append(Paragraph(
            f"Generated by BiasProbe  ·  {report.tested_at[:10]}  ·  Report ID: {report.report_id}",
            S_SMALL,
        ))

        # ==============================================================
        # Build PDF
        # ==============================================================
        doc.build(story)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Score gauge drawing
    # ------------------------------------------------------------------

    @staticmethod
    def _score_gauge(score: float, risk_level: str, width_pt: float) -> "Drawing":
        """Draw a semicircular score gauge using ReportLab graphics."""
        from reportlab.graphics.shapes import Drawing, Wedge, Circle, String, Rect
        from reportlab.lib import colors

        d_width  = min(width_pt, 220)
        d_height = 120
        cx       = d_width / 2
        cy       = 20
        radius   = 80
        inner_r  = 54

        risk_color_t = SEVERITY_COLORS.get(risk_level, (0.5, 0.5, 0.5))
        risk_color   = colors.Color(*risk_color_t)
        grey_color   = colors.Color(0.87, 0.87, 0.90)
        dark_bg      = colors.Color(*C_DARK_BG)

        drawing = Drawing(d_width, d_height + 30)

        # Background
        bg = Rect(0, 0, d_width, d_height + 30, fillColor=dark_bg, strokeColor=None)
        drawing.add(bg)

        # Grey background arc (180°)
        arc_bg = Wedge(cx, cy, radius, 0, 180,
                       fillColor=grey_color, strokeColor=None, innerRadius=inner_r)
        drawing.add(arc_bg)

        # Score arc — proportional to score/100
        arc_angle = score / 100.0 * 180
        if arc_angle > 0:
            arc_fg = Wedge(cx, cy, radius, 0, arc_angle,
                           fillColor=risk_color, strokeColor=None, innerRadius=inner_r)
            drawing.add(arc_fg)

        # Inner circle mask (creates arc look)
        mask = Circle(cx, cy, inner_r - 2,
                      fillColor=dark_bg, strokeColor=None)
        drawing.add(mask)

        # Score text
        score_str = String(cx, cy + 28, f"{score:.0f}",
                           fontName="Helvetica-Bold", fontSize=32,
                           fillColor=colors.white, textAnchor="middle")
        drawing.add(score_str)

        out_of = String(cx, cy + 14, "/ 100",
                        fontName="Helvetica", fontSize=11,
                        fillColor=colors.Color(*C_MID_GREY), textAnchor="middle")
        drawing.add(out_of)

        label = String(cx, cy + 2, "FAIRNESS SCORE",
                       fontName="Helvetica", fontSize=8,
                       fillColor=colors.Color(*C_ACCENT2), textAnchor="middle")
        drawing.add(label)

        return drawing

    # ------------------------------------------------------------------
    # Shared table style
    # ------------------------------------------------------------------

    @staticmethod
    def _table_style():
        from reportlab.platypus import TableStyle
        from reportlab.lib import colors
        header_bg = colors.Color(*C_ACCENT)
        row_alt   = colors.Color(*C_LIGHT_GREY)
        return TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  header_bg),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0),  9),
            ("BOTTOMPADDING", (0, 0), (-1, 0),  8),
            ("TOPPADDING",    (0, 0), (-1, 0),  8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, row_alt]),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 9),
            ("TOPPADDING",    (0, 1), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.Color(0.82, 0.82, 0.85)),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ])

    # ==================================================================
    # INTERNAL — GCS
    # ==================================================================

    def _upload(self, gcs_path: str, pdf_bytes: bytes) -> str:
        bucket = self.gcs_client.bucket(GCS_BUCKET)
        blob   = bucket.blob(gcs_path)
        blob.upload_from_string(pdf_bytes, content_type="application/pdf")
        uri = f"gs://{GCS_BUCKET}/{gcs_path}"
        return uri

    def _sign_url(self, gcs_path: str) -> str:
        """
        Generate a V4 signed URL valid for SIGNED_URL_TTL seconds.
        Requires the service account to have the
        ``iam.serviceAccounts.signBlob`` permission.
        """
        bucket = self.gcs_client.bucket(GCS_BUCKET)
        blob   = bucket.blob(gcs_path)
        url    = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(seconds=SIGNED_URL_TTL),
            method="GET",
        )
        return url
