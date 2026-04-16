"""
routers/report.py
=================
BiasProbe — Report generation router.

POST /api/report/{audit_id}/generate
    Trigger async report + PDF generation, return report_id immediately.

GET  /api/report/{audit_id}/latest
    Return the latest AuditReport for an audit.

GET  /api/report/{audit_id}/{report_id}
    Return a specific AuditReport by ID.

GET  /api/report/{audit_id}/{report_id}/pdf
    Return a fresh signed PDF URL (re-signs; valid 1 hour).

GET  /api/report/{audit_id}/list
    List all reports generated for an audit.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from google.cloud import firestore as _fs
from pydantic import BaseModel

from services.report_generator import ReportGenerator

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/report", tags=["Report"])

_generator: ReportGenerator | None = None


def _get_generator() -> ReportGenerator:
    global _generator
    if _generator is None:
        _generator = ReportGenerator()
    return _generator


def _db() -> _fs.Client:
    return _fs.Client()


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def _generate_task(audit_id: str) -> None:
    db = _db()
    ref = db.collection("audits").document(audit_id)
    try:
        ref.set({"status": "generating_report"}, merge=True)
        gen = _get_generator()
        report = await gen.generate_audit_report(audit_id)
        log.info(
            "Report complete: audit=%s report=%s score=%.1f risk=%s pdf=%s",
            audit_id, report.report_id,
            report.fairness_score, report.risk_level,
            "yes" if report.pdf_signed_url else "no",
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Report generation failed for audit=%s: %s", audit_id, exc)
        ref.set({"status": "report_failed", "report_error": str(exc)}, merge=True)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class GenerateReportResponse(BaseModel):
    audit_id:  str
    status:    str
    message:   str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/{audit_id}/generate",
    response_model=GenerateReportResponse,
    summary="Generate audit report + PDF",
)
async def generate_report(
    audit_id: str,
    background_tasks: BackgroundTasks,
) -> GenerateReportResponse:
    """
    Trigger async Gemini-powered report generation and PDF export.

    Prerequisites: audit must be in ``analysed`` status
    (stats engine must have run first via POST /api/stats/{id}/run).

    Returns immediately with ``status: generating_report``.
    Poll ``GET /api/report/{audit_id}/latest`` for results.
    """
    db = _db()
    doc = db.collection("audits").document(audit_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")

    status = doc.to_dict().get("status", "")
    if status == "generating_report":
        raise HTTPException(status_code=409, detail="Report generation already in progress.")
    if status == "report_ready":
        raise HTTPException(
            status_code=409,
            detail="Report already generated. Use GET /api/report/{id}/latest to retrieve it. "
                   "Create a new audit to re-generate.",
        )
    if status not in ("analysed", "judged", "complete"):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Audit status is '{status}'. Complete the pipeline first: "
                "run probes → judge → analyse → then generate report."
            ),
        )

    background_tasks.add_task(asyncio.ensure_future, _generate_task(audit_id))

    return GenerateReportResponse(
        audit_id=audit_id,
        status="generating_report",
        message=(
            f"Report generation started for audit '{audit_id}'. "
            f"Poll GET /api/report/{audit_id}/latest for results (typically 15-30 seconds)."
        ),
    )


@router.get("/{audit_id}/latest", summary="Get latest audit report")
async def get_latest_report(audit_id: str) -> dict:
    """
    Return the most recently generated AuditReport for this audit.
    Returns HTTP 202 (Accepted) if the report is not yet ready.
    """
    db = _db()
    audit_doc = db.collection("audits").document(audit_id).get()
    if not audit_doc.exists:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")

    audit_data  = audit_doc.to_dict()
    report_id   = audit_data.get("latest_report_id")
    status      = audit_data.get("status", "")

    if not report_id:
        raise HTTPException(
            status_code=202,
            detail=f"Report not yet available (status='{status}'). Try again shortly.",
        )

    report_doc = (
        db.collection("audits").document(audit_id)
          .collection("reports").document(report_id).get()
    )
    if not report_doc.exists:
        raise HTTPException(status_code=404, detail=f"Report document '{report_id}' not found.")

    return report_doc.to_dict()


@router.get("/{audit_id}/{report_id}", summary="Get specific report by ID")
async def get_report(audit_id: str, report_id: str) -> dict:
    """Retrieve a specific AuditReport by its report_id."""
    db = _db()
    doc = (
        db.collection("audits").document(audit_id)
          .collection("reports").document(report_id).get()
    )
    if not doc.exists:
        raise HTTPException(
            status_code=404,
            detail=f"Report '{report_id}' not found for audit '{audit_id}'.",
        )
    return doc.to_dict()


@router.get("/{audit_id}/{report_id}/pdf", summary="Get fresh signed PDF URL")
async def get_pdf_url(audit_id: str, report_id: str) -> dict:
    """
    Generate a fresh signed URL for the PDF report (valid 1 hour).
    Re-signs on every call, so the link is always fresh.
    """
    db = _db()
    doc = (
        db.collection("audits").document(audit_id)
          .collection("reports").document(report_id).get()
    )
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found.")

    data = doc.to_dict()
    gcs_pdf_uri = data.get("gcs_pdf_uri", "")
    if not gcs_pdf_uri:
        raise HTTPException(
            status_code=404,
            detail="PDF not available for this report (may have failed during generation).",
        )

    # Re-sign the URL
    try:
        from services.pdf_exporter import PdfExporter
        exporter  = PdfExporter()
        gcs_path  = gcs_pdf_uri.replace(f"gs://{exporter.gcs_client.project}/", "").split("/", 1)[-1]
        # Simpler: strip gs://bucket/ prefix
        import os, re
        bucket_name = os.getenv("GCS_BUCKET_NAME", "biasprobeaudit-batteries")
        gcs_path    = gcs_pdf_uri.replace(f"gs://{bucket_name}/", "")
        signed_url  = exporter._sign_url(gcs_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"URL signing failed: {exc}") from exc

    return {
        "audit_id":   audit_id,
        "report_id":  report_id,
        "signed_url": signed_url,
        "expires_in": "3600 seconds",
    }


@router.get("/{audit_id}/list", summary="List all reports for an audit")
async def list_reports(audit_id: str) -> dict:
    """Return all report IDs and metadata for an audit (lightweight listing)."""
    db = _db()
    if not db.collection("audits").document(audit_id).get().exists:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")

    docs = (
        db.collection("audits").document(audit_id)
          .collection("reports")
          .stream()
    )
    reports = [
        {
            "report_id":       d.id,
            "fairness_score":  d.to_dict().get("fairness_score"),
            "risk_level":      d.to_dict().get("risk_level"),
            "tested_at":       d.to_dict().get("tested_at"),
            "pdf_signed_url":  d.to_dict().get("pdf_signed_url"),
            "certification_eligible": d.to_dict().get("certification_eligible"),
        }
        for d in docs
    ]
    return {"audit_id": audit_id, "count": len(reports), "reports": reports}
