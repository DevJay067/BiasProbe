"""
routers/stats.py
================
BiasProbe — Statistical analysis router.

POST /api/stats/{audit_id}/run
    Trigger full statistical analysis in bg task.

GET  /api/stats/{audit_id}/report
    Return the full StatisticalReport.

GET  /api/stats/{audit_id}/summary
    Return lightweight summary for dashboard cards.

GET  /api/stats/regulations
    Return the full regulatory framework registry.

GET  /api/stats/regulations/{regulation_id}
    Return one regulation detail.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from google.cloud import firestore as _fs

from services.stats_engine import StatsEngine
from utils.regulatory_mapper import RegulatoryMapper

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/stats", tags=["Statistical Analysis"])

_engine: StatsEngine | None = None


def _get_engine() -> StatsEngine:
    global _engine
    if _engine is None:
        _engine = StatsEngine()
    return _engine


def _db() -> _fs.Client:
    return _fs.Client()


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def _stats_task(audit_id: str) -> None:
    db = _db()
    ref = db.collection("audits").document(audit_id)
    try:
        ref.set({"status": "analysing"}, merge=True)
        engine = _get_engine()
        report = await engine.analyse(audit_id)
        log.info(
            "Stats analysis done: audit=%s score=%.1f severity=%s",
            audit_id, report.overall_fairness_score, report.overall_severity,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Stats analysis failed for audit=%s: %s", audit_id, exc)
        ref.set({"status": "stats_failed", "stats_error": str(exc)}, merge=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/{audit_id}/run", summary="Run statistical analysis")
async def run_stats(audit_id: str, background_tasks: BackgroundTasks) -> dict:
    """
    Trigger statistical analysis in the background.
    The audit must already be in ``judged`` status.
    """
    db = _db()
    doc = db.collection("audits").document(audit_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")

    status = doc.to_dict().get("status", "")
    if status == "analysed":
        raise HTTPException(status_code=409, detail="Analysis already completed.")
    if status == "analysing":
        raise HTTPException(status_code=409, detail="Analysis already in progress.")
    if status not in ("judged", "complete"):
        raise HTTPException(
            status_code=409,
            detail=f"Audit status is '{status}'. Judge the battery first via POST /api/judge/{{id}}/run.",
        )

    background_tasks.add_task(asyncio.ensure_future, _stats_task(audit_id))
    return {
        "audit_id": audit_id,
        "status": "analysing",
        "message": f"Statistical analysis started. Poll GET /api/stats/{audit_id}/report.",
    }


@router.get("/{audit_id}/report", summary="Get full statistical report")
async def get_report(audit_id: str) -> dict:
    """Return the full StatisticalReport including p-values, effect sizes, and regulatory flags."""
    db = _db()
    doc = db.collection("audits").document(audit_id) \
            .collection("statistical_report").document("main").get()
    if not doc.exists:
        audit_doc = db.collection("audits").document(audit_id).get()
        if not audit_doc.exists:
            raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")
        status = audit_doc.to_dict().get("status", "unknown")
        raise HTTPException(
            status_code=202,
            detail=f"Report not yet available (status='{status}'). Try again later.",
        )
    return doc.to_dict()


@router.get("/{audit_id}/summary", summary="Get dashboard summary")
async def get_summary(audit_id: str) -> dict:
    """
    Return lightweight summary fields from the audit document.
    Designed for dashboard stat cards — fast single-document read.
    """
    db = _db()
    doc = db.collection("audits").document(audit_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")

    d = doc.to_dict()
    return {
        "audit_id":                  audit_id,
        "scenario":                  d.get("scenario"),
        "label":                     d.get("label"),
        "status":                    d.get("status"),
        "overall_fairness_score":    d.get("overall_fairness_score"),
        "overall_severity":          d.get("overall_severity"),
        "overall_severity_color":    d.get("overall_severity_color"),
        "total_significant":         d.get("total_significant"),
        "total_highly_significant":  d.get("total_highly_significant"),
        "unique_regulations":        d.get("unique_regulations", []),
        "probe_count":               d.get("probe_count"),
        "pair_count":                d.get("pair_count"),
    }


@router.get("/regulations", summary="List all regulatory frameworks")
async def list_regulations() -> dict:
    """Return the full regulatory framework registry for UI display."""
    mapper = RegulatoryMapper()
    return {
        "regulations": [r.to_dict() for r in mapper.all_regulations()]
    }


@router.get("/regulations/{regulation_id}", summary="Get one regulation detail")
async def get_regulation(regulation_id: str) -> dict:
    """Return detail for a specific regulatory framework."""
    mapper = RegulatoryMapper()
    reg = mapper.get_regulation(regulation_id)
    if reg is None:
        raise HTTPException(status_code=404, detail=f"Regulation '{regulation_id}' not found.")
    return reg.to_dict()
