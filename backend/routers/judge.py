"""
routers/judge.py
================
FastAPI router — expose judge engine endpoints.

POST /api/judge/{audit_id}/run
    Trigger full battery judging in background.

GET  /api/judge/{audit_id}/status
    Fetch judging status (delegates to audit status).

GET  /api/judge/{audit_id}/analysis
    Return the BiasAnalysis summary from Firestore.

GET  /api/judge/{audit_id}/judgements
    Return individual Judgement records (paginated, filterable).

POST /api/judge/pair
    Score a single probe pair on demand (useful for testing).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from google.cloud import firestore as _fs
from pydantic import BaseModel

from services.judge_engine import JudgeEngine, ScoreCard, Judgement

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/judge", tags=["Judge Engine"])

# Module-level singleton — shares rate limiter across requests
_engine: JudgeEngine | None = None


def _get_engine() -> JudgeEngine:
    global _engine
    if _engine is None:
        _engine = JudgeEngine()
    return _engine


def _db() -> _fs.Client:
    return _fs.Client()


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def _judge_battery_task(audit_id: str) -> None:
    """Run judging pipeline; update audit status on completion/failure."""
    db = _db()
    audit_ref = db.collection("audits").document(audit_id)
    try:
        audit_ref.set({"status": "judging"}, merge=True)
        engine = _get_engine()
        analysis = await engine.judge_full_battery(audit_id)
        log.info(
            "Judging complete: audit=%s pairs=%d biased=%d (%.1f%%)",
            audit_id, analysis.total_pairs,
            analysis.total_biased, analysis.overall_bias_rate_percent,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Judging failed for audit=%s: %s", audit_id, exc)
        audit_ref.set({"status": "judge_failed", "judge_error": str(exc)}, merge=True)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PairJudgeRequest(BaseModel):
    """On-demand single pair scoring."""
    probe_a: dict[str, Any]
    probe_b: dict[str, Any]


class RunJudgeResponse(BaseModel):
    audit_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/{audit_id}/run", response_model=RunJudgeResponse, summary="Start judging a battery")
async def run_judging(audit_id: str, background_tasks: BackgroundTasks) -> RunJudgeResponse:
    """
    Kick off bias judging for an audit in the background.
    The audit must already be in ``complete`` status (probes executed).
    Returns immediately — poll ``GET /api/judge/{audit_id}/analysis`` for results.
    """
    db = _db()
    doc = db.collection("audits").document(audit_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")

    status = doc.to_dict().get("status", "")
    if status == "judging":
        raise HTTPException(status_code=409, detail="Judging is already in progress.")
    if status == "judged":
        raise HTTPException(status_code=409, detail="Audit already judged. Check /analysis for results.")
    if status not in ("complete", "battery_ready"):
        raise HTTPException(
            status_code=409,
            detail=f"Audit is in status '{status}'. Run probes first via POST /api/audit/{{id}}/run.",
        )

    background_tasks.add_task(asyncio.ensure_future, _judge_battery_task(audit_id))

    return RunJudgeResponse(
        audit_id=audit_id,
        status="judging",
        message=f"Judging started. Poll GET /api/judge/{audit_id}/analysis for results.",
    )


@router.get("/{audit_id}/analysis", summary="Get bias analysis results")
async def get_analysis(audit_id: str) -> dict:
    """
    Return the aggregated ``BiasAnalysis`` for the audit.
    Available after judging completes (status = 'judged').
    """
    db = _db()
    doc = db.collection("audits").document(audit_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")

    data = doc.to_dict()
    if "bias_analysis" not in data:
        status = data.get("status", "unknown")
        raise HTTPException(
            status_code=202,
            detail=f"Analysis not yet available (status='{status}'). Try again later.",
        )
    return data["bias_analysis"]


@router.get("/{audit_id}/judgements", summary="List individual pair judgements")
async def list_judgements(
    audit_id: str,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    attribute: str | None = None,
    biased_only: bool = False,
) -> dict:
    """
    Return individual pair Judgement records.

    Filters
    -------
    attribute   : e.g. "gender", "race"
    biased_only : if True, return only pairs flagged as biased
    limit       : max records (default 100, max 500)
    """
    db = _db()
    if not db.collection("audits").document(audit_id).get().exists:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")

    ref = db.collection("audits").document(audit_id).collection("judgements")
    query = ref
    if attribute:
        query = query.where("attribute_tested", "==", attribute)
    if biased_only:
        query = query.where("is_biased", "==", True)
    query = query.limit(limit)

    docs = [d.to_dict() for d in query.stream()]
    return {"audit_id": audit_id, "count": len(docs), "judgements": docs}


@router.post("/pair", summary="Score a single probe pair on demand")
async def judge_single_pair(req: PairJudgeRequest) -> dict:
    """
    Score one probe pair synchronously.  Useful for testing the judge engine
    without running a full audit.  Both ``probe_a`` and ``probe_b`` must have
    at least: ``probe_id``, ``response_text``, ``scenario``, ``pair_id``,
    ``demographic_group``, ``attribute_tested``, ``audit_id``.
    """
    from types import SimpleNamespace

    def _to_ns(d: dict) -> SimpleNamespace:
        ns = SimpleNamespace(**d)
        # Ensure required fields have fallbacks
        for attr in ("probe_id", "pair_id", "response_text", "scenario",
                     "demographic_group", "attribute_tested", "audit_id"):
            if not hasattr(ns, attr):
                setattr(ns, attr, "")
        return ns

    engine = _get_engine()
    try:
        judgement = await engine.judge_probe_pair(
            _to_ns(req.probe_a),
            _to_ns(req.probe_b),
        )
        return judgement.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
