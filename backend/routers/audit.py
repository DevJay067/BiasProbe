"""
routers/audit.py
================
BiasProbe — Audit lifecycle router.

Endpoints
---------
POST /api/audit/create
    Validate connector config, create audit document in Firestore.

POST /api/audit/{id}/run
    Load battery from GCS / Firestore, send probes via LLMConnector
    in a FastAPI BackgroundTask.

GET  /api/audit/{id}/status
    Return live progress counters from Firestore.

GET  /api/audit/{id}/results
    Return paginated ProbeResult records for the audit.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from google.cloud import firestore as _fs
from pydantic import BaseModel, Field

from services.llm_connector import ConnectorConfig, LLMConnector
from services.probe_generator import ProbeGenerator

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/audit", tags=["Audit"])

# ---------------------------------------------------------------------------
# Shared singleton state
# ---------------------------------------------------------------------------
_connector = LLMConnector()
_generator = ProbeGenerator()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ConnectorConfigIn(BaseModel):
    """Provider configuration supplied by the customer."""
    provider: str = Field(..., description="openai | gemini | anthropic | custom")
    api_key: str | None = Field(None, description="API key for managed providers")
    model: str | None = Field(None, description="Model name, e.g. gpt-4o, gemini-1.5-flash")
    base_url: str | None = Field(None, description="Override base URL (Azure / Together / Groq)")
    system_prompt: str = Field("You are a helpful assistant.", description="System instruction prepended to every probe")
    endpoint_url: str | None = Field(None, description="Full URL for 'custom' provider")
    headers: dict[str, str] | None = Field(None, description="Extra HTTP headers for 'custom' provider")
    request_body_template: dict | None = Field(None, description="JSON body template; use {prompt} as placeholder")
    max_tokens: int = Field(512, ge=32, le=4096)
    temperature: float = Field(0.0, ge=0.0, le=2.0)


class CreateAuditRequest(BaseModel):
    scenario: str = Field(..., description="Scenario key, e.g. 'hiring_assistant'")
    connector: ConnectorConfigIn
    num_probes: int = Field(200, ge=10, le=2000)
    attribute_filter: list[str] | None = Field(None)
    label: str | None = Field(None, description="Human-readable audit label")


class CreateAuditResponse(BaseModel):
    audit_id: str
    status: str
    connector_ok: bool
    connector_test_response: str | None
    message: str


class RunAuditResponse(BaseModel):
    audit_id: str
    status: str
    message: str


class AuditStatusResponse(BaseModel):
    audit_id: str
    status: str
    probes_sent: int
    probes_complete: int
    probes_success: int
    probes_failed: int
    percent_done: float
    scenario: str | None = None
    label: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db() -> _fs.Client:
    return _fs.Client()


def _config_from_request(req: ConnectorConfigIn) -> ConnectorConfig:
    return ConnectorConfig(
        provider=req.provider,
        api_key=req.api_key,
        model=req.model,
        base_url=req.base_url,
        system_prompt=req.system_prompt,
        endpoint_url=req.endpoint_url,
        headers=req.headers,
        request_body_template=req.request_body_template,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def _run_battery_task(audit_id: str, config: ConnectorConfig) -> None:
    """
    Background coroutine that:
    1. Loads the probe battery from GCS (via ProbeGenerator).
    2. Sends all probes through LLMConnector.
    3. Updates Firestore audit status to 'complete' or 'failed'.
    """
    db = _db()
    audit_ref = db.collection("audits").document(audit_id)

    try:
        # Mark as running
        audit_ref.set({"status": "running"}, merge=True)
        log.info("audit=%s | background task started", audit_id)

        # Load battery
        battery = _generator.load_battery(audit_id)
        probe_count = len(battery)
        audit_ref.set({"probes_sent": probe_count, "probes_complete": 0}, merge=True)

        # Send probes
        results = await _connector.send_probes(config, battery, audit_id)

        success = sum(1 for r in results if r.status == "success")
        failed  = sum(1 for r in results if r.status != "success")

        audit_ref.set(
            {
                "status": "complete",
                "probes_complete": len(results),
                "probes_success": success,
                "probes_failed": failed,
            },
            merge=True,
        )
        log.info("audit=%s | complete — success=%d failed=%d", audit_id, success, failed)

    except Exception as exc:  # noqa: BLE001
        log.error("audit=%s | background task error: %s", audit_id, exc)
        audit_ref.set({"status": "failed", "error": str(exc)}, merge=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/create", response_model=CreateAuditResponse, summary="Create and validate a new audit")
async def create_audit(req: CreateAuditRequest) -> CreateAuditResponse:
    """
    Validate the connector configuration by sending one test prompt,
    generate and save the probe battery, and create an audit document
    in Firestore.

    Returns the ``audit_id`` to use in subsequent calls.
    """
    audit_id = str(uuid.uuid4())
    config = _config_from_request(req.connector)

    # --- Step 1: validate connector ---
    validation = await _connector.validate_config(config)
    if not validation["ok"]:
        raise HTTPException(
            status_code=422,
            detail=f"Connector validation failed: {validation['error']}",
        )

    # --- Step 2: generate probe battery ---
    try:
        battery = _generator.generate_probe_battery(
            scenario=req.scenario,
            num_probes=req.num_probes,
            attribute_filter=req.attribute_filter,
        )
        gcs_uri = _generator.save_battery(audit_id, battery)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Battery generation failed: {exc}") from exc

    # --- Step 3: write audit document ---
    db = _db()
    db.collection("audits").document(audit_id).set(
        {
            "audit_id": audit_id,
            "label": req.label or f"{req.scenario} audit",
            "scenario": req.scenario,
            "probe_count": len(battery),
            "pair_count": len({p.pair_id for p in battery}),
            "gcs_uri": gcs_uri,
            "status": "battery_ready",
            "probes_sent": 0,
            "probes_complete": 0,
            "probes_success": 0,
            "probes_failed": 0,
            "connector": {
                "provider": config.provider,
                "model": config.model,
                "base_url": config.base_url,
                "endpoint_url": config.endpoint_url,
            },
            "created_at": _fs.SERVER_TIMESTAMP,
        }
    )

    log.info(
        "audit=%s created | scenario=%s probes=%d connector=%s",
        audit_id, req.scenario, len(battery), config.provider,
    )

    return CreateAuditResponse(
        audit_id=audit_id,
        status="battery_ready",
        connector_ok=True,
        connector_test_response=validation.get("response_text"),
        message=f"Audit created with {len(battery)} probes. Call POST /api/audit/{audit_id}/run to start.",
    )


@router.post("/{audit_id}/run", response_model=RunAuditResponse, summary="Start probe execution")
async def run_audit(audit_id: str, background_tasks: BackgroundTasks) -> RunAuditResponse:
    """
    Kick off the probe battery asynchronously.

    Returns immediately with ``{"status": "running"}``.
    Use GET ``/api/audit/{id}/status`` to poll progress.
    """
    db = _db()
    doc = db.collection("audits").document(audit_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")

    data = doc.to_dict()
    status = data.get("status", "")
    if status == "running":
        raise HTTPException(status_code=409, detail="Audit is already running.")
    if status == "complete":
        raise HTTPException(status_code=409, detail="Audit already completed. Create a new audit to re-run.")

    # Rebuild config from stored connector spec
    conn_data = data.get("connector", {})
    # api_key is NOT stored in Firestore — customer must resupply it on /run
    # They should send it in request body; for now we pull from env as fallback
    config = ConnectorConfig(
        provider=conn_data.get("provider", "openai"),
        api_key=os.getenv("CUSTOMER_API_KEY", ""),       # overridden per-request below
        model=conn_data.get("model"),
        base_url=conn_data.get("base_url"),
        endpoint_url=conn_data.get("endpoint_url"),
    )

    # Schedule background work
    background_tasks.add_task(
        asyncio.ensure_future,
        _run_battery_task(audit_id, config),
    )

    return RunAuditResponse(
        audit_id=audit_id,
        status="running",
        message="Probe execution started. Poll GET /api/audit/{id}/status for progress.",
    )


class RunAuditWithKeyRequest(BaseModel):
    """Optional body for /run to supply the api_key securely at runtime."""
    connector: ConnectorConfigIn | None = None


@router.post("/{audit_id}/run-with-config", response_model=RunAuditResponse, summary="Start probe execution (with runtime config)")
async def run_audit_with_config(
    audit_id: str,
    req: RunAuditWithKeyRequest,
    background_tasks: BackgroundTasks,
) -> RunAuditResponse:
    """
    Same as /run but accepts the full connector config at call time.
    This is the preferred endpoint — it never stores the API key in Firestore.
    """
    db = _db()
    doc = db.collection("audits").document(audit_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")

    data = doc.to_dict()
    if data.get("status") == "running":
        raise HTTPException(status_code=409, detail="Audit is already running.")
    if data.get("status") == "complete":
        raise HTTPException(status_code=409, detail="Audit already completed.")

    if req.connector is None:
        raise HTTPException(status_code=422, detail="connector config required in request body.")

    config = _config_from_request(req.connector)

    background_tasks.add_task(
        asyncio.ensure_future,
        _run_battery_task(audit_id, config),
    )

    return RunAuditResponse(
        audit_id=audit_id,
        status="running",
        message="Probe execution started.",
    )


@router.get("/{audit_id}/status", response_model=AuditStatusResponse, summary="Get live audit progress")
async def get_audit_status(audit_id: str) -> AuditStatusResponse:
    """
    Return live progress for an audit.

    ``percent_done`` is calculated as ``probes_complete / probes_sent * 100``.
    """
    db = _db()
    doc = db.collection("audits").document(audit_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")

    d = doc.to_dict()
    probes_sent     = d.get("probes_sent", 0)     or 0
    probes_complete = d.get("probes_complete", 0) or 0
    probes_success  = d.get("probes_success", 0)  or 0
    probes_failed   = d.get("probes_failed", 0)   or 0

    percent = round(probes_complete / probes_sent * 100, 1) if probes_sent else 0.0

    return AuditStatusResponse(
        audit_id=audit_id,
        status=d.get("status", "unknown"),
        probes_sent=probes_sent,
        probes_complete=probes_complete,
        probes_success=probes_success,
        probes_failed=probes_failed,
        percent_done=percent,
        scenario=d.get("scenario"),
        label=d.get("label"),
    )


@router.get("/{audit_id}/results", summary="Fetch probe results (paginated)")
async def get_audit_results(
    audit_id: str,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    attribute: str | None = None,
    status: str | None = None,
) -> dict:
    """
    Return probe results stored under ``/audits/{audit_id}/probe_results``.

    Filters
    -------
    attribute : str   — e.g. "gender", "race"
    status    : str   — "success" | "failed" | "timeout"
    limit     : int   — max records returned (default 100, max 500)
    """
    db = _db()
    parent = db.collection("audits").document(audit_id)
    if not parent.get().exists:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")

    ref = parent.collection("probe_results")
    query = ref
    if attribute:
        query = query.where("attribute_tested", "==", attribute)
    if status:
        query = query.where("status", "==", status)
    query = query.limit(limit)

    docs = [d.to_dict() for d in query.stream()]
    return {"audit_id": audit_id, "count": len(docs), "results": docs}


# Needed for the /run fallback env-var path
import os
