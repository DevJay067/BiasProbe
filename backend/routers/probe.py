"""
routers/probe.py
================
FastAPI router exposing probe battery generation endpoints.

POST /probes/generate
    Generate and save a probe battery for a given scenario.

GET  /probes/{audit_id}
    Retrieve battery metadata from Firestore.

GET  /probes/{audit_id}/download
    Stream the battery JSON from GCS.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from services.probe_generator import ProbeGenerator, SCENARIO_FILE_MAP

router = APIRouter(prefix="/probes", tags=["Probe Generator"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    scenario: str = Field(..., description="Scenario key, e.g. 'hiring_assistant'")
    num_probes: int = Field(200, ge=10, le=2000, description="Target number of probes (must be even)")
    attribute_filter: list[str] | None = Field(
        None, description="Restrict to specific protected attributes; null = all"
    )
    audit_id: str | None = Field(
        None, description="Custom audit ID; auto-generated if omitted"
    )


class GenerateResponse(BaseModel):
    audit_id: str
    scenario: str
    probe_count: int
    pair_count: int
    gcs_uri: str
    status: str


# ---------------------------------------------------------------------------
# Shared generator instance (module-level singleton)
# ---------------------------------------------------------------------------
_generator: ProbeGenerator | None = None


def _get_generator() -> ProbeGenerator:
    global _generator
    if _generator is None:
        _generator = ProbeGenerator()
    return _generator


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/scenarios", summary="List available scenarios")
def list_scenarios() -> dict:
    """Return all supported scenario keys and their template file names."""
    return {"scenarios": SCENARIO_FILE_MAP}


@router.post("/generate", response_model=GenerateResponse, summary="Generate probe battery")
async def generate_battery(
    req: GenerateRequest,
    background_tasks: BackgroundTasks,
) -> GenerateResponse:
    """
    Generate a Gemini-powered probe battery for the specified scenario.
    The battery is saved to Firestore and GCS asynchronously (background task).

    - **scenario**: one of the keys returned by GET /probes/scenarios
    - **num_probes**: target battery size (even number; default 200)
    - **attribute_filter**: optional list of protected attributes to include
    """
    if req.scenario not in SCENARIO_FILE_MAP:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown scenario '{req.scenario}'. Use GET /probes/scenarios for valid keys.",
        )

    audit_id = req.audit_id or str(uuid.uuid4())
    gen = _get_generator()

    # Run generation synchronously (could be moved to Celery/Cloud Tasks for large runs)
    try:
        battery = gen.generate_probe_battery(
            scenario=req.scenario,
            num_probes=req.num_probes,
            attribute_filter=req.attribute_filter,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    pair_count = len({p.pair_id for p in battery})

    # Persist in background so the API returns promptly
    try:
        gcs_uri = gen.save_battery(audit_id, battery)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Persist failed: {exc}") from exc

    return GenerateResponse(
        audit_id=audit_id,
        scenario=req.scenario,
        probe_count=len(battery),
        pair_count=pair_count,
        gcs_uri=gcs_uri,
        status="battery_ready",
    )


@router.get("/{audit_id}", summary="Get battery metadata")
async def get_battery_metadata(audit_id: str) -> dict:
    """Retrieve probe battery metadata from Firestore."""
    from google.cloud import firestore as _fs
    db = _fs.Client()
    doc = db.collection("probe_batteries").document(audit_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")
    return doc.to_dict()


@router.get("/{audit_id}/probes", summary="List probes in battery")
async def list_probes(
    audit_id: str,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    attribute: str | None = None,
) -> dict:
    """
    Return a sample of probes from the battery stored in Firestore.

    Optionally filter by `attribute` (e.g. 'gender', 'race').
    """
    from google.cloud import firestore as _fs
    db = _fs.Client()
    ref = db.collection("probe_batteries").document(audit_id).collection("probes")

    query = ref.limit(limit)
    if attribute:
        query = ref.where("attribute_tested", "==", attribute).limit(limit)

    docs = query.stream()
    probes = [d.to_dict() for d in docs]
    if not probes and not db.collection("probe_batteries").document(audit_id).get().exists:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")

    return {"audit_id": audit_id, "count": len(probes), "probes": probes}
