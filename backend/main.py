"""
main.py — BiasProbe FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.audit import router as audit_router
from routers.judge import router as judge_router
from routers.probe import router as probe_router
from routers.report import router as report_router
from routers.stats import router as stats_router

app = FastAPI(
    title="BiasProbe API",
    version="0.6.0",
    description="AI bias auditing platform — probe generation, LLM execution, bias judging, statistical analysis, report generation, and PDF export.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(audit_router)
app.include_router(judge_router)
app.include_router(probe_router)
app.include_router(report_router)
app.include_router(stats_router)


@app.get("/", tags=["health"])
def health() -> dict:
    return {"status": "ok", "service": "BiasProbe API", "version": "0.6.0"}
