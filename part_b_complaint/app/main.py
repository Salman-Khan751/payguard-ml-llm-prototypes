"""FastAPI service.  Run from repo root:  uvicorn part_b_complaint.app.main:app --reload"""
from fastapi import FastAPI

from . import llm
from .schemas import (BatchRequest, BatchResponse, Metrics, TriageRequest,
                      TriageResponse)
from .triage import triage_one

app = FastAPI(title="PayGuard Complaint Triage", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "llm_configured": llm.configured(), "model": llm.MODEL}


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest):
    # Sync route: FastAPI runs it in a thread pool, so slow LLM calls don't block the loop.
    return triage_one(req.narrative)


@app.post("/triage/batch", response_model=BatchResponse)
def triage_batch(req: BatchRequest):
    return BatchResponse(results=[triage_one(r.narrative) for r in req.narratives])


@app.get("/metrics", response_model=Metrics)
def metrics():
    u = llm.usage
    cost = u.cost()
    per = cost / u.calls if u.calls else 0.0
    return Metrics(calls=u.calls, tokens_in=u.tokens_in, tokens_out=u.tokens_out,
                   est_cost_usd=round(cost, 6), est_cost_per_complaint_usd=round(per, 8),
                   est_cost_50k_per_month_usd=round(per * 50_000, 2))
