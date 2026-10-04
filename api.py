"""
api.py
------
FastAPI REST backend for the AI Hallucination & Factuality Detector.
Serves real-time claim decomposition, vector retrieval, and NLI verification
for the Chrome Browser Extension and external integrations.
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from claim_extractor import extract_claims
from retriever import (
    load_documents_from_folder,
    build_vector_store,
    retrieve_evidence,
    VectorIndex
)
from evaluator import evaluate_claim, ClaimResult
from scorer import compute_score, ScoreSummary
import llm_provider

app = FastAPI(
    title="AI Hallucination Detector API",
    description="REST API for real-time hallucination evaluation and claim verification",
    version="2.0.0"
)

# Enable CORS for browser extensions and local web portals
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows browser extensions and local/remote hosts
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KB_INDEX: Optional[VectorIndex] = None


def get_kb() -> Optional[VectorIndex]:
    global KB_INDEX
    if KB_INDEX is None:
        print("[API] Indexing local knowledge base...")
        docs = load_documents_from_folder("knowledge_base")
        KB_INDEX = build_vector_store(docs)
        chunk_count = len(KB_INDEX.texts) if KB_INDEX else 0
        print(f"[API] Knowledge Base Ready! Indexed {len(docs)} documents ({chunk_count} chunks).")
    return KB_INDEX


@app.on_event("startup")
def startup_event():
    get_kb()


class CheckRequest(BaseModel):
    text: str
    llm_source: Optional[str] = "unknown"  # chatgpt, claude, gemini, manual
    use_wikipedia: Optional[bool] = True


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "AI Hallucination Detector API",
        "provider": llm_provider.get_active_provider(),
        "provider_display": llm_provider.provider_name(),
        "active_llms_supported": ["chatgpt", "claude", "gemini"]
    }


@app.get("/api/health")
def health():
    kb = get_kb()
    chunks = len(kb.texts) if kb else 0
    unique_sources = len(set(kb.sources)) if kb else 0
    return {
        "status": "online",
        "provider": llm_provider.get_active_provider(),
        "provider_display": llm_provider.provider_name(),
        "indexed_sources": unique_sources,
        "indexed_chunks": chunks
    }


@app.post("/api/check")
def check_hallucination(payload: CheckRequest):
    """
    Decomposes incoming text into atomic claims, retrieves factual evidence,
    and runs NLI verification to compute reliability and hallucination scores.
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # 1. Extract claims
    extracted_claims = extract_claims(text)
    if not extracted_claims:
        return {
            "success": True,
            "text": text,
            "llm_source": payload.llm_source,
            "reliability_score": 100.0,
            "hallucination_rate": 0.0,
            "risk_label": "No Factual Claims Detected",
            "risk_emoji": "⚪",
            "risk_color": "#94a3b8",
            "total_claims": 0,
            "supported_count": 0,
            "contradicted_count": 0,
            "insufficient_count": 0,
            "claims": []
        }

    # 2. Retrieve & Evaluate each claim
    results: list[ClaimResult] = []
    for claim_obj in extracted_claims:
        evidence = retrieve_evidence(
            vector_store=get_kb(),
            claim=claim_obj.claim,
            k=3,
            include_web_search=payload.use_wikipedia
        )
        res = evaluate_claim(claim_obj, evidence)
        results.append(res)

    # 3. Compute aggregate scores
    summary: ScoreSummary = compute_score(results)

    # 4. Format clean JSON response
    claims_out = []
    for r in results:
        claims_out.append({
            "claim": r.claim,
            "original_span": r.original_span,
            "verdict": r.verdict,
            "confidence": r.confidence,
            "explanation": r.explanation,
            "hallucination_type": r.hallucination_type,
            "evidence": [
                {
                    "text": e.text[:400] + ("..." if len(e.text) > 400 else ""),
                    "source": e.source,
                    "score": round(float(e.score), 3)
                }
                for e in r.evidence
            ]
        })

    return {
        "success": True,
        "text": text,
        "llm_source": payload.llm_source,
        "reliability_score": round(summary.reliability_score, 1),
        "hallucination_rate": round(summary.hallucination_rate, 1),
        "risk_label": summary.risk_label,
        "risk_emoji": summary.risk_emoji,
        "risk_color": summary.risk_color,
        "total_claims": summary.total_claims,
        "supported_count": summary.supported_count,
        "contradicted_count": summary.contradicted_count,
        "insufficient_count": summary.insufficient_count,
        "claims": claims_out
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
