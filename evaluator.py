"""
evaluator.py
-------------
Step 3 of the pipeline: given a claim and the evidence retrieved for it,
ask the LLM to classify the claim as SUPPORTED, CONTRADICTED, or
INSUFFICIENT_EVIDENCE, with a short explanation and confidence score.

INSUFFICIENT_EVIDENCE exists deliberately: if the knowledge base has
nothing relevant, the system must say so instead of guessing FALSE.
"""

import json
import re
from dataclasses import dataclass, field
from langchain_core.messages import SystemMessage, HumanMessage
from llm_provider import get_chat_model

EVAL_PROMPT = """You are a fact-verification engine. You will be given a CLAIM and a
set of EVIDENCE passages retrieved from a trusted knowledge base.

Decide the verdict for the claim using ONLY the evidence provided (do not use
outside knowledge). Choose exactly one verdict:
- "SUPPORTED": the evidence confirms the claim.
- "CONTRADICTED": the evidence directly conflicts with the claim.
- "INSUFFICIENT_EVIDENCE": the evidence does not clearly confirm or deny the claim
  (including when no evidence was retrieved at all).

Also give:
- "explanation": one short sentence justifying the verdict, referencing the evidence.
- "confidence": an integer 0-100 for how confident you are in this verdict.

Return ONLY valid JSON, no markdown fences, in this exact format:
{"verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE", "explanation": "...", "confidence": 0}
"""


@dataclass
class ClaimResult:
    claim: str
    verdict: str
    explanation: str
    confidence: int
    evidence: list = field(default_factory=list)


def evaluate_claim(claim: str, evidence: list[str]) -> ClaimResult:
    model = get_chat_model(temperature=0.0)

    if evidence:
        evidence_block = "\n\n".join(f"- {e}" for e in evidence)
    else:
        evidence_block = "(no relevant evidence was found in the knowledge base)"

    messages = [
        SystemMessage(content=EVAL_PROMPT),
        HumanMessage(
            content=f"CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence_block}"
        ),
    ]
    response = model.invoke(messages)
    raw = response.content.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "INSUFFICIENT_EVIDENCE")
        if verdict not in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"):
            verdict = "INSUFFICIENT_EVIDENCE"
        explanation = parsed.get("explanation", "")
        confidence = int(parsed.get("confidence", 50))
    except (json.JSONDecodeError, ValueError):
        verdict = "INSUFFICIENT_EVIDENCE"
        explanation = "Could not parse model output; defaulting to insufficient evidence."
        confidence = 0

    return ClaimResult(
        claim=claim,
        verdict=verdict,
        explanation=explanation,
        confidence=confidence,
        evidence=evidence,
    )


def evaluate_all(claims: list[str], vector_store, retrieve_fn, k: int = 3) -> list[ClaimResult]:
    """Convenience helper: retrieve evidence + evaluate for a whole list of claims."""
    results = []
    for claim in claims:
        evidence = retrieve_fn(vector_store, claim, k=k)
        results.append(evaluate_claim(claim, evidence))
    return results
