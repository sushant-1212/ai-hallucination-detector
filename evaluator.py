"""
evaluator.py
-------------
Step 3 of the pipeline: 
Evaluates each atomic claim strictly against the retrieved evidence using
Natural Language Inference (NLI) logic. 

Produces:
  - Verdict: SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE
  - Confidence (0 - 100%)
  - Hallucination Type (if contradicted)
  - Clear justification referencing cited evidence
"""

import json
import re
from dataclasses import dataclass, field
from retriever import RetrievedChunk
from claim_extractor import ExtractedClaim
from llm_provider import generate_chat_response

EVAL_SYSTEM_PROMPT = """You are an automated Natural Language Inference (NLI) and Fact-Verification engine.
Your mission is to evaluate whether a CLAIM is supported or contradicted by the provided EVIDENCE passages.

Rules:
1. Ground your judgment ONLY in the provided EVIDENCE passages. Do not assume facts not explicitly stated.
2. Choose exactly ONE verdict:
   - "SUPPORTED": The evidence explicitly or semantically confirms the claim.
   - "CONTRADICTED": The evidence directly conflicts with or disproves the claim (AI Hallucination).
   - "INSUFFICIENT_EVIDENCE": The evidence neither confirms nor contradicts the claim (or no relevant evidence was found).
3. If CONTRADICTED, identify the "hallucination_type":
   - "Entity Error" (wrong person, place, or tool)
   - "Date / Numerical Inaccuracy" (wrong year, number, or metric)
   - "Factual Fabrication" (completely untrue statement)
   - "Relational Inconsistency" (misattributed action or causality)
   If SUPPORTED or INSUFFICIENT_EVIDENCE, set hallucination_type to "None".
4. Provide a concise, academic "explanation" (1-2 sentences) citing the exact evidence facts.
5. Provide a "confidence" score between 0 and 100.

Output strictly valid JSON with this format and NO surrounding text or markdown formatting:
{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE",
  "confidence": 95,
  "hallucination_type": "None" | "Entity Error" | "Date / Numerical Inaccuracy" | "Factual Fabrication" | "Relational Inconsistency",
  "explanation": "Brief reasoning referencing specific evidence facts"
}
"""


@dataclass
class ClaimResult:
    claim: str
    original_span: str
    verdict: str  # "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"
    confidence: int
    explanation: str
    hallucination_type: str = "None"
    evidence: list[RetrievedChunk] = field(default_factory=list)

    @property
    def max_similarity(self) -> float:
        """Maximum cosine similarity among retrieved evidence chunks."""
        if not self.evidence:
            return 0.0
        return max(e.score for e in self.evidence)


def evaluate_claim(extracted: ExtractedClaim, evidence: list[RetrievedChunk]) -> ClaimResult:
    """Evaluate a single claim against its retrieved evidence chunks."""
    claim_text = extracted.claim

    if evidence:
        evidence_text = "\n\n".join(
            f"[Source: {e.source} | Similarity: {e.score:.2f}]\n{e.text}"
            for e in evidence
        )
    else:
        evidence_text = "(No relevant evidence passages found in knowledge base)"

    user_prompt = f"CLAIM:\n{claim_text}\n\nRETRIEVED EVIDENCE:\n{evidence_text}"

    try:
        raw = generate_chat_response(
            prompt=user_prompt,
            system_prompt=EVAL_SYSTEM_PROMPT,
            temperature=0.0
        )
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        data = json.loads(cleaned)

        verdict = data.get("verdict", "INSUFFICIENT_EVIDENCE").upper()
        if verdict not in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"):
            verdict = "INSUFFICIENT_EVIDENCE"

        confidence = int(data.get("confidence", 50))
        confidence = max(0, min(100, confidence))
        hallucination_type = data.get("hallucination_type", "None")
        explanation = data.get("explanation", "").strip()

    except Exception as e:
        print(f"[evaluator] Evaluation parsing error: {e}")
        verdict = "INSUFFICIENT_EVIDENCE"
        confidence = 0
        hallucination_type = "None"
        explanation = "Automated parsing fallback: insufficient direct evidence identified."

    return ClaimResult(
        claim=claim_text,
        original_span=extracted.original_span,
        verdict=verdict,
        confidence=confidence,
        explanation=explanation,
        hallucination_type=hallucination_type,
        evidence=evidence
    )
