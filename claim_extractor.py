"""
claim_extractor.py
-------------------
Step 1 of the pipeline: 
Breaks a raw AI-generated response into atomic, independently checkable factual claims,
and maps each claim back to its original sentence span for visual inline highlighting.
"""

import json
import re
from dataclasses import dataclass
from llm_provider import generate_chat_response

EXTRACTION_SYSTEM_PROMPT = """You are a rigorous Natural Language Processing (NLP) claim-extraction engine.
Given an AI-generated text, decompose it into a list of atomic, self-contained factual claims.

Rules:
1. Each claim must be a single, testable proposition (Subject + Predicate + Object/Fact).
2. Split compound or multi-fact sentences into separate atomic claims:
   - "Java was created by Dennis Ritchie in 1995" ->
     Claim 1: "Java was created by Dennis Ritchie"
     Claim 2: "Java was released in 1995"
3. Identify the EXACT original sentence or text snippet from the input that each claim originated from.
4. Exclude opinions, greetings, pleasantries, hedging ("I think", "Perhaps"), or meta-instructions.
5. If the input contains no factual claims, return an empty list.

Output ONLY valid JSON in this exact structure with no markdown backticks or commentary:
{
  "claims": [
    {
      "claim": "Atomic claim statement here",
      "original_span": "Exact sentence or clause from input"
    }
  ]
}
"""


@dataclass
class ExtractedClaim:
    claim: str
    original_span: str = ""


def extract_claims(ai_answer: str) -> list[ExtractedClaim]:
    """Extract atomic factual claims and map to original spans."""
    if not ai_answer or not ai_answer.strip():
        return []

    prompt = f"Decompose this AI-generated text into atomic claims:\n\n{ai_answer}"

    try:
        raw_output = generate_chat_response(
            prompt=prompt,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            temperature=0.0
        )
        # Strip potential markdown fences
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw_output.strip(), flags=re.MULTILINE).strip()
        data = json.loads(cleaned)
        items = data.get("claims", [])
        results = []
        for it in items:
            if isinstance(it, dict):
                c_text = it.get("claim", "").strip()
                span = it.get("original_span", "").strip()
            elif isinstance(it, str):
                c_text = it.strip()
                span = ""
            else:
                continue

            if c_text:
                results.append(ExtractedClaim(claim=c_text, original_span=span or c_text))

        if results:
            return results

    except Exception as e:
        print(f"[claim_extractor] LLM extraction error ({e}), falling back to syntactic sentence splitting.")

    # Fallback: Syntactic sentence tokenizer
    sentences = re.split(r"(?<=[.!?])\s+", ai_answer.strip())
    fallback_claims = []
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) > 10:
            fallback_claims.append(ExtractedClaim(claim=s_clean, original_span=s_clean))

    return fallback_claims
