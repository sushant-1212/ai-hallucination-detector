"""
claim_extractor.py
-------------------
Step 1 of the pipeline: break a raw AI-generated answer into a list of
small, independently-checkable factual claims.

We ask the LLM to do this instead of naive sentence-splitting because a
single sentence can contain multiple claims ("Java was created by Dennis
Ritchie in 1985" = 2 claims: WHO and WHEN), and a claim can also span
multiple sentences.
"""

import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from llm_provider import get_chat_model

EXTRACTION_PROMPT = """You are a claim extraction engine used inside a fact-checking pipeline.

Given an AI-generated answer, break it down into a list of atomic, independently
verifiable factual claims. Rules:
- Each claim must be a single, self-contained statement of fact.
- Split compound sentences into separate claims (e.g. "who did X" and "when X happened"
  are two separate claims even if they appear in the same sentence).
- Ignore opinions, hedges, greetings, or filler text ("I think", "Great question!", etc.)
- Ignore instructions or meta-commentary that isn't a factual claim.
- If the answer contains no checkable factual claims, return an empty list.

Return ONLY valid JSON in this exact format, with no markdown fences and no extra text:
{"claims": ["claim 1 text", "claim 2 text", ...]}
"""


def extract_claims(ai_answer: str) -> list[str]:
    """Return a list of atomic factual claim strings extracted from ai_answer."""
    if not ai_answer or not ai_answer.strip():
        return []

    model = get_chat_model(temperature=0.0)
    messages = [
        SystemMessage(content=EXTRACTION_PROMPT),
        HumanMessage(content=f"AI-generated answer to analyze:\n\n{ai_answer}"),
    ]
    response = model.invoke(messages)
    raw = response.content.strip()

    # Models sometimes wrap JSON in ```json fences despite instructions - strip them.
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
        claims = parsed.get("claims", [])
        return [c.strip() for c in claims if isinstance(c, str) and c.strip()]
    except json.JSONDecodeError:
        # Fallback: if the model didn't return clean JSON, do a naive
        # sentence split so the app still produces a result.
        sentences = re.split(r"(?<=[.!?])\s+", ai_answer.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 8]
