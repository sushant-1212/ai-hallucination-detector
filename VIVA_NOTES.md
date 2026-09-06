# Viva / Demo Notes

## One-line pitch
"LLMs sometimes state wrong facts confidently. This tool splits an AI's
answer into individual claims and checks each one against a trusted
knowledge base using Retrieval-Augmented Generation (RAG), instead of just
trusting the AI's own confidence."

## What LangChain is actually doing (this gets asked every time)
LangChain is the **orchestration layer** connecting four independent stages:
1. `claim_extractor.py` — a LangChain chat model call with a structured-output
   prompt, turning free text into a JSON list of atomic claims.
2. `retriever.py` — LangChain's document loaders + text splitter chunk the
   knowledge base, LangChain's embeddings wrapper turns chunks into vectors,
   and LangChain's FAISS integration stores/searches them.
3. `evaluator.py` — another LangChain chat model call, this time given the
   claim + retrieved evidence, forced to answer using *only* that evidence.
4. `scorer.py` — plain Python, no LLM call; aggregates verdicts into a score.

LangChain's value here is standardizing the interface across these steps
(same `invoke()` pattern for every LLM call, swappable embeddings/vector
store) so the pipeline stays provider-agnostic — swapping Gemini for OpenAI
is a one-line env var change, not a rewrite.

## Why three verdicts, not two
A binary TRUE/FALSE system is forced to guess when the knowledge base simply
doesn't cover a claim. `INSUFFICIENT_EVIDENCE` prevents the system from
falsely accusing a correct-but-unverifiable claim of being a hallucination —
this is a deliberate design choice worth calling out.

## Why claims are extracted before checking (not checking the whole answer at once)
A single sentence can bundle multiple facts of different truth values (e.g.
"Java was invented by Dennis Ritchie [false] in 1985 [also false, separate
fact]"). Checking the whole answer as one blob would only produce one verdict
and hide which specific part was wrong. Atomic claims give a precise,
per-fact report.

## Known limitations (be upfront about these — it shows maturity)
- The reliability score is a **heuristic**, not a calibrated statistical
  probability — say this explicitly if asked.
- Verdicts are only as good as the knowledge base; a claim can be technically
  true but marked "insufficient evidence" if the KB doesn't cover it.
- The evaluator LLM can itself make mistakes — this is a decision-support
  tool, not a guaranteed ground truth.
- Retrieval quality depends on chunk size and embedding model; very short or
  very ambiguous claims retrieve noisier evidence.

## Possible extensions to mention if asked "what would you add next?"
- Let users flag incorrect verdicts to build a feedback dataset.
- Add source citations with page numbers for uploaded PDFs.
- Support multi-turn conversations (checking an AI's answer in context of the
  preceding chat, not just a single pasted answer).
- Add a browser extension that runs this automatically on ChatGPT/Gemini
  responses.
