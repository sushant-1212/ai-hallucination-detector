# AI Hallucination Detector

A retrieval-augmented fact-checking tool that verifies AI-generated answers against a trusted knowledge base — claim by claim — instead of taking an LLM's confidence at face value.

## Why

Language models can state incorrect information fluently and confidently. This tool breaks a given answer into individual, atomic factual claims and checks each one independently against a source of truth, so you get a precise, per-claim verdict instead of a single "trust it or don't" judgment.

## How it works

```
AI-Generated Answer
        │
        ▼
 Claim Extraction        →  splits the answer into atomic, checkable claims
        │
        ▼
    Retriever             →  embeds claims and searches a FAISS vector index
        │                     built from a curated or user-supplied knowledge base
        ▼
Relevant Evidence
        │
        ▼
  LLM Evaluator           →  judges each claim strictly against retrieved evidence
        │
        ▼
Supported / Contradicted / Insufficient Evidence
        │
        ▼
 Reliability Score + Report
```

Claims are extracted before verification because a single sentence can bundle multiple facts with different truth values — atomic extraction gives a precise report on exactly which part of an answer is wrong, not just a single blended verdict.

The `INSUFFICIENT_EVIDENCE` verdict exists deliberately: if the knowledge base doesn't cover a claim, the system says so rather than forcing a false positive or negative.

## Features

- **Claim-level verification** — every factual statement in an answer is checked independently, with its own verdict, confidence, and cited evidence
- **Pluggable knowledge base** — verify against a bundled curated reference set, your own uploaded documents (`.txt`, `.md`, `.pdf`), or both at once
- **Provider-agnostic** — switch between Gemini and OpenAI models with a single environment variable, no code changes
- **Reliability scoring** — an aggregate score and risk band (Low / Medium / High) computed from per-claim verdicts
- **No external database required** — uses an in-memory FAISS index, so there's nothing to host or provision

## Tech stack

| Layer      | Choice                                                              |
|------------|----------------------------------------------------------------------|
| Frontend   | Streamlit                                                           |
| Backend    | Python                                                              |
| Orchestration | LangChain                                                        |
| LLM        | Gemini (`gemini-1.5-flash`) or OpenAI (`gpt-4o-mini`) — switchable  |
| Embeddings | Gemini `text-embedding-004` or OpenAI `text-embedding-3-small`     |
| Vector store | FAISS (in-memory)                                                 |

## Project structure

```
ai-hallucination-detector/
├── app.py                 # Streamlit UI — entry point
├── llm_provider.py        # Provider abstraction (Gemini / OpenAI)
├── claim_extractor.py     # Answer → list of atomic claims
├── retriever.py           # Documents → FAISS index → evidence retrieval
├── evaluator.py           # Claim + evidence → verdict
├── scorer.py              # Verdicts → reliability score + risk band
├── knowledge_base/        # Sample curated reference documents
├── requirements.txt
└── .env.example
```

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/sushant-1212/ai-hallucination-detector.git
cd ai-hallucination-detector
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure your API key**
```bash
cp .env.example .env
```
Set `LLM_PROVIDER` to `gemini` or `openai` in `.env`, then add the matching key:
- Gemini: https://aistudio.google.com/apikey
- OpenAI: https://platform.openai.com/api-keys

**3. Run**
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`.

## Usage

Paste any AI-generated answer into the text box, choose what to verify it against in the sidebar, and click **Analyze**:

- **Curated knowledge base** — the bundled reference documents in `knowledge_base/`
- **Uploaded documents** — your own `.txt`, `.md`, or `.pdf` files
- **Both** — merges the two into a single search index

Each claim in the answer is reported with a verdict, a confidence score, a short explanation, and the retrieved evidence backing it.

## Limitations

- The reliability score is a heuristic aggregate, not a calibrated statistical probability
- Verdicts are only as reliable as the knowledge base supplied — an unindexed but true claim will surface as "insufficient evidence," not "correct"
- The evaluator model can itself make mistakes; this is a decision-support tool, not a guaranteed ground truth

## License

MIT
