# 🔍 AI Hallucination Detector

A RAG-based tool that fact-checks AI-generated answers by breaking them into
individual claims and verifying each one against a trusted knowledge base —
instead of trusting the AI's own confidence.

Built for an AI/ML college project. See [`VIVA_NOTES.md`](VIVA_NOTES.md) for
exam/demo talking points.

## How it works

```
        USER
          │
          ▼
  AI-Generated Answer
          │
          ▼
 ┌───────────────────┐
 │ Claim Extraction   │   (LLM splits the answer into atomic factual claims)
 └─────────┬──────────┘
           │
      Individual
        Claims
           │
           ▼
 ┌───────────────────┐
 │     Retriever      │   (FAISS similarity search over embedded KB chunks)
 └─────────┬──────────┘
           │
           ▼
   Relevant Evidence
           │
           ▼
 ┌───────────────────┐
 │   LLM Evaluator    │   (judges claim vs. evidence only, no outside knowledge)
 └─────────┬──────────┘
           │
           ▼
  Supported / Contradicted /
     Insufficient Evidence
           │
           ▼
      Final Report + Reliability Score
```

## Tech stack

| Layer       | Choice                                   |
|-------------|-------------------------------------------|
| Frontend    | Streamlit                                  |
| Backend     | Python                                     |
| Framework   | LangChain (orchestrates extraction → retrieval → evaluation) |
| LLM         | Gemini (`gemini-1.5-flash`) or OpenAI (`gpt-4o-mini`) — switchable via `.env` |
| Embeddings  | Gemini `text-embedding-004` or OpenAI `text-embedding-3-small` |
| Vector DB   | FAISS (in-memory, no server needed)        |
| Knowledge   | Curated `.txt` files in `knowledge_base/`, plus optional user-uploaded `.txt`/`.md`/`.pdf` |

## Project structure

```
ai-hallucination-detector/
├── app.py                 # Streamlit UI — entry point
├── llm_provider.py        # Switches between Gemini / OpenAI
├── claim_extractor.py     # Step 1: answer -> list of atomic claims
├── retriever.py           # Step 2: documents -> FAISS -> evidence per claim
├── evaluator.py           # Step 3: claim + evidence -> verdict
├── scorer.py              # Step 4: verdicts -> reliability score + risk band
├── knowledge_base/        # Sample curated CS reference docs (Java, Python, C, OOP concepts)
├── requirements.txt
├── .env.example
└── VIVA_NOTES.md          # Talking points for your demo/viva
```

## Setup

1. **Clone and install dependencies**
   ```bash
   git clone <your-repo-url>
   cd ai-hallucination-detector
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add your API key**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and set `LLM_PROVIDER` to `gemini` or `openai`, then fill in the
   matching API key.
   - Get a free Gemini key at https://aistudio.google.com/apikey
   - Get an OpenAI key at https://platform.openai.com/api-keys

3. **Run the app**
   ```bash
   streamlit run app.py
   ```
   This opens the app at `http://localhost:8501`.

4. **Try it**
   - Paste an AI answer, e.g.:
     > "Java was invented by Dennis Ritchie in 1985. It uses automatic garbage collection."
   - Click **Analyze**.
   - You should see Claim 1 marked ❌ Contradicted (it was James Gosling, 1991/95),
     and the garbage-collection claim marked ✅ Supported.

## Verification modes

In the sidebar, choose what to verify the answer against:
- **Computer Science knowledge base** — the bundled sample docs in `knowledge_base/`.
- **My uploaded documents** — upload your own `.txt`/`.md`/`.pdf` (e.g. lecture notes).
- **Both** — merges the two into one search index.

This is what turns the project from a generic demo into a practical academic
tool: a professor's lecture notes become the ground truth, and the tool checks
whether a chatbot's answer actually matches what was taught.

## Extending it

- Add more files to `knowledge_base/` to widen the default coverage.
- Swap `gemini-1.5-flash` / `gpt-4o-mini` in `llm_provider.py` for a stronger
  (and slower/costlier) model if you want higher accuracy for the demo.
- Increase `top_k` (evidence chunks per claim) in the sidebar for longer, more
  detailed source documents.

## Disclaimer

The "Reliability Score" and risk labels (🟢 Low / 🟡 Medium / 🔴 High) are a
**heuristic demo metric** built for this project, not a scientifically
calibrated probability of correctness.
