# 🛡️ AI Hallucination & Factuality Detector

[![Streamlit](https://img.shields.io/badge/Streamlit-1.63-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.14-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com/)
[![Vector Search](https://img.shields.io/badge/Vector%20Search-NumPy%20Cosine%20Similarity-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An advanced, retrieval-augmented factuality verification engine that detects hallucinations in AI-generated answers. Decomposes responses into atomic factual propositions, retrieves evidence from multi-source knowledge bases (local curated documents, custom uploads, and real-time live Wikipedia search), and performs Natural Language Inference (NLI) classification with interactive inline visual highlighting and an academic benchmark evaluation suite.

---

## 📌 Project Highlights (Phase 2 Prototype)

1. **Atomic Proposition Decomposition:** Deconstructs compound sentences into independent, testable claims (Subject + Predicate + Object) mapped directly back to original text spans.
2. **Interactive Inline Color Highlighter:** Visualizes hallucinations directly over the input text:
   - 🟢 **Green:** Verified by Evidence (`SUPPORTED`)
   - 🔴 **Red:** Contradicted Hallucination (`CONTRADICTED`)
   - 🟡 **Yellow:** Epistemic Uncertainty (`INSUFFICIENT_EVIDENCE`)
3. **Multi-Source Knowledge Retrieval:**
   - **Curated Technical Knowledge Base:** Bundled references for Computer Science, Python, Java, C, and Algorithms.
   - **Custom Uploads:** Upload your own `.pdf`, `.txt`, `.md`, or `.csv` files.
   - **🌐 Free Real-time Live Wikipedia Search:** Verifies *any* general knowledge statement with zero extra API keys.
4. **Vectorized NumPy Cosine Similarity Engine:** Dense vector similarity computation without heavy or fragile external database dependencies.
5. **Academic ML Benchmark Suite:** Displays a real-time **Confusion Matrix**, **Accuracy**, **Precision**, **Recall**, and **F1-Score** on standard test pairs.
6. **Hallucination Taxonomy:** Categorizes contradictions into *Entity Errors*, *Numerical/Date Inaccuracies*, *Factual Fabrications*, and *Relational Inconsistencies*.
7. **Audit Report Export:** Downloadable JSON / Markdown fact-checking certificates.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────┐
│               AI-Generated Answer Input                │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Step 1: NLP Atomic Claim Extractor                    │
│  - Splits complex sentences into atomic propositions   │
│  - Preserves token span for inline visual highlighting │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Step 2: Multi-Source Dense Vector Retrieval           │
│  ├── Local Curated Documents                           │
│  ├── User-Uploaded Files (.pdf, .txt, .md)             │
│  └── 🌐 Real-Time Live Wikipedia Knowledge             │
│  - Metric: Cosine Similarity S = (u · v) / (||u||||v||)│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Step 3: Natural Language Inference (NLI) Judge        │
│  - Classifies: SUPPORTED / CONTRADICTED / INSUFFICIENT │
│  - Assigns Confidence & Hallucination Taxonomy         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Step 4: Interactive Dashboard & Academic Metrics      │
│  ├── 🎨 Inline Color-Coded Text Markup                 │
│  ├── 📊 Academic Benchmark Tab & Confusion Matrix      │
│  └── 📥 Downloadable Audit Reports                     │
└────────────────────────────────────────────────────────┘
```

---

## 🔬 Mathematical Formulations

### 1. Vector Cosine Similarity
For a claim query vector $\mathbf{q}$ and document chunk vector $\mathbf{d}$:
$$\text{Sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \frac{\sum_{i=1}^D q_i d_i}{\sqrt{\sum_{i=1}^D q_i^2} \sqrt{\sum_{i=1}^D d_i^2}}$$

### 2. Confidence-Weighted Reliability Score
$$\text{Reliability} = \left( \frac{\sum_{i=1}^N w_i \cdot c_i}{\sum_{i=1}^N c_i} \right) \times 100$$
Where:
* $w_i = 1.0$ if $\text{Verdict}_i = \text{SUPPORTED}$
* $w_i = 0.4$ if $\text{Verdict}_i = \text{INSUFFICIENT\_EVIDENCE}$
* $w_i = 0.0$ if $\text{Verdict}_i = \text{CONTRADICTED}$
* $c_i \in [0, 1]$ represents the NLI confidence score.

### 3. Hallucination Rate (HR)
$$\text{HR} = \left( \frac{N_{\text{Contradicted}}}{N_{\text{Total Claims}}} \right) \times 100$$

---

## 🚀 Quick Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sushant-1212/ai-hallucination-detector.git
cd ai-hallucination-detector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file (or copy `.env.example`):
```bash
cp .env.example .env
```
Add your free Google Gemini API key:
```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_gemini_api_key_here
```
> *Get a 100% free Gemini API key in 30 seconds at [Google AI Studio](https://aistudio.google.com/apikey).*

### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Verification & Testing

Run the automated integration test suite:
```bash
python test_pipeline.py
```

---

## 🎓 Faculty Viva & Evaluation Q&A

* **Q: Why decompose into atomic claims instead of verifying the entire text?**
  * *A:* A single sentence frequently couples true statements with false ones (e.g., *"Java was created by Dennis Ritchie in 1995"*). Evaluating the whole sentence yields ambiguous verdicts, while atomic decomposition isolates the specific entity error (*Dennis Ritchie*) while confirming the release year (*1995*).
* **Q: Why is "INSUFFICIENT EVIDENCE" distinct from "CONTRADICTED"?**
  * *A:* Absence of evidence is not evidence of falsity. If an unindexed but true fact is checked, categorizing it as contradicted would inflate false positives. Modeling epistemic uncertainty preserves detector precision.
* **Q: How does the vector index perform similarity matching?**
  * *A:* We utilize normalized dense embeddings in a NumPy-vectorized matrix. Cosine similarity reduces to an optimized single BLAS matrix-vector dot product $\mathbf{S} = \mathbf{M} \mathbf{q}$, eliminating heavy database infrastructure.

---

## 📄 License
MIT License
