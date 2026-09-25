# 🛡️ AI Hallucination & Factuality Detector

[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.0-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-3072--dim%20Dense%20Vectors-8E24AA?style=for-the-badge)](https://ai.google.dev/)
[![Vector Search](https://img.shields.io/badge/Vector%20Search-NumPy%20Cosine%20Similarity-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An advanced, retrieval-augmented factuality verification engine that detects hallucinations in AI-generated responses. Decomposes outputs into atomic factual propositions, retrieves evidence from multi-source knowledge bases (curated technical documents, custom uploads, and real-time Wikipedia search), and performs Natural Language Inference (NLI) classification with interactive inline visual highlighting and a comprehensive academic benchmark suite.

---

## 🛠️ Tech Stack & Architecture

| Layer | Technology | Implementation Details |
|:---|:---|:---|
| **Frontend UI** | **Streamlit (v1.42.0)** | Multi-tab dashboard with custom CSS-styled inline visual highlighting |
| **Backend Core** | **Python (3.10–3.14)** | Modular pipeline with asynchronous HTTP clients (zero external framework bloat) |
| **LLM Reasoning & NLI** | **Google Gemini 2.5 Flash** (Default) | Multi-provider architecture supporting Gemini 2.5 Flash, Groq (Llama-3.3-70B), OpenAI (GPT-4o-mini), and xAI (Grok-2) |
| **Dense Vector Embeddings** | **Google `gemini-embedding-001`** | High-dimensional semantic embeddings (**3,072 dimensions**) |
| **Vector Search Engine** | **Custom In-Memory NumPy Engine** | High-speed normalized Cosine Similarity (`S = M · q`, no heavy external vector databases) |
| **Dynamic Knowledge Retrieval** | **Wikipedia REST API** | Live factual background retrieval for open-domain fact verification |
| **Document Ingestion** | **PyPDF & Semantic Chunking** | Ingests `.pdf`, `.txt`, `.md`, and `.csv` files with sentence-boundary chunking |

---

## 📌 Key System Features

### 1. 🎨 Interactive Inline Visual Highlighter
Unlike tools that merely list errors in separate logs, the engine maps every extracted claim back to its exact character and token spans in the original text, color-coding the paragraph in real time:
* 🟢 **Green Highlight (`SUPPORTED`):** Factually verified by retrieved evidence passages.
* 🔴 **Red Highlight (`CONTRADICTED`):** Flagged as an AI hallucination with an error category tooltip.
* 🟡 **Yellow Highlight (`INSUFFICIENT_EVIDENCE`):** Epistemic uncertainty where knowledge sources neither confirm nor deny the proposition.

### 2. 📊 Academic ML Benchmark Suite
To ensure quantitative scientific rigor, the system includes a dedicated evaluation benchmark with standardized ground-truth test pairs across Computer Science, programming paradigms, and algorithms. It computes live:
* **Confusion Matrix:** Measures True Positives, False Positives, True Negatives, and False Negatives.
* **Classification Metrics:**

```math
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad \text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
```

* **Latency Tracking:** Real-time verification latency per claim.

### 3. 🌐 Multi-Source Dense Retrieval
* **Curated Technical Knowledge Base:** Bundled verified documentation covering Python, Java, C, and algorithm complexity.
* **Custom Document Uploads:** Ingests user-supplied `.pdf`, `.txt`, `.md`, and `.csv` files.
* **Real-Time Wikipedia Retrieval:** Queries Wikipedia's REST API dynamically to retrieve authoritative contexts for open-domain statements.

### 4. 🔬 In-Memory Vectorized Cosine Similarity (NumPy)
Avoids cumbersome external vector databases by utilizing an in-memory linear algebra engine. Chunk embeddings are normalized to the unit sphere ($L_2$ norm), transforming similarity search into an optimized BLAS matrix-vector product:

```math
\text{Sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \sum_{i=1}^D q_i d_i \quad (\text{when } \|\mathbf{q}\| = \|\mathbf{d}\| = 1)
```

### 5. 🏷️ Fine-Grained Hallucination Taxonomy
When an assertion is contradicted, the NLI classifier categorizes the defect into a specific failure mode:
* **Entity Error:** Misattributed creator, company, or tool.
* **Date / Numerical Inaccuracy:** Discrepancies in years, versions, or quantitative metrics.
* **Factual Fabrication:** Statements that have no basis in factual reality.
* **Relational Inconsistency:** Misrepresented causality or structural relationships.

### 6. 🔌 Provider-Agnostic LLM Backend
Switch seamlessly between multiple model providers via environment configuration with zero code refactoring:
* **Google Gemini 2.5 Flash** (Default, generous free tier & high speed)
* **Groq** (`llama-3.3-70b-versatile` running on ultra-fast LPU inference)
* **OpenAI** (`gpt-4o-mini`)
* **xAI** (`grok-2-latest` with local embeddings fallback)

---

## 🏗️ Pipeline Architecture

```
┌────────────────────────────────────────────────────────┐
│               AI-Generated Answer Input                │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Step 1: NLP Atomic Claim Decomposition                │
│  - Deconstructs compound clauses into single facts     │
│  - Preserves token span for inline visual highlighting │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Step 2: Multi-Source Dense Vector Retrieval           │
│  ├── Local Curated Documents (.txt, .md)               │
│  ├── User-Uploaded Files (.pdf, .txt, .csv)            │
│  └── Real-Time Wikipedia REST API Knowledge            │
│  - Metric: Vectorized Cosine Similarity S = M · q      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Step 3: Natural Language Inference (NLI) Judge        │
│  - Classifies: SUPPORTED / CONTRADICTED / INSUFFICIENT │
│  - Categorizes Hallucination Taxonomy & Confidence %   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Step 4: Interactive Dashboard & Academic Metrics      │
│  ├── 🎨 Inline Color-Coded Text Markup                 │
│  ├── 📊 Academic Benchmark Suite & Confusion Matrix    │
│  └── 📥 Exportable JSON Audit Reports                  │
└────────────────────────────────────────────────────────┘
```

---

## 🔬 Mathematical Formulations

### 1. Confidence-Weighted Reliability Score

```math
\text{Reliability} = \left( \frac{\sum_{i=1}^N w_i \cdot c_i}{\sum_{i=1}^N c_i} \right) \times 100
```

Where $c_i \in [0, 1]$ represents the NLI confidence score, and weights $w_i$ are assigned as:

| Verification Verdict | State | Weight ($w_i$) | Interpretation |
| :--- | :--- | :---: | :--- |
| **`SUPPORTED`** | Entailment | `1.0` | Maximum factual credibility |
| **`INSUFFICIENT_EVIDENCE`** | Neutral / Epistemic | `0.4` | Unpenalized absence of ground truth |
| **`CONTRADICTED`** | Hallucination | `0.0` | Direct factual fabrication or contradiction |

### 2. Hallucination Rate (HR)

```math
\text{Hallucination Rate} = \left( \frac{N_{\text{Contradicted}}}{N_{\text{Total Claims}}} \right) \times 100
```

---

## 📁 Repository Structure

```
ai-hallucination-detector/
├── app.py                 # Streamlit UI (Live Detector, Benchmark Suite, Architecture Tab)
├── claim_extractor.py     # NLP atomic claim decomposition & span mapping
├── retriever.py           # NumPy Cosine VectorIndex + Wikipedia REST retriever
├── evaluator.py           # Natural Language Inference (NLI) & taxonomy classifier
├── scorer.py              # Confidence-weighted reliability metrics & risk assessment
├── benchmark.py           # Academic evaluation suite & Confusion Matrix calculation
├── llm_provider.py        # Gemini 2.5 Flash & 3072-dim embedding abstraction
├── test_pipeline.py       # End-to-end automated integration test suite
├── knowledge_base/        # Bundled curated technical reference documents
├── requirements.txt       # Pinned production dependencies
├── .env.example           # Environment configuration template
└── LICENSE                # MIT License
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sushant-1212/ai-hallucination-detector.git
cd ai-hallucination-detector
```

### 2. Install Pinned Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file:
```bash
cp .env.example .env
```
Add your Google Gemini API key:
```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing

Execute the complete end-to-end integration test:
```bash
python test_pipeline.py
```

---

## 📄 License
MIT License
