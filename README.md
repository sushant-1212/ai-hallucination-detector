# 🛡️ AI Hallucination & Factuality Detector

[![Streamlit](https://img.shields.io/badge/Streamlit-1.63-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.14-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com/)
[![Vector Search](https://img.shields.io/badge/Vector%20Search-NumPy%20Cosine%20Similarity-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An advanced, retrieval-augmented factuality verification engine that detects hallucinations in AI-generated answers. Decomposes responses into atomic factual propositions, retrieves evidence from multi-source knowledge bases (curated technical documents, custom uploads, and real-time Wikipedia search), and performs Natural Language Inference (NLI) classification with interactive inline visual highlighting and a comprehensive evaluation benchmark.

---

## 📌 Key Features

* **Atomic Proposition Decomposition:** Deconstructs compound sentences into independent, testable claims mapped directly back to original text spans.
* **Interactive Inline Color Highlighter:** Visualizes factual credibility directly over the input text:
  * 🟢 **Green:** Verified by Evidence (`SUPPORTED`)
  * 🔴 **Red:** Contradicted Hallucination (`CONTRADICTED`)
  * 🟡 **Yellow:** Epistemic Uncertainty (`INSUFFICIENT_EVIDENCE`)
* **Multi-Source Knowledge Retrieval:**
  * **Curated Technical Knowledge Base:** Bundled references for Computer Science, Python, Java, C, and Algorithms.
  * **Custom Document Uploads:** Ingests user-supplied `.pdf`, `.txt`, `.md`, or `.csv` files.
  * **Real-time Wikipedia Search:** Retrieves live reference contexts dynamically for general domain queries.
* **Vectorized Cosine Similarity Engine:** Dense vector similarity computation using optimized matrix-vector operations ($\mathbf{S} = \mathbf{M} \mathbf{q}$).
* **Academic ML Benchmark Suite:** Automated evaluation battery measuring **Confusion Matrix**, **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
* **Hallucination Taxonomy:** Classifies contradicted statements into *Entity Errors*, *Numerical/Date Inaccuracies*, *Factual Fabrications*, and *Relational Inconsistencies*.
* **Audit Report Export:** Downloadable JSON verification reports for auditability.

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
│  ├── Curated Knowledge Base                            │
│  ├── User-Uploaded Files (.pdf, .txt, .md, .csv)       │
│  └── Real-Time Wikipedia Knowledge Search              │
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
│  ├── Inline Color-Coded Text Markup                    │
│  ├── Benchmark Suite & Confusion Matrix                │
│  └── Downloadable JSON Audit Reports                   │
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

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sushant-1212/ai-hallucination-detector.git
cd ai-hallucination-detector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```
Set your API key in `.env`:
```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```
Access the web dashboard at `http://localhost:8501`.

---

## 🧪 Testing

Run the automated integration test suite:
```bash
python test_pipeline.py
```

---

## 📄 License
MIT License
