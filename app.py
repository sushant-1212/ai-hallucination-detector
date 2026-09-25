"""
app.py
-------
Upgraded Streamlit Interface for AI Hallucination Detector.

Features:
  - Tab 1: Live Interactive Hallucination Detector with Inline Visual Highlighting
  - Tab 2: Academic ML Evaluation & Confusion Matrix Benchmark Suite
  - Tab 3: System Architecture & Faculty Viva Defense Guide
"""

import os
import json
import tempfile
import streamlit as st
import pandas as pd

from claim_extractor import extract_claims, ExtractedClaim
from retriever import (
    build_vector_store,
    load_documents_from_folder,
    retrieve_evidence,
    VectorIndex,
    Document
)
from evaluator import evaluate_claim, ClaimResult
from scorer import compute_score, ScoreSummary
from benchmark import evaluate_benchmark
import llm_provider

# Page Configuration
st.set_page_config(
    page_title="AI Hallucination Detector | Trustworthy AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .highlight-box {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 18px;
        line-height: 1.8;
        font-size: 16px;
    }
    .badge-supported {
        background-color: #dcfce7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        border-bottom: 2px solid #22c55e;
    }
    .badge-contradicted {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        border-bottom: 2px solid #ef4444;
    }
    .badge-insufficient {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        border-bottom: 2px solid #eab308;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_curated_kb() -> VectorIndex | None:
    """Load and index the curated reference documents once."""
    docs = load_documents_from_folder("knowledge_base")
    return build_vector_store(docs)


def build_uploaded_kb(uploaded_files) -> VectorIndex | None:
    """Index user-uploaded documents on demand."""
    if not uploaded_files:
        return None
    docs = []
    for f in uploaded_files:
        name = f.name
        content = f.read()
        if name.lower().endswith((".txt", ".md", ".csv")):
            text = content.decode("utf-8", errors="ignore")
            docs.append(Document(text=text, source=f"Upload: {name}"))
        elif name.lower().endswith(".pdf"):
            try:
                import pypdf
                import io
                reader = pypdf.PdfReader(io.BytesIO(content))
                text = "".join(page.extract_text() or "" for page in reader.pages)
                docs.append(Document(text=text, source=f"Upload: {name}"))
            except Exception as e:
                st.sidebar.error(f"Error parsing {name}: {e}")
    return build_vector_store(docs)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("Settings & Controls")
    
    provider_info = llm_provider.provider_name()
    st.success(f"**Engine:** {provider_info}")

    st.subheader("📚 Knowledge Base Sources")
    use_curated = st.checkbox("Curated CS Documents", value=True, help="Includes bundled references on Python, Java, C, and Algorithms")
    use_web = st.checkbox("🌐 Live Wikipedia Search", value=True, help="Enables real-time fact retrieval for any general knowledge topic")
    use_uploads = st.checkbox("Custom Document Uploads", value=False)

    uploaded_files = None
    if use_uploads:
        uploaded_files = st.file_uploader(
            "Upload files (.txt, .md, .pdf, .csv)",
            type=["txt", "md", "pdf", "csv"],
            accept_multiple_files=True
        )

    st.divider()
    top_k = st.slider("Evidence Chunks per Claim", min_value=1, max_value=5, value=3)

    st.divider()
    st.markdown("### 🧪 Quick Demo Presets")
    preset_choice = st.selectbox(
        "Load test scenario:",
        [
            "-- Select a scenario --",
            "Example 1: Subtle Hallucination (Java & C)",
            "Example 2: Pure Factual Truth (Python)",
            "Example 3: Severe Hallucinations (Algorithms & Tech)",
            "Example 4: General Knowledge (Wikipedia test)"
        ]
    )


# Preset text mapping
DEMO_PRESETS = {
    "Example 1: Subtle Hallucination (Java & C)": (
        "Java was developed by Dennis Ritchie at Bell Labs in 1995. "
        "It compiles source code into platform-independent bytecode executed by the JVM. "
        "Dennis Ritchie also created the C programming language to develop the Unix operating system."
    ),
    "Example 2: Pure Factual Truth (Python)": (
        "Python was created by Guido van Rossum and first released in 1991. "
        "It emphasizes code readability with significant indentation. "
        "Python is dynamically typed and supports automatic garbage collection."
    ),
    "Example 3: Severe Hallucinations (Algorithms & Tech)": (
        "Binary search operates with a worst-case time complexity of O(N^2) on unsorted arrays. "
        "The C language has built-in automatic garbage collection and does not support raw pointers. "
        "James Gosling invented Python at Microsoft in 2018."
    ),
    "Example 4: General Knowledge (Wikipedia test)": (
        "The Eiffel Tower is located in Paris, France and was designed by Gustave Eiffel. "
        "It was completed in 1889 as the entrance arch for the World's Fair."
    )
}

# ---------------------------------------------------------------------------
# MAIN INTERFACE TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Live Hallucination Detector",
    "📊 Academic ML Evaluation & Benchmark",
    "🎓 Viva & Architecture Defense Guide"
])

# ===========================================================================
# TAB 1: LIVE DETECTOR
# ===========================================================================
with tab1:
    st.title("🛡️ AI Hallucination & Factuality Detector")
    st.markdown(
        "A retrieval-augmented factuality verification engine that decomposes AI-generated text into "
        "atomic factual claims and rigorously checks them against verified multi-source knowledge bases."
    )

    default_text = DEMO_PRESETS.get(preset_choice, "") if preset_choice != "-- Select a scenario --" else ""

    user_text = st.text_area(
        "Enter AI-Generated Response to Analyze:",
        value=default_text,
        height=150,
        placeholder="Paste text generated by ChatGPT, Claude, Gemini, or any LLM here..."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_analysis = st.button("🚀 Analyze Hallucinations", type="primary", use_container_width=True)

    if run_analysis:
        if not user_text.strip():
            st.warning("Please enter or select a text prompt to analyze.")
            st.stop()

        # Step A: Build Knowledge Base
        with st.spinner("Indexing knowledge bases..."):
            combined_kb = VectorIndex()
            if use_curated:
                curated_kb = load_curated_kb()
                if curated_kb:
                    combined_kb.merge_from(curated_kb)
            if use_uploads and uploaded_files:
                uploaded_kb = build_uploaded_kb(uploaded_files)
                if uploaded_kb:
                    combined_kb.merge_from(uploaded_kb)

        # Step B: Claim Extraction
        with st.spinner("Extracting atomic factual claims via NLP parser..."):
            extracted_claims = extract_claims(user_text)

        if not extracted_claims:
            st.info("No checkable factual claims found in the text.")
            st.stop()

        # Step C: Verification Pipeline
        results: list[ClaimResult] = []
        progress_bar = st.progress(0.0, text="Verifying claims against knowledge sources...")

        for idx, clm in enumerate(extracted_claims):
            # Retrieve
            evidence_chunks = retrieve_evidence(
                combined_kb,
                clm.claim,
                k=top_k,
                include_web_search=use_web
            )
            # Evaluate
            eval_res = evaluate_claim(clm, evidence_chunks)
            results.append(eval_res)
            progress_bar.progress((idx + 1) / len(extracted_claims), text=f"Checked {idx + 1}/{len(extracted_claims)} claims...")

        progress_bar.empty()

        # Step D: Scoring
        summary: ScoreSummary = compute_score(results)

        st.markdown("---")
        st.subheader("📊 Verification Overview")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Factual Reliability", f"{summary.reliability_score}%")
        with m2:
            st.metric("Hallucination Rate", f"{summary.hallucination_rate}%")
        with m3:
            st.metric("Risk Level", f"{summary.risk_emoji} {summary.risk_label}")
        with m4:
            st.metric("Claims Evaluated", f"{summary.total_claims} ({summary.supported_count} OK / {summary.contradicted_count} Bad)")

        # Step E: Interactive Inline Text Highlighting
        st.markdown("### 🎨 Visual Hallucination Highlighter")
        st.caption("Sentences color-coded based on claim verification against retrieved evidence:")

        # Build highlighted text
        highlighted_html = user_text
        for res in results:
            span = res.original_span.strip()
            if span and span in highlighted_html:
                if res.verdict == "SUPPORTED":
                    tag = f'<span class="badge-supported" title="Supported (Confidence: {res.confidence}%)">{span}</span>'
                elif res.verdict == "CONTRADICTED":
                    tag = f'<span class="badge-contradicted" title="Hallucination ({res.hallucination_type}): {res.explanation}">{span}</span>'
                else:
                    tag = f'<span class="badge-insufficient" title="Unverified: Insufficient evidence">{span}</span>'
                # Replace once to preserve ordering
                highlighted_html = highlighted_html.replace(span, tag, 1)

        st.markdown(f'<div class="highlight-box">{highlighted_html}</div>', unsafe_allow_html=True)

        col_l1, col_l2, col_l3 = st.columns(3)
        col_l1.markdown("🟢 **Green**: Verified by Evidence")
        col_l2.markdown("🔴 **Red**: Contradicted / Hallucination")
        col_l3.markdown("🟡 **Yellow**: Insufficient Evidence")

        # Step F: Detailed Claim Breakdown
        st.markdown("---")
        st.subheader("📑 Claim-by-Claim Verification Breakdown")

        for idx, res in enumerate(results, 1):
            if res.verdict == "SUPPORTED":
                icon, label, color = "✅", "SUPPORTED", "#166534"
            elif res.verdict == "CONTRADICTED":
                icon, label, color = "❌", f"HALLUCINATION ({res.hallucination_type})", "#991b1b"
            else:
                icon, label, color = "⚠️", "INSUFFICIENT EVIDENCE", "#854d0e"

            with st.container(border=True):
                st.markdown(f"**{icon} Claim {idx}:** `{res.claim}`")
                
                c_a, c_b, c_c = st.columns([2, 1, 1])
                c_a.markdown(f"**Verdict:** <span style='color:{color}; font-weight:bold;'>{label}</span>", unsafe_allow_html=True)
                c_b.markdown(f"**NLI Confidence:** `{res.confidence}%`")
                c_c.markdown(f"**Max Cosine Sim:** `{res.max_similarity:.2f}`")

                st.markdown(f"**Reasoning:** *{res.explanation}*")

                if res.evidence:
                    with st.expander(f"🔍 Inspect {len(res.evidence)} Retrieved Evidence Passages"):
                        for e_idx, ev in enumerate(res.evidence, 1):
                            st.markdown(f"**Passage [{e_idx}]** — Source: `{ev.source}` (Similarity: `{ev.score:.3f}`)")
                            st.info(ev.text)
                else:
                    st.caption("No relevant passages retrieved from knowledge store.")

        # Step G: Export Report
        st.markdown("---")
        report_data = {
            "summary": {
                "reliability_score": summary.reliability_score,
                "hallucination_rate": summary.hallucination_rate,
                "risk_label": summary.risk_label,
                "total_claims": summary.total_claims,
            },
            "claims": [
                {
                    "claim": r.claim,
                    "verdict": r.verdict,
                    "confidence": r.confidence,
                    "hallucination_type": r.hallucination_type,
                    "explanation": r.explanation,
                    "sources": [e.source for e in r.evidence]
                }
                for r in results
            ]
        }
        json_report = json.dumps(report_data, indent=2)
        st.download_button(
            label="📥 Download JSON Verification Report",
            data=json_report,
            file_name="hallucination_audit_report.json",
            mime="application/json"
        )


# ===========================================================================
# TAB 2: ACADEMIC ML BENCHMARK & EVALUATION
# ===========================================================================
with tab2:
    st.header("📊 Academic ML Benchmark & Performance Evaluation")
    st.markdown(
        "Demonstrate rigorous scientific validation of the detector. "
        "Runs an automated test battery on standardized ground-truth benchmark pairs and computes "
        "a **Confusion Matrix, Accuracy, Precision, Recall, and F1-Score**."
    )

    col_bench_1, col_bench_2 = st.columns([1, 3])
    with col_bench_1:
        run_bench = st.button("⚡ Run Benchmark Battery", type="primary")

    if run_bench:
        curated_kb = load_curated_kb()
        if not curated_kb:
            curated_kb = VectorIndex()

        bench_progress = st.progress(0.0, text="Initializing benchmark suite...")
        metrics = evaluate_benchmark(curated_kb, progress_callback=lambda p, t: bench_progress.progress(p, text=t))
        bench_progress.empty()

        st.success(f"Benchmark completed across {metrics.total_samples} ground-truth test cases!")

        # Metrics cards
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Model Accuracy", f"{metrics.accuracy}%")
        b2.metric("Precision", f"{metrics.precision}%", help="TP / (TP + FP) - Ability to avoid false alarms")
        b3.metric("Recall / Sensitivity", f"{metrics.recall}%", help="TP / (TP + FN) - Ability to catch all hallucinations")
        b4.metric("F1-Score", f"{metrics.f1_score}%", help="Harmonic mean of precision and recall")

        # Confusion Matrix Table
        st.subheader("🎯 Confusion Matrix")
        cm_df = pd.DataFrame(
            [
                [metrics.true_positives, metrics.fn_ignored if hasattr(metrics, 'fn_ignored') else metrics.false_negatives],
                [metrics.false_positives, metrics.true_negatives]
            ],
            columns=["Predicted Hallucination", "Predicted Factual"],
            index=["Actual Hallucination", "Actual Factual"]
        )
        st.table(cm_df)

        st.caption(f"⏱️ **Mean Verification Latency:** {metrics.avg_latency_sec:.2f} seconds per claim")

        # Detailed breakdown table
        st.subheader("📋 Detailed Sample Classification Results")
        table_data = []
        for r in metrics.detailed_results:
            table_data.append({
                "ID": r["id"],
                "Statement": r["statement"],
                "Ground Truth": r["ground_truth"],
                "Model Verdict": r["verdict"],
                "Outcome": r["classification"],
                "Confidence": f"{r['confidence']}%",
                "Latency (s)": r["latency"]
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)


# ===========================================================================
# TAB 3: VIVA & ARCHITECTURE DEFENSE GUIDE
# ===========================================================================
with tab3:
    st.header("🎓 Faculty Viva Defense & Technical Architecture")
    st.markdown("Use this tab during your presentation to answer theoretical and algorithmic questions with confidence.")

    st.subheader("1. Pipeline Architecture")
    st.code("""
┌────────────────────────┐
│  AI-Generated Text     │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  Step 1: NLP Atomic Claim Extraction                   │
│  - Decomposes compound clauses into single facts       │
│  - Maps each proposition back to original token span   │
└───────────┬────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  Step 2: Multi-Source Dense Retrieval (NumPy Cosine)   │
│  - Curated Knowledge Base (.txt, .md, .pdf)            │
│  - Real-Time Wikipedia Search Engine                   │
│  - Cosine Similarity Metric: S = (u . v) / (||u||||v||)│
└───────────┬────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  Step 3: Natural Language Inference (NLI) Judge        │
│  - Classifies: SUPPORTED / CONTRADICTED / INSUFFICIENT │
│  - Extracts Hallucination Taxonomy & Confidence Score  │
└───────────┬────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  Step 4: Metric Aggregation & Inline Highlighting      │
│  - Factual Reliability Score & Hallucination Rate      │
│  - Visual Color Coded Highlighting UI                  │
└────────────────────────────────────────────────────────┘
    """, language="text")

    st.subheader("2. Core Mathematical Formulations")
    st.markdown(r"""
    **A. Vector Cosine Similarity:**
    $$\text{Sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \frac{\sum_{i=1}^D q_i d_i}{\sqrt{\sum_{i=1}^D q_i^2} \sqrt{\sum_{i=1}^D d_i^2}}$$

    **B. Reliability Heuristic Formulation:**
    $$\text{Reliability} = \left( \frac{\sum_{i=1}^N w_i \cdot c_i}{\sum_{i=1}^N c_i} \right) \times 100$$
    Where $w_i \in \{1.0 \text{ (Supported)}, 0.4 \text{ (Insufficient)}, 0.0 \text{ (Contradicted)}\}$ and $c_i \in [0, 1]$ is the NLI confidence score.

    **C. Hallucination Rate:**
    $$\text{HR} = \left( \frac{N_{\text{Contradicted}}}{N_{\text{Total}}} \right) \times 100$$
    """)

    st.subheader("3. Top Viva Questions & Model Answers")

    with st.expander("Q1: Why decompose into atomic claims instead of checking the whole text?"):
        st.write(
            "**Answer:** A single sentence often bundles multiple facts with conflicting truth values. "
            "For instance, 'Java was created by Dennis Ritchie in 1995' contains a true fact (1995 release) "
            "and a hallucinated fact (Dennis Ritchie). Evaluating the whole paragraph produces a blended verdict, "
            "whereas atomic claim decomposition pinpoints precisely which entity, number, or relation is false."
        )

    with st.expander("Q2: What is the purpose of the 'INSUFFICIENT_EVIDENCE' category?"):
        st.write(
            "**Answer:** In epistemic AI safety, Absence of Evidence is NOT Evidence of Absence. "
            "If a knowledge base does not contain information about a statement, labeling it as false causes "
            "severe False Positives. Explicitly modeling epistemic uncertainty ensures high precision."
        )

    with st.expander("Q3: How does your vector search scale and avoid external database hosting?"):
        st.write(
            "**Answer:** We implemented a vectorized NumPy in-memory cosine similarity engine that normalizes "
            "3072-dimensional dense embeddings to unit spheres ($L_2$ norm), transforming similarity computation into "
            "a single optimized BLAS matrix-vector product $\\mathbf{S} = \\mathbf{M} \\mathbf{q}$."
        )

    with st.expander("Q4: How does your prototype address real-time web retrieval?"):
        st.write(
            "**Answer:** In addition to local curated documents and user PDFs, we integrated a zero-overhead "
            "live Wikipedia knowledge retriever. When testing arbitrary general knowledge queries, it performs "
            "real-time query extraction and injects factual reference contexts dynamically into the vector space."
        )
