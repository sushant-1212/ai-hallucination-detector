"""
benchmark.py
-------------
Academic ML Benchmark Suite for AI Hallucination Detector.
Computes formal statistical and classification metrics:
  - Confusion Matrix (TP, FP, TN, FN)
  - Accuracy, Precision, Recall, F1-Score
  - Hallucination Detection Specificity
  - Mean Inference Latency
"""

import time
from dataclasses import dataclass
from claim_extractor import ExtractedClaim
from retriever import retrieve_evidence, VectorIndex
from evaluator import evaluate_claim

# Standard benchmark evaluation dataset
# Ground Truth Labels:
# "HALLUCINATION": Statement is factually incorrect
# "FACTUAL": Statement is factually true
BENCHMARK_DATASET = [
    {
        "id": 1,
        "statement": "Python was created by Guido van Rossum and first released in 1991.",
        "ground_truth": "FACTUAL",
        "category": "Programming History"
    },
    {
        "id": 2,
        "statement": "Python was created by James Gosling at Bell Labs in 2005.",
        "ground_truth": "HALLUCINATION",
        "category": "Entity / Date Hallucination"
    },
    {
        "id": 3,
        "statement": "The C programming language was developed by Dennis Ritchie at Bell Labs.",
        "ground_truth": "FACTUAL",
        "category": "Computer Systems"
    },
    {
        "id": 4,
        "statement": "C is an interpreted, garbage-collected language invented in 2012 by Google.",
        "ground_truth": "HALLUCINATION",
        "category": "Paradigm Fabrication"
    },
    {
        "id": 5,
        "statement": "Java was developed by James Gosling at Sun Microsystems and released in 1995.",
        "ground_truth": "FACTUAL",
        "category": "Object-Oriented Languages"
    },
    {
        "id": 6,
        "statement": "Java compiles directly to x86 machine code and does not use any virtual machine.",
        "ground_truth": "HALLUCINATION",
        "category": "Technical Hallucination"
    },
    {
        "id": 7,
        "statement": "An algorithm with O(1) time complexity runs in constant time regardless of input size.",
        "ground_truth": "FACTUAL",
        "category": "Algorithms & Complexity"
    },
    {
        "id": 8,
        "statement": "Binary search has a worst-case time complexity of O(N^2) on sorted arrays.",
        "ground_truth": "HALLUCINATION",
        "category": "Complexity Hallucination"
    }
]


@dataclass
class BenchmarkMetrics:
    total_samples: int
    true_positives: int    # Correctly identified Hallucinations
    false_positives: int   # Factual statements falsely flagged as Hallucination
    true_negatives: int    # Factual statements correctly verified
    false_negatives: int   # Hallucinations missed (flagged as Supported or Insufficient)
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_latency_sec: float
    detailed_results: list[dict]


def evaluate_benchmark(vector_store: VectorIndex, progress_callback=None) -> BenchmarkMetrics:
    """Run benchmark suite across standard test cases and compute evaluation metrics."""
    tp = 0  # Actual Hallucination, Predicted Hallucination (Contradicted)
    fp = 0  # Actual Factual, Predicted Hallucination
    tn = 0  # Actual Factual, Predicted Factual (Supported)
    fn = 0  # Actual Hallucination, Predicted Factual or Insufficient

    details = []
    latencies = []
    total = len(BENCHMARK_DATASET)

    for i, item in enumerate(BENCHMARK_DATASET):
        t0 = time.time()
        claim_obj = ExtractedClaim(claim=item["statement"], original_span=item["statement"])
        # Retrieve evidence from vector store + web fallback if needed
        evidence = retrieve_evidence(vector_store, claim_obj.claim, k=3, include_web_search=True)
        result = evaluate_claim(claim_obj, evidence)
        elapsed = time.time() - t0
        latencies.append(elapsed)

        actual = item["ground_truth"]
        predicted = "HALLUCINATION" if result.verdict == "CONTRADICTED" else "FACTUAL"

        if actual == "HALLUCINATION" and predicted == "HALLUCINATION":
            tp += 1
            classification = "True Positive (TP)"
        elif actual == "FACTUAL" and predicted == "HALLUCINATION":
            fp += 1
            classification = "False Positive (FP)"
        elif actual == "FACTUAL" and predicted == "FACTUAL":
            tn += 1
            classification = "True Negative (TN)"
        else:  # actual == "HALLUCINATION" and predicted == "FACTUAL"
            fn += 1
            classification = "False Negative (FN)"

        details.append({
            "id": item["id"],
            "statement": item["statement"],
            "category": item["category"],
            "ground_truth": actual,
            "verdict": result.verdict,
            "predicted": predicted,
            "classification": classification,
            "confidence": result.confidence,
            "latency": round(elapsed, 2)
        })

        if progress_callback:
            progress_callback((i + 1) / total, f"Evaluating benchmark sample {i + 1}/{total}...")

    # Calculate statistical metrics
    accuracy = round(((tp + tn) / total) * 100, 1) if total > 0 else 0.0
    precision = round((tp / (tp + fp)) * 100, 1) if (tp + fp) > 0 else 0.0
    recall = round((tp / (tp + fn)) * 100, 1) if (tp + fn) > 0 else 0.0
    if (precision + recall) > 0:
        f1 = round((2 * precision * recall) / (precision + recall), 1)
    else:
        f1 = 0.0
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    return BenchmarkMetrics(
        total_samples=total,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        avg_latency_sec=avg_latency,
        detailed_results=details
    )
