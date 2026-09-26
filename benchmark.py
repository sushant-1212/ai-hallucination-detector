"""
benchmark.py
-------------
Academic ML Benchmark Suite for AI Hallucination Detector.
Contains 30 standardized ground-truth test pairs covering:
  - Language Specifications & History
  - Memory Management & Execution Models
  - Data Structures & Algorithms
  - Operating Systems & Networking
  - Subtle Hallucinations & Entity Inversions

Computes formal statistical metrics:
  - Confusion Matrix (TP, FP, TN, FN)
  - Accuracy, Precision, Recall, F1-Score
  - Mean Inference Latency
"""

import time
from dataclasses import dataclass
from claim_extractor import ExtractedClaim
from retriever import retrieve_evidence, VectorIndex
from evaluator import evaluate_claim

# Standard 30-sample benchmark evaluation dataset
# Ground Truth Labels:
# "HALLUCINATION": Statement is factually incorrect
# "FACTUAL": Statement is factually true
BENCHMARK_DATASET = [
    # --- Category 1: Programming Languages & History ---
    {
        "id": 1,
        "statement": "Python was created by Guido van Rossum and first released in 1991.",
        "ground_truth": "FACTUAL",
        "category": "Language History"
    },
    {
        "id": 2,
        "statement": "Python was created by James Gosling at Bell Labs in 2005.",
        "ground_truth": "HALLUCINATION",
        "category": "Entity / Date Inversion"
    },
    {
        "id": 3,
        "statement": "The C programming language was developed by Dennis Ritchie at Bell Labs.",
        "ground_truth": "FACTUAL",
        "category": "Systems Programming"
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
        "statement": "Java compiles directly to raw x86 machine code and does not execute via any virtual machine.",
        "ground_truth": "HALLUCINATION",
        "category": "Runtime Hallucination"
    },
    {
        "id": 7,
        "statement": "JavaScript was created in 10 days by Brendan Eich in 1995 at Netscape.",
        "ground_truth": "FACTUAL",
        "category": "Web Technology"
    },
    {
        "id": 8,
        "statement": "JavaScript was developed by Microsoft as a direct derivative of the Java runtime environment.",
        "ground_truth": "HALLUCINATION",
        "category": "Origin Hallucination"
    },

    # --- Category 2: Data Structures & Complexity ---
    {
        "id": 9,
        "statement": "An algorithm with O(1) time complexity executes in constant time regardless of input size.",
        "ground_truth": "FACTUAL",
        "category": "Algorithms & Complexity"
    },
    {
        "id": 10,
        "statement": "Binary search has a worst-case time complexity of O(N^2) on sorted arrays.",
        "ground_truth": "HALLUCINATION",
        "category": "Complexity Inversion"
    },
    {
        "id": 11,
        "statement": "Hash tables provide average-case O(1) time complexity for key lookup operations.",
        "ground_truth": "FACTUAL",
        "category": "Data Structures"
    },
    {
        "id": 12,
        "statement": "A binary search tree guarantees O(1) lookup even when degenerate and completely unbalanced.",
        "ground_truth": "HALLUCINATION",
        "category": "Algorithmic Inaccuracy"
    },
    {
        "id": 13,
        "statement": "Merge sort has a guaranteed worst-case time complexity of O(N log N).",
        "ground_truth": "FACTUAL",
        "category": "Sorting Algorithms"
    },
    {
        "id": 14,
        "statement": "QuickSort has a guaranteed best, average, and worst-case time complexity of O(N).",
        "ground_truth": "HALLUCINATION",
        "category": "Complexity Fabrication"
    },

    # --- Category 3: Memory Management & Systems ---
    {
        "id": 15,
        "statement": "The C language requires manual memory allocation and deallocation using functions like malloc and free.",
        "ground_truth": "FACTUAL",
        "category": "Memory Management"
    },
    {
        "id": 16,
        "statement": "C automatically detects and resolves memory leaks at compile time without any runtime overhead.",
        "ground_truth": "HALLUCINATION",
        "category": "Technical Misconception"
    },
    {
        "id": 17,
        "statement": "Java bytecode is executed by the Java Virtual Machine (JVM).",
        "ground_truth": "FACTUAL",
        "category": "Virtual Machines"
    },
    {
        "id": 18,
        "statement": "Stack memory in C is used for dynamic memory allocation allocated via malloc.",
        "ground_truth": "HALLUCINATION",
        "category": "Stack vs Heap Inversion"
    },

    # --- Category 4: Operating Systems & Concurrency ---
    {
        "id": 19,
        "statement": "The Linux operating system kernel was initially created by Linus Torvalds in 1991.",
        "ground_truth": "FACTUAL",
        "category": "Operating Systems"
    },
    {
        "id": 20,
        "statement": "Linux is a closed-source proprietary kernel owned and licensed exclusively by Apple Inc.",
        "ground_truth": "HALLUCINATION",
        "category": "Licensing Hallucination"
    },
    {
        "id": 21,
        "statement": "Threads within the same process typically share the same virtual address space and heap memory.",
        "ground_truth": "FACTUAL",
        "category": "Concurrency"
    },
    {
        "id": 22,
        "statement": "A deadlock can occur only when all processes release all shared locks simultaneously.",
        "ground_truth": "HALLUCINATION",
        "category": "Concurrency Inversion"
    },

    # --- Category 5: Computer Networking ---
    {
        "id": 23,
        "statement": "TCP is a connection-oriented transport protocol that guarantees ordered delivery of packets.",
        "ground_truth": "FACTUAL",
        "category": "Networking"
    },
    {
        "id": 24,
        "statement": "UDP establishes a mandatory 3-way handshake before transmitting any datagrams across the network.",
        "ground_truth": "HALLUCINATION",
        "category": "Protocol Misattribution"
    },
    {
        "id": 25,
        "statement": "The Domain Name System (DNS) translates human-readable domain names into numerical IP addresses.",
        "ground_truth": "FACTUAL",
        "category": "Internet Infrastructure"
    },
    {
        "id": 26,
        "statement": "HTTP status code 404 indicates that the server successfully processed the request with no errors.",
        "ground_truth": "HALLUCINATION",
        "category": "Status Code Inversion"
    },

    # --- Category 6: Nuanced & Subtle Hallucinations ---
    {
        "id": 27,
        "statement": "Python uses significant indentation to delimit code blocks instead of curly braces.",
        "ground_truth": "FACTUAL",
        "category": "Syntax & Semantics"
    },
    {
        "id": 28,
        "statement": "Python was named after the venomous snake family Colubridae by its creator.",
        "ground_truth": "HALLUCINATION",
        "category": "Etymology Hallucination"
    },
    {
        "id": 29,
        "statement": "The Git version control system was designed by Linus Torvalds to maintain Linux kernel development.",
        "ground_truth": "FACTUAL",
        "category": "Developer Tools"
    },
    {
        "id": 30,
        "statement": "Git and GitHub are the exact same software developed concurrently by the Apache Software Foundation.",
        "ground_truth": "HALLUCINATION",
        "category": "Tool Conflation"
    }
]


@dataclass
class BenchmarkMetrics:
    total_samples: int
    true_positives: int    # Correctly identified Hallucinations
    false_positives: int   # Factual statements falsely flagged as Hallucination
    true_negatives: int    # Factual statements correctly verified
    false_negatives: int   # Hallucinations missed
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_latency_sec: float
    detailed_results: list[dict]


def evaluate_benchmark(vector_store: VectorIndex, sample_limit: int | None = None, progress_callback=None) -> BenchmarkMetrics:
    """Run benchmark suite across standardized ground-truth test cases."""
    dataset = BENCHMARK_DATASET[:sample_limit] if sample_limit else BENCHMARK_DATASET
    total = len(dataset)

    tp = 0  # Actual Hallucination, Predicted Hallucination (Contradicted)
    fp = 0  # Actual Factual, Predicted Hallucination
    tn = 0  # Actual Factual, Predicted Factual (Supported)
    fn = 0  # Actual Hallucination, Predicted Factual / Insufficient

    details = []
    latencies = []

    for i, item in enumerate(dataset):
        t0 = time.time()
        claim_obj = ExtractedClaim(claim=item["statement"], original_span=item["statement"])
        
        # Retrieve evidence from vector store + web search for open knowledge
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
            progress_callback((i + 1) / total, f"Evaluating benchmark sample {i + 1}/{total} ({item['category']})...")
        time.sleep(1.0)

    # Statistical Metrics computation
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
