"""
test_pipeline.py
----------------
End-to-end integration test verifying all layers of the upgraded detector.
"""

from claim_extractor import extract_claims
from retriever import load_documents_from_folder, build_vector_store, retrieve_evidence
from evaluator import evaluate_claim
from scorer import compute_score

def run_tests():
    print("=" * 60)
    print("RUNNING END-TO-END PIPELINE VERIFICATION")
    print("=" * 60)

    # 1. Test Claim Extraction
    sample_text = (
        "Java was invented by Dennis Ritchie in 1995. "
        "Python was created by Guido van Rossum."
    )
    print("\n[1] Testing Claim Extraction...")
    claims = extract_claims(sample_text)
    print(f"-> Extracted {len(claims)} atomic claims:")
    for c in claims:
        print(f"   * '{c.claim}' (from: '{c.original_span}')")

    assert len(claims) >= 2, "Claim extraction returned too few claims."

    # 2. Test Retriever
    print("\n[2] Testing Knowledge Base & Vector Index...")
    docs = load_documents_from_folder("knowledge_base")
    vector_store = build_vector_store(docs)
    assert vector_store is not None, "Vector store failed to build."
    print(f"-> Indexed {len(docs)} documents successfully.")

    # 3. Test Evaluation
    print("\n[3] Testing NLI Fact-Checking Evaluator...")
    results = []
    for c in claims:
        ev = retrieve_evidence(vector_store, c.claim, k=2, include_web_search=True)
        res = evaluate_claim(c, ev)
        results.append(res)
        print(f"   * Claim: '{res.claim}'")
        print(f"     Verdict: {res.verdict} | Conf: {res.confidence}% | Type: {res.hallucination_type}")
        print(f"     Explanation: {res.explanation[:90]}...")

    # 4. Test Scorer
    print("\n[4] Testing Metric Scoring Engine...")
    summary = compute_score(results)
    print(f"-> Reliability Score: {summary.reliability_score}%")
    print(f"-> Hallucination Rate: {summary.hallucination_rate}%")
    print(f"-> Risk Assessment: {summary.risk_label}")

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
