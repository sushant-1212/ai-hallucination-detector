"""
scorer.py
----------
Aggregates claim-level evaluation results into holistic reliability metrics,
hallucination rates, risk classifications, and taxonomy distributions.

Mathematical formulation:
  - Factual Weight: w_supported = 1.0, w_insufficient = 0.5, w_contradicted = 0.0
  - Reliability Score = (sum(w_i * confidence_i) / sum(confidence_i)) * 100
  - Hallucination Rate = (Contradicted Claims / Total Claims) * 100
"""

from dataclasses import dataclass, field
from evaluator import ClaimResult

VERDICT_WEIGHTS = {
    "SUPPORTED": 1.0,
    "INSUFFICIENT_EVIDENCE": 0.4,
    "CONTRADICTED": 0.0,
}


@dataclass
class ScoreSummary:
    reliability_score: float
    hallucination_rate: float
    risk_label: str
    risk_emoji: str
    risk_color: str
    supported_count: int
    contradicted_count: int
    insufficient_count: int
    total_claims: int
    avg_confidence: float
    avg_similarity: float
    hallucination_types: dict[str, int] = field(default_factory=dict)


def compute_score(results: list[ClaimResult]) -> ScoreSummary:
    """Compute aggregate reliability metrics and risk profile."""
    total = len(results)
    if total == 0:
        return ScoreSummary(
            reliability_score=0.0,
            hallucination_rate=0.0,
            risk_label="No Claims Detected",
            risk_emoji="⚪",
            risk_color="#94a3b8",
            supported_count=0,
            contradicted_count=0,
            insufficient_count=0,
            total_claims=0,
            avg_confidence=0.0,
            avg_similarity=0.0,
            hallucination_types={},
        )

    supported = sum(1 for r in results if r.verdict == "SUPPORTED")
    contradicted = sum(1 for r in results if r.verdict == "CONTRADICTED")
    insufficient = sum(1 for r in results if r.verdict == "INSUFFICIENT_EVIDENCE")

    # Weighted confidence scoring
    total_weight_points = 0.0
    total_conf = 0.0
    sim_scores = []
    types_count = {}

    for r in results:
        w = VERDICT_WEIGHTS.get(r.verdict, 0.4)
        c = max(r.confidence, 10) / 100.0
        total_weight_points += w * c
        total_conf += c
        sim_scores.append(r.max_similarity)

        if r.verdict == "CONTRADICTED" and r.hallucination_type != "None":
            types_count[r.hallucination_type] = types_count.get(r.hallucination_type, 0) + 1

    reliability = (total_weight_points / total_conf * 100.0) if total_conf > 0 else 0.0
    reliability = round(min(100.0, max(0.0, reliability)), 1)

    hallucination_rate = round((contradicted / total) * 100.0, 1)
    avg_conf = round(sum(r.confidence for r in results) / total, 1)
    avg_sim = round(sum(sim_scores) / total, 2) if sim_scores else 0.0

    # Risk categorization
    if contradicted == 0 and reliability >= 80:
        risk_label = "Verified Reliable"
        risk_emoji = "🟢"
        risk_color = "#10b981"
    elif contradicted > 0 and hallucination_rate >= 40:
        risk_label = "Severe Hallucination Risk"
        risk_emoji = "🔴"
        risk_color = "#ef4444"
    elif contradicted > 0 or reliability < 65:
        risk_label = "Moderate Hallucination Risk"
        risk_emoji = "🟠"
        risk_color = "#f59e0b"
    else:
        risk_label = "Uncertain / Unverified"
        risk_emoji = "🟡"
        risk_color = "#eab308"

    return ScoreSummary(
        reliability_score=reliability,
        hallucination_rate=hallucination_rate,
        risk_label=risk_label,
        risk_emoji=risk_emoji,
        risk_color=risk_color,
        supported_count=supported,
        contradicted_count=contradicted,
        insufficient_count=insufficient,
        total_claims=total,
        avg_confidence=avg_conf,
        avg_similarity=avg_sim,
        hallucination_types=types_count,
    )
