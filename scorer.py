"""
scorer.py
----------
Turns a list of per-claim verdicts into an overall "reliability score" and
a risk band. This is explicitly a heuristic, not a calibrated probability -
say so in the demo/viva so nobody mistakes it for something more rigorous.

Scoring rule:
    SUPPORTED               -> 1.0 point
    INSUFFICIENT_EVIDENCE   -> 0.5 point (neutral - we don't punish the AI
                                for claims our knowledge base can't check)
    CONTRADICTED            -> 0.0 point

reliability_score = (sum of points / number of claims) * 100
"""

from dataclasses import dataclass

POINTS = {
    "SUPPORTED": 1.0,
    "INSUFFICIENT_EVIDENCE": 0.5,
    "CONTRADICTED": 0.0,
}


@dataclass
class ScoreSummary:
    reliability_score: float
    risk_label: str
    risk_emoji: str
    supported_count: int
    contradicted_count: int
    insufficient_count: int
    total_claims: int


def _risk_band(score: float) -> tuple[str, str]:
    if score >= 80:
        return "Low Risk", "🟢"
    if score >= 60:
        return "Medium Risk", "🟡"
    return "High Risk", "🔴"


def compute_score(claim_results: list) -> ScoreSummary:
    total = len(claim_results)
    if total == 0:
        return ScoreSummary(0.0, "No Claims Found", "⚪", 0, 0, 0, 0)

    supported = sum(1 for r in claim_results if r.verdict == "SUPPORTED")
    contradicted = sum(1 for r in claim_results if r.verdict == "CONTRADICTED")
    insufficient = sum(1 for r in claim_results if r.verdict == "INSUFFICIENT_EVIDENCE")

    points = sum(POINTS[r.verdict] for r in claim_results)
    score = round((points / total) * 100, 1)
    label, emoji = _risk_band(score)

    return ScoreSummary(
        reliability_score=score,
        risk_label=label,
        risk_emoji=emoji,
        supported_count=supported,
        contradicted_count=contradicted,
        insufficient_count=insufficient,
        total_claims=total,
    )
