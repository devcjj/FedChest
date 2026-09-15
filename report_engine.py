"""Deterministic report generation for FedChest AI predictions."""

from collections.abc import Sequence

MODERATE_RISK = 0.30
HIGH_RISK = 0.60


def build_report(
    labels: Sequence[str], probabilities: Sequence[float], age: int, sex: str, view_position: str
) -> dict[str, str]:
    findings = [
        f"{label}: {probability:.0%} estimated likelihood."
        for label, probability in zip(labels, probabilities)
        if probability >= MODERATE_RISK
    ]
    high_risk = [
        (label, probability)
        for label, probability in zip(labels, probabilities)
        if probability >= HIGH_RISK
    ]
    high_risk.sort(key=lambda item: item[1], reverse=True)

    if high_risk:
        impression = f"Highest estimated finding: {high_risk[0][0]} ({high_risk[0][1]:.0%})."
    elif findings:
        impression = "No high-risk estimated finding; moderate-risk findings require clinical review."
    else:
        impression = "No modeled finding exceeded the moderate-risk threshold."

    return {
        "indication": f"Chest radiograph, {view_position} view; age {age}; sex {sex}.",
        "findings": " ".join(findings) or "No modeled finding exceeded the moderate-risk threshold.",
        "impression": impression,
        "disclaimer": "Decision support only. This output is not a diagnosis and must be reviewed by a qualified clinician.",
    }
