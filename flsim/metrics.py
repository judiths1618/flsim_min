from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, List, Any

@dataclass
class DetectionMetrics:
    rounds: List[int] = field(default_factory=list)
    detected: List[Set[int]] = field(default_factory=list)
    truth: List[Set[int]] = field(default_factory=list)

    def log(self, round_idx: int, detected_ids: Set[int], true_malicious: Set[int]):
        self.rounds.append(int(round_idx))
        self.detected.append(set(map(int, detected_ids)))
        self.truth.append(set(map(int, true_malicious)))

    def summary(self):
        out = []
        for r, det, tru in zip(self.rounds, self.detected, self.truth):
            tp = len(det & tru)
            fp = len(det - tru)
            fn = len(tru - det)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            out.append({
                "round": r,
                "detected": sorted(det),
                "true_malicious": sorted(tru),
                "precision": precision,
                "recall": recall,
                "tp": tp, "fp": fp, "fn": fn
            })
        return out


def compute_round_statistics(results: List[Dict[str, Any]], detection_summary: List[Dict[str, Any]]):
    """Collect per-round statistics for accuracy, detection rate and balances.

    Parameters
    ----------
    results: List[Dict[str, Any]]
        Per-round outputs returned from the contract (each containing at least
        ``round`` and ``balances`` and optionally ``metrics`` with an ``acc``).
    detection_summary: List[Dict[str, Any]]
        Output from :meth:`DetectionMetrics.summary` describing detection
        performance each round.

    Returns
    -------
    Dict[str, List]
        Dictionary with three keys capturing the requested statistics:

        ``accuracy_per_round``
            List of ``(round, acc)`` tuples if accuracy data is present.
        ``detection_rate_per_round``
            List of ``(round, recall)`` tuples where recall acts as the
            malicious detection rate.
        ``balances_per_round``
            List of ``(round, balances_dict)`` for the balances among nodes.
    """

    acc_curve: List[tuple[int, float]] = []
    detection_rates: List[tuple[int, float]] = []
    balances_over_time: List[tuple[int, Dict[int, float]]] = []

    det_by_round = {d.get("round"): d for d in detection_summary}

    for res in results:
        r = int(res.get("round", 0))

        metrics = res.get("metrics") or {}
        acc = metrics.get("acc")
        if acc is not None:
            acc_curve.append((r, float(acc)))

        det = det_by_round.get(r)
        if det is not None:
            detection_rates.append((r, float(det.get("recall", 0.0))))

        bal = {int(k): float(v) for k, v in (res.get("balances") or {}).items()}
        balances_over_time.append((r, bal))
        

    return {
        "accuracy_per_round": acc_curve,
        "detection_rate_per_round": detection_rates,
        "balances_per_round": balances_over_time,
    }
