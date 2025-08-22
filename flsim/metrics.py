from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, List

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
