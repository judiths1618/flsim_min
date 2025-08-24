from __future__ import annotations
from dataclasses import dataclass
from ..core.registry import CONTRIB

@dataclass
class ContributionParams:
    pass

@CONTRIB.register("metric")
class MetricContribution:
    def __init__(self, params: ContributionParams | None = None, **kwargs) -> None:
        self.p = params or ContributionParams()

    def score(self, metrics: dict) -> float:
        # print(f"Metric: {metrics}")
        return float(metrics.get("eval_acc", 0.0))
