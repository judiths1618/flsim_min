from __future__ import annotations
from dataclasses import dataclass
from ..core.registry import PENALTY

@dataclass
class PenaltyParams:
    stake_penalty_factor: float = 0.02
    rep_penalty_factor: float = 0.5

@PENALTY.register("default")
class DefaultPenalty:
    def __init__(self, params: PenaltyParams | None = None, **kwargs) -> None:
        self.p = params or PenaltyParams(**kwargs) if kwargs else (params or PenaltyParams())
