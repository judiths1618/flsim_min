from __future__ import annotations
from dataclasses import dataclass
from ..core.registry import PENALTY

@dataclass
class PenaltyParams:
    stake_penalty_factor: float = 0.02
    rep_penalty_factor: float = 0.5

@PENALTY.register("ours")
class DefaultPenalty:
    def __init__(self, params: PenaltyParams | None = None, **kwargs) -> None:
        self.p = params or PenaltyParams(**kwargs) if kwargs else (params or PenaltyParams())


@PENALTY.register("none")
class NoOpPenalty:
    """Penalty strategy that leaves stake and reputation untouched."""

    def __init__(self, params: PenaltyParams | None = None, **kwargs) -> None:  # pragma: no cover
        # Zero multipliers so that ``apply_penalty`` in the contract has no effect
        self.p = PenaltyParams(stake_penalty_factor=0.0, rep_penalty_factor=0.0)
