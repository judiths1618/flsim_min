from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from ..core.types import NodeState
from ..core.registry import REWARD

@dataclass
class RewardParams:
    base_reward: float = 100.0
    hist_decay: float = 0.9
    stake_weight: float = 0.4

@REWARD.register("default")
class DefaultReward:
    def __init__(self, params: RewardParams | None = None, **kwargs) -> None:
        self.p = params or RewardParams(**kwargs) if kwargs else (params or RewardParams())

    def compute(self, node: NodeState, nodes: Dict[int, NodeState]) -> float:
        hist = 0.0
        recent = node.contrib_history[-5:]
        for t, c in enumerate(reversed(recent)):
            hist += float(c) * (self.p.hist_decay ** t)
        total_contrib = sum(n.contrib_history[-1] if n.contrib_history else 0.0 for n in nodes.values()) + 1e-8
        diversity = 1.0
        return float(self.p.base_reward * ((hist / total_contrib) * (1.0 - self.p.stake_weight) + self.p.stake_weight * 1.0) * diversity)
