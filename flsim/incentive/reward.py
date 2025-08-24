from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from ..core.types import NodeState
from ..core.registry import REWARD


def sigmoid(x: float) -> float:
    """Standard logistic function."""
    return float(1.0 / (1.0 + np.exp(-float(x))))


def jain_fairness(values: Sequence[float]) -> float:
    """Compute Jain's fairness index for a sequence of values."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    num = np.square(arr.sum())
    den = arr.size * np.square(arr).sum() + 1e-8
    return float(num / den)


@dataclass
class RewardParams:
    base_reward: float = 100.0
    hist_decay: float = 0.9
    stake_weight: float = 0.4


@REWARD.register("default")
class DefaultReward:
    def __init__(self, params: RewardParams | None = None, **kwargs) -> None:
        self.p = params or RewardParams(**kwargs) if kwargs else (params or RewardParams())

    def compute(
        self,
        node: NodeState,
        nodes: Dict[int, NodeState],
        *,
        in_committee: bool = False,
    ) -> float:
        """Calculate reward using stake, history and fairness factors."""
        if not node.contrib_history or node.contrib_history[-1] == 0:
            return 0.0

        avg_stake = (
            float(np.mean([n.stake for n in nodes.values()])) if nodes else 0.0
        )
        effective_stake = min(node.stake, 3.0 * avg_stake)

        recent_contribs = node.contrib_history[-5:]
        hist_contrib = sum(
            c * (self.p.hist_decay ** t) for t, c in enumerate(reversed(recent_contribs))
        )

        reputations = [n.reputation for n in nodes.values()]
        diversity_bonus = jain_fairness(reputations) if reputations else 0.0
        avg_rep = float(np.mean(reputations)) if reputations else 0.0

        node_rep = node.reputation
        alpha = sigmoid((avg_rep - node_rep) / 50.0) * self.p.stake_weight
        beta = 1.0 - alpha

        committee_bonus = 20.0 * diversity_bonus if in_committee else 0.0

        total_stake = sum(n.stake for n in nodes.values()) + 1e-8
        total_contrib = (
            sum(n.contrib_history[-1] if n.contrib_history else 0.0 for n in nodes.values())
            + 1e-8
        )

        reward = (
            (
                alpha * self.p.base_reward * (effective_stake / total_stake)
                + beta * self.p.base_reward * (hist_contrib / total_contrib)
            )
            * diversity_bonus
            + committee_bonus
        )
        return float(max(reward, 0.0))

