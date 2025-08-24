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
        
        # To refine
        """
        Calculate the dynamic hybrid reward for a node based on its effective stake, 
        time-decayed contribution history, diversity fairness, and committee membership.

        Parameters:
            node (Node): The node for which the reward is calculated.
            avg_rep (float): Average reputation across all nodes.

        Returns:
            float: The final reward allocated to the node.
        """

        # === Step 0: Participation check ===
        if not node.contrib_history or node.contrib_history[-1] == 0:
            # No contribution in this round or no history at all
            return 0.0  # or: return committee_bonus if committee_bonus is decoupled

        # === Step 1: Effective stake with anti-monopoly cap: S_eff = min(S_i, 3 * avg(S)) ===
        avg_stake = np.mean([n.stake for n in self.nodes])
        effective_stake = min(node.stake, 3 * avg_stake)

        # === Step 2: Time-decayed historical contribution: C_i^hist = sum(c_t * decay^t) ===
        recent_contribs = node.contrib_history[-5:]
        hist_contrib = sum(
            c * (self.hist_decay_factor ** t)
            for t, c in enumerate(reversed(recent_contribs))
        )

        # === Step 3: Diversity bonus using Jain's fairness index: J(r) ===
        reputations = [n.reputation for n in self.nodes]
        diversity_bonus = jain_fairness(reputations)

        # === Step 4: Dynamic alpha weight for stake vs. contribution ===
        # Adaptive to system fairness and node position
        node_rep = node.reputation
        alpha = sigmoid((avg_rep - node_rep) / 50) * self.stake_weight
        beta = 1 - alpha

        # === Step 5: Committee bonus if node participated in committee this round ===
        committee_bonus = 20 * diversity_bonus if node in self.committee_history[-1] else 0

        # === Step 6: Reward calculation based on Equation (25) ===
        total_stake = sum(n.stake for n in self.nodes) + 1e-8
        total_contrib = sum(n.contrib_history[-1] for n in self.nodes) + 1e-8

        reward = (
            (alpha * self.base_reward * (effective_stake / total_stake) +
            beta * self.base_reward * (hist_contrib / total_contrib)) * diversity_bonus
            + committee_bonus
        )

        return max(reward, 0.0)