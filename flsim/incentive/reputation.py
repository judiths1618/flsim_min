from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..core.registry import REPUTATION
from ..core.types import NodeState

@dataclass
class ReputationParams:
    X_c: float = 10.0
    X_s: float = 5.0
    rep_cap_early: float = 300.0
    rep_cap_late: float = 500.0
    rep_cap_round: int = 50
    contrib_base: float = 0.0
    contrib_thre: float = 1.0

@REPUTATION.register("default")
class DefaultReputation:
    def __init__(self, params: ReputationParams | None = None, **kwargs) -> None:
        self.p = params or ReputationParams(**kwargs) if kwargs else (params or ReputationParams())

    def update(self, node: NodeState, contribution: float, *, current_round: int) -> float:
        rep = float(node.reputation)
        hist = node.contrib_history
        participation = int(node.participation)
        age_factor = 1.0 - 1.0 / (1.0 + (participation / 100.0))
        delta = 0.88 + 0.07 * age_factor

        denom = max(1e-8, (self.p.contrib_thre - self.p.contrib_base))
        contrib_quality = 1.0 / (1.0 + np.exp(-((float(contribution) - self.p.contrib_base)/denom)))

        if len(hist) >= 5:
            stability = 1.0 - (float(np.std(hist[-5:], dtype=float)) / 5.0)
        else:
            stability = 0.8

        new_rep = (rep * delta) + (contrib_quality * self.p.X_c) + (stability * self.p.X_s)

        rep_cap = float(self.p.rep_cap_late if current_round > self.p.rep_cap_round else self.p.rep_cap_early)
        new_rep = float(np.clip(new_rep, 0.0, rep_cap))
        print(f"node [{node}]'s new rep: {new_rep}")
        return new_rep


        """
        Robust reputation update system
        
        Parameters:
            node (Node): The node whose reputation is being updated.
            contribution (float): The contribution value from the node.
            
        Returns:
            new_rep (float): The updated reputation value for the node
        """
        # Dynamic decay
        age_factor = 1 - 1/(1 + node.participation/100)
        delta = 0.88 + 0.07 * age_factor
        
        # Contribution quality
        contrib_base = 0
        contrib_thre = 10
        contrib_quality = sigmoid((contribution-contrib_base)/(contrib_thre - contrib_base))
        
        # Stability evaluation
        if len(node.contrib_history) >= 5:
            stability = 1 - np.std(node.contrib_history[-5:])/5
        else:
            stability = 0.8
        
        # New reputation calculation 
        new_rep = (node.reputation * delta + 
                   contrib_quality * self.X_c + 
                   stability * self.X_s)
        
        # Dynamic cap
        rep_cap = 500 if self.current_round > 50 else 300
        return np.clip(new_rep, 0, rep_cap)