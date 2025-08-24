from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from ..core.types import NodeState
from ..core.registry import SETTLEMENT

@dataclass
class SettlementParams:
    warmup_rounds: int = 1

@SETTLEMENT.register("plans_engine")
class SettlementEnginePlans:
    def __init__(self, params: SettlementParams | None = None, **kwargs) -> None:
        self.p = params or SettlementParams(**kwargs) if kwargs else (params or SettlementParams())

    def run(self, round_idx: int, nodes: Dict[int, NodeState], contributions: Dict[int, float], features: Dict[int, Dict[str, float]],
            pre_rewards: Dict[int, float], detector, reward_policy, penalty_policy, reputation_policy) -> Dict[str, Any]:
        """Execute settlement planning with optional detection.

        Attempts to use the detector's ``model_sift`` method when available. If the
        method is missing or raises ``ModuleNotFoundError`` (e.g. optional
        dependencies like ``torch`` absent), it falls back to the legacy
        ``detect`` API and finally to an empty dict.
        """
        detected: Dict[int, bool] = {}
        if hasattr(detector, "model_sift"):
            try:  # prefer new API
                res = detector.model_sift(round_idx, features, contributions, [], [])
                if isinstance(res, dict):
                    detected = res
            except ModuleNotFoundError:
                pass  # try legacy API below
        if not detected and hasattr(detector, "detect"):
            try:
                res = detector.detect(features, contributions)
                if isinstance(res, dict):
                    detected = res
            except Exception:
                detected = {}

        plans: Dict[str, Any] = {
            "apply_penalties": {},
            "credit_rewards": {},
            "set_reputations": {},
            "note_participation": set(),
            "append_contrib": {},
            "detected": detected,
        }

        for nid, node in nodes.items():
            if nid in contributions:
                plans["note_participation"].add(nid)
            last = float(contributions.get(nid, 0.0))
            plans["append_contrib"][nid] = last

            if round_idx < self.p.warmup_rounds:
                r = float(pre_rewards.get(nid, 0.0))
                plans["credit_rewards"][nid] = r
                new_rep = reputation_policy.update(node, contribution=max(0.0, last), current_round=round_idx)
                plans["set_reputations"][nid] = new_rep
                continue

            if detected.get(nid, False):
                plans["apply_penalties"][nid] = {
                    "stake_mul": (1.0 - getattr(penalty_policy.p, "stake_penalty_factor", 0.02)),
                    "rep_mul": (1.0 - getattr(penalty_policy.p, "rep_penalty_factor", 0.5)),
                }
                plans["credit_rewards"][nid] = 0.0
            else:
                r = float(pre_rewards.get(nid, 0.0))
                plans["credit_rewards"][nid] = r
                new_rep = reputation_policy.update(node, contribution=max(0.0, last), current_round=round_idx)
                plans["set_reputations"][nid] = new_rep

        plans["computed_rewards_next"] = {nid: reward_policy.compute(nodes[nid], nodes) for nid in nodes}
        return plans
