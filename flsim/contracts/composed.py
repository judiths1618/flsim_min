from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set
import numpy as np

from ..core.types import NodeState, ModelUpdate
from ..core.registry import AGGREGATION, CONTRIB, DETECTION, REWARD, PENALTY, REPUTATION, SELECTION, SETTLEMENT
from ..aggregation.base import AggregationStrategy
from ..metrics import DetectionMetrics


@dataclass
class ContractConfig:
    committee_size: int = 5
    committee_cooldown: int = 3
    rep_exponent: float = 1.0
    detection: str = "flame"
    contribution: str = "metric"
    reward: str = "default"
    penalty: str = "default"
    reputation: str = "default"
    selection: str = "stratified_softmax"
    settlement: str = "plans_engine"
    aggregation: str = "flame_agg"
    # aggregation: str = "base_Fedavg"

class ComposedContract:
    def __init__(self, cfg: ContractConfig | None = None, strategy_params: dict | None = None):
        self.cfg = cfg or ContractConfig()
        self.detector = DETECTION.get(self.cfg.detection)(**(strategy_params or {}).get('detection', {}))
        self.contrib = CONTRIB.get(self.cfg.contribution)(**(strategy_params or {}).get('contribution', {}))
        self.reward = REWARD.get(self.cfg.reward)(**(strategy_params or {}).get('reward', {}))
        self.penalty = PENALTY.get(self.cfg.penalty)(**(strategy_params or {}).get('penalty', {}))
        self.reputation = REPUTATION.get(self.cfg.reputation)(**(strategy_params or {}).get('reputation', {}))
        sel_params = {
            "committee_size": self.cfg.committee_size,
            "rep_exponent": self.cfg.rep_exponent,
            "cooldown": self.cfg.committee_cooldown,
            **(strategy_params or {}).get("selection", {}),
        }
        self.selector = SELECTION.get(self.cfg.selection)(**sel_params)

        self.settlement = SETTLEMENT.get(self.cfg.settlement)(**(strategy_params or {}).get('settlement', {}))
        self.aggregator: AggregationStrategy = AGGREGATION.get(self.cfg.aggregation)(**(strategy_params or {}).get('aggregation', {}))

        self.nodes: Dict[int, NodeState] = {}
        self.features: Dict[int, Dict[str, float]] = {}
        self.contributions: Dict[int, float] = {}
        self.rewards: Dict[int, float] = {}
        self.balances: Dict[int, float] = {}
        self.committee: List[int] = []
        self.committee_history: List[List[int]] = []
        self.prev_global = None
        self.cooldowns: Dict[int, float] = {}
        self.metrics = DetectionMetrics()

    def register_node(self, node_id: int, *, stake: float, reputation: float):
        self.nodes[node_id] = NodeState(node_id=node_id, stake=float(stake), reputation=float(reputation))
        self.cooldowns.setdefault(node_id, 0.0)

    def set_features(self, node_id: int, **feats: Any):
        out = {}
        for k, v in feats.items():
            if v is None:
                continue
            # vectors/matrices -> numpy array (float)
            if isinstance(v, (np.ndarray, list, tuple)):
                try:
                    out[k] = np.asarray(v, dtype=float)
                except Exception:
                    out[k] = np.asarray(v)
            # numeric scalars -> float
            elif isinstance(v, (int, float, np.integer, np.floating)):
                out[k] = float(v)
            # fallback: keep as-is (strings, bools, etc.)
            else:
                try:
                    out[k] = float(v)
                except Exception:
                    out[k] = v
        # print(f"Set features for node {node_id}: {out}")
        self.features[int(node_id)] = out

    # def set_features(self, node_id: int, **feats: float):
    #     self.features[int(node_id)] = {k: float(v) for k, v in feats.items()}

    def set_contribution(self, node_id: int, score: float):


        self.contributions[int(node_id)] = float(score)

    def credit_reward(self, node_id: int, amount: float):
        self.rewards[int(node_id)] = self.rewards.get(int(node_id), 0.0) + float(amount)

    def select_committee(self):
        sel = self.selector.select(self.nodes, self.cooldowns)
        self.committee = sel
        self.committee_history.append(sel)
        for nid in list(self.cooldowns.keys()):
            self.cooldowns[nid] = max(0.0, self.cooldowns[nid] - 1.0)
        for nid in sel:
            self.cooldowns[nid] = float(self.cfg.committee_cooldown)
        return sel

    def _execute_plans(self, plans: Dict):
        for nid in plans.get("note_participation", set()):
            if nid in self.nodes:
                self.nodes[nid].participation += 1
        for nid, c in plans.get("append_contrib", {}).items():
            if nid in self.nodes:
                arr = self.nodes[nid].contrib_history
                arr.append(float(c))
                if len(arr) > 200:
                    del arr[:len(arr)-200]
        for nid, d in plans.get("apply_penalties", {}).items():
            if nid in self.nodes:
                node = self.nodes[nid]
                node.stake = max(0.0, node.stake * float(d.get("stake_mul", 1.0)))
                node.reputation = max(0.0, node.reputation * float(d.get("rep_mul", 1.0)))
        for nid, amt in plans.get("credit_rewards", {}).items():
            if nid in self.nodes:
                node = self.nodes[nid]
                a = float(amt)
                node.stake += a
                self.balances[nid] = self.balances.get(nid, 0.0) + a
        for nid, rep in plans.get("set_reputations", {}).items():
            if nid in self.nodes:
                self.nodes[nid].reputation = float(rep)

        detected_map = plans.get("detected", {})
        detected_ids = {int(k) for k, v in detected_map.items() if v}
        return detected_ids

    def run_round(self, round_idx: int, updates: Optional[List[ModelUpdate]] = None,
                  true_malicious: Optional[Sequence[int]] = None):

        print(f"Selected committee: {self.select_committee()} for round {round_idx}")
        plans = self.settlement.run(
            round_idx,
            self.nodes,
            self.contributions,
            self.features,
            self.rewards,
            self.detector,
            self.reward,
            self.penalty,
            self.reputation,
        )
        detected_ids = self._execute_plans(plans)
        print(f"detected ids: {detected_ids}")

        global_params = None
        if updates:
            filtered_updates = [u for u in updates if u.node_id not in detected_ids]
            admitted_ids = [u.node_id for u in filtered_updates]
            print(f"Admitted client ids: {admitted_ids}")
            try:
                global_params = self.aggregator.aggregate(
                    filtered_updates,
                    prev_global=self.prev_global,
                    admitted_ids=admitted_ids,

                )

            except TypeError:
                try:
                    global_params = self.aggregator.aggregate(
                        filtered_updates, prev_global=self.prev_global
                    )
                except TypeError:
                    global_params = self.aggregator.aggregate(filtered_updates)
            self.prev_global = global_params

        truth_set: Set[int] = set(map(int, true_malicious or []))
        self.metrics.log(round_idx, detected_ids, truth_set)
        print(f"[Round {round_idx}] Detected malicious: {sorted(detected_ids)}; Truth: {sorted(truth_set)}")

        out = {
            "round": round_idx,
            "committee": self.committee,
            "global_params": global_params,  # may remain from previous round
            "balances": dict(self.balances),
            "reputations": {nid: n.reputation for nid, n in self.nodes.items()},
            "detected": sorted(detected_ids),
            "truth": sorted(truth_set),
            # "plans": plans,
            # "metrics": self.metrics.summary()[-1] if self.metrics.summary() else {},
        }
        self.features.clear()
        self.contributions.clear()
        self.rewards.clear()

        return out
