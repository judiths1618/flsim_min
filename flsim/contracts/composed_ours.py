from dataclasses import asdict, dataclass

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
    # detection: str = "flame"
    contribution: str = "metric"
    reward: str = "ours"
    penalty: str = "default"
    reputation: str = "default"
    selection: str = "stratified_softmax"
    settlement: str = "plans_engine"
    aggregation: str = "flame_agg"


class ComposedContract:
    def __init__(self, cfg: ContractConfig | None = None, strategy_params: dict | None = None):
        self.cfg = cfg or ContractConfig()

        # self.detector = DETECTION.get(self.cfg.detection)(**(strategy_params or {}).get('detection', {}))
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
            if isinstance(v, (np.ndarray, list, tuple)):
                try:
                    out[k] = np.asarray(v, dtype=float)
                except Exception:
                    out[k] = np.asarray(v)
            elif isinstance(v, (int, float, np.integer, np.floating)):
                out[k] = float(v)
            else:
                try:
                    out[k] = float(v)
                except Exception:
                    out[k] = v
        self.features[int(node_id)] = out

    def set_contribution(self, node_id: int, score: float):  # quantify the contribution score using eval acc
        # print(f"Node [{node_id}]'s Score: {score}")
        self.contributions[int(node_id)] = round(float(10*score), 4)
        # print(self.contributions)

    def credit_reward(self, node_id: int, amount: float | None = None, *, in_committee: bool = False) -> float:
        node = self.nodes.get(int(node_id))
        if node is None:
            return 0.0
        reward_amt = self.reward.compute(node, self.nodes, in_committee=in_committee)
        if amount is not None:
            reward_amt = float(amount)
        self.rewards[int(node_id)] = float(reward_amt)
        # print(f"node {node_id}'s credit reward amount: {reward_amt}, {node}")
        return float(reward_amt)

    def update_reputation(self, node_id: int, contribution: float, *, current_round: int) -> float:
        node = self.nodes.get(int(node_id))
        if node is None:
            return 0.0
        new_rep = self.reputation.update(node, contribution=float(contribution), current_round=current_round)
        self.nodes[int(node_id)].reputation = float(new_rep)
        # print(f"{node} update rep: {new_rep}")
        return float(new_rep)

    def apply_penalty(self, node_id: int, *, stake_mul: float | None = None, rep_mul: float | None = None) -> None:
        node = self.nodes.get(int(node_id))
        if node is None:
            return
        stake_mul = float(stake_mul) if stake_mul is not None else (
            1.0 - getattr(self.penalty.p, "stake_penalty_factor", 0.02)
        )
        rep_mul = float(rep_mul) if rep_mul is not None else (
            1.0 - getattr(self.penalty.p, "rep_penalty_factor", 0.5)
        )
        node.stake = max(0.0, node.stake * stake_mul)
        node.reputation = max(0.0, node.reputation * rep_mul)

    def select_committee(self, round_idx: int | None = None):
        sel = self.selector.select(self.nodes, self.cooldowns)
        self.committee = sel
        self.committee_history.append(sel)
        for nid in list(self.cooldowns.keys()):
            self.cooldowns[nid] = max(0.0, self.cooldowns[nid] - 1.0)
            if nid in self.nodes:
                self.nodes[nid].cooldown = self.cooldowns[nid]
        for nid in sel:
            self.cooldowns[nid] = float(self.cfg.committee_cooldown)
            if nid in self.nodes:
                self.nodes[nid].cooldown = self.cooldowns[nid]
                if round_idx is not None:
                    self.nodes[nid].committee_history.append(round_idx)
        return sel

    def _execute_plans(self, plans: Dict, detected_ids, *, round_idx: int):
        # record the participation
        for nid in plans.get("note_participation", set()):
            if nid in self.nodes:
                self.nodes[nid].participation += 1
        # append the contributions
        for nid, c in plans.get("append_contrib", {}).items():
            if nid in self.nodes:
                arr = self.nodes[nid].contrib_history
                arr.append(float(c))
                if len(arr) > 200:
                    del arr[: len(arr) - 200]
        # apply penalties according to the detection
        for nid, d in plans.get("apply_penalties", {}).items():
            if nid in detected_ids:
                # print(nid)
                self.apply_penalty(nid, stake_mul=d.get("stake_mul"), rep_mul=d.get("rep_mul"))
        # compute rewards
        for nid, amt in plans.get("credit_rewards", {}).items():
            if nid in self.nodes:
                node = self.nodes[nid]
                a = float(amt)
                # node.stake += a
                node.balance += a   # add reward to balance
                self.balances[nid] = self.balances.get(nid, 0.0) + a
        # update the reputation
        for nid in plans.get("set_reputations", {}).keys():
            self.update_reputation(nid, contribution=self.contributions.get(nid, 0.0), current_round=round_idx)

        # detected_map = plans.get("detected", {})
        # print(f"detected map: {detected_map}")
        # detected_ids = {int(k) for k, v in detected_map.items() if v}
        return "plan has been executed!\n"

    def run_round(self, round_idx: int, detected_ids, updates: Optional[List[ModelUpdate]] = None,
                  true_malicious: Optional[Sequence[int]] = None):

        # print(f"Selected committee: {self.select_committee(round_idx)} for round {round_idx}")
        plans = self.settlement.run(
            round_idx,
            self.nodes,
            self.contributions,
            self.features,
            self.rewards,
            # self.detected,
            detected_ids,
            self.committee,
            self.reward,
            self.penalty,
            self.reputation,
        )
        # executed = self._execute_plans(plans)
        executed = self._execute_plans(plans, detected_ids, round_idx=round_idx)
        # print(f"{executed} \n detected ids: {detected_ids}")

        global_params = self.prev_global

        if updates:
            filtered_updates = [u for u in updates if u.node_id not in detected_ids]
            admitted_ids = [u.node_id for u in filtered_updates]
            print(f"Admitted client ids: {admitted_ids}")
            try:
                # only aggregate the model updates fron filtered ones
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
        print(
            f"[Round {round_idx}] \n Detected malicious: {sorted(detected_ids)}; Truth: {sorted(truth_set)}; Committee: {self.select_committee(round_idx)}\n",
            f""
        )
        
        out = {
            "round": round_idx,
            "committee": self.committee,
            "global_params": global_params,  # may remain from previous round
            "balances": dict(self.balances),
            "reputations": {nid: n.reputation for nid, n in self.nodes.items()},
            "detected": sorted(detected_ids),
            "truth": sorted(truth_set),
            "plans": plans,
            "node_states": {nid: asdict(n) for nid, n in self.nodes.items()},
            "metrics": self.metrics.summary()[-1] if self.metrics.summary() else {},
        }
        self.features.clear()
        self.contributions.clear()
        # self.rewards.clear()
        self.rewards = {
            int(nid): float(r)
            for nid, r in plans.get("computed_rewards_next", {}).items()
        }
        # print(self.reward)

        return out
