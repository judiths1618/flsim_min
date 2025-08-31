from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from ..core.types import NodeState
from ..core.registry import SELECTION

@dataclass
class SelectionParams:
    committee_size: int = 5
    rep_exponent: float = 1.0
    cooldown: int = 3
    num_strata: int = 3

@SELECTION.register("stratified_softmax")
class StratifiedSoftmaxSelector:
    def __init__(self, params: SelectionParams | None = None, **kwargs) -> None:
        self.p = params or SelectionParams(**kwargs) if kwargs else (params or SelectionParams())

    def select(self, nodes: Dict[int, NodeState], cooldowns: Dict[int, float]) -> List[int]:
        ordered = sorted(nodes.values(), key=lambda n: n.reputation, reverse=True)
        if not ordered:
            return []
        s = max(1, int(self.p.num_strata))
        stride = max(1, len(ordered) // s)
        buckets = [ordered[i*stride:(i+1)*stride] for i in range(s)]
        rest = s * stride
        if rest < len(ordered):
            buckets[-1].extend(ordered[rest:])

        quotas = [self.p.committee_size // s] * s
        for i in range(self.p.committee_size - sum(quotas)):
            quotas[i % s] += 1

        selected: List[int] = []
        for bucket, q in zip(buckets, quotas):
            pool = [b for b in bucket if cooldowns.get(b.node_id, 0.0) <= 0.0]
            if not pool or q <= 0:
                continue
            if len(pool) <= q:
                selected.extend([p.node_id for p in pool])
            else:
                reps = np.asarray([p.reputation for p in pool], dtype=float)
                probs = np.exp((reps - reps.max()) * float(self.p.rep_exponent))
                ps = probs / (probs.sum() if probs.sum() > 0 else 1.0)
                idx = list(range(len(pool)))
                picks = []
                for _ in range(q):
                    r = np.random.random(); acc = 0.0; pick = 0
                    for j, pv in enumerate(ps):
                        acc += pv
                        if r <= acc:
                            pick = j; break
                    picks.append(idx[pick])
                    ps = np.delete(ps, pick); idx.pop(pick)
                selected.extend([pool[i].node_id for i in picks])

        if len(selected) < self.p.committee_size:
            need = self.p.committee_size - len(selected)
            chosen = set(selected)
            pool = [n for n in ordered if cooldowns.get(n.node_id, 0.0) <= 0.0 and n.node_id not in chosen]
            selected.extend([n.node_id for n in pool[:need]])
        return selected[: self.p.committee_size]


@SELECTION.register("none")
class NoSelection:
    """Selector that always returns an empty committee."""

    def __init__(self, params: SelectionParams | None = None, **kwargs) -> None:  # noqa: D401
        pass

    def select(self, nodes: Dict[int, NodeState], cooldowns: Dict[int, float]) -> List[int]:
        return []
