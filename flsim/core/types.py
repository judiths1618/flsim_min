from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class NodeState:
    node_id: int
    stake: float = 0.0
    reputation: float = 0.0
    balance: float = 0.0
    cooldown: float = 0.0
    contrib_history: List[float] = field(default_factory=list)
    participation: int = 0
    committee_history: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelUpdate:
    node_id: int
    params: Any               # numpy arrays or dict of arrays
    weight: float = 1.0
    metrics: Dict[str, float] = field(default_factory=dict)
    update_type: str = "gradient"

@dataclass
class RoundContext:
    round_idx: int
    node_states: Dict[int, NodeState]
    contributions: Dict[int, float]
    features: Dict[int, Dict[str, float]] = field(default_factory=dict)
    pre_rewards: Dict[int, float] = field(default_factory=dict)
    committee: List[int] = field(default_factory=list)
    rng: Any = None
