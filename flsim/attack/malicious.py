# flsim/attack/malicious.py
from __future__ import annotations
from typing import Dict, Iterable, Set, Tuple
import numpy as np

from ..core.types import ModelUpdate

def choose_malicious_nodes(
    all_node_ids: Iterable[int],
    mal_frac: float = 0.0,
    explicit_ids: Iterable[int] | None = None,
    seed: int = 42,
) -> Set[int]:
    """选择恶意节点集合。若提供 explicit_ids 则优先生效，否则按比例随机抽样。"""
    all_ids = sorted(set(int(i) for i in all_node_ids))
    # print(all_ids)
    if explicit_ids is not None:
        # print(explicit_ids)
        return {int(i) for i in explicit_ids if int(i) in all_ids}
    if mal_frac <= 0.0:
        # print()
        return set()
    rng = np.random.default_rng(seed)

    k = max(1, int(np.floor(len(all_ids) * float(mal_frac))))
    # print(mal_frac, k)
    return set(rng.choice(all_ids, size=k, replace=False).tolist())

def _scale_params(params: Dict[str, np.ndarray], factor: float) -> Dict[str, np.ndarray]:
    return {k: (v.astype(float) * factor) for k, v in params.items()}

def _signflip_params(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return _scale_params(params, -1.0)

def _zero_params(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.zeros_like(v, dtype=float) for k, v in params.items()}

def _noise_params(params: Dict[str, np.ndarray], std: float, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}
    for k, v in params.items():
        noise = rng.normal(loc=0.0, scale=float(std), size=v.shape)
        out[k] = v.astype(float) + noise
    return out

def apply_malicious_updates(
    updates: list[ModelUpdate],
    malicious_ids: Set[int],
    *,
    behavior: str = "scale",       # ["scale", "signflip", "zero", "noise"]
    scale: float = -10.0,          # 用于 "scale"
    noise_std: float = 0.1,        # 用于 "noise"
    seed: int = 42,
) -> Tuple[list[ModelUpdate], Set[int]]:
    """对 updates 中属于恶意节点的条目应用篡改并返回 (updates, true_malicious)。"""
    true_mal = set(malicious_ids)
    if not true_mal:
        return updates, true_mal

    mutated: list[ModelUpdate] = []
    for u in updates:
        if int(u.node_id) not in true_mal:
            mutated.append(u)
            continue

        p = u.params
        if behavior == "scale":
            p2 = _scale_params(p, factor=float(scale))
        elif behavior == "signflip":
            p2 = _signflip_params(p)
        elif behavior == "zero":
            p2 = _zero_params(p)
        elif behavior == "noise":
            p2 = _noise_params(p, std=float(noise_std), seed=seed + int(u.node_id))
        else:
            # 未知行为则不改动
            p2 = p

        mutated.append(
            ModelUpdate(
                node_id=int(u.node_id),
                params=p2,                     # 篡改后的绝对参数
                weight=float(u.weight),
                metrics=dict(u.metrics or {}),
                update_type=u.update_type,
            )
        )
    return mutated, true_mal
