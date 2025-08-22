# base.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
from ..core.registry import AGGREGATION  # 如需注册，解开并在类上加装饰器


@AGGREGATION.register("fedavg")
class AggregationStrategy:
    """FedAvg on absolute model weights (no deltas).
    - Expects each update to have: .params (dict[str, np.ndarray]),
      optional .num_samples or .weight, and update_type == "weights".
    - Deterministic key order, strict shape checks, float64 accumulation.
    """

    def __init__(self, eps: float = 1e-12, debug: bool = True) -> None:
        self.eps = float(eps)
        self.debug = bool(debug)

    def _key_order(self, params: Dict[str, np.ndarray]) -> Tuple[str, ...]:
        return tuple(sorted(params.keys()))

    def _validate(self, updates: List, key_order: Tuple[str, ...]) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        if not updates:
            raise ValueError("aggregate() received no updates")
        ref = {}
        first = updates[0].params
        for k in key_order:
            if k not in first:
                raise ValueError(f"First update missing key '{k}'.")
            v = np.asarray(first[k])
            ref[k] = (v.shape, v.dtype)

        for idx, u in enumerate(updates):
            if getattr(u, "update_type", "weights") != "weights":
                raise ValueError(f"FedAvg expects absolute weights; got {getattr(u,'update_type', None)} at update[{idx}]")
            if set(u.params.keys()) != set(key_order):
                missing = set(key_order) - set(u.params.keys())
                extra = set(u.params.keys()) - set(key_order)
                raise ValueError(f"Key mismatch at update[{idx}]; missing={missing}, extra={extra}")
            for k in key_order:
                v = np.asarray(u.params[k])
                shp, _ = ref[k]
                if v.shape != shp:
                    raise ValueError(f"Shape mismatch at key '{k}' for update[{idx}]: {v.shape} vs {shp}")
                if not np.isfinite(v).all():
                    raise ValueError(f"Non-finite values in update[{idx}] key '{k}'")
        return ref

    def _get_weight(self, u) -> float:
        if hasattr(u, "num_samples") and u.num_samples is not None:
            return float(u.num_samples)
        if hasattr(u, "weight") and u.weight is not None:
            return float(u.weight)
        return 1.0

    def aggregate(self, updates: List) -> Dict[str, np.ndarray]:
        key_order = self._key_order(updates[0].params)
        ref_map = self._validate(updates, key_order)

        raw_w = np.asarray([max(0.0, self._get_weight(u)) for u in updates], dtype=np.float64)
        tot_w = float(np.sum(raw_w))
        if not np.isfinite(tot_w) or tot_w <= self.eps:
            raw_w[:] = 1.0
            tot_w = float(len(updates))
        norm_w = (raw_w / tot_w).astype(np.float64, copy=False)

        out: Dict[str, np.ndarray] = {k: np.zeros(ref_map[k][0], dtype=np.float64) for k in key_order}
        for w, u in zip(norm_w, updates):
            for k in key_order:
                out[k] += w * np.asarray(u.params[k], dtype=np.float64, order="C")

        for k in key_order:
            out[k] = out[k].astype(ref_map[k][1], copy=False)

        if self.debug:
            print(f"[FedAvg] clients={len(updates)} | weights(min/med/max)={norm_w.min():.4f}/{np.median(norm_w):.4f}/{norm_w.max():.4f}")
        return out
