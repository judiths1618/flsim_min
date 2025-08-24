from __future__ import annotations
from typing import List, Iterable, Any, Optional, Tuple, Dict
import numpy as np
from .base import AggregationStrategy
from ..core.types import ModelUpdate
from ..core.registry import AGGREGATION


# ----------------- 强校验与有序展平 -----------------

def _canonical_key_order(template: Any) -> Optional[Tuple[str, ...]]:
    """确定性键顺序：优先按名称排序，避免受构造顺序影响。"""
    if isinstance(template, dict):
        return tuple(sorted(template.keys()))
    return None

def _validate_keys_and_shapes(updates: List[ModelUpdate],
                              key_order: Optional[Tuple[str, ...]]) -> None:
    """确保所有客户端与模板键集合、形状一致，并无 NaN/Inf。"""
    if key_order is None:
        # 非 dict，不需要校验
        return
    # 基于第一位客户端建立参考形状
    ref = {}
    first = updates[0].params
    for k in key_order:
        if k not in first:
            raise ValueError(f"Client[0] missing key '{k}' expected by template/order.")
        v = np.asarray(first[k])
        ref[k] = (v.shape, v.dtype)

    for idx, u in enumerate(updates):
        if getattr(u, "update_type", "weights") != "weights":
            raise ValueError(f"Update[{idx}] has update_type={getattr(u,'update_type', None)}; FLAME expects absolute 'weights'.")
        for k in key_order:
            if k not in u.params:
                raise ValueError(f"Client[{idx}] missing key '{k}'.")
            v = np.asarray(u.params[k])
            shp, _ = ref[k]
            if v.shape != shp:
                raise ValueError(f"Shape mismatch at key '{k}' for client[{idx}]: {v.shape} vs {shp}")
            if not np.isfinite(v).all():
                raise ValueError(f"Client[{idx}] key '{k}' contains NaN/Inf.")

def _flatten_ordered(params: Any, key_order: Optional[Tuple[str, ...]]) -> Tuple[np.ndarray, Dict[str, np.dtype]]:
    """按 key_order 展平为 float64，同时记录原 dtype 用于回填。"""
    if isinstance(params, dict):
        if key_order is None:
            key_order = tuple(sorted(params.keys()))
        parts = []
        dtypes = {}
        for k in key_order:
            v = np.asarray(params[k])
            dtypes[k] = v.dtype
            parts.append(np.asarray(v, dtype=np.float64, order="C").ravel())
        flat = np.concatenate(parts, axis=0) if parts else np.array([], dtype=np.float64)
        return flat, dtypes
    else:
        arr = np.asarray(params, order="C")
        return arr.astype(np.float64, copy=False).ravel(), {"__scalar__": arr.dtype}

def _shape_like_ordered(template: Any,
                        flat: np.ndarray,
                        key_order: Optional[Tuple[str, ...]],
                        dtype_map: Dict[str, np.dtype]) -> Any:
    """按模板形状回填并恢复 dtype。"""
    if isinstance(template, dict):
        if key_order is None:
            key_order = tuple(sorted(template.keys()))
        out = {}
        i = 0
        for k in key_order:
            v_t = np.asarray(template[k])
            n = v_t.size
            seg = flat[i:i+n]
            if seg.size != n:
                raise ValueError(f"Flat length mismatch at '{k}': need {n}, have {seg.size}")
            out[k] = seg.reshape(v_t.shape).astype(dtype_map.get(k, v_t.dtype), copy=False)
            i += n
        if i != flat.size:
            raise ValueError(f"Flat length mismatch: consumed {i}, total {flat.size}")
        return out
    else:
        arr_t = np.asarray(template)
        if flat.size != arr_t.size:
            raise ValueError(f"Flat length mismatch: {flat.size} vs template {arr_t.size}")
        return flat.reshape(arr_t.shape).astype(dtype_map.get("__scalar__", arr_t.dtype), copy=False)

def _zeros_like_params(p: Any) -> Any:
    if isinstance(p, dict):
        return {k: np.zeros_like(v) for k, v in p.items()}
    return np.zeros_like(p)


# ----------------- 主类实现（加强版） -----------------

@AGGREGATION.register("flame_agg")
class FlameAggregation(AggregationStrategy):
    """FLAME-style aggregation (robust median clip + optional DP noise), with:
       - canonical key order (sorted keys)
       - strict key/shape/finite validation
       - float64 accumulation
       - first-round safeguards
       - optional sample weighting
       - optional debug logs
    """
    def __init__(self,
                 percentile: float = 0.9,      # 如需用分位数替代中位数，可把 S_t = np.quantile(dists, percentile)
                 epsilon: float = 8.0,
                 delta: float = 1e-5,
                 use_noise: bool = False,      # 定位阶段建议关闭
                 use_clipping: bool = True,    # 可一键关闭裁剪，排查影响
                 weight_by_samples: bool = True,  # 是否按样本数加权平均（在裁剪之后）
                 disable_clip_if_zero_init: bool = True,  # 首轮若全局权重近零，则禁用裁剪
                 debug: bool = True):
        self.percentile = float(np.clip(percentile, 0.5, 0.99))
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.use_noise = bool(use_noise)
        self.use_clipping = bool(use_clipping)
        self.weight_by_samples = bool(weight_by_samples)
        self.disable_clip_if_zero_init = bool(disable_clip_if_zero_init)
        self.debug = bool(debug)

    def aggregate(self,
                  updates: List[ModelUpdate],
                  *,
                  prev_global: Optional[Any] = None,
                  admitted_ids: Optional[Iterable[int]] = None) -> Any:
        if not updates:
            raise ValueError("No updates to aggregate")

        # 1) 准备模板与有序键
        if prev_global is None:
            prev_global = _zeros_like_params(updates[0].params)
        key_order = _canonical_key_order(prev_global)

        # 2) 强校验
        _validate_keys_and_shapes(updates, key_order)

        # 3) 过滤接纳集合
        ids = set(int(u.node_id) for u in updates)
        A = set(int(i) for i in (admitted_ids or [])) & ids
        target = [u for u in updates if not A or int(u.node_id) in A] or updates

        # 4) 基准与距离
        base_flat, base_dtype_map = _flatten_ordered(prev_global, key_order)

        def dist_to_base(u: ModelUpdate) -> float:
            w, _ = _flatten_ordered(u.params, key_order)
            d = w - base_flat
            return float(np.linalg.norm(d))

        dists = np.asarray([dist_to_base(u) for u in target], dtype=np.float64)
        if not np.isfinite(dists).all():
            raise ValueError("Distances contain NaN/Inf; check client updates.")

        S_t = float(np.median(dists))  # or np.quantile(dists, self.percentile)
        if S_t == 0.0:
            S_t = float(np.max(dists)) or 1e-12

        # 5) 裁剪与平均
        clipped_vecs = []
        gammas = []
        # 首轮防炸：若全局范数近 0，且开启 disable_clip_if_zero_init，则不裁剪
        base_norm = float(np.linalg.norm(base_flat))
        actually_clip = self.use_clipping and not (self.disable_clip_if_zero_init and base_norm < 1e-12)

        # 若裁剪，是否放宽门限
        clip_scale = 1.0
        if actually_clip and base_norm < 1e-8 and np.median(dists) > 0:
            clip_scale = 2.0  # 轻微放宽
        S_eff = S_t * clip_scale

        for u in target:
            w_u, _ = _flatten_ordered(u.params, key_order)
            if actually_clip:
                delta_u = w_u - base_flat
                e_i = float(np.linalg.norm(delta_u))
                gamma = 1.0 if e_i == 0.0 else min(1.0, S_eff / (e_i + 1e-12))
                w_proc = base_flat + gamma * delta_u
                gammas.append(gamma)
            else:
                w_proc = w_u
                gammas.append(1.0)
            clipped_vecs.append(w_proc)

        stacked = np.stack(clipped_vecs, axis=0).astype(np.float64, copy=False)

        # 样本数（或权重）加权
        if self.weight_by_samples:
            raw_w = np.asarray([float(getattr(u, "num_samples", None) or getattr(u, "weight", 1.0)) for u in target], dtype=np.float64)
            raw_w = np.maximum(raw_w, 0.0)
            tot_w = float(np.sum(raw_w))
            if not np.isfinite(tot_w) or tot_w <= 0.0:
                raw_w[:] = 1.0
                tot_w = float(len(target))
            norm_w = (raw_w / tot_w).reshape(-1, 1)
            avg_flat = np.sum(stacked * norm_w, axis=0)
        else:
            avg_flat = np.mean(stacked, axis=0)

        # 6) 可选加噪
        if self.use_noise and self.epsilon > 0.0 and self.delta > 0.0:
            sigma = (S_t / self.epsilon) * np.sqrt(2.0 * np.log(1.25 / self.delta))
            if sigma > 0.0:
                avg_flat = avg_flat + np.random.normal(loc=0.0, scale=float(sigma), size=avg_flat.shape)

        # 7) 调试信息（与裁剪路径完全一致的参数）
        if self.debug:
            clipped_ratio = 0.0
            if actually_clip:
                clipped_ratio = float(np.mean([g < 0.999999 for g in gammas]))
            print(
                "[FLAME DEBUG] | "
                f"num_clients={len(target)}: {ids} | "
                f"S_t={S_t:.6g} | "
                f"dists[min/med/max]=[{dists.min():.6g}/{np.median(dists):.6g}/{dists.max():.6g}] | "
                f"clip={'on' if actually_clip else 'off'} (S_eff={S_eff:.6g}) | "
                f"clipped_ratio={clipped_ratio:.3f} | "
                f"weighted={'yes' if self.weight_by_samples else 'no'} | "
                f"use_noise={self.use_noise}"
            )

        # 8) 回填结构并恢复 dtype
        merged = _shape_like_ordered(prev_global, avg_flat, key_order, base_dtype_map)
        return merged
