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
    """确保所有客户端与模板键集合、形状一致。"""
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
        for k in key_order:
            if k not in u.params:
                raise ValueError(f"Client[{idx}] missing key '{k}'.")
            v = np.asarray(u.params[k])
            shp, dt = ref[k]
            if v.shape != shp:
                raise ValueError(f"Shape mismatch at key '{k}' for client[{idx}]: {v.shape} vs {shp}")
            # 不强制 dtype 完全一致（有时 float32/float64 混用），但记录一下
    # 通过则无异常

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
       - strict key/shape validation
       - float64 accumulation
       - first-round safeguards
       - optional debug logs
    """
    def __init__(self,
                 percentile: float = 0.9,      # 预留，如果想用分位数替代中位数
                 epsilon: float = 8.0,
                 delta: float = 1e-5,
                 use_noise: bool = False,      # 建议先关闭噪声，定位核心问题
                 use_clipping: bool = True,    # 可一键关闭裁剪，排查影响
                 debug: bool = True):
        self.percentile = float(np.clip(percentile, 0.5, 0.99))
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.use_noise = bool(use_noise)
        self.use_clipping = bool(use_clipping)
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

        # 2) 强校验：所有客户端键、形状必须与模板一致
        _validate_keys_and_shapes(updates, key_order)

        # 3) 过滤接纳集合
        ids = set(int(u.node_id) for u in updates)
        A = set(int(i) for i in (admitted_ids or [])) & ids
        target = [u for u in updates if not A or int(u.node_id) in A]
        if not target:
            # 若 admitted_ids 结果为空，回退到所有
            target = updates

        # 4) 计算基准向量与距离分布
        base_flat, base_dtype_map = _flatten_ordered(prev_global, key_order)

        def dist_to_base(u: ModelUpdate) -> float:
            w, _ = _flatten_ordered(u.params, key_order)
            return float(np.linalg.norm(w - base_flat))

        dists = np.asarray([dist_to_base(u) for u in target], dtype=np.float64)
        # 中位数（或分位数）作为 S_t 的尺度
        S_t = float(np.median(dists))
        # 如果 S_t 异常为 0（极端同构/首轮均等），给一个极小正数，避免数值问题
        if S_t == 0.0:
            S_t = float(np.max(dists))  # 先试更宽容的尺度
            if S_t == 0.0:
                S_t = 1e-12

        # 5) 是否进行裁剪与平均
        clipped_vecs = []
        if self.use_clipping:
            # 首轮防炸：如果 prev_global 全 0 或范数极小，裁剪容易把更新拉回去，导致学习停滞
            # 策略：当 ||prev_global|| 很小且大多数 e_i >> 0 时，轻微放宽裁剪（提高门限或直接不裁剪）
            base_norm = float(np.linalg.norm(base_flat))
            loosen_clip = (base_norm < 1e-8 and np.median(dists) > 0)
            clip_scale = 1.0 if not loosen_clip else 2.0  # 首轮稍微放松
            S_eff = S_t * clip_scale

            for u in target:
                w_u, _ = _flatten_ordered(u.params, key_order)
                delta_u = w_u - base_flat
                e_i = float(np.linalg.norm(delta_u))
                gamma = 1.0 if e_i == 0.0 else min(1.0, S_eff / (e_i + 1e-12))
                w_clip = base_flat + gamma * delta_u
                clipped_vecs.append(w_clip)
        else:
            for u in target:
                w_u, _ = _flatten_ordered(u.params, key_order)
                clipped_vecs.append(w_u)

        stacked = np.stack(clipped_vecs, axis=0).astype(np.float64, copy=False)
        avg_flat = np.mean(stacked, axis=0)

        # 6) 可选加噪（建议问题定位阶段关闭）
        if self.use_noise and self.epsilon > 0.0 and self.delta > 0.0:
            sigma = (S_t / self.epsilon) * np.sqrt(2.0 * np.log(1.25 / self.delta))
            if sigma > 0.0:
                avg_flat = avg_flat + np.random.normal(loc=0.0, scale=float(sigma), size=avg_flat.shape)

        # 7) 调试信息（只打印统计，不暴露大数组）
        if self.debug:
            clipped_flags = []
            if self.use_clipping:
                # 被裁剪的判定：gamma<1 等价于 e_i > S_eff
                S_eff = S_t * (2.0 if (float(np.linalg.norm(base_flat)) < 1e-8 and np.median(dists) > 0) else 1.0)
                clipped_flags = [dist > S_eff + 1e-12 for dist in dists]
            print(
                "[FLAME DEBUG] | "
                f"num_clients={len(target)} | "
                f"S_t={S_t:.6g} | "
                f"dists[min/med/max]=[{dists.min():.6g}/{np.median(dists):.6g}/{dists.max():.6g}] | "
                f"clipped_ratio={(np.mean(clipped_flags) if clipped_flags else 0):.3f} | "
                f"use_clipping={self.use_clipping} | use_noise={self.use_noise}"
            )

        # 8) 回填结构并恢复 dtype
        merged = _shape_like_ordered(prev_global, avg_flat, key_order, base_dtype_map)
        return merged
