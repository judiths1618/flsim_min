from __future__ import annotations
from typing import Dict, Set
import numpy as np
from ...core.registry import DETECTION


from typing import Any, Dict, List, Tuple
import numpy as np

def _to_1d_float(x: Any) -> np.ndarray:
    """Recursively flatten x -> 1D float array.
    Supports dict (sorted by key), list/tuple, array-like, and scalars.
    """
    if isinstance(x, dict):
        parts = []
        # 保持确定性：按键名排序后拼接
        for k in sorted(x.keys()):
            parts.append(_to_1d_float(x[k]))
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    elif isinstance(x, (list, tuple)):
        parts = [_to_1d_float(e) for e in x]
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    else:
        arr = np.asarray(x, dtype=float)
        return arr.ravel()

def _flattened(features: Dict[int, Dict[str, Any]]) -> Tuple[List[int], np.ndarray | None]:
    ids: List[int] = []
    vecs: List[np.ndarray] = []

    for nid, f in features.items():
        if "flat_update" not in f:
            continue
        v = _to_1d_float(f["flat_update"])
        if v.size > 0 and np.isfinite(v).all():
            ids.append(int(nid))
            vecs.append(v)

    if not vecs:
        return ids, None

    # 对齐长度：用 0 填充到统一长度，避免 stack 失败
    maxd = max(v.size for v in vecs)
    if any(v.size != maxd for v in vecs):
        vecs = [v if v.size == maxd else np.pad(v, (0, maxd - v.size), constant_values=0.0) for v in vecs]

    X = np.stack(vecs, axis=0)
    return ids, X

# def _flattened(features: Dict[int, Dict[str, float]]):
#     """Flatten features to a list of node IDs and a 2D array of vectors."""
#     ids, vecs = [], []
#     for nid, f in features.items():
#         if "flat_update" in f:
#             v = np.asarray(f["flat_update"], dtype=float).ravel()
#             if v.size > 0 and np.isfinite(v).all():
#                 ids.append(int(nid)); vecs.append(v)
#     return ids, (np.stack(vecs, axis=0) if vecs else None)

def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / nrm

def _largest_label(labels: np.ndarray) -> int | None:
    labs, counts = np.unique(labels[labels != -1], return_counts=True)
    if labs.size == 0:
        return None
    return int(labs[np.argmax(counts)])


def _pairwise_euclidean(Xn: np.ndarray) -> np.ndarray:
    G = Xn @ Xn.T
    nrm = np.sum(Xn * Xn, axis=1, keepdims=True)
    D2 = np.maximum(0.0, nrm + nrm.T - 2.0 * G)
    return np.sqrt(D2) + 1e-12


@DETECTION.register("flame")
class FlameDetector:
    """FLAME-style filtering via clustering on cosine distance with fallbacks."""
    def __init__(self, *, min_points: int = 4, min_cluster_frac: float = 0.2, dbscan_eps: float = 0.3,
                 detect_score_thresh: float = 0.05):
        self.min_points = int(min_points)
        self.min_cluster_frac = float(np.clip(float(min_cluster_frac), 0.05, 0.8))
        self.dbscan_eps = float(dbscan_eps)
        self.detect_score_thresh = float(detect_score_thresh)

    def detect(self, features: Dict[int, Dict[str, float]], scores: Dict[int, float]) -> Dict[int, bool]:
        ids, X = _flattened(features)
        if X is None or X.shape[0] < self.min_points:
            keys = set(list(scores.keys()) + list(features.keys()))
            return {int(n): (float(scores.get(n, 0.0)) < self.detect_score_thresh) for n in keys}

        norms = np.linalg.norm(X, axis=1)

        med = float(np.median(norms))
        iqr = np.subtract(*np.percentile(norms, [75, 25]))
        thresh = med + 1.5 * max(1e-8, float(iqr))
        norm_flags = norms > thresh
        if np.any(norm_flags):
            flagged = {ids[i]: bool(norm_flags[i]) for i in range(len(ids))}

            for nid in (set(scores.keys()) - set(ids)):
                flagged[int(nid)] = bool(float(scores.get(nid, 0.0)) < self.detect_score_thresh)
            return flagged

        Xn = _l2_normalize(X)

        labels = None
        try:
            import hdbscan  # type: ignore
            min_cluster_size = max(self.min_points, int(self.min_cluster_frac * Xn.shape[0]))
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
            labels = clusterer.fit_predict(Xn)
        except Exception:
            try:
                from sklearn.cluster import DBSCAN  # type: ignore
                min_samples = max(2, int(self.min_cluster_frac * Xn.shape[0]))
                clusterer = DBSCAN(eps=self.dbscan_eps, min_samples=min_samples, metric='cosine')
                labels = clusterer.fit_predict(Xn)
            except Exception:
                labels = None

        if labels is not None and np.any(labels != -1):
            keep_label = _largest_label(labels)
            admitted = {ids[i] for i, lb in enumerate(labels) if lb == keep_label}
            flagged = {nid: (nid not in admitted) for nid in ids}
            for nid in (set(scores.keys()) - set(ids)):
                flagged[int(nid)] = bool(float(scores.get(nid, 0.0)) < self.detect_score_thresh)
            return flagged

        # Fallback: medoid + IQR
        D = _pairwise_euclidean(Xn)
        # D = np.sqrt(np.maximum(0.0, (Xn @ Xn.T * -2.0) + (np.sum(Xn*Xn,1,keepdims=True) + np.sum(Xn*Xn,1,keepdims=True).T))) + 1e-12
        s = np.sum(D, axis=1)
        med = int(np.argmin(s))
        d_med = D[med]
        q1, q3 = np.percentile(d_med, [25, 75])
        iqr = max(1e-8, float(q3 - q1))
        tau = float(q3 + 1.5 * iqr)
        flags = (d_med > tau)
        flagged = {ids[i]: bool(flags[i]) for i in range(len(ids))}
        for nid in (set(scores.keys()) - set(ids)):
            flagged[int(nid)] = bool(float(scores.get(nid, 0.0)) < self.detect_score_thresh)
        return flagged
