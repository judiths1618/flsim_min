from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

try:
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
except Exception:
    FederatedDataset = None  # type: ignore

try:  # optional dependency for label-column discovery
    from datasets import load_dataset as _load_hf_dataset
except Exception:  # pragma: no cover - dependency may be missing
    _load_hf_dataset = None  # type: ignore

# Heuristic list of common label field names; extend as needed for specific datasets
_LABEL_CANDIDATES = ["label", "labels", "target", "class", "y", "logS"]

def _find_label_column(ds) -> str:
    cols = set(ds.column_names)
    for c in _LABEL_CANDIDATES:
        if c in cols:
            return c
    return "label"

def _num_classes_from_features(ds, label_col: str) -> Optional[int]:
    try:
        feats = ds.features
        if label_col in feats and hasattr(feats[label_col], "num_classes"):
            return int(feats[label_col].num_classes)
    except Exception:
        pass
    return None

def load_flower_partitions(dataset: str, num_clients: int, *, iid: bool = True, alpha: float = 0.5, split: str = "train", seed: int = 42):
    if FederatedDataset is None:
        raise ImportError("flwr-datasets is not installed. Run: pip install flwr-datasets datasets pillow")
    if iid:
        # part = IidPartitioner(num_partitions=int(num_clients), seed=seed)
        part = IidPartitioner(num_partitions=int(num_clients))
    else:
        # part = DirichletPartitioner(num_partitions=int(num_clients), partition_by="label", alpha=float(alpha), seed=seed)
                # Attempt to infer label column for non-IID partitioning
        part_col = "label"
        if _load_hf_dataset is not None:
            try:
                ds_tmp = _load_hf_dataset(dataset, split=split)
                part_col = _find_label_column(ds_tmp)
            except Exception:
                pass
        part = DirichletPartitioner(
            num_partitions=int(num_clients),
            partition_by=part_col,
            alpha=float(alpha),
            seed=seed,
        )

    fds = FederatedDataset(dataset=dataset, partitioners={split: part})
    partitions = {}
    label_col = None
    num_classes = None
    for cid in range(num_clients):
        ds = fds.load_partition(cid, split)
        partitions[cid] = ds
        if label_col is None:
            label_col = _find_label_column(ds)
            num_classes = _num_classes_from_features(ds, label_col)
    return partitions, (label_col or "label"), num_classes

def label_hist_vector(ds, label_col: str, *, num_classes: Optional[int] = None) -> np.ndarray:
    y = np.array(ds[label_col], dtype=int)
    if num_classes is None:
        K = int(y.max() + 1) if y.size > 0 else 1
    else:
        K = int(num_classes)
    hist, _ = np.histogram(y, bins=np.arange(K+1+1) - 0.5)
    if hist.sum() == 0:
        return np.zeros(K, dtype=float)
    return (hist / hist.sum()).astype(float)

def load_flower_label_vectors(dataset: str, num_clients: int, *, iid: bool = True, alpha: float = 0.5, split: str = "train", seed: int = 42):
    parts, label_col, num_classes = load_flower_partitions(dataset, num_clients, iid=iid, alpha=alpha, split=split, seed=seed)
    vecs: Dict[int, np.ndarray] = {}
    K_global = 0
    for cid, ds in parts.items():
        v = label_hist_vector(ds, label_col, num_classes=num_classes)
        vecs[int(cid)] = v
        K_global = max(K_global, v.shape[0])
    for cid, v in vecs.items():
        if v.shape[0] < K_global:
            pad = np.zeros(K_global - v.shape[0], dtype=float)
            vecs[cid] = np.concatenate([v, pad], axis=0)
    return vecs, K_global

def project_vectors(vecs: Dict[int, np.ndarray], out_dim: int, seed: int = 42) -> Dict[int, np.ndarray]:
    if len(vecs) == 0:
        return {}
    K = next(iter(vecs.values())).shape[0]
    rng = np.random.default_rng(seed)
    R = rng.normal(loc=0.0, scale=1.0/np.sqrt(K), size=(out_dim, K))
    proj = {int(cid): (R @ v).astype(float) for cid, v in vecs.items()}
    return proj



def load_flower_arrays(
    dataset: str = "cifar10",
    n_clients: int = 10,
    iid: bool = True,
    alpha: float = 0.5,
    split: str = "train",
    flatten: bool = True,
    normalize: bool = True,
    seed: int = 42,
    eval_split: str = "test",
    normalization: str = "client",   # "client" | "global" | "none"
    max_per_client: Optional[int] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], int, int, np.ndarray, np.ndarray]:
    """Return client train partitions and a *global* eval set.

    Returns:
        Xp: Dict[cid, X_cid] where X_cid has shape [N_cid, D] (float32)
        yp: Dict[cid, y_cid] where y_cid has shape [N_cid] (int)
        D:  feature dimension
        K:  number of classes
        X_eval: ndarray [N_eval, D] (float32)
        y_eval: ndarray [N_eval] (int)

    Notes:
      - Requires flwr-datasets (`pip install .[data]` or your project extras).
      - normalization="client": per-client min–max; "global": global min–max over all clients;
        "none": disable normalization (the `normalize` flag must also be True to enable any normalization).
      - If `max_per_client` is set, each client's data is truncated deterministically.
    """
    try:
        from flwr_datasets import FederatedDataset
        from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
    except Exception as e:
        raise ImportError("flwr-datasets is not installed. Install with extras: `pip install .[data]`") from e

    # Local helper shared by train/eval
    _IMG_KEYS = {"img", "image", "images", "pixel_values"}

    def _find_label_column(ds) -> str:
        # light heuristic
        candidates = {"label", "labels", "target", "class", "y"}
        cols = set(ds.column_names)
        for c in candidates:
            if c in cols:
                return c
        return "label"

    def _to_matrix(ds, ycol: str) -> Tuple[np.ndarray, np.ndarray]:
        y = ds[ycol].astype(int).ravel()
        X_keys = [k for k in ds.column_names if k != ycol]

        # Vision heuristic: single image-like field
        if len(X_keys) == 1 and X_keys[0] in _IMG_KEYS:
            X_list = ds[X_keys[0]]
            # Flatten each sample (handles HxW or HxWxC)
            X = np.array([np.asarray(x).ravel() for x in X_list], dtype=np.float32)
        else:
            cols = []
            for k in X_keys:
                arr = np.asarray(ds[k])
                # Some tabular datasets store feature vectors as lists, leading to
                # ``object`` dtype arrays. Attempt to convert those to a numeric
                # matrix before applying the dtype filter below. If conversion
                # fails, fall back to skipping the column.
                if arr.dtype.kind == "O":
                    try:
                        arr = np.array([
                            np.asarray(row, dtype=np.float32).ravel() for row in arr
                        ])
                    except Exception:
                        # Non-numeric object column (e.g. strings), skip it
                        continue

                # Skip obviously non-numeric columns
                if arr.dtype.kind not in "iuifb":  # ints, unsigned, floats, bools
                    continue
                cols.append(arr.reshape(len(arr), -1))
            if not cols:
                # Fallback: at least produce zeros if nothing numeric (unlikely)
                X = np.zeros((len(y), 1), dtype=np.float32)
            else:
                X = np.concatenate(cols, axis=1).astype(np.float32, copy=False)

        if flatten and X.ndim > 2:
            X = X.reshape(X.shape[0], -1).astype(np.float32, copy=False)

        return X, y

    # ---- Build partitioner for training split ----
    if iid:
        part_train = IidPartitioner(num_partitions=int(n_clients))
        # , seed=seed)
    else:
        # Use label column name for partitioning; if not available, defaults to "label"
        part_col = "label"
        if _load_hf_dataset is not None:
            try:
                ds_tmp = _load_hf_dataset(dataset, split=split)
                part_col = _find_label_column(ds_tmp)
            except Exception:
                pass
        part_train = DirichletPartitioner(
            num_partitions=int(n_clients),
            # partition_by="label",
            partition_by=part_col,
            alpha=float(alpha),
            seed=seed,
        )

    fds_train = FederatedDataset(dataset=dataset, partitioners={split: part_train})
    Xp: Dict[int, np.ndarray] = {}
    yp: Dict[int, np.ndarray] = {}
    D = 0
    K = 0

    # Optionally compute global min/max across clients
    global_min: Optional[np.ndarray] = None
    global_max: Optional[np.ndarray] = None

    # ---- Load client train partitions ----
    for cid in range(n_clients):
        ds = fds_train.load_partition(cid, split=split).with_format("numpy")
        ycol = _find_label_column(ds)
        X, y = _to_matrix(ds, ycol)

        # deterministic truncation if requested
        if max_per_client is not None and X.shape[0] > max_per_client:
            rng = np.random.default_rng(seed + cid)
            idx = rng.permutation(X.shape[0])[:int(max_per_client)]
            X = X[idx]
            y = y[idx]

        # gather global stats (pre-normalization)
        if normalize and normalization == "global" and X.size > 0:
            if global_min is None:
                global_min = X.min(axis=0, keepdims=True)
                global_max = X.max(axis=0, keepdims=True)
            else:
                global_min = np.minimum(global_min, X.min(axis=0, keepdims=True))
                global_max = np.maximum(global_max, X.max(axis=0, keepdims=True))

        Xp[int(cid)] = X.astype(np.float32, copy=False)
        yp[int(cid)] = y.astype(int, copy=False)
        if X.size > 0:
            D = int(X.shape[1])
        if y.size > 0:
            K = max(K, int(np.max(y) + 1))

    # Apply normalization
    if normalize and normalization in {"client", "global"}:
        if normalization == "global":
            # if no data, skip
            if global_min is not None and global_max is not None:
                denom = np.maximum(1e-8, (global_max - global_min))
                for cid in list(Xp.keys()):
                    X = Xp[cid]
                    if X.size > 0:
                        Xp[cid] = ((X - global_min) / denom).astype(np.float32, copy=False)
        else:  # per-client
            for cid, X in Xp.items():
                if X.size == 0:
                    continue
                x_min = X.min(axis=0, keepdims=True)
                x_max = X.max(axis=0, keepdims=True)
                denom = np.maximum(1e-8, (x_max - x_min))
                Xp[cid] = ((X - x_min) / denom).astype(np.float32, copy=False)

    # ---- Build eval set (single IID partition) ----
    from flwr_datasets.partitioner import IidPartitioner as _Iid  # local alias
    fds_eval = FederatedDataset(dataset=dataset, partitioners={eval_split: _Iid(num_partitions=1)})
                                                                                # , seed=seed + 997)})
    ds_eval = fds_eval.load_partition(0, split=eval_split).with_format("numpy")
    ycol_e = _find_label_column(ds_eval)
    X_eval, y_eval = _to_matrix(ds_eval, ycol_e)

    # Apply normalization to eval
    if normalize and normalization in {"client", "global"} and X_eval.size > 0:
        if normalization == "global" and global_min is not None and global_max is not None:
            denom_e = np.maximum(1e-8, (global_max - global_min))
            X_eval = ((X_eval - global_min) / denom_e).astype(np.float32, copy=False)
        else:
            # per-eval-split normalization
            x_min_e = X_eval.min(axis=0, keepdims=True)
            x_max_e = X_eval.max(axis=0, keepdims=True)
            denom_e = np.maximum(1e-8, (x_max_e - x_min_e))
            X_eval = ((X_eval - x_min_e) / denom_e).astype(np.float32, copy=False)

    # Final shape-derived fallbacks
    if D == 0 and X_eval.size > 0:
        D = int(X_eval.shape[1])
    if K == 0 and y_eval.size > 0:
        K = int(np.max(y_eval) + 1)

    return Xp, yp, D, K, X_eval.astype(np.float32, copy=False), y_eval.astype(int, copy=False)
