# flsim/eval/global.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from ..core.registry import MODEL

def _softmax(z: np.ndarray) -> np.ndarray:
    zmax = np.max(z, axis=1, keepdims=True)
    e = np.exp(z - zmax)
    return e / np.sum(e, axis=1, keepdims=True)

def _ce_loss_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    p = _softmax(logits)
    eps = 1e-12
    return float(-np.mean(np.log(p[np.arange(y.shape[0]), y] + eps)))

def evaluate_global_params(
    model_name: str,
    global_params: Dict[str, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 2048,
    device: str = "mps",
) -> Dict[str, float]:
    """Evaluate a global model (given params) on arrays X,y."""
    if X.size == 0 or y.size == 0:
        return {"n": 0.0, "loss": float("nan"), "acc": float("nan")}
    D = int(X.shape[1])
    K = int(np.max(y) + 1)

    # Convert all global_params to float32 to avoid MPS float64 error
    global_params = {k: v.astype(np.float32) for k, v in global_params.items()}

    ModelCls = MODEL.get(model_name)
    model = ModelCls(D, K)
    model.set_parameters(global_params)

    N = int(X.shape[0])
    preds = np.empty((N,), dtype=int)
    ce_losses = []
    for s in range(0, N, int(batch_size)):
        j = slice(s, min(s + int(batch_size), N))
        logits = model.predict_logits(X[j])
        ce_losses.append(_ce_loss_from_logits(logits, y[j]))
        preds[j] = np.argmax(logits, axis=1)

    acc = float(np.mean(preds == y)) # Overall accuracy indicates how well the model performs on the entire dataset
    loss = float(np.mean(ce_losses))    # Cross-entropy loss indicates how well the predicted probabilities match the true labels, averaged over all batches
    return {"n": float(N), "loss": loss, "acc": acc}

def evaluate_global_on_flower(
    model_name: str,
    global_params: Dict[str, np.ndarray],
    *,
    dataset: str = "cifar10",
    split: str = "test",
    flatten: bool = True,
    normalize: bool = True,
    seed: int = 123,
    batch_size: int = 2048,
) -> Dict[str, float]:
    """Load a single (server) partition from Flower datasets and evaluate."""
    try:
        from flwr_datasets import FederatedDataset
        from flwr_datasets.partitioner import IidPartitioner
    except Exception as e:
        raise ImportError(
            "flwr-datasets not installed. Install extras: `pip install .[data]`"
        ) from e

    from ..data.flower import _find_label_column  # reuse helper

    fds = FederatedDataset(dataset=dataset, partitioners={split: IidPartitioner(num_partitions=1)})
    ds = fds.load_partition(0, split=split).with_format("numpy")
    ycol = _find_label_column(ds)
    y = ds[ycol].astype(int).ravel()

    # Heuristic to build X
    X_keys = [k for k in ds.column_names if k != ycol]
    if len(X_keys) == 1 and X_keys[0] in ("img", "image", "images", "pixel_values") and hasattr(ds, "features") and hasattr(ds.features, "image"):
        X = np.array([x.flatten() for x in ds[X_keys[0]]], dtype=float)
    else:
        cols = [np.asarray(ds[k]).reshape(len(ds[k]), -1) for k in X_keys]
        X = np.concatenate(cols, axis=1).astype(float)

    if normalize and X.size > 0:
        x_min = X.min(axis=0, keepdims=True)
        x_max = X.max(axis=0, keepdims=True)
        denom = np.maximum(1e-8, (x_max - x_min))
        X = (X - x_min) / denom

    if flatten and X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    return evaluate_global_params(model_name, global_params, X, y, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Convenience wrappers for specific models


def evaluate_cnn_cifar(
    model_name: str,
    global_params: Dict[str, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    # global_params: Dict[str, np.ndarray],
    # X: np.ndarray,
    # y: np.ndarray,
    *,
    batch_size: int = 2048,
) -> Dict[str, float]:
    """Evaluate ``cnn_cifar`` global parameters on arrays ``X`` and ``y``."""

    return evaluate_global_params("cnn_cifar", global_params, X, y, batch_size=batch_size)


def evaluate_cnn_mnist(
    global_params: Dict[str, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 2048,
) -> Dict[str, float]:
    """Evaluate ``cnn_mnist`` global parameters on arrays ``X`` and ``y``."""

    return evaluate_global_params("cnn_mnist", global_params, X, y, batch_size=batch_size)


def evaluate_logreg(
    global_params: Dict[str, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 2048,
) -> Dict[str, float]:
    """Evaluate ``logreg`` global parameters on arrays ``X`` and ``y``."""

    return evaluate_global_params("logreg", global_params, X, y, batch_size=batch_size)


def evaluate_cnn_cifar_on_flower(
    global_params: Dict[str, np.ndarray],
    *,
    dataset: str = "cifar10",
    split: str = "test",
    flatten: bool = True,
    normalize: bool = True,
    seed: int = 123,
    batch_size: int = 2048,
) -> Dict[str, float]:
    """Evaluate ``cnn_cifar`` global parameters on a Flower dataset."""

    return evaluate_global_on_flower(
        "cnn_cifar",
        global_params,
        dataset=dataset,
        split=split,
        flatten=flatten,
        normalize=normalize,
        seed=seed,
        batch_size=batch_size,
    )


def evaluate_cnn_mnist_on_flower(
    global_params: Dict[str, np.ndarray],
    *,
    dataset: str = "mnist",
    split: str = "test",
    flatten: bool = True,
    normalize: bool = True,
    seed: int = 123,
    batch_size: int = 2048,
) -> Dict[str, float]:
    """Evaluate ``cnn_mnist`` global parameters on a Flower dataset."""

    return evaluate_global_on_flower(
        "cnn_mnist",
        global_params,
        dataset=dataset,
        split=split,
        flatten=flatten,
        normalize=normalize,
        seed=seed,
        batch_size=batch_size,
    )


def evaluate_logreg_on_flower(
    global_params: Dict[str, np.ndarray],
    *,
    dataset: str = "cifar10",
    split: str = "test",
    flatten: bool = True,
    normalize: bool = True,
    seed: int = 123,
    batch_size: int = 2048,
) -> Dict[str, float]:
    """Evaluate ``logreg`` global parameters on a Flower dataset."""

    return evaluate_global_on_flower(
        "logreg",
        global_params,
        dataset=dataset,
        split=split,
        flatten=flatten,
        normalize=normalize,
        seed=seed,
        batch_size=batch_size,
    )