from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from ..core.types import ModelUpdate
from ..core.registry import MODEL


def infer_dims(
    partitions: Dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[int, int]:
    """Infer (D, K) from the first non-empty client partition."""
    for _, (X, y, _Xe, _ye) in partitions.items():
        if X.size > 0 and y.size > 0:
            D = int(X.shape[1])
            K = int(np.max(y) + 1)
            return D, K
    raise ValueError("Cannot infer input_dim/num_classes from empty partitions.")


def train_locally_on_partitions(
    model_name: str,
    partitions: Dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    global_params: dict[str, np.ndarray] | None = None,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.1,
    seed: int = 42,
) -> tuple[list[ModelUpdate], dict[str, np.ndarray]]:
    """Train a registered model locally on each client's partition.

    The input partitions dict maps client id -> (X_train, y_train, X_val, y_val).

    Returns:
        updates: list of ModelUpdate, each with ABSOLUTE params (not deltas)
        reference_global: the global params used as this round's starting point
    """
    D, K = infer_dims(partitions)
    ModelCls = MODEL.get(model_name)

    # Initialize / set reference global
    tmp_model = ModelCls(D, K)
    if global_params is None:
        global_params = tmp_model.init_parameters()
    else:
        tmp_model.set_parameters(global_params)

    updates: list[ModelUpdate] = []
    for cid, (X, y, X_val, y_val) in partitions.items():
        # Fresh local model from the current global
        local = ModelCls(D, K)
        local.set_parameters(global_params)

        # Deterministic, per-client RNG (order-invariant)
        rng_c = np.random.default_rng(seed + int(cid))

        # Train (with optional val)
        # Some models may not accept X_val/y_val — fall back gracefully.
        try:
            metrics = local.fit_local(
                X=X,
                y=y,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                shuffle=True,
                rng=rng_c,
                X_val=X_val,
                y_val=y_val,
            )
        except TypeError:
            metrics = local.fit_local(
                X=X,
                y=y,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                shuffle=True,
                rng=rng_c,
            )

        # ---- Make metrics robust & consistent ----
        m = dict(metrics or {})
        n = float(len(y))

        # Train acc
        if "acc" not in m and "train_acc" not in m:
            try:
                preds = local.predict(X)
                tr_acc = float(np.mean(preds == y)) if y.size > 0 else float("nan")
            except Exception:
                tr_acc = float("nan")
            m["acc"] = tr_acc
            m["train_acc"] = tr_acc
        else:
            # normalize keys
            tr = float(m.get("acc", m.get("train_acc", float("nan"))))
            m["acc"] = tr
            m["train_acc"] = tr

        # Val acc (compute if missing and val provided)
        if "val_acc" not in m:
            if X_val is not None and y_val is not None and X_val.size > 0 and y_val.size > 0:
                try:
                    val_preds = local.predict(X_val)
                    m["val_acc"] = float(np.mean(val_preds == y_val))
                except Exception:
                    m["val_acc"] = float("nan")
            else:
                m["val_acc"] = float("nan")

        # Standard fields
        m.setdefault("n", n)
        m.setdefault("loss", float(m.get("loss", float("nan"))))
        # For scorers that look for eval_acc/claimed_acc
        m.setdefault("eval_acc", float(m.get("val_acc", float("nan"))))
        m.setdefault("claimed_acc", float(m.get("acc", float("nan"))))

        print(
            f"Client {cid} trained: "
            f"loss={m.get('loss', float('nan')):.4f}, "
            f"train_acc={m.get('train_acc', float('nan')):.4f}, "
            f"val_acc={m.get('val_acc', float('nan')):.4f}, "
            f"n={n:.0f}"
        )

        # Build absolute-params update
        upd = ModelUpdate(
            node_id=int(cid),
            params=local.get_parameters(),
            weight=n,
            metrics=m,
            update_type="weights",
        )
        updates.append(upd)

    return updates, global_params
