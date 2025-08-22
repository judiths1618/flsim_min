from __future__ import annotations
from typing import Any
import numpy as np

from .base import BaseModel, register_model


def _softmax(z: np.ndarray) -> np.ndarray:
    zmax = np.max(z, axis=1, keepdims=True)
    exp = np.exp(z - zmax)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _one_hot(y: np.ndarray, K: int) -> np.ndarray:
    oh = np.zeros((y.shape[0], K), dtype=float)
    oh[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return oh


@register_model("mlp")
class MLP(BaseModel):
    """A simple two-layer MLP implemented with NumPy."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, **kwargs: Any) -> None:
        super().__init__(input_dim, num_classes, **kwargs)
        self.hidden_dim = int(hidden_dim)
        self.W1 = np.zeros((self.input_dim, self.hidden_dim), dtype=float)
        self.b1 = np.zeros((self.hidden_dim,), dtype=float)
        self.W2 = np.zeros((self.hidden_dim, self.num_classes), dtype=float)
        self.b2 = np.zeros((self.num_classes,), dtype=float)

    def init_parameters(self) -> dict[str, np.ndarray]:
        scale1 = 1.0 / np.sqrt(max(1, self.input_dim))
        scale2 = 1.0 / np.sqrt(max(1, self.hidden_dim))
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * scale1
        self.b1 = np.zeros((self.hidden_dim,), dtype=float)
        self.W2 = np.random.randn(self.hidden_dim, self.num_classes) * scale2
        self.b2 = np.zeros((self.num_classes,), dtype=float)
        return self.get_parameters()

    def get_parameters(self) -> dict[str, np.ndarray]:
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def set_parameters(self, params: dict[str, np.ndarray]) -> None:
        self.W1 = params["W1"].astype(float).copy()
        self.b1 = params["b1"].astype(float).copy()
        self.W2 = params["W2"].astype(float).copy()
        self.b2 = params["b2"].astype(float).copy()

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, X @ self.W1 + self.b1[None, :])
        return h @ self.W2 + self.b2[None, :]

    def _batch_grad(self, X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        z1 = X @ self.W1 + self.b1[None, :]
        h1 = np.maximum(0.0, z1)
        logits = h1 @ self.W2 + self.b2[None, :]
        probs = _softmax(logits)
        Y = _one_hot(y, self.num_classes)
        eps = 1e-12
        loss = -np.mean(np.sum(Y * np.log(probs + eps), axis=1))
        diff = (probs - Y) / X.shape[0]
        gW2 = h1.T @ diff
        gb2 = np.sum(diff, axis=0)
        dh1 = diff @ self.W2.T
        dz1 = dh1 * (z1 > 0)
        gW1 = X.T @ dz1
        gb1 = np.sum(dz1, axis=0)
        grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
        return loss, grads

    def fit_local(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.1,
        shuffle: bool = True,
        rng: np.random.Generator | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        if rng is None:
            rng = np.random.default_rng(42)
        N = int(X.shape[0])
        if N == 0:
            return {"loss": float("nan"), "acc": float("nan"), "n": 0.0}
        for _ in range(int(epochs)):
            idx = np.arange(N)
            if shuffle:
                rng.shuffle(idx)
            for s in range(0, N, int(batch_size)):
                j = idx[s:s+int(batch_size)]
                loss, grads = self._batch_grad(X[j], y[j])
                self.W1 -= lr * grads["W1"]
                self.b1 -= lr * grads["b1"]
                self.W2 -= lr * grads["W2"]
                self.b2 -= lr * grads["b2"]

        logits = self.predict_logits(X)
        preds = np.argmax(_softmax(logits), axis=1)
        acc = float(np.mean(preds == y)) if N > 0 else float("nan")
        loss, _ = self._batch_grad(X, y)

        if X_val is not None and y_val is not None and y_val.size > 0:
            v_logits = self.predict_logits(X_val)
            v_preds = np.argmax(_softmax(v_logits), axis=1)
            v_acc = float(np.mean(v_preds == y_val))
        else:
            v_acc = float("nan")

        metrics = {
            "loss": float(loss),
            "acc": float(acc),
            "train_acc": float(acc),
            "val_acc": v_acc,
            "n": float(N),
        }
        # print("mlp: ", metrics)
        return metrics
