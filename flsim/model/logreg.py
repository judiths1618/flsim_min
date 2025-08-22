
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

@register_model("logreg")
class LogisticRegression(BaseModel):
    """Multinomial logistic regression implemented with NumPy.
    Params:
      W: [D, K], b: [K]
    """
    def __init__(self, input_dim: int, num_classes: int, **kwargs: Any) -> None:
        super().__init__(input_dim, num_classes, **kwargs)
        self.W = np.zeros((self.input_dim, self.num_classes), dtype=float)
        self.b = np.zeros((self.num_classes,), dtype=float)

    def init_parameters(self) -> dict[str, np.ndarray]:
        # Xavier-like small init
        scale = 1.0 / np.sqrt(max(1, self.input_dim))
        self.W = np.random.randn(self.input_dim, self.num_classes) * scale
        self.b = np.zeros((self.num_classes,), dtype=float)
        return self.get_parameters()

    def get_parameters(self) -> dict[str, np.ndarray]:
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_parameters(self, params: dict[str, np.ndarray]) -> None:
        self.W = params["W"].astype(float).copy()
        self.b = params["b"].astype(float).copy()

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b[None, :]

    def _batch_grad(self, X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        logits = self.predict_logits(X)
        probs = _softmax(logits)
        Y = _one_hot(y, self.num_classes)
        # Cross-entropy
        eps = 1e-12
        loss = -np.mean(np.sum(Y * np.log(probs + eps), axis=1))
        # Gradients
        diff = (probs - Y) / X.shape[0]  # [N,K]
        gW = X.T @ diff                   # [D,K]
        gb = np.sum(diff, axis=0)        # [K]
        return loss, {"W": gW, "b": gb}

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
                self.W = self.W - lr * grads["W"]
                self.b = self.b - lr * grads["b"]

        # Train metrics
        logits = self.predict_logits(X)
        preds = np.argmax(_softmax(logits), axis=1)
        acc = float(np.mean(preds == y)) if N > 0 else float("nan")
        loss, _ = self._batch_grad(X, y)

        nan = "unknown"
        # Optional val metrics
        if X_val is not None and y_val is not None and y_val.size > 0:
            v_logits = self.predict_logits(X_val)
            v_preds = np.argmax(_softmax(v_logits), axis=1)
            v_acc = float(np.mean(v_preds == y_val))
            metrics = {"loss": float(loss), "train acc": float(acc), "n": float(N)}
        else:
            metrics = {"loss": float(loss), "train acc": float(acc), "n": float(N)}
        print("logreg: ", metrics)
        return metrics
