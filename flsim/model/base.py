
from __future__ import annotations
from typing import Any
import numpy as np
from ..core.registry import MODEL

class BaseModel:
    """Abstract base class for local models.
    Implementations must provide parameters as a dict[str, np.ndarray].
    """
    def __init__(self, input_dim: int, num_classes: int, **kwargs: Any) -> None:
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

    def init_parameters(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def get_parameters(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def set_parameters(self, params: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(X)
        return np.argmax(logits, axis=1)

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
        raise NotImplementedError

def register_model(name: str):
    return MODEL.register(name)
