from __future__ import annotations
from typing import Any
import numpy as np

try:
    import torch
    from torch import nn
    import torch, time
    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu")
    print("Using:", device)
    x = torch.randn(8192, 8192, device=device)
    t0=time.time(); y = x @ x; torch.cuda.synchronize() if device=="cuda" else None
    print("OK, elapsed:", time.time()-t0, "s", y.norm().item())

except Exception as e:  # pragma: no cover - torch is optional
    raise RuntimeError("PyTorch is required to use flsim.model.nets") from e


from .base import BaseModel, register_model


class _TorchModel(BaseModel):
    """Base class for PyTorch-backed models following BaseModel API."""

    def __init__(self, input_dim: int, num_classes: int, **kwargs: Any) -> None:
        super().__init__(input_dim, num_classes, **kwargs)
        self.model = self._build_model()

    def _build_model(self) -> nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- parameter handling -------------------------------------------------
    def init_parameters(self) -> dict[str, np.ndarray]:
        def _reset(m: nn.Module) -> None:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        self.model.apply(_reset)
        return self.get_parameters()

    def get_parameters(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().copy() for k, v in self.model.state_dict().items()}

    def set_parameters(self, params: dict[str, np.ndarray]) -> None:
        state = {k: torch.tensor(v) for k, v in params.items()}
        self.model.load_state_dict(state, strict=True)

    # ---- inference ----------------------------------------------------------
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_t)
            return logits.detach().cpu().numpy()

    # ---- training -----------------------------------------------------------
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

        self.model.train()
        optim = torch.optim.SGD(self.model.parameters(), lr=float(lr))
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(int(epochs)):
            idx = np.arange(N)
            if shuffle:
                rng.shuffle(idx)
            for s in range(0, N, int(batch_size)):
                j = idx[s : s + int(batch_size)]
                Xb = torch.tensor(X[j], dtype=torch.float32)
                yb = torch.tensor(y[j], dtype=torch.long)
                optim.zero_grad()
                logits = self.model(Xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optim.step()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
            loss = loss_fn(logits, torch.tensor(y, dtype=torch.long)).item()
            preds = logits.argmax(dim=1).cpu().numpy()
            acc = float(np.mean(preds == y)) if N > 0 else float("nan")

            if X_val is not None and y_val is not None and y_val.size > 0:
                v_logits = self.model(torch.tensor(X_val, dtype=torch.float32))
                v_preds = v_logits.argmax(dim=1).cpu().numpy()
                v_acc = float(np.mean(v_preds == y_val))
            else:
                v_acc = float("nan")

        metrics = {
            "loss": float(loss),
            "acc": acc,
            "train_acc": acc,
            "val_acc": v_acc,
            "n": float(N),
        }
        return metrics


@register_model("torchmlp")
class MLP(_TorchModel):
    """Two-layer MLP implemented with PyTorch."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, **kwargs: Any) -> None:
        self.hidden_dim = int(hidden_dim)
        super().__init__(input_dim, num_classes, **kwargs)

    def _build_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )


@register_model("cnn_mnist")
class CNNMnist(_TorchModel):
    """Simple CNN suitable for MNIST-sized (28x28) images."""

    def __init__(self, input_dim: int, num_classes: int, **kwargs: Any) -> None:
        super().__init__(input_dim, num_classes, **kwargs)

    def _build_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Unflatten(1, (1, 28, 28)),
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )




@register_model("cnn_cifar")
class CNNCifar(_TorchModel):
    """Convolutional network tailored for CIFAR-10 images."""

    def __init__(self, input_dim: int, num_classes: int, **kwargs: Any) -> None:
        super().__init__(input_dim, num_classes, **kwargs)

    def _build_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Unflatten(1, (3, 32, 32)),
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes),
        )
    


class _BasicBlock(nn.Module):
    """Minimal ResNet basic block used for ResNet18Light."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.downsample(identity)
        out = self.relu(out)
        return out


class _ResNet18Light(nn.Module):
    """Lightweight ResNet-18 style network for CIFAR-10."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.unflatten = nn.Unflatten(1, (3, 32, 32))
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(
        self, in_channels: int, out_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [_BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(_BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unflatten(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


@register_model("resnet18_light")
class ResNet18Light(_TorchModel):
    """ResNet-18 style network with reduced width for CIFAR-10."""

    def __init__(self, input_dim: int, num_classes: int, **kwargs: Any) -> None:
        super().__init__(input_dim, num_classes, **kwargs)

    def _build_model(self) -> nn.Module:
        return _ResNet18Light(self.num_classes)