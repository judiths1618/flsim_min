
from __future__ import annotations
# Public surface of flsim.model
from .base import BaseModel, register_model

# Import built-in models so their registration executes on package import
from . import logreg as _logreg  # noqa: F401
from . import nn as _nn  # noqa: F401  # ensure "mlp" registers on import

# Optional PyTorch models; ignore import errors if torch is missing
try:  # pragma: no cover - torch may be unavailable
    from . import nets as _nets  # noqa: F401
except Exception:  # pragma: no cover
    _nets = None

# device = "mps"  # Apple GPU
# model.to(device); tensor.to(device)
