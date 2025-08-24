from . import selection, settlement, contribution, reward, penalty, reputation  # noqa: F401

# Optional detection module; gracefully handle missing dependencies like torch
try:  # pragma: no cover - import side-effect
    from .detection import flame as detection_flame  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - torch or other deps missing
    detection_flame = None
