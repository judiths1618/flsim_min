import numpy as np
from typing import Dict, List, Tuple, Any

def to_1d_float(x: Any) -> np.ndarray:
    if isinstance(x, dict):
        parts = [to_1d_float(x[k]) for k in sorted(x.keys())]
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    elif isinstance(x, (list, tuple)):
        parts = [to_1d_float(e) for e in x]
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    else:
        return np.asarray(x, dtype=float).ravel()

def flatten_features(features: Dict[int, Dict[str, Any]]) -> Tuple[List[int], np.ndarray | None]:
    ids, vecs = [], []
    for nid, f in features.items():
        if "flat_update" not in f:
            continue
        v = to_1d_float(f["flat_update"])
        if v.size > 0 and np.isfinite(v).all():
            ids.append(nid)
            vecs.append(v)
    if not vecs:
        return ids, None
    max_len = max(v.size for v in vecs)
    vecs = [np.pad(v, (0, max_len - v.size), constant_values=0.0) if v.size < max_len else v for v in vecs]
    return ids, np.stack(vecs)

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def aggregate_models(model_list: List[Dict[str, np.ndarray]], base_model) -> Any:
    # Assume model_list is a list of state_dicts
    import copy
    agg = copy.deepcopy(base_model.state_dict())
    for k in agg:
        agg[k] = sum(m[k] for m in model_list) / len(model_list)
    base_model.load_state_dict(agg)
    return base_model

def evaluate_on_trigger_data(model, dataloader) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in dataloader:
        with torch.no_grad():
            preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    return correct / total

def evaluate_on_clean_data(model, dataloader) -> float:
    return evaluate_on_trigger_data(model, dataloader)  # same logic
