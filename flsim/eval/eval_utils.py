# eval_utils.py
from typing import Dict, Iterable, Tuple
import torch
from torch.utils.data import DataLoader
import numpy as np

def assign_numpy_params_to_model(model: torch.nn.Module, params: Dict[str, np.ndarray], strict: bool = False):
    """将 numpy 参数字典加载进 model。params 的 key 需与 model.state_dict() 对齐或提供映射。"""
    sd = model.state_dict()
    new_sd = {}
    # 尝试直接对齐：若 params 的 key 和 state_dict 一致，最简单
    for k in sd.keys():
        if k in params:
            arr = params[k]
        elif k.endswith(".weight") and k[:-7] in params:
            # 兼容你的简化命名：比如 'W1' 对应 'layer1.weight'
            arr = params[k[:-7]]
        elif k.endswith(".bias") and k[:-5] in params:
            arr = params[k[:-5]]
        else:
            # 保留原值
            new_sd[k] = sd[k]
            continue
        t = torch.from_numpy(np.asarray(arr))
        if t.shape != sd[k].shape:
            raise ValueError(f"Shape mismatch for '{k}': {tuple(t.shape)} vs {tuple(sd[k].shape)}")
        new_sd[k] = t.to(sd[k].dtype)

    # 保留剩余未覆盖的键
    for k in sd.keys():
        if k not in new_sd:
            new_sd[k] = sd[k]

    model.load_state_dict(new_sd, strict=strict)


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, int]:
    """返回 (accuracy, num_samples)。"""
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    acc = (correct / total) if total > 0 else 0.0
    return acc, total


def evaluate_on_clients(model: torch.nn.Module, test_loaders: Dict[int, DataLoader], device: torch.device):
    """对每个客户端 test set 评估并打印；返回宏平均与样本加权平均。"""
    per_client = {}
    total_n = 0
    weighted_sum = 0.0
    for cid, loader in test_loaders.items():
        acc, n = evaluate_model(model, loader, device)
        per_client[cid] = (acc, n)
        total_n += n
        weighted_sum += acc * n

    macro = float(np.mean([a for a, _ in per_client.values()])) if per_client else 0.0
    micro = (weighted_sum / total_n) if total_n > 0 else 0.0

    # 打印
    line = " | ".join([f"client {cid}: {acc*100:.2f}% (n={n})" for cid, (acc, n) in sorted(per_client.items())])
    print(f"[Eval] per-client: {line}")
    print(f"[Eval] macro-avg acc: {macro*100:.2f}% | micro-avg (weighted) acc: {micro*100:.2f}%")
    return per_client, macro, micro
