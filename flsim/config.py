from __future__ import annotations
from typing import Any, Dict
import json

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from .contracts.composed_ours import ContractConfig, ComposedContract
from .contracts.composed_ours_zero import ContractConfig, ComposedContract
from .contracts.composed_flame import ContractConfig, ComposedContract
from .contracts.composed_fedavg import ContractConfig, ComposedContract


def _load_yaml(path: str) -> Dict[str, Any]:
    print(f"load yaml path: {path}")
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    if yaml is not None:
        return yaml.safe_load(txt)  # type: ignore
    return json.loads(txt)

def build_contract_from_dict(cfg: Dict[str, Any]) -> ComposedContract:
    def _get(section: str, default_name: str):
        sec = cfg.get(section, default_name)
        if isinstance(sec, dict):
            return sec.get("name", default_name), sec.get("params", {}) or {}
        return str(sec), {}

    names = {}
    params = {}
    for section, default_name in [
        ("detection", "flame"),
        ("contribution", "metric"),
        ("reward", "default"),
        ("penalty", "default"),
        ("reputation", "default"),
        ("selection", "stratified_softmax"),
        ("settlement", "plans_engine"),
        ("aggregation", "flame_agg"),
    ]:
        n, p = _get(section, default_name)
        names[section] = n
        params[section] = p

    cc = ContractConfig(
        committee_size=int(cfg.get("committee_size", 5)),
        committee_cooldown=int(cfg.get("committee_cooldown", 3)),
        rep_exponent=float(cfg.get("rep_exponent", 1.0)),
        # detection=names["detection"],
        contribution=names["contribution"],
        reward=names["reward"],
        penalty=names["penalty"],
        reputation=names["reputation"],
        selection=names["selection"],
        settlement=names["settlement"],
        aggregation=names["aggregation"],
    )
    return ComposedContract(cc, strategy_params=params)

def build_contract_from_yaml(path: str) -> ComposedContract:
    cfg = _load_yaml(path)
    return build_contract_from_dict(cfg)
