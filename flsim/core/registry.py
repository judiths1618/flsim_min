from __future__ import annotations
from typing import Dict, Any

class Registry:
    def __init__(self) -> None:
        self._map: Dict[str, Any] = {}

    def register(self, name: str):
        def deco(obj):
            key = name.lower()
            if key in self._map:
                raise ValueError(f"Duplicate registry name: {name}")
            self._map[key] = obj
            return obj
        return deco

    def get(self, name: str) -> Any:
        obj = self._map.get(name.lower())
        if obj is None:
            raise KeyError(f"Not found in registry: {name}")
        return obj

    def available(self):
        return sorted(self._map.keys())

AGGREGATION = Registry()
CONTRIB = Registry()
DETECTION = Registry()
REWARD = Registry()
PENALTY = Registry()
REPUTATION = Registry()
SELECTION = Registry()
SETTLEMENT = Registry()

MODEL = Registry()
