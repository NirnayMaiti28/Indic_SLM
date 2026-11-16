"""Typed configuration loading with path resolution."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict[str, Any]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        rel = self.get(dotted)
        if rel is None:
            raise KeyError(f"missing path key: {dotted}")
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else ROOT / "config.yaml"
    with p.open(encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))
