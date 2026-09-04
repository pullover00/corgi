from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}``/``$VAR`` placeholders in string values.

    Used for host-specific paths (e.g. a local SAM3 checkout) that must not
    be hardcoded into a shared config. An unset variable is left as a
    literal ``${VAR}`` string, which fails loudly and legibly downstream
    rather than silently resolving to an empty path.
    """
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a versioned YAML config and apply optional dotted-key overrides."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported configuration schema")
    config = _expand_env(deepcopy(config))
    for dotted, value in (overrides or {}).items():
        # Turn an override such as ``tracking.visibility_alpha`` into nested
        # dictionary access without requiring a second configuration library.
        cursor = config
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return config
