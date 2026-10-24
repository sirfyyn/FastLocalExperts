"""Global fle settings: one TOML file, defaults below.

CLI flags beat environment variables, environment variables beat this file, and the file beats the built-in profile
defaults. That order is deliberate: a one-off run stays a one-off run, and a config file can be shared.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .util import FleError, as_bool, as_int

PROFILES = ("off", "auto", "balanced", "max", "nvme", "debug")
SPECS = ("auto", "none", "mtp", "ngram-mod", "ngram-simple")

DEFAULTS: dict[str, Any] = {
    "profile": "auto",
    "spec": "auto",
    "engine": "",
    "model": "",
    "host": "127.0.0.1",
    "port": 8080,
    "context": 0,
    "threads": 0,
    "gpu_cache_mib": 0,
    "fit_target_mib": 0,  # 0 = the profile's default (1115 auto, 1536 balanced, 512 max, 1024 nvme)
    "pinned_load": "auto",
    "mtp": True,
    "env": {},
}

ENV_KEYS = {
    "FLE_PROFILE": "profile",
    "FLE_SPEC": "spec",
    "FLE_ENGINE": "engine",
    "FLE_MODEL": "model",
    "FLE_HOST": "host",
    "FLE_PORT": "port",
    "FLE_CONTEXT": "context",
    "FLE_THREADS": "threads",
    "FLE_GPU_CACHE_MIB": "gpu_cache_mib",
    "FLE_FIT_TARGET_MIB": "fit_target_mib",
    "FLE_MTP": "mtp",
}


def config_path(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("FLE_CONFIG", "").strip()
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME", "").strip()
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "fle" / "config.toml"


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    if text[0:1] in ("'", '"') and text[-1:] == text[0]:
        try:
            return json.loads(text) if text[0] == '"' else text[1:-1]
        except ValueError:
            return text[1:-1]
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(text, 10)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_fallback(text: str) -> dict[str, Any]:
    """A small TOML reader for Python without tomllib. Enough for the keys this file uses."""
    out: dict[str, Any] = {}
    section: dict[str, Any] = out
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip().strip('"').strip("'")
            section = out.setdefault(name, {})
            if not isinstance(section, dict):
                section = {}
                out[name] = section
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().strip('"').strip("'")
        section[key] = _parse_scalar(value)
    return out


def _read_toml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib  # type: ignore

        return tomllib.loads(text)
    except ImportError:
        return _parse_fallback(text)
    except ValueError as exc:
        raise FleError(f"Cannot read {path}: {exc}", "Fix the TOML or delete the file to start from defaults.") from None


def load(explicit: str | None = None) -> tuple[dict[str, Any], Path]:
    path = config_path(explicit)
    data = {key: (dict(value) if isinstance(value, dict) else value) for key, value in DEFAULTS.items()}
    if not path.exists():
        return data, path
    raw = _read_toml(path)
    for key, default in DEFAULTS.items():
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(default, dict):
            if isinstance(value, dict):
                data[key] = {str(k): str(v) for k, v in value.items()}
            continue
        if isinstance(default, bool):
            data[key] = as_bool(value, default)
        elif isinstance(default, int):
            data[key] = as_int(value, default)
        elif isinstance(default, str):
            data[key] = "" if value is None else str(value)
    profile = str(data.get("profile") or "auto").lower()
    if profile not in PROFILES:
        raise FleError(f"Unknown profile {profile!r} in {path}", f"Use one of: {', '.join(PROFILES)}.")
    return data, path


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return json.dumps(str(value), ensure_ascii=True)


def save(data: dict[str, Any], explicit: str | None = None) -> Path:
    path = config_path(explicit)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# fle settings. CLI flags > environment > this file > profile defaults.",
        "# Run `fle config show` to see the values in effect.",
        "",
    ]
    for key in DEFAULTS:
        if key == "env":
            continue
        lines.append(f"{key} = {_toml_value(data.get(key, DEFAULTS[key]))}")
    env = data.get("env") or {}
    if isinstance(env, dict) and env:
        lines += ["", "[env]"]
        for key in sorted(env):
            lines.append(f"{key} = {_toml_value(env[key])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def get_dotted(data: dict[str, Any], key: str) -> Any:
    node: Any = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(key)
        node = node[part]
    return node


def set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def unset_dotted(data: dict[str, Any], key: str) -> None:
    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            return
        node = child
    node.pop(parts[-1], None)


def env_settings() -> dict[str, Any]:
    """The few FLE_* variables that set the helper itself. All other FLE_* values go to the engine."""
    out: dict[str, Any] = {}
    for env_key, key in ENV_KEYS.items():
        if env_key not in os.environ:
            continue
        value: Any = os.environ[env_key]
        if key in ("port", "context", "threads", "gpu_cache_mib", "fit_target_mib"):
            value = as_int(value, 0)
        elif key == "mtp":
            value = as_bool(value, True)
        out[key] = value
    return out


def effective(cli: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Merge CLI > environment > file > defaults. CLI values must already be None when not given."""
    out = {key: (dict(value) if isinstance(value, dict) else value) for key, value in DEFAULTS.items()}
    for layer in (data, env_settings(), cli):
        for key, value in layer.items():
            if value is None:
                continue
            if key == "env" and isinstance(value, dict):
                out["env"] = {**out.get("env", {}), **{str(k): str(v) for k, v in value.items()}}
            else:
                out[key] = value
    profile = str(out.get("profile") or "auto").lower()
    if profile not in PROFILES:
        raise FleError(f"Unknown profile {profile!r}", f"Use one of: {', '.join(PROFILES)}.")
    out["profile"] = profile
    return out
