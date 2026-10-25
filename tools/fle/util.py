"""Small helpers shared by the fle commands."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

MiB = 1024 * 1024
GiB = 1024 * MiB


class FleError(Exception):
    """A problem the user can fix. The message is the whole story; hint adds the next step."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


def human_bytes(n: float | int | None) -> str:
    if n is None:
        return "?"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


def human_int(n: int | float | None) -> str:
    if n is None:
        return "?"
    return f"{int(n):,}"  # English convention: 32,768


def prepare_run_env(env: dict[str, str]) -> None:
    """Create what a plan's environment points at. Plans only compute; this runs just before a server starts."""
    target = env.get("FLE_MOE_PROFILE")
    if target:
        try:
            Path(target).expanduser().parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # the engine reports an unwritable profile path itself


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def run(
    args: Iterable[str],
    *,
    timeout: float = 10.0,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(cwd) if cwd else None,
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(list(args), 127, "", str(exc))


def which(name: str) -> str:
    found = shutil.which(name)
    return found or ""


def as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return default


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def redact(value: Any) -> Any:
    """Remove secrets before a report leaves the machine."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            upper = str(key).upper()
            if any(word in upper for word in ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL")):
                out[key] = "<redacted>"
            else:
                out[key] = redact(item)
        return out
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=True, indent=2, sort_keys=False, default=str))


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def format_table(rows: list[tuple[str, str]], indent: int = 0) -> str:
    if not rows:
        return ""
    width = max(len(left) for left, _ in rows)
    pad = " " * indent
    return "\n".join(f"{pad}{left.ljust(width)}  {right}" for left, right in rows)


def model_basename(path: str | Path) -> str:
    p = Path(path)
    name = p.name
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    return name
