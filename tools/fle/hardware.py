"""What the machine has and which llama-server it can run. Read-only, no GPU libraries."""

from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from .util import GiB, MiB, as_int, human_bytes, run

IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"


def _linux_mem() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable", "MemFree"):
                out[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    total = out.get("MemTotal", 0)
    avail = out.get("MemAvailable", out.get("MemFree", 0))
    return {"total": total, "available": avail}


def memory() -> dict[str, int]:
    if IS_MAC:
        total = as_int(run(["sysctl", "-n", "hw.memsize"], timeout=3).stdout, 0)
        return {"total": total, "available": total}
    if IS_WIN:
        return {"total": 0, "available": 0}
    return _linux_mem()


def _linux_physical() -> int:
    try:
        pairs = set()
        phys = core = None
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("physical id"):
                phys = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core = line.split(":", 1)[1].strip()
            elif not line.strip() and phys is not None and core is not None:
                pairs.add((phys, core))
                phys = core = None
        if pairs:
            return len(pairs)
    except OSError:
        pass
    return 0


def cpu() -> dict[str, Any]:
    logical = os.cpu_count() or 1
    physical = 0
    if IS_MAC:
        physical = as_int(run(["sysctl", "-n", "hw.physicalcpu"], timeout=3).stdout, 0)
    elif not IS_WIN:
        physical = _linux_physical()
    if not physical:
        physical = logical
    name = ""
    if IS_MAC:
        name = run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=3).stdout.strip()
    elif not IS_WIN:
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return {"name": name or platform.processor() or "CPU", "physical": physical, "logical": logical}


def gpus() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    proc = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        timeout=8,
    )
    out: list[dict[str, Any]] = []
    if proc.returncode != 0:
        return out
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            out.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total": int(float(parts[2])) * MiB,
                    "memory_used": int(float(parts[3])) * MiB,
                    "memory_free": int(float(parts[4])) * MiB,
                    "vendor": "nvidia",
                }
            )
        except ValueError:
            continue
    return out


def disk_free(path: str | Path) -> int:
    probe = Path(path).expanduser()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


def os_name() -> str:
    if IS_MAC:
        return f"macOS {platform.mac_ver()[0]}"
    if IS_WIN:
        return f"Windows {platform.release()}"
    try:
        text = Path("/etc/os-release").read_text(encoding="utf-8")
        m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', text, re.M)
        if m:
            return m.group(1)
    except OSError:
        pass
    return f"{platform.system()} {platform.release()}"


def summary(model_path: str | Path | None = None) -> dict[str, Any]:
    mem = memory()
    return {
        "os": os_name(),
        "platform": "macos" if IS_MAC else "windows" if IS_WIN else "linux",
        "python": platform.python_version(),
        "cpu": cpu(),
        "memory": mem,
        "memory_human": human_bytes(mem.get("total", 0)),
        "gpus": gpus(),
        "disk_free": disk_free(model_path or Path.home()),
        "disk_free_human": human_bytes(disk_free(model_path or Path.home())),
    }


def port_free(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def engine_candidates(explicit: str = "") -> list[str]:
    out: list[str] = []
    if explicit:
        out.append(explicit)
    env = os.environ.get("FLE_ENGINE", "").strip()
    if env:
        out.append(env)
    found = shutil.which("llama-server")
    if found:
        out.append(found)
    here = Path(__file__).resolve().parents[2]
    for rel in (
        "build/bin/llama-server",
        "build/bin/llama-server.exe",
        "build/bin/Release/llama-server.exe",
    ):
        out.append(str(here / rel))
    seen: set[str] = set()
    unique: list[str] = []
    for item in out:
        key = str(Path(item).expanduser())
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def find_engine(explicit: str = "") -> str:
    for candidate in engine_candidates(explicit):
        p = Path(candidate).expanduser()
        if p.is_file():
            return str(p)
    return ""


def detect_engine(path: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_file():
        return {"ok": False, "path": str(p), "problem": f"No file at {p}"}
    env = os.environ.copy()
    if not IS_WIN:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(x for x in (str(p.parent), env.get("LD_LIBRARY_PATH")) if x)
    ver = run([str(p), "--version"], timeout=60, env=env, cwd=p.parent)
    hlp = run([str(p), "--help"], timeout=60, env=env, cwd=p.parent)
    dev = run([str(p), "--list-devices"], timeout=15, env=env, cwd=p.parent)
    text = (ver.stdout or "") + (ver.stderr or "")
    m = re.search(r"version:\s*(\S+)(?:\s*\(([0-9a-f]+)\))?", text)
    help_text = (hlp.stdout or "") + (hlp.stderr or "")
    devices = (dev.stdout or "") + (dev.stderr or "")
    flavor = "fle" if "--moe-expert-cache-inserts" in help_text else "upstream"
    backend = "cpu"
    blob = (text + " " + help_text + " " + devices).lower()
    if "cuda" in blob:
        backend = "cuda"
    elif "metal" in blob:
        backend = "metal"
    elif "vulkan" in blob:
        backend = "vulkan"
    elif "rocm" in blob or "hip" in blob:
        backend = "rocm"
    ok = ver.returncode == 0 and bool(m)
    return {
        "ok": ok,
        "path": str(p),
        "version": (m.group(1) + (f" ({m.group(2)})" if m.group(2) else "")) if m else "",
        "backend": backend,
        "flavor": flavor,
        "has_fit_target": "--fit-target" in help_text,
        "has_load_mode": "--load-mode" in help_text,
        "has_cmoe": "-cmoe" in help_text or "--cpu-moe" in help_text,
        "has_spec_mtp": "--spec-type" in help_text,
        "problem": "" if ok else ((text.strip().splitlines() or ["no output"])[-1][:300]),
    }


def check_engine_flags(engine: dict[str, Any], profile: str) -> list[str]:
    if not engine.get("ok"):
        return [f"The engine does not start: {engine.get('problem') or 'unknown error'}."]
    out: list[str] = []
    if profile != "off":
        if engine.get("flavor") != "fle":
            out.append("This is not an FLE engine; the FLE cache and profile settings will be ignored or rejected.")
        if not engine.get("has_fit_target"):
            out.append("This engine has no --fit-target; use a newer FLE build.")
        if not engine.get("has_cmoe"):
            out.append("This engine has no -cmoe; experts would not stay on the CPU.")
    return out


def gpu_free_total(gpus_list: list[dict[str, Any]] | None = None) -> int:
    return sum(max(0, int(g.get("memory_free", 0))) for g in (gpus_list or gpus()))
