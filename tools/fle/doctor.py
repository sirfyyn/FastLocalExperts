"""One read-only setup check: model, engine, memory, disk and the command that would start."""

from __future__ import annotations

from typing import Any

from . import config, gguf, hardware, plan as plan_mod
from .util import FleError, GiB, MiB, human_bytes, human_int

STATUS_ORDER = {"ok": 0, "warn": 1, "fail": 2}


def _check(name: str, status: str, detail: str, hint: str = "") -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail, "hint": hint}


def worst(checks: list[dict[str, str]]) -> str:
    status = "ok"
    for item in checks:
        if STATUS_ORDER.get(item["status"], 0) > STATUS_ORDER[status]:
            status = item["status"]
    return status


def run(model_path: str, settings: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    model: gguf.Model | None = None
    verification: dict[str, Any] | None = None
    parsed_plan: plan_mod.Plan | None = None
    system = hardware.summary(model_path)
    profile = str(settings.get("profile") or "auto")
    if not model_path:
        checks.append(_check("model", "warn", "no model selected", "Pass a model, or set model in `fle config`."))
        verification = None
    try:
        if not model_path:
            raise FileNotFoundError("no model selected")
        model = gguf.read_model(model_path)
        verification = gguf.verify_model(model)
        checks.append(_check("model", "ok" if verification["ok"] else "fail",
                             f"{model.name} ({model.arch}, {human_bytes(model.size)})",
                             "" if verification["ok"] else verification["errors"][0]))
        if not model.complete:
            detail = f"{len(model.missing)} part(s) missing"
            if model.incomplete_shards:
                detail += f", {len(model.incomplete_shards)} incomplete"
            checks.append(_check("model parts", "fail", detail, "Wait for the download to finish."))
        else:
            checks.append(_check("model parts", "ok", f"{len(model.parts)} part(s) present"))
        if verification["warnings"]:
            for warning in verification["warnings"]:
                checks.append(_check("model warning", "warn", warning))
        if model.unknown_types:
            checks.append(_check("tensor types", "fail", f"unknown type ids {model.unknown_types}",
                                 "This engine cannot load the file; use the engine that wrote it."))
        else:
            checks.append(_check("tensor types", "ok", f"{len(model.type_histogram)} known types"))
        side = verification["sidecars"]
        if side.get("mtp_head") and side.get("mtp_vocab"):
            checks.append(_check("MTP", "ok", "head and vocabulary found"))
        elif side.get("mtp_head") or side.get("mtp_vocab"):
            checks.append(_check("MTP", "warn", "head or vocabulary is missing", "Download both files, or turn MTP off."))
        else:
            checks.append(_check("MTP", "ok", "off (no MTP files)"))
    except (FleError, OSError) as exc:
        if model_path:
            checks.append(_check("model", "fail", str(exc), getattr(exc, "hint", "")))
        model = None
        verification = None

    engine_path = str(settings.get("engine") or "") or hardware.find_engine()
    engine: dict[str, Any] | None = None
    if not engine_path:
        checks.append(_check("engine", "fail", "no llama-server found",
                             "Set engine in `fle config`, or pass --engine."))
    else:
        engine = hardware.detect_engine(engine_path)
        if engine.get("ok"):
            checks.append(_check("engine", "ok", f"{engine['path']} ({engine.get('flavor')}, {engine.get('backend')})"))
            for problem in hardware.check_engine_flags(engine, profile):
                status = "fail" if profile not in ("off",) else "warn"
                checks.append(_check("engine flags", status, problem))
        else:
            checks.append(_check("engine", "fail", engine.get("problem") or "does not start", "Check the path."))

    gpus = hardware.gpus()
    mem = hardware.memory()
    if profile != "off":
        if not gpus:
            checks.append(_check("GPU", "fail", "no NVIDIA GPU found", "Use an upstream CPU build with --profile off."))
        else:
            free = sum(g.get("memory_free", 0) for g in gpus)
            checks.append(_check("GPU", "ok", f"{len(gpus)} GPU(s), {human_bytes(free)} free"))
            if model:
                kv = int(model.kv_bytes_per_token("q8_0", "q8_0") * plan_mod._context(profile, model, 0))
                need = model.gpu_bytes + kv + plan_mod.ENGINE_OVERHEAD_MIB * MiB
                if need > free:
                    checks.append(_check("GPU memory", "warn",
                                         f"non-expert weights and KV need about {human_bytes(need)}, {human_bytes(free)} free",
                                         "Stop other GPU work, lower the context, or use --profile balanced/nvme."))
                else:
                    checks.append(_check("GPU memory", "ok", f"{human_bytes(free - need)} left for the cache"))
    if model and model.expert_bytes:
        avail = mem.get("available", 0)
        status = "warn" if model.expert_bytes > avail else "ok"
        checks.append(_check("RAM", status, f"experts {human_bytes(model.expert_bytes)}, available {human_bytes(avail)}",
                             "Close other programs, or use --profile nvme." if status == "warn" else ""))
    if model and system["disk_free"] < model.size:
        checks.append(_check("disk", "warn", f"{human_bytes(system['disk_free'])} free for a {human_bytes(model.size)} model"))

    if model:
        try:
            parsed_plan = plan_mod.build(settings, model)
            checks.append(_check("plan", "ok", f"{parsed_plan.profile}, spec {parsed_plan.spec}, "
                                              f"{parsed_plan.cache_mib} MiB cache, "
                                              f"{human_int(parsed_plan.context)} tokens"))
            for warning in parsed_plan.warnings:
                checks.append(_check("plan warning", "warn", warning))
        except (FleError, OSError) as exc:
            checks.append(_check("plan", "fail", str(exc), getattr(exc, "hint", "")))
    else:
        checks.append(_check("plan", "warn", "no model to plan for", "Pass a model or set model in `fle config`."))

    status = worst(checks)
    return {
        "ok": status != "fail",
        "status": status,
        "profile": profile,
        "config": str(config.config_path()),
        "model": model.summary() if model else None,
        "verification": verification,
        "engine": engine,
        "plan": parsed_plan.as_dict() if parsed_plan else None,
        "system": system,
        "checks": checks,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"fle doctor: {report['status']}")
    for item in report["checks"]:
        mark = {"ok": "ok  ", "warn": "warn", "fail": "fail"}[item["status"]]
        print(f"[{mark}] {item['name']}: {item['detail']}")
        if item.get("hint"):
            print(f"       -> {item['hint']}")
    system = report.get("system") or {}
    print()
    print(f"system: {system.get('os')} / Python {system.get('python')}")
    cpu = system.get("cpu") or {}
    print(f"cpu:    {cpu.get('name')} ({cpu.get('physical')} cores, {cpu.get('logical')} threads)")
    print(f"ram:    {human_bytes((system.get('memory') or {}).get('total'))}")
    for gpu in system.get("gpus") or []:
        print(f"gpu {gpu.get('index')}: {gpu.get('name')} {human_bytes(gpu.get('memory_free'))} free of {human_bytes(gpu.get('memory_total'))}")
    print(f"disk:   {human_bytes(system.get('disk_free'))} free")
    if report.get("model"):
        model = report["model"]
        print(f"model:  {model['name']} ({model['arch']}, {model['size_human']}, {model['tensor_count']} tensors)")
    print(f"config: {report['config']}")
