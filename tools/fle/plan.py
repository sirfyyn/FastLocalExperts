"""Turn a model, a profile and this machine into one llama-server command.

The profile names are the same in the fle CLI and in FLE Studio: off, auto, balanced, max, nvme, debug.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config, gguf, hardware
from .util import FleError, GiB, MiB, as_bool, as_int, human_bytes, human_int

PROFILE_HELP = {
    "off": "FLE features off; a plain llama.cpp baseline for comparisons.",
    "auto": "Size the GPU expert cache and memory from this machine.",
    "balanced": "Leave more free memory; a little slower, less likely to run out.",
    "max": "Use most free GPU memory for the expert cache; needs a quiet machine.",
    "nvme": "For a model larger than RAM: memory-map from disk and keep the cache small.",
    "debug": "Auto settings plus timing and reuse diagnostics.",
}

PROFILE_CONTEXT = {
    "off": 32768,
    "auto": 32768,
    "balanced": 32768,
    "max": 65536,
    "nvme": 16384,
    "debug": 32768,
}

PROFILE_CACHE_FACTOR = {
    "off": 0.0,
    "auto": 1.0,
    "balanced": 0.70,
    "max": 1.0,
    "nvme": 0.0,
    "debug": 1.0,
}

# GPU memory besides the always-on weights, the KV cache, the expert cache and the fit margin: CUDA context, compute
# buffers and the MTP head. Measured with the reference command on the reference machine (FLE-IQ3_M, 262,144 tokens in
# two slots, -ub 4096 with ubatch lending): 32,623 MiB card - 4,460 always-on - 3,264 KV - 1,115 margin - 17,035 cache
# = 6,749 MiB. The same cautious estimate is used for other settings.
ENGINE_OVERHEAD_MIB = 6749
DEFAULT_FIT_TARGET_MIB = 1115


@dataclass
class Plan:
    profile: str
    engine: dict[str, Any]
    binary: str
    args: list[str]
    env: dict[str, str]
    host: str
    port: int
    model: gguf.Model
    sidecars: dict[str, str | None]
    context: int
    slots: int
    kv_unified: bool
    context_per_slot: int
    spec: str
    alias: str
    threads: int
    fit_target_mib: int
    cache_mib: int
    fallback: bool = False
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    estimate: dict[str, Any] = field(default_factory=dict)

    def command(self) -> str:
        if os.name == "nt":
            lines = [f"set {k}={v}" for k, v in self.env.items()]
            lines.append(" ".join([self.binary, *self.args]))
            return "\r\n".join(lines)
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
        cmd = " ".join(shlex.quote(x) for x in [self.binary, *self.args])
        return f"{env} \\\n  {cmd}" if env else cmd

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "profile_help": PROFILE_HELP.get(self.profile, ""),
            "engine": self.engine,
            "binary": self.binary,
            "args": self.args,
            "env": self.env,
            "command": self.command(),
            "host": self.host,
            "port": self.port,
            "context": self.context,
            "slots": self.slots,
            "kv_unified": self.kv_unified,
            "context_per_slot": self.context_per_slot,
            "spec": self.spec,
            "alias": self.alias,
            "fallback": self.fallback,
            "threads": self.threads,
            "fit_target_mib": self.fit_target_mib,
            "cache_mib": self.cache_mib,
            "model": self.model.summary(),
            "sidecars": self.sidecars,
            "notes": self.notes,
            "warnings": self.warnings,
            "estimate": self.estimate,
        }


def _threads(profile: str, requested: int) -> int:
    if requested > 0:
        return requested
    cpu = hardware.cpu()
    physical = int(cpu.get("physical") or 1)
    logical = int(cpu.get("logical") or physical)
    if logical > physical:
        return max(1, physical)
    return max(1, physical - 1)


def _context(profile: str, model: gguf.Model, requested: int) -> int:
    default = PROFILE_CONTEXT.get(profile, 32768)
    ctx = requested if requested > 0 else default
    cap = int(model.context_length or 0)
    if cap > 0:
        ctx = min(ctx, cap)
    return max(1024, ctx)


def _pinned(profile: str, setting: str, model: gguf.Model, mem: dict[str, int]) -> bool:
    if profile in ("off", "nvme"):
        return False
    if setting == "on":
        return True
    if setting == "off":
        return False
    if os.name == "nt" or sys.platform.startswith("linux"):
        return mem.get("total", 0) - model.expert_bytes >= 24 * GiB
    return False


def _default_profile_dir() -> Path:
    return config.config_path().parent / "profiles"


def _extra_parallel(args: list[str]) -> int:
    """The --parallel value a user passed through --arg, if any."""
    for index, arg in enumerate(args):
        if arg == "--parallel" and index + 1 < len(args):
            return as_int(args[index + 1], 0)
        if arg.startswith("--parallel="):
            return as_int(arg.split("=", 1)[1], 0)
    return 0


def _stable_alias(model: gguf.Model, model_path: str) -> str:
    """A model id that stays useful when the GGUF's general.name is just a quant type."""
    name = str(model.name or "").strip()
    generic = re.match(r"^(bf16|f16|f32|q[0-9].*|iq[0-9].*|mxfp[0-9].*)$", name, re.I)
    if name and not generic:
        return name
    stem = Path(model_path or model.first).stem.strip()
    return stem or "model"


def _cache_budget(
    profile: str,
    requested: int,
    model: gguf.Model,
    ctx: int,
    fit_target: int,
    gpus: list[dict[str, Any]],
    notes: list[str],
    warnings: list[str],
) -> tuple[int, dict[str, int]]:
    free = sum(max(0, int(g.get("memory_free", 0))) for g in gpus)
    gpu_weights = model.gpu_bytes
    kv = int(model.kv_bytes_per_token("q8_0", "q8_0") * ctx)
    overhead = ENGINE_OVERHEAD_MIB * MiB
    available = max(0, free - gpu_weights - kv - overhead - fit_target * MiB)
    base_mib = available // MiB
    factor = PROFILE_CACHE_FACTOR.get(profile, 1.0)
    auto_mib = int(base_mib * factor)
    if profile == "off":
        cache = 0
    elif requested > 0:
        cache = requested
        if cache > base_mib:
            warnings.append(
                f"The requested cache of {human_int(cache)} MiB is larger than the {human_int(base_mib)} MiB "
                "left after weights and KV cache; loading may fail."
            )
    elif profile == "nvme":
        cache = min(2048, auto_mib)
    else:
        cache = auto_mib
    if profile != "off" and cache > 0 and base_mib <= 0:
        warnings.append("No GPU memory is left for the expert cache; experts will run on the CPU.")
    if profile not in ("off", "nvme") and cache > 0:
        notes.append(
            f"GPU expert cache {cache} MiB: {human_bytes(free)} free, minus {human_bytes(gpu_weights)} weights, "
            f"{human_bytes(kv)} KV at {human_int(ctx)} tokens, {human_bytes(overhead)} overhead and "
            f"{human_int(fit_target)} MiB margin."
        )
    if profile == "nvme":
        notes.append("NVMe profile: the model is memory-mapped, so the filesystem can page experts in on demand.")
    return max(0, cache), {
        "vram_free": free,
        "gpu_weights": gpu_weights,
        "kv_cache": kv,
        "overhead": overhead,
        "fit_target": fit_target * MiB,
        "available": available,
        "base_mib": base_mib,
    }


def build(
    settings: dict[str, Any],
    model: gguf.Model,
    *,
    engine_path: str = "",
    model_path: str = "",
    extra_args: list[str] | None = None,
    env_override: dict[str, str] | None = None,
    gpu_ids: str = "",
    no_mtp: bool = False,
    mtp_head: str = "",
    mtp_vocab: str = "",
    spec: str = "",
) -> Plan:
    profile = str(settings.get("profile") or "auto").lower()
    if profile not in config.PROFILES:
        raise FleError(f"Unknown profile {profile!r}", f"Use one of: {', '.join(config.PROFILES)}.")
    path = engine_path or str(settings.get("engine") or "")
    if not path:
        path = hardware.find_engine()
    if not path:
        raise FleError(
            "No llama-server found.",
            "Pass --engine /path/to/llama-server, set engine in `fle config`, or build this checkout.",
        )
    engine = hardware.detect_engine(path)
    if not engine.get("ok"):
        raise FleError(f"The engine does not start: {engine.get('problem') or path}", "Check the path and its libraries.")
    notes: list[str] = []
    warnings: list[str] = []
    if settings.get("profile_help"):
        notes.append(str(settings["profile_help"]))
    if profile == "off":
        notes.append("FLE features are off. This is the reproducibility baseline.")
    if engine.get("flavor") != "fle" and profile != "off":
        raise FleError(
            "This is not an FLE engine.",
            "Choose an FLE build, or use --profile off for a plain llama.cpp baseline.",
        )
    if engine.get("flavor") == "fle" and profile == "off":
        notes.append("The binary is FLE, but every FLE_* setting is left out for the baseline.")
    cpu = hardware.cpu()
    mem = hardware.memory()
    gpus = hardware.gpus()
    host = str(settings.get("host") or "127.0.0.1")
    port = as_int(settings.get("port"), 8080)
    ctx = _context(profile, model, as_int(settings.get("context"), 0))
    if model.context_length and ctx == int(model.context_length) and as_int(settings.get("context"), 0) > ctx:
        notes.append(f"Context capped at the model maximum of {human_int(ctx)} tokens.")
    threads = _threads(profile, as_int(settings.get("threads"), 0))
    fit_target = as_int(settings.get("fit_target_mib"), 0) or DEFAULT_FIT_TARGET_MIB
    if profile == "balanced" and fit_target == DEFAULT_FIT_TARGET_MIB:
        fit_target = 1536
        notes.append("Balanced keeps a 1536 MiB fit margin (1115 MiB in auto) and 30 percent of the cache free.")
    elif profile == "max" and fit_target == DEFAULT_FIT_TARGET_MIB:
        fit_target = 512
        notes.append("Max uses a 512 MiB fit margin instead of 1115 MiB; stop other GPU work first.")
    elif profile == "nvme" and fit_target == DEFAULT_FIT_TARGET_MIB:
        fit_target = 1024
    if profile in ("off",):
        fit_target = 0

    if not gpus and engine.get("flavor") == "fle":
        raise FleError("The FLE engine needs an NVIDIA GPU, and none was found.", "Install the driver, or use an upstream CPU build with --profile off.")
    chosen_gpus = [g for g in gpus if not gpu_ids or str(g.get("index")) in {x.strip() for x in gpu_ids.split(",")}]
    if gpu_ids and not chosen_gpus:
        raise FleError(f"No GPU with index {gpu_ids}.", "Run nvidia-smi to see the indices.")
    cache_mib, estimate = _cache_budget(profile, as_int(settings.get("gpu_cache_mib"), 0), model, ctx, fit_target, chosen_gpus, notes, warnings)
    estimate.update(
        {
            "model_size": model.size,
            "model_total": model.size,
            "expert_bytes": model.expert_bytes,
            "input_bytes": model.input_bytes,
            "ram_total": mem.get("total", 0),
            "ram_available": mem.get("available", 0),
            "context": ctx,
            "threads": threads,
            "cache_mib": cache_mib,
        }
    )
    if model.missing or model.incomplete_shards:
        warnings.append(
            f"{len(model.missing) + len(model.incomplete_shards)} model part(s) are missing or incomplete; "
            "the size and cache estimate are incomplete."
        )
    if model.size > mem.get("total", 0) + sum(g.get("memory_total", 0) for g in chosen_gpus):
        warnings.append(
            "The model is larger than RAM plus VRAM. Use --profile nvme to stream experts from disk, or add memory."
        )
    if model.expert_bytes and mem.get("available", 0) and model.expert_bytes > mem.get("available", 0):
        warnings.append(
            f"Experts need {human_bytes(model.expert_bytes)} in RAM, but only {human_bytes(mem.get('available', 0))} is free."
        )

    side = gguf.sidecars(model.first)
    head = mtp_head or side.get("mtp_head") or ""
    vocab = mtp_vocab or side.get("mtp_vocab") or ""
    requested_spec = str(spec or settings.get("spec") or "auto").lower()
    if requested_spec not in config.SPECS:
        raise FleError(f"Unknown speculation mode {requested_spec!r}", f"Use one of: {', '.join(config.SPECS)}.")
    if no_mtp and requested_spec == "mtp":
        raise FleError("--no-mtp and --spec mtp contradict each other.", "Pick one, or use --spec auto.")
    if requested_spec in ("ngram-mod", "ngram-simple") and profile == "off":
        raise FleError(
            "The off profile is the reproducibility baseline; it does not add speculation.",
            f"Use --profile auto with --spec {requested_spec}, or keep --profile off.",
        )
    if (head and not vocab) or (vocab and not head):
        warnings.append("MTP needs both a head file and a vocabulary file; MTP stays off.")
    use_mtp = False
    ngram = ""
    if requested_spec == "none":
        notes.append("Speculation is off.")
    elif requested_spec == "mtp":
        if not (head and vocab):
            raise FleError(
                "MTP was requested, but this model has no MTP head and vocabulary file.",
                "Download the MTP sidecars, or use --spec auto / --spec ngram-mod.",
            )
        if profile in ("off", "nvme"):
            warnings.append(f"MTP is not used in the {profile} profile.")
        elif no_mtp:
            warnings.append("MTP is disabled by --no-mtp.")
        else:
            use_mtp = True
    elif requested_spec in ("ngram-mod", "ngram-simple"):
        if not engine.get("has_spec_mtp"):
            raise FleError(
                f"This engine does not support --spec-type, so {requested_spec} cannot be used.",
                "Use an FLE build or an upstream llama.cpp with speculative decoding.",
            )
        ngram = requested_spec
        if profile == "nvme":
            notes.append("Ngram speculation is on in the NVMe profile; it needs no draft model.")
    else:  # auto
        use_mtp = (
            bool(settings.get("mtp", True))
            and not no_mtp
            and profile not in ("off", "nvme")
            and bool(head and vocab)
        )
        if profile == "nvme" and head:
            notes.append("MTP stays off in the NVMe profile; it needs memory that the expert tier can use.")
        elif settings.get("mtp", True) and not no_mtp and not head and profile not in ("off", "nvme"):
            notes.append("MTP is off: no MTP head and vocabulary file were found next to the model. Use --spec ngram-mod for self-speculation.")
    effective_spec = "mtp" if use_mtp else ngram or "none"

    args = ["-m", model_path or str(model.first), "-c", str(ctx), "--jinja"]
    if engine.get("flavor") == "fle":
        args += ["-ngl", "99", "-cmoe"]
    elif model.is_moe:
        args += ["--cpu-moe"]
    args += ["--host", host, "--port", str(port), "--metrics", "-t", str(threads)]
    if profile != "off":
        args += ["-ctk", "q8_0", "-ctv", "q8_0"]
    if engine.get("flavor") == "fle" and profile != "off":
        args += ["-fit", "on", "--fit-target", str(fit_target), "--moe-expert-cache-inserts", "1"]
    if profile == "debug" and engine.get("log_verbosity", True):
        args += ["--log-verbosity", "4"]
    if use_mtp:
        args += [
            "-md",
            head,
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "2",
            "--spec-draft-p-min",
            "0.5",
            "--no-spec-draft-backend-sampling",
            "-cmoed",
        ]
    if ngram:
        args += ["--spec-type", ngram]
        notes.append(f"Self-speculation: {ngram}. It drafts from the prompt/context; acceptance is prompt-dependent.")
    pinned = _pinned(profile, str(settings.get("pinned_load") or "auto"), model, mem)
    if pinned:
        args += ["--load-mode", "none"]
        notes.append("Experts are read into pinned host memory (--load-mode none); this needs the expert bytes as free RAM.")
    if profile == "nvme":
        args += ["--no-warmup"]
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    env: dict[str, str] = {}
    if engine.get("flavor") == "fle" and profile != "off":
        profile_dir = _default_profile_dir()
        try:
            profile_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", model.name).strip("-") or "model"
        env.update(
            {
                "FLE_MOE_CACHE_BUDGET_FIT": "1",
                "FLE_MOE_PROFILE": str(profile_dir / f"{safe_name}.bin"),
                "FLE_MOE_PROFILE_SAVE_SEC": "600",
                "FLE_PREFILL_LEAN": "1",
                "FLE_HC_FUSED": "1",
                "FLE_SLOT_EARLY_SAVE": "1",
            }
        )
        if cache_mib > 0:
            env["FLE_MOE_CACHE_BUDGET_MIB"] = str(cache_mib)
            env["FLE_MC_OVERLAP"] = "2"
        if profile in ("auto", "max", "debug"):
            env["FLE_MOE_CACHE_BIAS"] = "0.001"
        if profile in ("balanced", "max"):
            env["FLE_MOE_ADMIT_FREQ"] = "2000"
            env["FLE_MOE_EVICT_OLDEST"] = "32"
        if profile == "debug":
            env["FLE_TIME_PROFILE"] = "20000"
            env["FLE_REUSE_REASON"] = "1"
            env["FLE_NAN_GUARD"] = "1"
        if use_mtp:
            env["FLE_MTP_VOCAB"] = vocab
            env["FLE_MTP_VOCAB_N"] = "32768"
            env["FLE_MTP_PENALTIES"] = "1"
    env.update({str(k): str(v) for k, v in (settings.get("env") or {}).items()})
    if env_override:
        env.update({str(k): str(v) for k, v in env_override.items()})
    if extra_args:
        args += list(extra_args)
    alias = _stable_alias(model, model_path or str(model.first))
    args += ["--alias", alias]

    parallel = _extra_parallel(args)
    slots = max(1, parallel)
    kv_unified = "--kv-unified" in args
    context_per_slot = ctx if slots == 1 or kv_unified else max(1, ctx // slots)
    if slots > 1 and kv_unified:
        notes.append(f"{slots} slots share one unified KV cache; the context is total, not per slot.")
    elif slots > 1:
        warnings.append(
            f"Explicit --parallel {slots} without --kv-unified: each slot gets about {human_int(context_per_slot)} "
            "tokens and unified KV is off. Add --arg=--kv-unified, or use one slot."
        )
    if not model.experts and profile in ("auto", "balanced", "max", "debug"):
        notes.append("Dense model: the GPU expert cache stays empty; the FLE launch path still applies.")
    if profile == "max" and not model.experts:
        notes.append("Max does not add an expert cache for a dense model; auto/off plus --spec ngram-mod is usually the speed lever.")

    if cache_mib > 0 and profile not in ("off", "nvme"):
        if profile == "balanced":
            notes.append("Balanced keeps 30 percent of the possible cache free for other GPU work and spikes.")
        elif profile == "max":
            notes.append("Max gives the cache all of the possible free memory; other GPU work may not fit.")
    if profile == "debug":
        notes.append("Debug adds timing and graph-reuse counters; it is not a speed profile.")
    if not hardware.port_free(host, port):
        warnings.append(f"Port {port} on {host} is already in use. Stop that server or choose --port.")
    return Plan(
        profile=profile,
        engine=engine,
        binary=path,
        args=args,
        env=env,
        host=host,
        port=port,
        model=model,
        sidecars={"mtp_head": head or None, "mtp_vocab": vocab or None, "mmproj": side.get("mmproj")},
        context=ctx,
        slots=slots,
        kv_unified=kv_unified,
        context_per_slot=context_per_slot,
        spec=effective_spec,
        alias=alias,
        threads=threads,
        fit_target_mib=fit_target,
        cache_mib=cache_mib,
        notes=notes,
        warnings=warnings,
        estimate=estimate,
    )
