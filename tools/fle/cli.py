"""Command line entry point for the FLE setup helper."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import traceback
from pathlib import Path
from typing import Any

from . import bench, config, doctor, gguf, hardware, plan as plan_mod, report
from .util import FleError, as_bool, eprint, human_bytes, human_int, print_json

VERSION = "0.1"


def _common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("model", nargs="?", help="model file, first shard, or folder")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--debug", action="store_true", help="show a traceback on unexpected errors")
    p.add_argument("--config", metavar="FILE", help="use another config file")
    p.add_argument("--engine", metavar="FILE", help="path to llama-server")
    p.add_argument("--profile", choices=config.PROFILES, help="off, auto, balanced, max, nvme, debug")
    return p


def _plan_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", help="listen address (default 127.0.0.1)")
    p.add_argument("--port", type=int, help="listen port (default 8080)")
    p.add_argument("--context", type=int, help="context tokens (0 = profile default)")
    p.add_argument("--threads", type=int, help="CPU threads (0 = profile default)")
    p.add_argument("--cache-mib", type=int, dest="gpu_cache_mib", help="GPU expert cache in MiB (0 = automatic)")
    p.add_argument("--fit-target", type=int, dest="fit_target_mib", help="free VRAM beside the cache in MiB")
    p.add_argument("--pinned", choices=("auto", "on", "off"), dest="pinned_load", help="experts in pinned RAM")
    p.add_argument(
        "--spec",
        choices=config.SPECS,
        help="speculation: auto, none, mtp, ngram-mod or ngram-simple (default: auto)",
    )
    p.add_argument("--no-mtp", action="store_true", help="do not use an MTP head even when it is there")
    p.add_argument("--mtp-head", metavar="FILE", help="MTP head GGUF")
    p.add_argument("--mtp-vocab", metavar="FILE", help="MTP vocabulary text file")
    p.add_argument("--gpu", dest="gpu_ids", metavar="IDS", help="CUDA_VISIBLE_DEVICES, for example 0 or 0,1")
    p.add_argument("--arg", action="append", default=[], dest="extra_arg", metavar="ARG", help="extra llama-server argument")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="extra engine environment")
    p.add_argument(
        "--fallback",
        action="store_true",
        help="if the FLE plan cannot be built, retry with an installed upstream llama.cpp and profile off",
    )


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    data, _path = config.load(getattr(args, "config", None))
    cli: dict[str, Any] = {}
    for key in (
        "profile",
        "spec",
        "engine",
        "host",
        "port",
        "context",
        "threads",
        "gpu_cache_mib",
        "fit_target_mib",
        "pinned_load",
    ):
        value = getattr(args, key, None)
        if value is not None:
            cli[key] = value
    if getattr(args, "no_mtp", False):
        cli["mtp"] = False
    settings = config.effective(cli, data)
    settings["config_path"] = str(_path)
    return settings


def _model_path(args: argparse.Namespace, settings: dict[str, Any], required: bool = True) -> str:
    path = getattr(args, "model", None) or str(settings.get("model") or "")
    if not path and required:
        raise FleError("No model given.", "Pass MODEL, or run `fle config set model /path/to/model.gguf`.")
    return path


def _env_pairs(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise FleError(f"--env needs KEY=VALUE, got {item!r}")
        out[key.strip()] = value
    return out


def _upstream_engine(explicit: str = "") -> str:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("FLE_UPSTREAM_ENGINE", "").strip()
    if env:
        candidates.append(env)
    candidates += hardware.engine_candidates()
    for candidate in candidates:
        if not Path(candidate).expanduser().is_file():
            continue
        info = hardware.detect_engine(candidate)
        if info.get("ok") and info.get("flavor") == "upstream":
            return candidate
    return ""


def _plan_for(args: argparse.Namespace, settings: dict[str, Any], model_path: str) -> plan_mod.Plan:
    model = gguf.read_model(model_path)
    common = {
        "model_path": str(model.first),
        "extra_args": list(getattr(args, "extra_arg", []) or []),
        "env_override": _env_pairs(list(getattr(args, "env", []) or [])),
        "gpu_ids": str(getattr(args, "gpu_ids", "") or ""),
        "mtp_head": str(getattr(args, "mtp_head", "") or ""),
        "mtp_vocab": str(getattr(args, "mtp_vocab", "") or ""),
    }
    try:
        return plan_mod.build(
            settings,
            model,
            engine_path=str(settings.get("engine") or ""),
            no_mtp=bool(getattr(args, "no_mtp", False)),
            spec=str(getattr(args, "spec", "") or ""),
            **common,
        )
    except FleError as exc:
        if not getattr(args, "fallback", False):
            raise
        requested_spec = str(getattr(args, "spec", "") or settings.get("spec") or "auto").lower()
        if requested_spec not in ("auto", "none"):
            # Do not silently drop an explicit speculation request while falling back.
            raise
        upstream = _upstream_engine(str(settings.get("engine") or ""))
        if not upstream:
            raise FleError(
                f"FLE plan failed: {exc}",
                "No upstream llama.cpp was found. Install one, or pass --engine /path/to/llama-server.",
            ) from None
        fallback = dict(settings)
        fallback.update({"profile": "off", "spec": "none", "engine": upstream, "mtp": False})
        plan = plan_mod.build(
            fallback,
            model,
            engine_path=upstream,
            no_mtp=True,
            spec="none",
            **common,
        )
        plan.fallback = True
        plan.notes.insert(0, f"FLE plan failed: {exc}")
        plan.notes.insert(1, "Falling back to upstream llama.cpp with profile off; FLE features are disabled.")
        return plan


def _print_inspect(model: gguf.Model, show_tensors: bool = False) -> None:
    side = gguf.sidecars(model.first)
    print(f"model:    {model.name}")
    print(f"path:     {model.first}")
    print(f"arch:     {model.arch}")
    print(f"layout:   {model.layout}")
    print(f"parts:    {len(model.shards)}/{len(model.parts)} present" + (f" (missing {len(model.missing)})" if model.missing else ""))
    print(f"size:     {human_bytes(model.size)}")
    print(f"tensors:  {human_int(model.tensor_count)}")
    if model.context_length:
        print(f"context:  {human_int(model.context_length)} tokens")
    if model.layers:
        print(f"layers:   {human_int(model.layers)}")
    if model.experts:
        print(f"experts:  {human_int(model.experts)} per layer, {human_int(model.experts_used)} used")
    print(f"experts:  {human_bytes(model.expert_bytes)}")
    print(f"input:    {human_bytes(model.input_bytes)}")
    print(f"gpu:      {human_bytes(model.gpu_bytes)}")
    print(f"kv q8:    {human_bytes(model.kv_bytes_per_token('q8_0', 'q8_0'))} per token")
    print(f"kv f16:   {human_bytes(model.kv_bytes_per_token('f16', 'f16'))} per token")
    print("types:")
    for name, item in sorted(model.type_histogram.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"  {name:<10} {item['tensors']:>5} tensors  {human_bytes(item['bytes'])}")
    if model.unknown_types:
        print(f"unknown types: {model.unknown_types}")
    for key, value in side.items():
        if value:
            print(f"{key}: {value}")
    if show_tensors:
        print("tensors:")
        for shard in model.shards:
            for tensor in shard.tensors:
                print(f"  {tensor.name:<60} {tensor.type_name:<10} {human_bytes(tensor.nbytes)}")


def _print_plan(plan: plan_mod.Plan) -> None:
    print(f"profile:  {plan.profile}  ({plan_mod.PROFILE_HELP.get(plan.profile, '')})")
    print(f"model:    {plan.model.name} ({human_bytes(plan.model.size)})")
    print(f"alias:    {plan.alias}")
    print(f"engine:   {plan.binary} ({plan.engine.get('flavor')}, {plan.engine.get('backend')})")
    slot_note = f"{plan.slots} slot(s), {'unified KV' if plan.kv_unified else 'separate KV'}"
    print(f"context:  {human_int(plan.context)} tokens total ({human_int(plan.context_per_slot)} per slot; {slot_note})")
    print(f"spec:     {plan.spec}")
    print(f"threads:  {plan.threads}")
    print(f"cache:    {human_int(plan.cache_mib)} MiB")
    print(f"fit:      {human_int(plan.fit_target_mib)} MiB")
    print(f"host:     {plan.host}:{plan.port}")
    print()
    print(plan.command())
    if plan.notes:
        print()
        print("notes:")
        for note in plan.notes:
            print(f"  - {note}")
    if plan.warnings:
        print()
        print("warnings:")
        for warning in plan.warnings:
            print(f"  - {warning}")


def _cmd_inspect(args: argparse.Namespace) -> int:
    settings = _settings(args)
    model = gguf.read_model(_model_path(args, settings))
    if args.json:
        data = model.summary()
        data["sidecars"] = gguf.sidecars(model.first)
        print_json(data)
    else:
        _print_inspect(model, show_tensors=bool(getattr(args, "tensors", False)))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    settings = _settings(args)
    model = gguf.read_model(_model_path(args, settings))
    result = gguf.verify_model(model)
    if args.json:
        print_json(result)
    else:
        print(f"verify: {'ok' if result['ok'] else 'failed'}")
        for check in result["checks"]:
            print(f"[{check['status']}] {check['name']}: {check['detail']}")
            if check.get("hint"):
                print(f"       -> {check['hint']}")
        for warning in result["warnings"]:
            print(f"[warn] {warning}")
        for error in result["errors"]:
            print(f"[fail] {error}")
    return 0 if result["ok"] else 1


def _cmd_doctor(args: argparse.Namespace) -> int:
    settings = _settings(args)
    path = _model_path(args, settings, required=False)
    result = doctor.run(path, settings)
    if args.json:
        print_json(result)
    else:
        doctor.print_report(result)
    if result["status"] == "fail":
        return 1
    if result["status"] == "warn" and getattr(args, "strict", False):
        return 1
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    settings = _settings(args)
    plan = _plan_for(args, settings, _model_path(args, settings))
    if args.json:
        print_json(plan.as_dict())
    else:
        _print_plan(plan)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    settings = _settings(args)
    path = _model_path(args, settings)
    plan = _plan_for(args, settings, path)
    if not args.no_preflight and not plan.fallback:
        result = doctor.run(path, settings)
        if result["status"] == "fail" and not args.force:
            doctor.print_report(result)
            eprint()
            eprint("Refusing to start. Fix the failed checks, or pass --force.")
            return 1
    elif plan.fallback:
        eprint("FLE preflight skipped: falling back to upstream llama.cpp.")
    if args.dry_run:
        if args.json:
            print_json(plan.as_dict())
        else:
            _print_plan(plan)
        return 0
    _print_plan(plan)
    eprint()
    eprint(f"starting {plan.binary}")
    env = os.environ.copy()
    env.update(plan.env)
    try:
        os.execvpe(plan.binary, [plan.binary, *plan.args], env)
    except OSError as exc:
        raise FleError(f"Cannot start {plan.binary}: {exc}", "Check the path and its shared libraries.") from None
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    settings = _settings(args)
    path = _model_path(args, settings, required=False)
    data = report.build(path, settings)
    out = report.write(data, args.out, as_zip=bool(getattr(args, "zip", False)))
    if out != "-":
        print(out)
    return 0 if data["status"] != "fail" else 1


def _cmd_engines(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in hardware.engine_candidates(str(getattr(args, "engine", "") or "")):
        key = str(Path(path).expanduser())
        if key in seen:
            continue
        seen.add(key)
        if not Path(path).expanduser().is_file() and not getattr(args, "all", False):
            continue
        rows.append(hardware.detect_engine(path))
    if args.json:
        print_json({"engines": rows})
        return 0
    if not rows:
        print("No llama-server build found.")
        return 1
    for info in rows:
        if not info.get("ok"):
            print(f"{info.get('path')}  unusable: {info.get('problem') or 'unknown error'}")
            continue
        spec = "spec-type" if info.get("has_spec_mtp") else "no spec-type"
        print(
            f"{info.get('path')}  {info.get('flavor')}  {info.get('backend')}  "
            f"{spec}  {info.get('version') or ''}"
        )
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    prompt = args.prompt
    if getattr(args, "prompt_file", ""):
        try:
            prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            raise FleError(f"Cannot read {args.prompt_file}: {exc}", "Check the path.") from None
    urls = [args.url]
    if getattr(args, "compare", ""):
        urls.append(args.compare)
    results = [
        bench.run(
            url,
            model=args.model_name or "",
            prompt=prompt,
            max_tokens=args.max_tokens,
            runs=args.runs,
            warmup=args.warmup,
            timeout=args.timeout,
        )
        for url in urls
    ]
    result: dict[str, Any] = results[0]
    payload: dict[str, Any] = result if len(results) == 1 else {"primary": result, "compare": results[1]}
    if getattr(args, "output", ""):
        try:
            Path(args.output).expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            raise FleError(f"Cannot write {args.output}: {exc}", "Check the path.") from None
    if args.json:
        print_json(payload)
    else:
        bench.print_result(result)
        if len(results) > 1:
            print()
            bench.print_result(results[1])
            print()
            print(
                f"compare: {result['median_tok_s']:.2f} vs {results[1]['median_tok_s']:.2f} tok/s "
                f"({result['url']} vs {results[1]['url']})"
            )
    return 0


def _parse_value(text: str) -> Any:
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(text, 10)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _cmd_config(args: argparse.Namespace) -> int:
    path = config.config_path(args.config)
    action = args.action
    if action == "path":
        print(path)
        return 0
    if action == "init":
        if path.exists() and not args.force:
            raise FleError(f"{path} already exists", "Use --force to replace it.")
        data, _ = config.load(args.config)
        saved = config.save(data, args.config)
        print(saved)
        return 0
    if action == "edit":
        if not path.exists():
            data, _ = config.load(args.config)
            config.save(data, args.config)
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
        return os.spawnvp(os.P_WAIT, editor, [editor, str(path)])
    data, resolved = config.load(args.config)
    if action == "show":
        settings = config.effective({}, data)
        if args.json:
            print_json({"path": str(resolved), "settings": settings})
        else:
            print(f"path: {resolved}")
            for key, value in settings.items():
                if key != "env":
                    print(f"{key} = {value}")
            if settings.get("env"):
                print("[env]")
                for key, value in settings["env"].items():
                    print(f"{key} = {value}")
        return 0
    if action == "set":
        if not args.key:
            raise FleError("config set needs a key")
        if args.key == "profile" and args.value not in config.PROFILES:
            raise FleError(f"Unknown profile {args.value!r}", f"Use one of: {', '.join(config.PROFILES)}.")
        value = args.value
        if args.key.startswith("env."):
            value = "" if value is None else str(value)
        elif value is not None:
            value = _parse_value(value)
        config.set_dotted(data, args.key, value)
        saved = config.save(data, args.config)
        print(saved)
        return 0
    if action == "unset":
        if not args.key:
            raise FleError("config unset needs a key")
        config.unset_dotted(data, args.key)
        saved = config.save(data, args.config)
        print(saved)
        return 0
    raise FleError(f"Unknown config action {action!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fle", description="Set up and run FLE without memorising the flags.")
    p.add_argument("--version", action="version", version=f"fle {VERSION}")
    sub = p.add_subparsers(dest="command", required=True)
    common = _common()

    inspect = sub.add_parser("inspect", parents=[common], help="read the model header and show what it is")
    inspect.add_argument("--tensors", action="store_true", help="list every tensor")
    inspect.set_defaults(func=_cmd_inspect)

    verify = sub.add_parser("verify", parents=[common], help="check parts, tensor count, offsets and types")
    verify.set_defaults(func=_cmd_verify)

    engines = sub.add_parser("engines", parents=[common], help="list the llama-server builds this helper can find")
    engines.add_argument("--all", action="store_true", help="also show candidate paths that do not exist")
    engines.set_defaults(func=_cmd_engines)

    doc = sub.add_parser("doctor", parents=[common], help="check model, engine, memory and the plan")
    doc.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    doc.set_defaults(func=_cmd_doctor)

    plan = sub.add_parser("plan", parents=[common], help="show the command and the numbers behind it")
    _plan_args(plan)
    plan.set_defaults(func=_cmd_plan)

    serve = sub.add_parser("serve", parents=[common], help="start llama-server with the plan")
    _plan_args(serve)
    serve.add_argument("--dry-run", action="store_true", help="print the plan, do not start")
    serve.add_argument("--no-preflight", action="store_true", help="skip doctor before starting")
    serve.add_argument("--force", action="store_true", help="start even if doctor finds a failure")
    serve.set_defaults(func=_cmd_serve)

    rep = sub.add_parser("report", parents=[common], help="write a redacted diagnostic report")
    rep.add_argument("--out", default="-", help="file to write, - for stdout (default -)")
    rep.add_argument("--zip", action="store_true", help="write a zip with report.json and plan.sh")
    rep.set_defaults(func=_cmd_report)

    b = sub.add_parser("bench", help="measure tokens per second through /v1")
    b.add_argument("--url", default="http://127.0.0.1:8080", help="server base URL")
    b.add_argument("--compare", metavar="URL", help="also benchmark a second running server/model at URL")
    b.add_argument("--model", dest="model_name", default="", help="model name (default: first in /v1/models)")
    b.add_argument("--prompt", default="Write a short paragraph about why local language models are useful.")
    b.add_argument("--prompt-file", metavar="FILE", help="read the prompt from a file")
    b.add_argument("--max-tokens", type=int, default=128)
    b.add_argument("--runs", type=int, default=3)
    b.add_argument("--warmup", type=int, default=1)
    b.add_argument("--timeout", type=float, default=600.0)
    b.add_argument("--output", metavar="FILE", help="write the JSON result to a file")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=_cmd_bench)

    cfg = sub.add_parser("config", help="show or change the global settings")
    cfg.add_argument("--config", dest="config", metavar="FILE", help="use another config file")
    cfg.add_argument("--json", action="store_true")
    cfg_sub = cfg.add_subparsers(dest="action", required=True)
    cfg_sub.add_parser("path", help="print the config path")
    init = cfg_sub.add_parser("init", help="write a default config")
    init.add_argument("--force", action="store_true")
    show = cfg_sub.add_parser("show", help="show the effective settings")
    setp = cfg_sub.add_parser("set", help="set KEY VALUE")
    setp.add_argument("key")
    setp.add_argument("value")
    unset = cfg_sub.add_parser("unset", help="remove KEY")
    unset.add_argument("key")
    cfg_sub.add_parser("edit", help="open the file in $EDITOR")
    cfg.set_defaults(func=_cmd_config)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FleError as exc:
        eprint(f"fle: {exc}")
        if exc.hint:
            eprint(f"hint: {exc.hint}")
        return 2
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        eprint("fle: interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 - last line for a non-traceback CLI
        if getattr(args, "debug", False):
            traceback.print_exc()
        else:
            eprint(f"fle: unexpected error: {exc}")
            eprint("hint: run again with --debug for a traceback.")
        return 2
