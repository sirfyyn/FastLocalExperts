"""A diagnostic report that a user can attach to a bug report without leaking keys."""

from __future__ import annotations

import datetime as _dt
import json
import zipfile
from pathlib import Path
from typing import Any

from . import config, doctor
from .util import redact

TOOL_VERSION = "0.1"


def build(model_path: str, settings: dict[str, Any]) -> dict[str, Any]:
    report = doctor.run(model_path, settings)
    raw_config, path = config.load()
    report["tool"] = "fle"
    report["tool_version"] = TOOL_VERSION
    report["generated"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    report["config_file"] = str(path)
    report["settings"] = redact({k: v for k, v in settings.items() if k != "env"})
    report["settings"]["env"] = redact(settings.get("env") or {})
    report["file_settings"] = redact(raw_config)
    return report


def write(report: dict[str, Any], out: str = "-", as_zip: bool = False) -> str:
    text = json.dumps(report, ensure_ascii=True, indent=2, default=str)
    if out == "-":
        print(text)
        return "-"
    path = Path(out).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if as_zip or path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("report.json", text)
            plan = report.get("plan") or {}
            command = plan.get("command") or ""
            if command:
                zf.writestr("plan.sh", "#!/usr/bin/env sh\n" + command + "\n")
            zf.writestr("checks.txt", _checks_text(report))
        return str(path)
    path.write_text(text + "\n", encoding="utf-8")
    return str(path)


def _checks_text(report: dict[str, Any]) -> str:
    lines = [f"fle doctor: {report.get('status')}"]
    for item in report.get("checks") or []:
        lines.append(f"[{item.get('status')}] {item.get('name')}: {item.get('detail')}")
        if item.get("hint"):
            lines.append(f"  -> {item['hint']}")
    return "\n".join(lines) + "\n"
