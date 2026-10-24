"""`fle fetch-sidecars`: download the MTP head and vocabulary that belong to a known model, next to the model.

Only models with a known source are served. The table below is copied from FLE Studio's catalog
(fle_studio/catalog.py: FLE_REPO, MTP_HEAD, MTP_VOCAB); the sizes and SHA-256 are the Hub's own values. A model that
is not in the table gets "no known source", never a guessed URL.

The download goes through a `fetcher(url, dest_path, headers)` that is injected, so tests never touch the network. Files
are written to a temporary name beside the target, checked for size and SHA-256, and only then renamed into place.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from . import gguf
from .util import FleError, human_bytes

HF = "https://huggingface.co"
USER_AGENT = "fle-helper"
FLE_REPO = "sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF"

Fetcher = Callable[[str, Path, dict[str, str]], None]


@dataclass(frozen=True)
class SidecarFile:
    kind: str  # "mtp_head" or "mtp_vocab": the keys gguf.sidecars() uses
    repo: str
    path: str
    size: int
    sha256: str

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        return f"{HF}/{self.repo}/resolve/main/{quote(self.path)}"


@dataclass(frozen=True)
class Source:
    family: str
    pattern: re.Pattern[str]  # matched against the model file name
    files: tuple[SidecarFile, ...]


KNOWN_SOURCES: tuple[Source, ...] = (
    Source(
        family="Qwen3.8-Flash-Next-FLE",
        pattern=re.compile(r"^Qwen3\.8-Flash-Next-FLE", re.IGNORECASE),
        files=(
            SidecarFile(
                "mtp_head",
                FLE_REPO,
                "mtp/mtp-Qwen3.8-Flash-Next-FLE-IQ4_NL.gguf",
                1528277056,
                "25217613e5e4659186db0b81ce2be77ab5955d01817732e4597b8f2bf0e292fa",
            ),
            SidecarFile(
                "mtp_vocab",
                FLE_REPO,
                "mtp/fle-mtp-vocab-32k.txt",
                281159,
                "8385f0457511fe600986bbd0ecb0b32766cbe31859c12e78a033685c28eef0d8",
            ),
        ),
    ),
)


def find_source(model: gguf.Model) -> Source | None:
    """The known source for this model, matched by its file name, or None."""
    name = model.first.name
    for source in KNOWN_SOURCES:
        if source.pattern.search(name):
            return source
    return None


@dataclass
class Item:
    file: SidecarFile
    dest: Path
    action: str  # "download", "present" or "replace" (present, but --force)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.file.kind,
            "name": self.file.name,
            "url": self.file.url,
            "size": self.file.size,
            "sha256": self.file.sha256,
            "dest": str(self.dest),
            "action": self.action,
        }


def plan_items(model: gguf.Model, source: Source, force: bool = False) -> list[Item]:
    """What to do for each sidecar. A kind that gguf.sidecars() already finds (under any name) counts as present."""
    present = gguf.sidecars(model.first)
    items: list[Item] = []
    for file in source.files:
        dest = model.first.parent / file.name
        if force:
            items.append(Item(file, dest, "replace" if dest.exists() else "download"))
        elif present.get(file.kind):
            items.append(Item(file, Path(str(present[file.kind])), "present"))
        else:
            items.append(Item(file, dest, "download"))
    return items


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def urllib_fetcher(url: str, dest: Path, headers: dict[str, str]) -> None:
    """Stream `url` into `dest`. The only place that uses the network."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, dest.open("wb") as out:
            shutil.copyfileobj(response, out, 1024 * 1024)
    except OSError as exc:
        raise FleError(f"Download failed for {url}: {exc}", "Check the network, or download the file by hand.") from None


def _headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def fetch(
    items: list[Item],
    *,
    force: bool = False,
    fetcher: Fetcher = urllib_fetcher,
    say: Callable[[str], None] = lambda text: None,
) -> list[Item]:
    """Download what is not present. Returns the items that were written. Raises FleError on any mismatch."""
    todo = [item for item in items if item.action != "present"]
    if not todo:
        return []
    folder = todo[0].dest.parent
    needed = sum(item.file.size for item in todo)
    free = shutil.disk_usage(folder).free
    if free < needed + 64 * 1024 * 1024:
        raise FleError(
            f"Not enough disk space in {folder}: {human_bytes(needed)} needed, {human_bytes(free)} free.",
            "Free some space or move the model.",
        )
    written: list[Item] = []
    for item in todo:
        if item.dest.exists() and not force:
            say(f"{item.dest.name}: already there, left alone (use --force to replace)")
            continue
        temp = item.dest.with_name(f".{item.dest.name}.part-{os.getpid()}")
        say(f"{item.dest.name}: downloading {human_bytes(item.file.size)}")
        try:
            fetcher(item.file.url, temp, _headers())
            size = temp.stat().st_size if temp.exists() else -1
            if size != item.file.size:
                raise FleError(
                    f"{item.dest.name}: got {size} bytes, expected {item.file.size}.",
                    "The download was cut short or the file changed upstream. Nothing was written.",
                )
            digest = sha256_of(temp)
            if digest != item.file.sha256:
                raise FleError(
                    f"{item.dest.name}: SHA-256 mismatch ({digest[:12]}... instead of {item.file.sha256[:12]}...).",
                    "Nothing was written. Try again, or download the file by hand.",
                )
            if item.dest.exists() and not force:  # appeared while downloading
                raise FleError(f"{item.dest} appeared during the download.", "It was left alone; use --force to replace it.")
            os.replace(temp, item.dest)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        written.append(item)
    return written
