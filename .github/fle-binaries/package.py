#!/usr/bin/env python3
"""Pack the build/bin directory of an FLE build into a portable archive.

Linux:   <name>.tar.gz  with the top-level directory <name>/ (as the upstream llama.cpp archives)
Windows: <name>.zip     with the files at the root          (as the upstream llama.cpp archives)

Besides the binaries the archive holds the licenses, a README.txt and fle-build-info.json
(commit, toolchain, CUDA architectures that are really inside the CUDA module). The CUDA
runtime (cudart, cublas, cublasLt) is NOT part of it: it comes from the upstream cudart package.
Windows additionally gets the Visual C++ runtime DLLs (app-local deployment).
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

CACHE_KEYS = (
    "CMAKE_BUILD_TYPE", "BUILD_SHARED_LIBS", "GGML_BACKEND_DL", "GGML_NATIVE", "GGML_CPU_ALL_VARIANTS",
    "GGML_CUDA", "GGML_OPENMP", "GGML_CUDA_GRAPHS", "LLAMA_CURL", "LLAMA_OPENSSL", "LLAMA_BUILD_BORINGSSL",
    "LLAMA_BUILD_SERVER", "LLAMA_BUILD_TOOLS", "LLAMA_BUILD_APP", "LLAMA_USE_PREBUILT_UI", "LLAMA_BUILD_UI",
)


def log(msg):
    print("[package] " + msg, flush=True)


def run(cmd, cwd=None):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", cwd=cwd)
        return p.returncode, p.stdout
    except OSError as exc:
        return 127, str(exc)


def natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def commit_time(repo, commit):
    rc, out = run(["git", "show", "-s", "--format=%ct", commit], cwd=repo)
    try:
        return int(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return int(time.time())


def cache_values(cache_file):
    values = {}
    if cache_file and Path(cache_file).exists():
        for line in Path(cache_file).read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^([A-Za-z0-9_]+):[A-Z]+=(.*)$", line)
            if m and m.group(1) in CACHE_KEYS:
                values[m.group(1)] = m.group(2)
    return values


def requested_archs(configure_log):
    if configure_log and Path(configure_log).exists():
        m = re.search(r"Using CMAKE_CUDA_ARCHITECTURES=(\S*)", Path(configure_log).read_text(encoding="utf-8", errors="replace"))
        if m:
            return m.group(1).strip(";")
    return ""


def cuda_archs(cuobjdump, module):
    """Machine code (SASS) and PTX architectures that are really inside the CUDA module."""
    sass, ptx = set(), set()
    if not cuobjdump or not module:
        return [], []
    for flag, target in (("--list-elf", sass), ("--list-ptx", ptx)):
        rc, out = run([cuobjdump, flag, str(module)])
        for m in re.finditer(r"\.sm_(\d+[a-z]?)\.", out):
            target.add("sm_" + m.group(1))
        if rc != 0:
            log("warning: %s %s exited with %d" % (cuobjdump, flag, rc))
    return sorted(sass, key=natural), sorted(ptx, key=natural)


def toolchain(platform):
    info = {}
    nvcc = shutil.which("nvcc")
    if nvcc:
        rc, out = run([nvcc, "--version"])
        m = re.search(r"release (\d+\.\d+), V(\d+\.\d+\.\d+)", out)
        if m:
            info["cuda_release"], info["cuda_version"] = m.group(1), m.group(2)
    if platform == "linux":
        rc, out = run(["gcc", "--version"])
        info["compiler"] = "GCC " + (re.search(r"(\d+\.\d+\.\d+)", out).group(1) if re.search(r"(\d+\.\d+\.\d+)", out) else "?")
    else:
        rc, out = run(["cl"])
        m = re.search(r"Version (\d+\.\d+\.\d+)", out)
        info["compiler"] = "MSVC " + (m.group(1) if m else "?")
    return info


def vc_redist_files():
    """Visual C++ runtime DLLs for app-local deployment (the directory comes from vcvarsall)."""
    redist = os.environ.get("VCToolsRedistDir", "")
    if not redist:
        raise SystemExit("[package] VCToolsRedistDir is not set (run vcvarsall.bat first)")
    files = []
    for pattern in ("x64/Microsoft.VC*.CRT/*.dll", "x64/Microsoft.VC*.OpenMP/vcomp140.dll"):
        files += sorted(glob.glob(os.path.join(redist, pattern)))
    if not files:
        raise SystemExit("[package] no Visual C++ runtime DLLs found below %s" % redist)
    return [Path(f) for f in files]


def readme(platform, name, info):
    major = info["cuda_major"]
    sass = ", ".join(info["cuda_sass"]) or "?"
    ptx = ", ".join(info["cuda_ptx"]) or "?"
    head = (
        "FLE engine - portable binary package\n"
        "====================================\n\n"
        "Package  : %s\n"
        "Commit   : %s (%s)\n"
        "Built    : %s by GitHub Actions %s\n"
        "Toolchain: %s, CUDA toolkit %s\n"
        "GPU code : machine code for %s; PTX (JIT-compiled by the driver for other GPUs) for %s\n"
        "CPU code : every x86-64 feature level as its own CPU backend module; the best one for the\n"
        "           machine is picked at start (no AVX2 -> the FLE CPU kernels are not available)\n\n"
    ) % (name, info["commit"], info["ref"], info["built_at"], info["run_url"] or "",
         info["compiler"], info.get("cuda_version", "?"), sass, ptx)
    if platform == "linux":
        body = (
            "Start\n-----\n"
            "    ./llama-server -m model.gguf --host 127.0.0.1 --port 8080 -ngl 99\n\n"
            "What the machine needs\n----------------------\n"
            "* An NVIDIA driver (libcuda.so.1) for the GPU. Without a driver the CUDA module is skipped\n"
            "  and the engine computes on the CPU.\n"
            "* The CUDA %(m)s runtime libraries libcudart.so.%(m)s, libcublas.so.%(m)s and libcublasLt.so.%(m)s.\n"
            "  They are not part of this package. Either install the CUDA %(m)s runtime, or unpack the matching\n"
            "  upstream package into this directory (the three .so files must end up next to llama-server):\n"
            "      cudart-llama-<tag>-bin-ubuntu-cuda-%(r)s-x64.tar.gz  from  https://github.com/ggml-org/llama.cpp/releases\n"
            "* glibc >= 2.35, libstdc++6, libgomp1 and libssl3 (part of every current Linux distribution).\n\n"
        ) % {"m": major, "r": info["cuda_release"]}
    else:
        body = (
            "Start\n-----\n"
            "    llama-server.exe -m model.gguf --host 127.0.0.1 --port 8080 -ngl 99\n\n"
            "What the machine needs\n----------------------\n"
            "* An NVIDIA driver (nvcuda.dll) for the GPU. Without a driver the CUDA module is skipped\n"
            "  and the engine computes on the CPU.\n"
            "* The CUDA %(m)s runtime DLLs cudart64_%(m)s.dll, cublas64_%(m)s.dll and cublasLt64_%(m)s.dll.\n"
            "  They are not part of this package. Unpack the matching upstream package into this directory:\n"
            "      cudart-llama-bin-win-cuda-%(r)s-x64.zip  from  https://github.com/ggml-org/llama.cpp/releases\n"
            "* The Visual C++ runtime DLLs (msvcp140*.dll, vcruntime140*.dll, vcomp140.dll) are included\n"
            "  (app-local deployment), nothing needs to be installed for them.\n"
            "* Windows 10 or newer, x64.\n\n"
        ) % {"m": major, "r": info["cuda_release"]}
    tail = (
        "Licenses\n--------\n"
        "LICENSE, NOTICE-*.md and the licenses/ directory contain the licenses and attributions of the\n"
        "software in this package (llama.cpp / ggml and the projects named in the NOTICE files).\n"
        "Build recipe: .github/workflows/fle-binaries.yml in https://github.com/sirfyyn/FastLocalExperts\n"
    )
    return head + body + tail


def tar_entries(top, files, extras):
    for src in files:
        yield src, "%s/%s" % (top, src.name)
    for src, arc in extras:
        yield src, "%s/%s" % (top, arc)


def write_tar(out, top, files, extras, mtime):
    with tarfile.open(out, "w:gz", compresslevel=9, format=tarfile.GNU_FORMAT) as tar:
        d = tarfile.TarInfo(top)
        d.type, d.mode, d.mtime, d.uname, d.gname = tarfile.DIRTYPE, 0o755, mtime, "root", "root"
        tar.addfile(d)
        for src, arc in sorted(tar_entries(top, files, extras), key=lambda e: e[1]):
            ti = tar.gettarinfo(str(src), arcname=arc)
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = "root"
            ti.mtime = mtime
            if ti.isreg():
                with open(src, "rb") as f:
                    tar.addfile(ti, f)
            else:
                tar.addfile(ti)


def write_zip(out, files, extras, mtime):
    stamp = time.gmtime(max(mtime, 315532800))[:6]
    entries = [(src, src.name) for src in files] + list(extras)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in sorted(entries, key=lambda e: e[1]):
            zi = zipfile.ZipInfo(arc, date_time=stamp)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            with open(src, "rb") as f, z.open(zi, "w", force_zip64=src.stat().st_size > (1 << 30)) as dest:
                shutil.copyfileobj(f, dest, 1 << 20)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", choices=("linux", "windows"), required=True)
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True, help="e.g. ubuntu-cuda-12.8-x64")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--commit", required=True, help="full commit sha that was built")
    ap.add_argument("--ref", default="", help="the ref that was built (for the README)")
    ap.add_argument("--run-url", default="")
    ap.add_argument("--build-os", default="")
    ap.add_argument("--cuobjdump", default="", help="path of cuobjdump (CUDA architectures inside the module)")
    ap.add_argument("--cmake-cache", default="", help="build/CMakeCache.txt")
    ap.add_argument("--configure-log", default="", help="output of the cmake configure step")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    bin_dir = Path(args.bin_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sha8 = args.commit[:8]
    name = "fle-%s-bin-%s" % (sha8, args.label)
    mtime = commit_time(repo, args.commit)

    # the files of build/bin
    files = []
    for p in sorted(bin_dir.iterdir()):
        if p.is_dir():
            continue
        if args.platform == "linux":
            if p.suffix in (".a", ".o", ".pdb") or p.name.endswith(".debug"):
                continue
        elif p.suffix.lower() not in (".exe", ".dll"):
            continue
        files.append(p)
    if not any(p.name in ("llama-server", "llama-server.exe") for p in files):
        raise SystemExit("[package] llama-server is missing in %s" % bin_dir)
    module = next((p for p in files if re.match(r"^(lib)?ggml-cuda\.(dll|so)(\.[\d.]+)?$", p.name) and not p.is_symlink()), None)
    if module is None:
        raise SystemExit("[package] the CUDA module (ggml-cuda) is missing in %s" % bin_dir)

    # licenses and attributions
    extras = []
    for pattern in ("LICENSE", "NOTICE-*.md"):
        for f in sorted(repo.glob(pattern)):
            extras.append((f, f.name))
    for f in sorted((repo / "licenses").glob("*")):
        extras.append((f, "licenses/" + f.name))
    if args.platform == "windows":
        for f in vc_redist_files():
            if f.name not in {p.name for p in files}:
                extras.append((f, f.name))

    tc = toolchain(args.platform)
    sass, ptx = cuda_archs(args.cuobjdump, module)
    if not sass:
        log("warning: no machine code (SASS) architectures found in %s" % module.name)
    info = {
        "name": name,
        "engine": "FastLocalExperts (FLE), llama.cpp fork",
        "commit": args.commit,
        "ref": args.ref,
        "platform": "linux-x86_64" if args.platform == "linux" else "windows-x86_64",
        "build_os": args.build_os,
        "compiler": tc.get("compiler", "?"),
        "cuda_release": tc.get("cuda_release", ""),
        "cuda_version": tc.get("cuda_version", ""),
        "cuda_major": (tc.get("cuda_release", "0.0").split(".")[0]),
        "cuda_archs_requested": requested_archs(args.configure_log),
        "cuda_sass": sass,
        "cuda_ptx": ptx,
        "cuda_runtime_files": (
            ["libcudart.so.%s", "libcublas.so.%s", "libcublasLt.so.%s"] if args.platform == "linux"
            else ["cudart64_%s.dll", "cublas64_%s.dll", "cublasLt64_%s.dll"]),
        "cmake": cache_values(args.cmake_cache),
        "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_url": args.run_url,
    }
    info["cuda_runtime_files"] = [f % info["cuda_major"] for f in info["cuda_runtime_files"]]

    stage = out_dir / "_stage"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    (stage / "README.txt").write_text(readme(args.platform, name, info), encoding="utf-8", newline="\n")
    (stage / "fle-build-info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")
    extras.append((stage / "README.txt", "README.txt"))
    extras.append((stage / "fle-build-info.json", "fle-build-info.json"))

    if args.platform == "linux":
        out = out_dir / (name + ".tar.gz")
        write_tar(out, name, files, extras, mtime)
    else:
        out = out_dir / (name + ".zip")
        write_zip(out, files, extras, mtime)

    digest = sha256_file(out)
    size = out.stat().st_size
    total = sum(p.stat().st_size for p in files if not p.is_symlink()) + sum(s.stat().st_size for s, _ in extras)
    log("package   : %s" % out.name)
    log("size      : %.1f MiB (unpacked %.1f MiB)" % (size / 2**20, total / 2**20))
    log("sha256    : %s" % digest)
    log("toolchain : %s, CUDA %s" % (info["compiler"], info["cuda_version"] or "?"))
    log("CUDA archs: requested %s" % (info["cuda_archs_requested"] or "(unknown)"))
    log("            machine code: %s" % (", ".join(sass) or "?"))
    log("            PTX         : %s" % (", ".join(ptx) or "?"))
    log("contents  : %d files" % (len(files) + len(extras)))
    for p in sorted(files, key=lambda p: p.name):
        if p.is_symlink():
            log("    %-46s -> %s" % (p.name, os.readlink(p)))
        else:
            log("    %-46s %10.2f MiB" % (p.name, p.stat().st_size / 2**20))
    for src, arc in extras:
        log("    %-46s %10.2f MiB" % (arc, src.stat().st_size / 2**20))
    outputs = {"name": name, "file": out.name, "path": str(out), "sha256": digest, "size": str(size), "sha8": sha8}
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a", encoding="utf-8") as f:
            for k, v in outputs.items():
                f.write("%s=%s\n" % (k, v))

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("### %s\n\n" % out.name)
            f.write("| | |\n|---|---|\n")
            f.write("| size | %.1f MiB (unpacked %.1f MiB) |\n" % (size / 2**20, total / 2**20))
            f.write("| sha256 | `%s` |\n" % digest)
            f.write("| commit | `%s` |\n" % args.commit)
            f.write("| toolchain | %s, CUDA %s |\n" % (info["compiler"], info["cuda_version"] or "?"))
            f.write("| CUDA machine code | %s |\n" % (", ".join(sass) or "?"))
            f.write("| CUDA PTX | %s |\n" % (", ".join(ptx) or "?"))
            f.write("| CUDA runtime to add | %s (upstream cudart package) |\n" % ", ".join(info["cuda_runtime_files"]))
            f.write("\n<details><summary>%d files</summary>\n\n```\n" % (len(files) + len(extras)))
            for p in sorted(files, key=lambda p: p.name):
                f.write("%-46s %s\n" % (p.name, ("-> " + os.readlink(p)) if p.is_symlink() else "%.2f MiB" % (p.stat().st_size / 2**20)))
            for src, arc in extras:
                f.write("%-46s %.2f MiB\n" % (arc, src.stat().st_size / 2**20))
            f.write("```\n</details>\n\n")

    shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
