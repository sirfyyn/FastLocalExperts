#!/usr/bin/env python3
"""Static checks of the Linux package. Needs binutils (readelf) and, for --ldd, ldd.

  * every ELF file only needs libraries that are in the package, on a normal Linux system,
    or the NVIDIA/CUDA runtime (libcuda.so.1 = driver, libcudart / libcublas / libcublasLt =
    the upstream cudart package that is unpacked next to the binaries)
  * RUNPATH/RPATH is $ORIGIN only (no build directory leaks into the package)
  * the highest glibc / libstdc++ symbol versions that are required (the portability baseline)
  * --ldd: resolve everything with the real dynamic loader; only libcuda.so.1 may be missing
    (and the CUDA runtime libraries unless --cudart-present is given)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SYSTEM_LIBS = re.compile(
    r"^(ld-linux-x86-64\.so\.2|linux-vdso\.so\.1"
    r"|lib(c|m|dl|pthread|rt|util|resolv|nsl)\.so\.\d+"
    r"|libstdc\+\+\.so\.6|libgcc_s\.so\.1|libgomp\.so\.1|libatomic\.so\.1"
    r"|libssl\.so\.3|libcrypto\.so\.3)$"
)
DRIVER_LIB = "libcuda.so.1"


def fail(errors):
    for e in errors:
        print("[elf] ERROR: " + e, flush=True)
    sys.exit(1)


def is_elf(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def sh(cmd, env=None):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", env=env)
    return p.returncode, p.stdout


def dynamic(path):
    rc, out = sh(["readelf", "-d", "-W", str(path)])
    needed, runpath = [], []
    for line in out.splitlines():
        m = re.search(r"\(NEEDED\)\s+Shared library: \[(.+?)\]", line)
        if m:
            needed.append(m.group(1))
        m = re.search(r"\((RUNPATH|RPATH)\)\s+Library (?:runpath|rpath): \[(.*?)\]", line)
        if m:
            runpath.append((m.group(1), m.group(2)))
    return needed, runpath


def version_needs(path):
    rc, out = sh(["readelf", "-V", "-W", str(path)])
    names = set()
    for line in out.splitlines():
        m = re.search(r"Name: (\S+)\s+Flags", line)
        if m:
            names.add(m.group(1))
    return names


def version_tuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="the package directory")
    ap.add_argument("--cuda-major", required=True, help="major version of the CUDA runtime libraries (e.g. 12)")
    ap.add_argument("--max-glibc", default="", help="fail if a higher GLIBC_ version is needed (e.g. 2.35)")
    ap.add_argument("--max-glibcxx", default="", help="fail if a higher GLIBCXX_ version is needed (e.g. 3.4.29)")
    ap.add_argument("--ldd", action="store_true", help="also resolve every file with the dynamic loader")
    ap.add_argument("--cudart-present", action="store_true",
                    help="the CUDA runtime libraries must resolve (upstream cudart package unpacked next to the binaries)")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    cuda_runtime = {"libcudart.so.%s" % args.cuda_major, "libcublas.so.%s" % args.cuda_major,
                    "libcublasLt.so.%s" % args.cuda_major}
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink() and is_elf(p))
    provided = {p.name for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    print("[elf] %d ELF files in %s" % (len(files), root), flush=True)

    errors = []
    external = {}
    maxver = {}
    for f in files:
        needed, runpath = dynamic(f)
        for kind, value in runpath:
            if value != "$ORIGIN":
                errors.append("%s: %s is '%s', expected '$ORIGIN'" % (f.name, kind, value))
        for lib in needed:
            if lib in provided:
                continue
            if lib in cuda_runtime or lib == DRIVER_LIB:
                external.setdefault(lib, []).append(f.name)
            elif SYSTEM_LIBS.match(lib):
                external.setdefault(lib, []).append(f.name)
            else:
                errors.append("%s needs %s, which is neither in the package nor a standard system library" % (f.name, lib))
        for name in version_needs(f):
            m = re.match(r"^([A-Z]+)_(\d[\d.]*)$", name)
            if not m:
                continue
            key, ver = m.group(1), version_tuple(m.group(2))
            if key not in maxver or ver > maxver[key][0]:
                maxver[key] = (ver, name, f.name)

    print("[elf] libraries the package does not contain (system or NVIDIA):", flush=True)
    for lib in sorted(external):
        who = external[lib]
        kind = "driver" if lib == DRIVER_LIB else ("CUDA runtime" if lib in cuda_runtime else "system")
        print("    %-26s %-12s needed by %d file(s), e.g. %s" % (lib, kind, len(who), who[0]), flush=True)

    print("[elf] highest symbol versions required (portability baseline):", flush=True)
    for key in ("GLIBC", "GLIBCXX", "CXXABI", "GCC", "GOMP", "OPENSSL"):
        if key in maxver:
            print("    %-8s %-18s (%s)" % (key, maxver[key][1], maxver[key][2]), flush=True)
    if args.max_glibc and "GLIBC" in maxver and maxver["GLIBC"][0] > version_tuple(args.max_glibc):
        errors.append("needs %s (%s), but the baseline is glibc %s" % (maxver["GLIBC"][1], maxver["GLIBC"][2], args.max_glibc))
    if args.max_glibcxx and "GLIBCXX" in maxver and maxver["GLIBCXX"][0] > version_tuple(args.max_glibcxx):
        errors.append("needs %s (%s), but the baseline is GLIBCXX_%s" % (maxver["GLIBCXX"][1], maxver["GLIBCXX"][2], args.max_glibcxx))

    if args.ldd:
        env = dict(os.environ)
        env.pop("LD_LIBRARY_PATH", None)
        allowed_missing = {DRIVER_LIB}
        if not args.cudart_present:
            allowed_missing |= cuda_runtime
        print("[elf] ldd (no LD_LIBRARY_PATH); allowed to be missing: %s" % ", ".join(sorted(allowed_missing)), flush=True)
        for f in files:
            rc, out = sh(["ldd", str(f)], env=env)
            missing = re.findall(r"^\s*(\S+) => not found", out, re.M)
            bad = [m for m in missing if m not in allowed_missing]
            if f.name.startswith("libggml-cuda") or f.name == "llama-server":
                print("  -- %s" % f.name, flush=True)
                for ln in out.splitlines():
                    print("       " + ln.strip(), flush=True)
            for m in bad:
                errors.append("ldd %s: %s => not found" % (f.name, m))
        if args.cudart_present:
            # the CUDA runtime must now resolve from the package directory itself
            rc, out = sh(["ldd", str(next(p for p in files if p.name.startswith("libggml-cuda")))], env=env)
            for lib in cuda_runtime:
                m = re.search(r"^\s*%s => (\S+)" % re.escape(lib), out, re.M)
                if not m or not m.group(1).startswith(str(root)):
                    errors.append("%s does not resolve to the package directory (%s)" % (lib, m.group(1) if m else "not found"))
                else:
                    print("[elf] %s -> %s" % (lib, m.group(1)), flush=True)

    if errors:
        fail(errors)
    print("[elf] OK", flush=True)


if __name__ == "__main__":
    main()
