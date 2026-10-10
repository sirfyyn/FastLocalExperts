#!/usr/bin/env python3
"""Static check of the Windows package. Needs `pip install pefile`; no GPU, no NVIDIA driver.

The Windows loader gives no hint which dependency of ggml-cuda.dll is missing, and a runner
without an NVIDIA driver can never load it. So the import tables are checked instead:

  * every DLL a file imports is in the package, a Windows system DLL, the NVIDIA driver
    (nvcuda.dll) or one of the three CUDA runtime DLLs (cudart / cublas / cublasLt) that come
    from the upstream cudart package unpacked next to the binaries
  * the Visual C++ runtime DLLs it imports are bundled (app-local deployment)
  * every function imported from a DLL of the package is exported by that DLL
  * all files are x64
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pefile

VC_RUNTIME = re.compile(r"^(vcruntime140(_1)?|msvcp140(_\w+)?|vcomp140|concrt140|vccorlib140)\.dll$")
API_SET = re.compile(r"^(api|ext)-ms-win-.*\.dll$")
DRIVER = "nvcuda.dll"
MACHINE_AMD64 = 0x8664


def load(path):
    pe = pefile.PE(str(path), fast_load=True)
    pe.parse_data_directories(directories=[
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"],
    ])
    return pe


def imports_of(pe):
    out = []
    for attr in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
        for entry in getattr(pe, attr, []):
            dll = entry.dll.decode("ascii", "replace").lower()
            for imp in entry.imports:
                name = imp.name.decode("ascii", "replace") if imp.name else None
                out.append((dll, name, imp.ordinal))
    return out


def exports_of(pe):
    names, ordinals = set(), set()
    exp = getattr(pe, "DIRECTORY_ENTRY_EXPORT", None)
    if exp:
        for sym in exp.symbols:
            if sym.name:
                names.add(sym.name.decode("ascii", "replace"))
            ordinals.add(sym.ordinal)
    return names, ordinals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="the package directory")
    ap.add_argument("--cuda-major", required=True, help="major version of the CUDA runtime DLLs (e.g. 13)")
    ap.add_argument("--cudart-present", action="store_true",
                    help="the CUDA runtime DLLs of the upstream cudart package are unpacked next to the binaries")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    cuda_runtime = {"cudart64_%s.dll" % args.cuda_major, "cublas64_%s.dll" % args.cuda_major,
                    "cublasLt64_%s.dll" % args.cuda_major}
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in (".dll", ".exe"))
    by_name = {p.name.lower(): p for p in files}
    print("[pe] %d exe/dll files in %s" % (len(files), root), flush=True)

    errors = []
    parsed = {}
    for p in files:
        try:
            pe = load(p)
        except pefile.PEFormatError as exc:
            errors.append("%s: not a valid PE file (%s)" % (p.name, exc))
            continue
        parsed[p.name.lower()] = pe
        if pe.FILE_HEADER.Machine != MACHINE_AMD64:
            errors.append("%s: machine type 0x%x, expected x64" % (p.name, pe.FILE_HEADER.Machine))

    exports = {name: exports_of(pe) for name, pe in parsed.items()}
    external = {}   # dll -> importing files
    checked_symbols = 0
    for name, pe in sorted(parsed.items()):
        for dll, func, ordinal in imports_of(pe):
            if dll in by_name:
                names, ordinals = exports.get(dll, (set(), set()))
                checked_symbols += 1
                if func is not None and func not in names:
                    errors.append("%s imports %s from %s, which that DLL does not export" % (name, func, dll))
                elif func is None and ordinal not in ordinals:
                    errors.append("%s imports ordinal %s from %s, which that DLL does not export" % (name, ordinal, dll))
                continue
            if VC_RUNTIME.match(dll):
                errors.append("%s imports %s, which is not bundled in the package" % (name, dll))
            elif dll in cuda_runtime:
                external.setdefault(dll, set()).add(name)
                if args.cudart_present:
                    errors.append("%s imports %s, but it is not in the package although the cudart package was unpacked" % (name, dll))
            elif dll == DRIVER:
                external.setdefault(dll, set()).add(name)
            elif API_SET.match(dll) or (system32 / dll).exists():
                external.setdefault(dll, set()).add(name)
            else:
                errors.append("%s imports %s, which is neither in the package nor a Windows system DLL" % (name, dll))

    print("[pe] %d imported symbols of package DLLs checked against the exports of those DLLs" % checked_symbols, flush=True)
    print("[pe] DLLs outside the package (Windows, NVIDIA driver, CUDA runtime):", flush=True)
    for dll in sorted(external):
        kind = "driver" if dll == DRIVER else ("CUDA runtime" if dll in cuda_runtime else "Windows")
        who = sorted(external[dll])
        print("    %-28s %-12s %d file(s), e.g. %s" % (dll, kind, len(who), who[0]), flush=True)

    bundled = sorted(n for n in by_name if VC_RUNTIME.match(n))
    imported_vc = sorted({d for imps in (imports_of(pe) for pe in parsed.values()) for d, _, _ in imps if VC_RUNTIME.match(d)})
    print("[pe] bundled Visual C++ runtime DLLs: %s" % ", ".join(bundled), flush=True)
    print("[pe] Visual C++ runtime DLLs actually imported: %s" % ", ".join(imported_vc), flush=True)
    unused = [n for n in bundled if n not in imported_vc]
    if unused:
        print("[pe] note: bundled but not imported by any file: %s" % ", ".join(unused), flush=True)

    cuda = parsed.get("ggml-cuda.dll")
    if cuda is None:
        errors.append("ggml-cuda.dll is missing")
    else:
        print("[pe] ggml-cuda.dll imports from: %s" % ", ".join(sorted({d for d, _, _ in imports_of(cuda)})), flush=True)

    if errors:
        for e in errors:
            print("[pe] ERROR: " + e, flush=True)
        sys.exit(1)
    print("[pe] OK", flush=True)


if __name__ == "__main__":
    main()
