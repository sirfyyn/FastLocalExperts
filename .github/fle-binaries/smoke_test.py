#!/usr/bin/env python3
"""Smoke test for an extracted FLE binary package. Needs no GPU and no NVIDIA driver.

With GGML_BACKEND_DL the CUDA backend is a separate module (libggml-cuda.so /
ggml-cuda.dll). Without a driver it cannot be loaded, the loader skips it and the best
CPU variant computes. This script checks exactly that on the package as it is shipped:

  1. llama-server --version (and the commit it was built from)
  2. llama-server --help lists the FLE options (so it is the FLE build, not plain llama.cpp)
  3. llama-server --list-devices: a CPU backend variant is loaded from the package directory
  4. llama-server on a tiny GGUF (CPU only): wait for /health, one /completion request
     with 16 tokens, check the answer, stop the server
  5. optional (--cuda-stub-dir, Linux): with a stub libcuda.so.1 the CUDA module must load,
     i.e. every symbol of it resolves (libggml-base, cudart, cublas, driver API)

Only the Python standard library is used.
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# stories15M (llama2.c, 15M parameters), q4_0, 19 MB. ggml-org moved the repo; the old URL redirects.
MODEL_URLS = (
    "https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/stories15M-q4_0.gguf",
    "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf",
)
MODEL_SHA256 = "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739"

# options that only exist in the FLE fork
FLE_OPTIONS = ("--moe-expert-cache", "--moe-active-experts")

# lines of the server log worth showing in the CI log
INTERESTING = (
    "load_backend",
    "backend",
    "CUDA",
    "system_info",
    "n_threads",
    "model loaded",
    "listening",
    "rror",
    "failed",
)


def log(msg):
    print("[smoke] " + msg, flush=True)


def fail(msg):
    print("[smoke] FAIL: " + msg, flush=True)
    sys.exit(1)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_model(dest):
    if dest.exists() and sha256_of(dest) == MODEL_SHA256:
        log("model already present: %s" % dest)
        return
    last = None
    for attempt in range(1, 7):
        url = MODEL_URLS[(attempt - 1) % len(MODEL_URLS)]
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fle-binaries-smoke"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
                shutil.copyfileobj(resp, out)
            got = sha256_of(dest)
            if got != MODEL_SHA256:
                raise RuntimeError("sha256 mismatch: %s" % got)
            log("model downloaded: %s (%d bytes, sha256 ok)" % (dest.name, dest.stat().st_size))
            return
        except Exception as exc:  # noqa: BLE001 - any failure is retried
            last = exc
            log("model download attempt %d failed: %s" % (attempt, exc))
            time.sleep(5 * attempt)
    fail("cannot fetch the smoke-test model: %s" % last)


def run(cmd, env=None, timeout=180):
    log("run: " + " ".join(str(c) for c in cmd))
    proc = subprocess.run(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=timeout,
        env=env,
        cwd=str(Path(cmd[0]).parent),
    )
    return proc.returncode, proc.stdout


def show(text, limit=60):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for ln in lines[:limit]:
        print("    " + ln, flush=True)
    if len(lines) > limit:
        print("    ... (%d more lines)" % (len(lines) - limit), flush=True)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method, url, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.headers.get("Content-Type", ""), resp.read()


def check_version(server, expect_commit):
    rc, out = run([server, "--version"])
    show(out)
    if rc != 0 or "version:" not in out:
        fail("--version (rc=%d)" % rc)
    if expect_commit and expect_commit[:7] not in out:
        fail("--version does not name the commit %s" % expect_commit[:7])
    log("OK  --version%s" % (" (commit %s)" % expect_commit[:7] if expect_commit else ""))


def check_help(server):
    rc, out = run([server, "--help"])
    if rc != 0:
        fail("--help (rc=%d)" % rc)
    for opt in FLE_OPTIONS:
        if opt not in out:
            fail("%s is missing in --help, this is not the FLE build" % opt)
    log("OK  --help lists %s (FLE build)" % ", ".join(FLE_OPTIONS))


def check_backends(server):
    """The backends are modules in the package directory: at least one CPU variant must load."""
    rc, out = run([server, "--list-devices"])
    show(out)
    if rc != 0:
        fail("--list-devices (rc=%d)" % rc)
    if "loaded CPU backend" not in out:
        fail("no CPU backend module was loaded from the package directory")
    cuda = "loaded" if "loaded CUDA backend" in out else "not loaded (expected without an NVIDIA driver)"
    log("OK  CPU backend variant loaded; CUDA module: %s" % cuda)


def check_cuda_module_loads(server, stub_dir):
    """With a stub libcuda.so.1 the CUDA module can be dlopen()ed even without a GPU."""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(stub_dir)
    rc, out = run([server, "--list-devices"], env=env)
    show(out)
    if "loaded CUDA backend" not in out:
        fail("the CUDA backend module did not load with a stub driver (unresolved symbols or missing libraries?)")
    log("OK  CUDA backend module loads with a stub driver (all symbols resolve)")


def check_server(server, model, log_path, n_predict):
    port = free_port()
    base = "http://127.0.0.1:%d" % port
    cmd = [server, "-m", model, "--host", "127.0.0.1", "--port", str(port),
           "-c", "256", "-np", "1", "-ngl", "0", "-t", "2", "--no-warmup"]
    log("start: " + " ".join(str(c) for c in cmd))
    logf = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([str(c) for c in cmd], stdout=logf, stderr=subprocess.STDOUT,
                            cwd=str(Path(server).parent))
    ok = False
    ui = "unknown"
    err = None
    try:
        deadline = time.time() + 240
        while True:
            if proc.poll() is not None:
                err = "llama-server exited early (rc=%s)" % proc.returncode
                break
            if time.time() > deadline:
                err = "/health did not become ready within 240 s"
                break
            try:
                status, _, body = http("GET", base + "/health", timeout=5)
                if status == 200:
                    log("/health -> %s" % body.decode(errors="replace").strip())
                    break
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(1)

        if err is None:
            status, _, body = http("POST", base + "/completion", {
                "prompt": "Once upon a time",
                "n_predict": n_predict,
                "temperature": 0,
                "seed": 1,
                "cache_prompt": False,
                "stream": False,
            }, timeout=180)
            reply = json.loads(body.decode("utf-8", errors="replace"))
            content = reply.get("content", "")
            n_tok = reply.get("tokens_predicted", 0)
            log("/completion -> %d tokens: %r" % (n_tok, content))
            if status != 200 or not isinstance(content, str) or not content.strip() or n_tok < 1:
                err = "/completion gave no usable answer: %s" % reply
            elif n_tok != n_predict:
                log("note: %d tokens instead of %d (stopped early)" % (n_tok, n_predict))

        if err is None:
            # information only: is the web UI embedded?
            try:
                status, ctype, body = http("GET", base + "/", timeout=10)
                if status == 200 and "html" in ctype.lower():
                    ui = "embedded (%d bytes)" % len(body)
                else:
                    ui = "missing (status %d, %s)" % (status, ctype)
            except urllib.error.HTTPError as exc:
                ui = "missing (HTTP %d)" % exc.code
            except (urllib.error.URLError, OSError) as exc:
                ui = "unknown (%s)" % exc
            log("web UI: %s" % ui)
            ok = True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        logf.close()
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        picked = [ln for ln in text.splitlines() if any(k in ln for k in INTERESTING)]
        log("server log, key lines (%d of %d):" % (min(len(picked), 80), len(text.splitlines())))
        show("\n".join(picked), limit=80)
    if not ok:
        fail(err or "server test")
    log("OK  server /health + /completion on the CPU")
    return ui


def main():
    # CI consoles on Windows are not UTF-8; never die on a printable character
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin-dir", required=True, help="directory of the extracted package")
    ap.add_argument("--work-dir", required=True, help="scratch directory (model, logs)")
    ap.add_argument("--expect-commit", default="", help="commit the binaries must report in --version")
    ap.add_argument("--cuda-stub-dir", help="directory with a stub libcuda.so.1 (Linux only)")
    ap.add_argument("--n-predict", type=int, default=16)
    ap.add_argument("--summary", help="append a short result line to this file (job summary)")
    args = ap.parse_args()

    bin_dir = Path(args.bin_dir).resolve()
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    server = bin_dir / exe
    if not server.exists():
        fail("%s not found" % server)

    log("python %s on %s" % (sys.version.split()[0], sys.platform))
    check_version(server, args.expect_commit)
    check_help(server)
    check_backends(server)
    model = work / "stories15M-q4_0.gguf"
    fetch_model(model)
    ui = check_server(server, model, work / "server.log", args.n_predict)
    if args.cuda_stub_dir:
        check_cuda_module_loads(server, Path(args.cuda_stub_dir).resolve())
    log("ALL OK")
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write("- smoke test: version, FLE options, CPU backend, CPU server (/health, /completion, %d tokens) OK; "
                    "web UI %s%s\n" % (args.n_predict, ui,
                                       "; CUDA module loads with a stub driver" if args.cuda_stub_dir else ""))


if __name__ == "__main__":
    main()
