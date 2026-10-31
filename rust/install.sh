#!/bin/sh
# fle installer for macOS and Linux (POSIX sh).
#
#   curl -fsSL https://raw.githubusercontent.com/sirfyyn/FastLocalExperts/main/rust/install.sh | sh
#
# What it does: downloads ONE file (the `fle` program for this machine), checks its SHA-256, puts it in ~/.local/bin and
# asks how you want to use FLE. No Python, no uv, no git, no sudo; it writes only below your home directory.
#
# Options:  --yes (no questions)   --bin-dir DIR   --mode terminal|studio|both
# Environment: FLE_MODE, FLE_BIN_DIR, FLE_REPO (default sirfyyn/FastLocalExperts), FLE_URL_BASE (default: the newest "cli-v*" release)
set -eu

REPO="${FLE_REPO:-sirfyyn/FastLocalExperts}"
BASE="${FLE_URL_BASE:-}"
BIN_DIR="${FLE_BIN_DIR:-$HOME/.local/bin}"
MODE="${FLE_MODE:-}"
YES=0

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y) YES=1 ;;
    --bin-dir) shift; BIN_DIR="${1:?--bin-dir needs a folder}" ;;
    --mode) shift; MODE="${1:?--mode needs terminal, studio or both}" ;;
    -h|--help) sed -n '2,12p' "$0" 2>/dev/null || true; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done

os="$(uname -s)"; arch="$(uname -m)"
case "$os:$arch" in
  Darwin:arm64)           target="aarch64-apple-darwin" ;;
  Darwin:x86_64)          target="x86_64-apple-darwin" ;;
  Linux:x86_64|Linux:amd64) target="x86_64-unknown-linux-gnu" ;;
  *) die "no build for $os $arch yet (macOS arm64/x64 and Linux x64 are available; Windows: install.ps1)" ;;
esac

have curl || die "curl is needed"
if have sha256sum; then sha() { sha256sum "$1" | cut -d' ' -f1; }
elif have shasum;  then sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
else die "sha256sum or shasum is needed to check the download"; fi

if [ -z "$BASE" ]; then
  # The engine's own releases (v1.x) share this repository; the terminal program is released under tags "cli-v*".
  tag="$(curl -fsSL "https://api.github.com/repos/$REPO/releases?per_page=50" | sed -n 's/.*"tag_name": *"\(cli-v[^"]*\)".*/\1/p' | head -1)"
  [ -n "$tag" ] || die "no terminal-program release (cli-v*) found in $REPO yet"
  BASE="https://github.com/$REPO/releases/download/$tag"
fi
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT INT TERM
file="fle-cli-$target.tar.gz"
say "Downloading $file"
curl -fSL --progress-bar -o "$tmp/$file" "$BASE/$file" || die "download failed: $BASE/$file"
curl -fsSL -o "$tmp/$file.sha256" "$BASE/$file.sha256" || die "checksum file missing: $BASE/$file.sha256"
want="$(cut -d' ' -f1 < "$tmp/$file.sha256")"
got="$(sha "$tmp/$file")"
[ "$want" = "$got" ] || die "checksum mismatch (wanted $want, got $got); nothing was installed"
say "Checksum ok"

mkdir -p "$BIN_DIR"
tar -xzf "$tmp/$file" -C "$tmp"
[ -f "$tmp/fle" ] || die "the archive has no 'fle' program"
install -m 0755 "$tmp/fle" "$BIN_DIR/fle"
say "Installed $BIN_DIR/fle"

if [ -z "$MODE" ]; then
  if [ "$YES" = 1 ] || ! ( : < /dev/tty ) 2>/dev/null; then MODE=terminal; else
    printf '\nHow do you want to use FLE?\n  [1] Terminal  (fle)\n  [2] Browser   (FLE Studio)\n  [3] Both\nChoose 1-3 [1]: ' >&2
    read -r a < /dev/tty || a=1
    case "${a:-1}" in 2) MODE=studio ;; 3) MODE=both ;; *) MODE=terminal ;; esac
  fi
fi
case "$MODE" in terminal|studio|both) ;; *) die "mode must be terminal, studio or both" ;; esac

if [ "$MODE" = studio ] || [ "$MODE" = both ]; then
  say "Installing FLE Studio (the browser interface)"
  curl -fsSL https://raw.githubusercontent.com/sirfyyn/fle-studio/main/install.sh | sh -s -- --no-start \
    || printf 'warning: FLE Studio could not be installed; `fle` itself is ready.\n' >&2
fi

case ":$PATH:" in *":$BIN_DIR:"*) ;; *) printf '\nAdd this to your shell profile so you can type `fle`:\n  export PATH="%s:$PATH"\n' "$BIN_DIR" ;; esac
printf '\nDone. Start with:\n'
[ "$MODE" != studio ] && printf '  fle            the terminal interface (first run sets everything up)\n'
[ "$MODE" != terminal ] && printf '  fle-studio     the browser interface\n'
printf '  fle doctor     what this machine can do\n'
