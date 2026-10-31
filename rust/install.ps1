# fle installer for Windows (PowerShell 5.1+).
#   irm https://raw.githubusercontent.com/sirfyyn/FastLocalExperts/main/rust/install.ps1 | iex
# Downloads ONE file, checks its SHA-256, installs to %LOCALAPPDATA%\FLE\bin and adds it to your user PATH.
# No administrator rights, no Python, no git.
param([string]$Mode = $env:FLE_MODE, [string]$BinDir = $(if ($env:FLE_BIN_DIR) { $env:FLE_BIN_DIR } else { Join-Path $env:LOCALAPPDATA 'FLE\bin' }),
      [string]$Repo = $(if ($env:FLE_REPO) { $env:FLE_REPO } else { 'sirfyyn/FastLocalExperts' }), [string]$Base = $env:FLE_URL_BASE, [switch]$Yes)
$ErrorActionPreference = 'Stop'
if (-not $Base) {
  # The engine releases (v1.x) share this repository; the terminal program is released under tags "cli-v*".
  $rel = (Invoke-RestMethod "https://api.github.com/repos/$Repo/releases?per_page=50") | Where-Object { $_.tag_name -like 'cli-v*' } | Select-Object -First 1
  if (-not $rel) { throw "no terminal-program release (cli-v*) found in $Repo yet" }
  $Base = "https://github.com/$Repo/releases/download/$($rel.tag_name)"
}
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw "no build for $($env:PROCESSOR_ARCHITECTURE) yet (Windows x64 only)" }
$file = 'fle-cli-x86_64-pc-windows-msvc.zip'
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("fle-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
  Write-Host "==> Downloading $file"
  Invoke-WebRequest "$Base/$file" -OutFile "$tmp\$file"
  $want = ((Invoke-WebRequest "$Base/$file.sha256").Content.Trim() -split '\s+')[0].ToLower()
  $got = (Get-FileHash "$tmp\$file" -Algorithm SHA256).Hash.ToLower()
  if ($want -ne $got) { throw "checksum mismatch (wanted $want, got $got); nothing was installed" }
  Write-Host "==> Checksum ok"
  Expand-Archive "$tmp\$file" -DestinationPath $tmp -Force
  New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
  Copy-Item "$tmp\fle.exe" "$BinDir\fle.exe" -Force
  Write-Host "==> Installed $BinDir\fle.exe"
  $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
  if (($userPath -split ';') -notcontains $BinDir) { [Environment]::SetEnvironmentVariable('Path', "$userPath;$BinDir", 'User'); Write-Host "==> Added $BinDir to your PATH (open a new terminal)" }
  if (-not $Mode) {
    if ($Yes) { $Mode = 'terminal' } else {
      $a = Read-Host "`nHow do you want to use FLE?`n  [1] Terminal (fle)`n  [2] Browser (FLE Studio)`n  [3] Both`nChoose 1-3 [1]"
      $Mode = switch ($a) { '2' { 'studio' } '3' { 'both' } default { 'terminal' } }
    }
  }
  if ($Mode -in 'studio', 'both') {
    Write-Host "==> Installing FLE Studio (the browser interface)"
    try { & ([scriptblock]::Create((Invoke-RestMethod 'https://raw.githubusercontent.com/sirfyyn/fle-studio/main/install.ps1'))) -NoStart } catch { Write-Warning "FLE Studio could not be installed; fle itself is ready." }
  }
  Write-Host "`nDone. Start with:  fle   (terminal)   |   fle doctor   (what this machine can do)"
} finally { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
