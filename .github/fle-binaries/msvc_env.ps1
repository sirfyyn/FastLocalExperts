<#
Dot-source this file (". .\.github\fle-binaries\msvc_env.ps1") to get the x64 MSVC build environment
(cl, link, VCToolsRedistDir, ...) in the current PowerShell session. It runs vcvarsall.bat in a
cmd.exe (through a small batch file, quoting paths with spaces on a cmd.exe command line is fragile)
and copies the resulting environment over.
#>
$vcvars = 'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat'
if (-not (Test-Path $vcvars)) {
    $found = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'VC\Auxiliary\Build\vcvarsall.bat'
    if (-not $found) { throw 'vcvarsall.bat not found' }
    $vcvars = @($found)[0]
}
$tempDir = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
$bat = Join-Path $tempDir 'fle_msvc_env.bat'
Set-Content -Path $bat -Encoding ascii -Value @(
    '@echo off',
    ('call "' + $vcvars + '" x64 >nul 2>&1'),
    'if errorlevel 1 exit /b 1',
    'set'
)
$lines = & cmd.exe /d /c $bat
if ($LASTEXITCODE -ne 0) { throw "vcvarsall.bat failed (exit code $LASTEXITCODE)" }
foreach ($line in $lines) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}
Write-Host ("MSVC environment: " + (Get-Command cl).Source)
