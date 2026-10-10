<#
Installs the CUDA toolkit on a Windows runner from the NVIDIA redistributable archives
(https://developer.download.nvidia.com/compute/cuda/redist/), the same way as the upstream
llama.cpp release workflow does. Only the parts that are needed to compile and link ggml-cuda
are installed; every archive is verified against the SHA-256 of the redist manifest.

  -Version      name of the redist manifest, e.g. 13.4.1 (redistrib_13.4.1.json pins every component)
  -Destination  toolkit root, e.g. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4

CUDA_PATH and PATH are exported to the following steps of the job (GITHUB_ENV / GITHUB_PATH).
#>
param(
    [Parameter(Mandatory = $true)] [string] $Version,
    [Parameter(Mandatory = $true)] [string] $Destination,
    [string[]] $Components = @(
        'cuda_crt', 'cuda_cudart', 'cuda_nvcc', 'libnvvm', 'libcublas', 'cccl',
        'cuda_nvtx', 'cuda_profiler_api', 'visual_studio_integration', 'cuda_cuobjdump'
    )
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$base = 'https://developer.download.nvidia.com/compute/cuda/redist'
$work = Join-Path $env:RUNNER_TEMP 'cuda-redist'
New-Item -ItemType Directory -Force -Path $work | Out-Null
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

$manifestFile = Join-Path $work "redistrib_$Version.json"
& curl.exe --fail --silent --show-error --location --retry 5 --retry-delay 5 --retry-all-errors -o $manifestFile "$base/redistrib_$Version.json"
if ($LASTEXITCODE -ne 0) { throw "cannot download redistrib_$Version.json" }
$manifest = Get-Content -Raw -Path $manifestFile | ConvertFrom-Json
Write-Host "CUDA redist manifest $Version, release date $($manifest.release_date)"

foreach ($name in $Components) {
    $component = $manifest.$name
    if ($null -eq $component) { throw "component $name is not in redistrib_$Version.json" }
    $entry = $component.'windows-x86_64'
    if ($null -eq $entry) { throw "component $name has no windows-x86_64 archive" }

    $archive = Join-Path $work ([System.IO.Path]::GetFileName($entry.relative_path))
    Write-Host ("{0,-28} {1,-14} {2,10:N1} MiB" -f $name, $component.version, ($entry.size / 1MB))
    & curl.exe --fail --silent --show-error --location --retry 5 --retry-delay 5 --retry-all-errors -o $archive "$base/$($entry.relative_path)"
    if ($LASTEXITCODE -ne 0) { throw "download of $name failed" }

    $hash = (Get-FileHash -Path $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $entry.sha256) { throw "SHA-256 mismatch for $name : $hash, expected $($entry.sha256)" }

    $unpacked = Join-Path $work "x-$name"
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $unpacked
    & 7z x -y "-o$unpacked" $archive | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "7z could not unpack $name" }

    # every archive holds one top-level directory <name>-windows-x86_64-<version>-archive
    $inner = Get-ChildItem -Path $unpacked -Directory | Select-Object -First 1
    & robocopy $inner.FullName $Destination /E /MOVE /NFL /NDL /NJH /NJS /NP /MT:8 | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed for $name (exit code $LASTEXITCODE)" }
    $global:LASTEXITCODE = 0
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $unpacked
    Remove-Item -Force $archive
}

$nvcc = Join-Path $Destination 'bin\nvcc.exe'
if (-not (Test-Path $nvcc)) { throw "nvcc.exe is missing in $Destination\bin" }

$parts = $Version.Split('.')
$suffix = "$($parts[0])_$($parts[1])"
Add-Content -Path $env:GITHUB_ENV -Value "CUDA_PATH=$Destination"
Add-Content -Path $env:GITHUB_ENV -Value "CUDA_PATH_V$suffix=$Destination"
Add-Content -Path $env:GITHUB_PATH -Value (Join-Path $Destination 'bin')
Add-Content -Path $env:GITHUB_PATH -Value (Join-Path $Destination 'bin\x64')

& $nvcc --version
Write-Host 'Runtime DLLs and import libraries of the toolkit:'
Get-ChildItem -Path $Destination -Recurse -Include 'cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll', 'cudart.lib', 'cublas.lib', 'cuda.lib' |
    ForEach-Object { Write-Host ("  {0}  {1:N1} MiB" -f $_.FullName.Substring($Destination.Length), ($_.Length / 1MB)) }
