# Downloads the Linux Python packages for the Docker image into .\wheels,
# using your normal Windows internet connection (faster and more reliable than
# downloading inside Docker). Run once from the project folder:
#
#     .\scripts\download_wheels.ps1
#
# If PowerShell blocks the script:  powershell -ExecutionPolicy Bypass -File .\scripts\download_wheels.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) { Write-Error "Python was not found. Install Python 3.10+ from python.org first." }

Write-Host "Downloading Linux packages into .\wheels (this can take a few minutes)..."
python -m pip download -r requirements.txt --dest wheels `
    --only-binary=:all: --python-version 3.10 --implementation cp `
    --abi cp310 --abi abi3 --abi none `
    --platform manylinux_2_28_x86_64 --platform manylinux_2_24_x86_64 `
    --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 `
    --platform manylinux_2_5_x86_64 --platform manylinux1_x86_64 `
    --platform linux_x86_64 --platform any `
    --retries 10 --timeout 120

$count = (Get-ChildItem wheels -Filter *.whl).Count
Write-Host ""
Write-Host "Done: $count packages in .\wheels"
Write-Host "Now run:  docker compose up --build"
