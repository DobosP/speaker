# One-command native install for the local voice assistant (Windows / PowerShell).
#
#   .\install.ps1                full install (venv, deps, models, doctor)
#   .\install.ps1 -Recreate      rebuild the venv from scratch (fixes conda/venv mixes)
#   .\install.ps1 -DryRun        print the plan, change nothing
#   .\install.ps1 -SkipModels    deps + venv only, no model download
#
# PortAudio ships inside the sounddevice wheel on Windows, so there is no system
# step -- this just finds a Python and runs the cross-platform installer
# (tools/install.py), the same code path Linux/macOS use.
#
# If PowerShell blocks the script, run it once as:
#   powershell -ExecutionPolicy Bypass -File .\install.ps1
param(
  [switch]$Recreate,
  [switch]$DryRun,
  [switch]$SkipModels
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Find a launchable Python: prefer the 'py' launcher, then 'python'.
$py = $null
foreach ($cand in @("py", "python")) {
  if (Get-Command $cand -ErrorAction SilentlyContinue) { $py = $cand; break }
}
if (-not $py) {
  Write-Error "No Python found on PATH. Install Python 3.10+ from https://python.org (tick 'Add to PATH')."
  exit 1
}

$forward = @()
if ($Recreate)   { $forward += "--recreate" }
if ($DryRun)     { $forward += "--dry-run" }
if ($SkipModels) { $forward += "--skip-models" }

Write-Host "==> Cross-platform install (venv, deps, models, doctor)"
& $py tools/install.py @forward
exit $LASTEXITCODE
