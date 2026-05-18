# run_daily.ps1 — Activate venv and run the daily email script.
# Task Scheduler calls this file directly.

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$python     = Join-Path $projectDir "venv\Scripts\python.exe"
$script     = Join-Path $projectDir "daily_email.py"
$logFile    = Join-Path $projectDir "logs\daily_run.log"

# Ensure log directory exists
$logDir = Split-Path $logFile
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content $logFile "`n`n=== $timestamp ==="

& $python $script 2>&1 | Tee-Object -FilePath $logFile -Append
