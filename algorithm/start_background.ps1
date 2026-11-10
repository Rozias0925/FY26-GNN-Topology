param([string]$Output = 'algorithm/results/overnight_20260930')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python environment is missing.' }
$batchDirectory = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Output))
$lockPath = Join-Path $batchDirectory 'batch.lock'
if (Test-Path -LiteralPath $lockPath) {
    $batchPid = [int](Get-Content -LiteralPath $lockPath -Raw)
    if (Get-Process -Id $batchPid -ErrorAction SilentlyContinue) { throw "Batch PID $batchPid is already running." }
}
New-Item -ItemType Directory -Force -Path $batchDirectory | Out-Null
$env:PYTHONUTF8 = '1'
$arguments = @('-u', ('"' + (Join-Path $PSScriptRoot 'run_batch.py') + '"'), '--output', ('"' + $batchDirectory + '"'))
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $batchDirectory 'launcher.stdout.log') -RedirectStandardError (Join-Path $batchDirectory 'launcher.stderr.log')
$result = @{ pid = $process.Id; output = $batchDirectory; launched_at = [DateTime]::UtcNow.ToString('o') }
$result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batchDirectory 'launcher.json') -Encoding utf8
$result | ConvertTo-Json
