param(
    [string]$Output = 'algorithm/results/v2_stage_A',
    [string]$Config = 'algorithm/configs/mechanism_v2.json',
    [ValidateSet('A', 'B', 'C', 'D', 'sample_size', 'robustness')]
    [string[]]$Stages = @('A')
)

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$entryPath = Join-Path $PSScriptRoot 'run_mechanisms.py'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python environment is missing.' }
if ([System.IO.Path]::IsPathRooted($Output)) {
    $batchDirectory = [System.IO.Path]::GetFullPath($Output)
} else {
    $batchDirectory = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Output))
}
$workspacePrefix = $projectRoot.TrimEnd([char[]]'\/') + [System.IO.Path]::DirectorySeparatorChar
if (-not $batchDirectory.StartsWith($workspacePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Simulation output must remain inside this project.'
}
if ([System.IO.Path]::IsPathRooted($Config)) {
    $configPath = [System.IO.Path]::GetFullPath($Config)
} else {
    $configPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Config))
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) { throw 'Simulation config is missing.' }
$null = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$lockPath = Join-Path $batchDirectory 'run.lock'
if (Test-Path -LiteralPath $lockPath) {
    $workerId = [int](Get-Content -LiteralPath $lockPath -Raw)
    if (Get-Process -Id $workerId -ErrorAction SilentlyContinue) {
        throw "Simulation worker PID $workerId is already running."
    }
    throw 'A stale run.lock exists. Verify the worker has stopped before removing that file and resuming.'
}
if (Test-Path -LiteralPath (Join-Path $batchDirectory 'complete.json')) {
    throw 'This simulation is already complete. Choose a new output for a new run.'
}
$launcherPath = Join-Path $batchDirectory 'launcher.json'
if (Test-Path -LiteralPath $launcherPath) {
    $previous = Get-Content -LiteralPath $launcherPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (Get-Process -Id $previous.pid -ErrorAction SilentlyContinue) {
        throw "Simulation launcher PID $($previous.pid) is already running."
    }
}
New-Item -ItemType Directory -Force -Path $batchDirectory | Out-Null
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$launchedAt = [DateTime]::UtcNow
$logStamp = $launchedAt.ToString('yyyyMMddTHHmmssfffZ')
$stdoutPath = Join-Path $batchDirectory ("simulation_$logStamp.stdout.log")
$stderrPath = Join-Path $batchDirectory ("simulation_$logStamp.stderr.log")
$arguments = @('-u', ('"' + $entryPath + '"'), 'simulate', '--config',
    ('"' + $configPath + '"'), '--stages') + $Stages + @('--output', ('"' + $batchDirectory + '"'))
$statusPath = Join-Path $batchDirectory 'status.json'
$encoding = New-Object System.Text.UTF8Encoding($false)
$starting = @{ status = 'starting'; stages = $Stages; updated_at = $launchedAt.ToString('o') }
[System.IO.File]::WriteAllText($statusPath, ($starting | ConvertTo-Json -Depth 5), $encoding)
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$result = @{
    pid = $process.Id
    output = $batchDirectory
    config = $configPath
    stages = $Stages
    launched_at = $launchedAt.ToString('o')
    stdout = $stdoutPath
    stderr = $stderrPath
}
[System.IO.File]::WriteAllText($launcherPath, ($result | ConvertTo-Json -Depth 5), $encoding)
$result | ConvertTo-Json -Depth 5
