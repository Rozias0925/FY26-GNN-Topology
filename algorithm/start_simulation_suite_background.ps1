param(
    [string]$Output = 'algorithm/results/v2_full_suite_20261003',
    [string]$Config = 'algorithm/configs/mechanism_v2_suite.json',
    [string]$ResumeFrom = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$entryPath = Join-Path $PSScriptRoot 'run_simulation_suite.py'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python environment is missing.' }
if ([System.IO.Path]::IsPathRooted($Output)) {
    $batchDirectory = [System.IO.Path]::GetFullPath($Output)
} else {
    $batchDirectory = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Output))
}
$workspacePrefix = $projectRoot.TrimEnd([char[]]'\/') + [System.IO.Path]::DirectorySeparatorChar
if (-not $batchDirectory.StartsWith($workspacePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Suite output must remain inside this project.'
}
if ([System.IO.Path]::IsPathRooted($Config)) {
    $configPath = [System.IO.Path]::GetFullPath($Config)
} else {
    $configPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Config))
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) { throw 'Suite config is missing.' }
$lockPath = Join-Path $batchDirectory 'suite.lock'
if (Test-Path -LiteralPath $lockPath) {
    $workerId = [int](Get-Content -LiteralPath $lockPath -Raw)
    if (Get-Process -Id $workerId -ErrorAction SilentlyContinue) { throw "Suite PID $workerId is already running." }
}
if (Test-Path -LiteralPath (Join-Path $batchDirectory 'complete.json')) { throw 'This suite is already complete.' }
$launcherPath = Join-Path $batchDirectory 'launcher.json'
if (Test-Path -LiteralPath $launcherPath) {
    $previous = Get-Content -LiteralPath $launcherPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (Get-Process -Id $previous.pid -ErrorAction SilentlyContinue) { throw "Launcher PID $($previous.pid) is active." }
}
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$plan = & $pythonPath $entryPath --config $configPath --dry-run
if ($LASTEXITCODE -ne 0) { throw 'Suite preflight rejected the configuration.' }
$null = ($plan -join "`n") | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $batchDirectory | Out-Null
$launchTime = [DateTime]::UtcNow
$stamp = $launchTime.ToString('yyyyMMddTHHmmssfffZ')
$stdoutPath = Join-Path $batchDirectory ("suite_$stamp.stdout.log")
$stderrPath = Join-Path $batchDirectory ("suite_$stamp.stderr.log")
$arguments = @('-u', ('"' + $entryPath + '"'), '--config', ('"' + $configPath + '"'), '--output', ('"' + $batchDirectory + '"'))
if ($ResumeFrom) {
    if ([System.IO.Path]::IsPathRooted($ResumeFrom)) { $recoveryPath = [System.IO.Path]::GetFullPath($ResumeFrom) }
    else { $recoveryPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $ResumeFrom)) }
    $arguments += @('--resume-from', ('"' + $recoveryPath + '"'))
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
if (-not (Test-Path -LiteralPath (Join-Path $batchDirectory 'status.json'))) {
    $starting = @{status='starting';updated_at=$launchTime.ToString('o')}
    [System.IO.File]::WriteAllText((Join-Path $batchDirectory 'status.json'), ($starting | ConvertTo-Json), $utf8)
}
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$result = @{pid=$process.Id;output=$batchDirectory;config=$configPath;launched_at=$launchTime.ToString('o');stdout=$stdoutPath;stderr=$stderrPath}
[System.IO.File]::WriteAllText($launcherPath, ($result | ConvertTo-Json -Depth 5), $utf8)
$result | ConvertTo-Json -Depth 5
