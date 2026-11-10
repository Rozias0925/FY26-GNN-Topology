<#
.SYNOPSIS
Launch one CPU-only real-data benchmark without disturbing other applications.
.DESCRIPTION
DryRun writes no files and never starts a helper. WaitForResources starts only
an Idle-priority, single-core standard-library helper while resources are low.
It checks every 30 seconds for at most 24 hours. Three consecutive two-second
checks with at least 1.5 GiB available RAM and at most 70 percent system CPU are
required before the benchmark's mandatory tests, preflight, and training.
.EXAMPLE
.\algorithm\launch_real_benchmark.ps1 -DryRun -WaitForResources
.EXAMPLE
.\algorithm\launch_real_benchmark.ps1 -OutputPath .\algorithm\results\real_20261019 -WaitForResources
#>
[CmdletBinding()]
param(
    [string]$ConfigPath = (Join-Path $PSScriptRoot 'configs\real_benchmark_regression.json'),
    [string]$OutputPath = (Join-Path $PSScriptRoot 'results\real_benchmark_regression'),
    [switch]$DryRun,
    [switch]$WaitForResources
)

# This launcher only manages its own benchmark. It never changes another process.
$ErrorActionPreference = 'Stop'
$algorithmRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$pythonPath = Join-Path $algorithmRoot '.venv\Scripts\python.exe'
$entryPath = Join-Path $algorithmRoot 'run_real_benchmark.py'
$configAbsolute = [IO.Path]::GetFullPath($ConfigPath)
$outputAbsolute = [IO.Path]::GetFullPath($OutputPath)
$lockPath = Join-Path $outputAbsolute '.benchmark_launcher.lock.json'
$statusPath = Join-Path $outputAbsolute 'launcher_status.json'
$launcherDirectory = Join-Path $outputAbsolute '.launcher'
$guardPath = Join-Path $launcherDirectory 'launcher.guard'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Write-JsonFile([string]$Path, $Value) {
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 12), $utf8)
}

function Initialize-NativeApi {
    if ('GnnBenchmarkWindowsNative' -as [type]) { return }
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
public static class GnnBenchmarkWindowsNative {
    [StructLayout(LayoutKind.Sequential)] public struct FT { public uint Low; public uint High; }
    [StructLayout(LayoutKind.Sequential)] public class MEM {
        public uint Length = (uint)Marshal.SizeOf(typeof(MEM)); public uint Load;
        public ulong TotalPhys, AvailPhys, TotalPage, AvailPage, TotalVirtual, AvailVirtual, AvailExtended;
    }
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetSystemTimes(out FT idle, out FT kernel, out FT user);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool GlobalMemoryStatusEx([In,Out] MEM memory);
    [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
    [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)] static extern bool QueryFullProcessImageNameW(IntPtr process, uint flags, StringBuilder name, ref uint length);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetProcessTimes(IntPtr process, out FT created, out FT exited, out FT kernel, out FT user);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    static ulong Value(FT value) { return ((ulong)value.High << 32) | value.Low; }
    public static ulong[] Resources() {
        FT idle, kernel, user; MEM memory = new MEM();
        if (!GetSystemTimes(out idle, out kernel, out user)) throw new Win32Exception(Marshal.GetLastWin32Error());
        if (!GlobalMemoryStatusEx(memory)) throw new Win32Exception(Marshal.GetLastWin32Error());
        return new ulong[] { Value(idle), Value(kernel), Value(user), memory.AvailPhys, memory.TotalPhys };
    }
    public static string[] Identity(uint pid) {
        IntPtr process = OpenProcess(0x1000, false, pid);
        if (process == IntPtr.Zero) {
            int error = Marshal.GetLastWin32Error();
            if (error == 87) return null; // ERROR_INVALID_PARAMETER: no such PID.
            throw new Win32Exception(error); // Access denied is unknown, never dead.
        }
        try {
            uint size = 32768; StringBuilder path = new StringBuilder((int)size);
            FT created, exited, kernel, user;
            if (!QueryFullProcessImageNameW(process, 0, path, ref size)) throw new Win32Exception(Marshal.GetLastWin32Error());
            if (!GetProcessTimes(process, out created, out exited, out kernel, out user)) throw new Win32Exception(Marshal.GetLastWin32Error());
            return new string[] { path.ToString(), DateTime.FromFileTimeUtc((long)Value(created)).ToString("o") };
        } finally { CloseHandle(process); }
    }
}
'@
}

function Get-NativeIdentity([int]$ProcessId) {
    Initialize-NativeApi
    $identity = [GnnBenchmarkWindowsNative]::Identity([uint32]$ProcessId)
    if ($null -eq $identity) { return $null }
    return [pscustomobject]@{ ExecutablePath = $identity[0]; StartUtc = $identity[1]; verification = 'native_limited_query_executable_and_creation_time' }
}

function Get-ResourceSnapshot {
    $cpuSamples = @()
    $memorySamples = @()
    $problems = @()
    for ($sample = 0; $sample -lt 2; $sample++) {
        try {
            $cpu = Get-CimInstance -ClassName Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'" -OperationTimeoutSec 2 -ErrorAction Stop
            if ($null -eq $cpu -or $null -eq $cpu.PercentProcessorTime) { throw 'CPU counter returned no value.' }
            $cpuSamples += [double]$cpu.PercentProcessorTime
        } catch { $problems += ('CPU CIM sample unavailable: ' + $_.Exception.Message); break }
        try {
            $os = Get-CimInstance -ClassName Win32_OperatingSystem -OperationTimeoutSec 2 -ErrorAction Stop
            if ($null -eq $os -or $null -eq $os.FreePhysicalMemory) { throw 'RAM counter returned no value.' }
            $memorySamples += ([double]$os.FreePhysicalMemory * 1024.0 / 1GB)
        } catch { $problems += ('Available RAM CIM sample unavailable: ' + $_.Exception.Message); break }
        if ($sample -eq 0) { Start-Sleep -Seconds 1 }
    }
    $known = ($cpuSamples.Count -eq 2 -and $memorySamples.Count -eq 2 -and $problems.Count -eq 0)
    $method = 'CIM_PerfOS_Processor_and_OperatingSystem'
    if (-not $known) {
        try {
            Initialize-NativeApi
            $before = [GnnBenchmarkWindowsNative]::Resources()
            Start-Sleep -Seconds 1
            $after = [GnnBenchmarkWindowsNative]::Resources()
            $total = ([double]$after[1] - [double]$before[1]) + ([double]$after[2] - [double]$before[2])
            $idle = [double]$after[0] - [double]$before[0]
            if ($total -le 0 -or $idle -lt 0 -or $idle -gt $total) { throw 'Invalid native CPU counter delta.' }
            $cpuSamples = @([Math]::Max(0.0, [Math]::Min(100.0, 100.0 * ($total - $idle) / $total)))
            $memorySamples = @(([double]$before[3] / 1GB), ([double]$after[3] / 1GB))
            $method = 'Win32_GetSystemTimes_and_GlobalMemoryStatusEx'
            $known = $true
        } catch { $problems += ('Native resource sample unavailable: ' + $_.Exception.Message) }
    }
    $cpuMaximum = $null
    $memoryMinimum = $null
    if ($cpuSamples.Count) { $cpuMaximum = ($cpuSamples | Measure-Object -Maximum).Maximum }
    if ($memorySamples.Count) { $memoryMinimum = ($memorySamples | Measure-Object -Minimum).Minimum }
    return [pscustomobject]@{
        known = $known
        counter_method = $method
        cpu_percent_samples = $cpuSamples
        cpu_percent_max = $cpuMaximum
        available_gib_samples = $memorySamples
        available_gib_min = $memoryMinimum
        allowed = ($known -and $cpuMaximum -le 70.0 -and $memoryMinimum -ge 1.5)
        limits = [pscustomobject]@{ max_cpu_percent = 70; min_available_gib = 1.5 }
        errors = $problems
    }
}

function Get-ExistingRun {
    if (-not (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
        if (Test-Path -LiteralPath $statusPath -PathType Leaf) {
            try {
                $oldStatus = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
                if ($oldStatus.state -eq 'completed') {
                    return [pscustomobject]@{ state = 'completed'; message = 'This output folder already has a completed run; choose another output folder.'; details = $oldStatus }
                }
            } catch {
                return [pscustomobject]@{ state = 'unknown'; message = 'Existing launcher status is unreadable; refusing to overwrite it.' }
            }
        }
        return [pscustomobject]@{ state = 'none'; message = 'No active launcher lock.' }
    }
    try {
        $previous = Get-Content -LiteralPath $lockPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if (-not $previous.job_id -or -not $previous.process_id -or -not $previous.process_start_utc -or -not $previous.command_line -or -not $previous.executable) { throw 'Lock identity fields are missing.' }
        # Sandboxed Get-Process can omit a live process outside its boundary.
        # Only a native ERROR_INVALID_PARAMETER (87) proves a missing PID.
        # Native access denied and other errors escape to the unknown branch.
        $identity = Get-NativeIdentity -ProcessId ([int]$previous.process_id)
        if ($null -eq $identity) {
            return [pscustomobject]@{ state = 'stale'; message = 'Native limited query confirms the recorded PID no longer exists (error 87).'; details = $previous }
        }
        $actualStart = [DateTimeOffset]::Parse($identity.StartUtc).UtcDateTime
        $verification = $identity.verification
        try {
            $cimIdentity = Get-CimInstance -ClassName Win32_Process -Filter ('ProcessId=' + [int]$previous.process_id) -OperationTimeoutSec 2 -ErrorAction Stop
            if ($null -eq $cimIdentity -or -not $cimIdentity.ExecutablePath -or -not $cimIdentity.CommandLine) { throw 'CIM identity is unavailable; retain verified native identity.' }
            $identity = $cimIdentity
            $verification = 'CIM_executable_command_line_and_creation_time'
        } catch { }
        # ConvertFrom-Json may already parse ISO timestamps as DateTime. Do not
        # stringify that object and discard its UTC/local Kind before comparing.
        if ($previous.process_start_utc -is [DateTime]) {
            $recordedStart = $previous.process_start_utc.ToUniversalTime()
        } elseif ($previous.process_start_utc -is [DateTimeOffset]) {
            $recordedStart = $previous.process_start_utc.UtcDateTime
        } else {
            $recordedStart = [DateTimeOffset]::Parse([string]$previous.process_start_utc).UtcDateTime
        }
        if ([Math]::Abs(($actualStart - $recordedStart).TotalSeconds) -gt 1.0) {
            return [pscustomobject]@{ state = 'stale'; message = 'PID has been reused; recorded run has exited.'; details = $previous; verification = $verification }
        }
        if (-not [string]::Equals([string]$identity.ExecutablePath, [string]$previous.executable, [StringComparison]::OrdinalIgnoreCase)) { throw 'PID exists but its executable does not match the recorded identity.' }
        if ($verification -like 'CIM_*' -and -not [string]::Equals([string]$identity.CommandLine, [string]$previous.command_line, [StringComparison]::Ordinal)) { throw 'PID exists but its command line does not match the recorded identity.' }
        return [pscustomobject]@{ state = 'running'; message = 'Verified existing benchmark or launcher; a duplicate will not be started.'; details = $previous; verification = $verification }
    } catch {
        return [pscustomobject]@{ state = 'unknown'; message = ('Cannot safely verify the existing run: ' + $_.Exception.Message) }
    }
}

foreach ($required in @($pythonPath, $entryPath, $configAbsolute)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw ('Required file does not exist: ' + $required) }
}
# A syntax or schema error must be caught by the runner's own preflight; still
# refuse malformed JSON before creating a background process.
$null = Get-Content -LiteralPath $configAbsolute -Raw -Encoding UTF8 | ConvertFrom-Json
$existing = Get-ExistingRun
$resources = Get-ResourceSnapshot
$canStart = ($resources.allowed -and $existing.state -in @('none', 'stale'))
$canQueue = ([bool]$WaitForResources -and $existing.state -in @('none', 'stale'))
$reason = 'Ready to start one CPU-only background process.'
if (-not $resources.known) { $reason = 'Resource availability is unknown; nothing will be started.' }
elseif (-not $resources.allowed) { $reason = 'CPU or available RAM exceeds the conservative launch limits; nothing will be started.' }
elseif ($existing.state -notin @('none', 'stale')) { $reason = $existing.message }
if (-not $canStart -and $canQueue) { $reason += ' WaitForResources permits only a lightweight Idle-priority waiting helper; no model imports until three successful resource checks.' }
$summary = [pscustomobject]@{
    dry_run = [bool]$DryRun
    can_start = $canStart
    can_queue_lightweight_helper = $canQueue
    wait_for_resources = [bool]$WaitForResources
    reason = $reason
    config = $configAbsolute
    output = $outputAbsolute
    command = @($pythonPath, '-u', $entryPath, '--config', $configAbsolute, '--output', $outputAbsolute)
    resources = $resources
    previous_run = $existing
    policy = 'Idle priority; one allowed logical CPU; one thread per numerical library; CPU only; no other process is modified.'
}
if ($DryRun -or (-not $canStart -and -not $canQueue)) {
    $summary | ConvertTo-Json -Depth 15
    if (-not $canStart -and $existing.state -ne 'completed') { exit 2 }
    exit 0
}

$null = [IO.Directory]::CreateDirectory($launcherDirectory)
$guard = $null
$spawned = $null
$jobId = [Guid]::NewGuid().ToString('N')
try {
    # FileShare.None serializes launchers. Windows releases the handle even if
    # a launcher crashes, so a persistent guard file is harmless.
    $guard = [IO.File]::Open($guardPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $existing = Get-ExistingRun
    if ($existing.state -notin @('none', 'stale')) { throw $existing.message }
    if ($existing.state -eq 'stale') {
        $staleArchive = Join-Path $launcherDirectory ('stale_lock_' + $jobId + '.json')
        Move-Item -LiteralPath $lockPath -Destination $staleArchive -ErrorAction Stop
    }
    try {
        $selfIdentity = Get-CimInstance -ClassName Win32_Process -Filter ('ProcessId=' + $PID) -OperationTimeoutSec 2 -ErrorAction Stop
        if (-not $selfIdentity.CommandLine -or -not $selfIdentity.ExecutablePath) { throw 'Cannot verify this launcher process identity.' }
        $selfStart = (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
    } catch {
        $nativeSelf = Get-NativeIdentity -ProcessId $PID
        if ($null -eq $nativeSelf) { throw 'Native query could not verify this launcher PID; nothing was started.' }
        $selfIdentity = [pscustomobject]@{ ExecutablePath = $nativeSelf.ExecutablePath; CommandLine = [Environment]::CommandLine }
        $selfStart = $nativeSelf.StartUtc
    }
    $launcherRecord = [ordered]@{
        state = 'launching'; job_id = $jobId; process_id = $PID
        process_start_utc = $selfStart
        executable = [string]$selfIdentity.ExecutablePath
        command_line = [string]$selfIdentity.CommandLine
        config = $configAbsolute; output = $outputAbsolute
        resources_before_launch = $resources
    }
    $lockStream = [IO.File]::Open($lockPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $bytes = $utf8.GetBytes(($launcherRecord | ConvertTo-Json -Depth 12))
        $lockStream.Write($bytes, 0, $bytes.Length)
    } finally { $lockStream.Dispose() }
    $helperPath = Join-Path $launcherDirectory ('runner_' + $jobId + '.py')
    $stdoutPath = Join-Path $launcherDirectory ($jobId + '.stdout.log')
    $stderrPath = Join-Path $launcherDirectory ($jobId + '.stderr.log')
    $payload = [ordered]@{
        job_id = $jobId; entry = $entryPath; config = $configAbsolute
        output = $outputAbsolute; lock = $lockPath; status = $statusPath
        stdout = $stdoutPath; stderr = $stderrPath; resources_before_launch = $resources
        wait_for_resources = [bool]$WaitForResources
        runtime_process_memory_limit_mib = 768
    } | ConvertTo-Json -Depth 12 -Compress
    $payloadBase64 = [Convert]::ToBase64String($utf8.GetBytes($payload))
    $helper = @'
"""Standard-library-only bootstrap; lowers its own resources before model imports."""
import base64
import ctypes
from ctypes import wintypes
import datetime as dt
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

JOB = json.loads(base64.b64decode("__PAYLOAD__").decode("utf-8"))
ENV = {
    "GNN_LOW_IMPACT": "1", "GNN_REQUIRE_TESTS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "-1", "PYTHONUTF8": "1",
}
os.environ.update(ENV)
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)

def atomic_json(path, value):
    target = Path(path)
    staging = target.with_name(target.name + "." + JOB["job_id"] + ".tmp")
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(staging, target)

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong) for name in (
            "total_phys", "avail_phys", "total_page", "avail_page",
            "total_virtual", "avail_virtual", "avail_extended")]

def native_resources(kernel, interval=2.0):
    def counters():
        idle, system, user = (wintypes.FILETIME() for _ in range(3))
        if not kernel.GetSystemTimes(ctypes.byref(idle), ctypes.byref(system), ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())
        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        if not kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise ctypes.WinError(ctypes.get_last_error())
        values = [(value.dwHighDateTime << 32) | value.dwLowDateTime for value in (idle, system, user)]
        return values, memory.avail_phys / 2 ** 30
    try:
        previous, before_ram = counters()
        time.sleep(interval)
        current, after_ram = counters()
        total = current[1] - previous[1] + current[2] - previous[2]
        idle = current[0] - previous[0]
        if total <= 0 or idle < 0 or idle > total:
            raise RuntimeError("Native CPU counters returned an invalid delta.")
        cpu = max(0.0, min(100.0, 100.0 * (total - idle) / total))
        available = min(before_ram, after_ram)
        return {"known": True, "counter_method": "Win32_GetSystemTimes_and_GlobalMemoryStatusEx",
                "cpu_percent": round(cpu, 3), "available_gib": round(available, 6),
                "allowed": cpu <= 70.0 and available >= 1.5, "sampled_at_utc": now()}
    except Exception as error:
        return {"known": False, "allowed": False, "error": repr(error), "sampled_at_utc": now()}

def publish():
    atomic_json(JOB["lock"], record)
    atomic_json(JOB["status"], record)

def stop_wait_if_requested():
    if not (Path(JOB["output"]) / "STOP_REQUESTED").exists():
        return False
    record.update(state="stopped", reason="graceful stop requested", heavy_imports_allowed=False)
    publish()
    return True

def sleep_with_stop(seconds):
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        if stop_wait_if_requested():
            return False
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    return not stop_wait_if_requested()

def runner_stopped():
    try:
        status = json.loads((Path(JOB["output"]) / "status.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    if status.get("pid") not in (None, os.getpid()):
        return False
    if status.get("status", status.get("state")) != "stopped":
        return False
    record.update(state="stopped", reason=status.get("reason", "graceful stop requested"), runner_status=status)
    record.pop("error", None)
    return True

def await_resources(kernel):
    deadline = started + 24 * 60 * 60
    record["resource_wait_deadline_utc"] = (
        dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=max(0, deadline - time.monotonic()))
    ).isoformat()
    good_count = 0
    while time.monotonic() < deadline:
        if stop_wait_if_requested():
            return False
        poll_started = time.monotonic()
        resource = native_resources(kernel, interval=2.0)
        if stop_wait_if_requested():
            return False
        record["resources_current"] = resource
        good_count = good_count + 1 if resource["allowed"] else 0
        record["consecutive_ready_checks"] = good_count
        if good_count >= 3:
            record.update(state="running", reason="Three consecutive two-second resource checks passed.", heavy_imports_allowed=True)
            publish()
            return True
        record.update(state="waiting_for_resources", heavy_imports_allowed=False,
                      reason=("Confirming resource stability before imports." if resource["allowed"] else
                              "Resources unknown or below thresholds; numerical libraries and tests have not been imported."))
        publish()
        print(json.dumps({"state": record["state"], "resource": resource, "consecutive_ready_checks": good_count}, ensure_ascii=False), flush=True)
        if not resource["allowed"]:
            if not JOB["wait_for_resources"]:
                record.update(state="blocked", reason="Resource limits failed and WaitForResources was not enabled.")
                return False
            remaining = deadline - time.monotonic()
            if not sleep_with_stop(max(0.0, min(remaining, 30.0 - (time.monotonic() - poll_started)))):
                return False
        # Good checks immediately take another two-second measurement. Failed
        # checks are polled every 30 seconds. Only standard-library code runs.
    record.update(state="stopped", reason="Resource wait timed out after 24 hours; no numerical tests or training were started.", heavy_imports_allowed=False)
    return False

record = {"job_id": JOB["job_id"], "state": "starting", "process_id": os.getpid(),
          "config": JOB["config"], "output": JOB["output"], "stdout": JOB["stdout"],
          "stderr": JOB["stderr"], "environment": ENV,
          "resources_before_launch": JOB["resources_before_launch"], "started_at_utc": now(),
          "runtime_process_memory_limit_mib": JOB["runtime_process_memory_limit_mib"],
          "runtime_memory_limit_enforced_by": "main runner low_impact_runtime; not the lightweight waiting helper",
          "wait_for_resources": JOB["wait_for_resources"], "heavy_imports_allowed": False}
started = time.monotonic()
exit_code = 1
try:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel.GetCurrentProcess()
    kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.GetPriorityClass.argtypes = (wintypes.HANDLE,)
    kernel.GetPriorityClass.restype = wintypes.DWORD
    kernel.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
    kernel.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
    kernel.GetCommandLineW.restype = wintypes.LPWSTR
    kernel.GetModuleFileNameW.argtypes = (wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD)
    kernel.GetModuleFileNameW.restype = wintypes.DWORD
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME))
    kernel.GetSystemTimes.argtypes = (ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME))
    kernel.GlobalMemoryStatusEx.argtypes = (ctypes.POINTER(MemoryStatus),)
    if not kernel.SetPriorityClass(handle, 0x40):
        raise ctypes.WinError(ctypes.get_last_error())
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(allowed), ctypes.byref(system)) or not allowed.value:
        raise ctypes.WinError(ctypes.get_last_error())
    chosen = allowed.value & -allowed.value
    if not kernel.SetProcessAffinityMask(handle, chosen):
        raise ctypes.WinError(ctypes.get_last_error())
    verified, _system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(verified), ctypes.byref(_system)):
        raise ctypes.WinError(ctypes.get_last_error())
    priority = kernel.GetPriorityClass(handle)
    if priority != 0x40 or verified.value != chosen:
        raise RuntimeError("Idle priority or single-CPU affinity could not be verified; model was not imported.")
    creation, end, system_time, user_time = (wintypes.FILETIME() for _ in range(4))
    if not kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(end), ctypes.byref(system_time), ctypes.byref(user_time)):
        raise ctypes.WinError(ctypes.get_last_error())
    ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
    creation_utc = dt.datetime(1601, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(microseconds=ticks // 10)
    image_path = ctypes.create_unicode_buffer(32768)
    if not kernel.GetModuleFileNameW(None, image_path, len(image_path)):
        raise ctypes.WinError(ctypes.get_last_error())
    # Windows venv launchers can spawn the base Python image. PID validation
    # must record that actual image, while preserving the selected venv path.
    record.update(state="starting", executable=image_path.value,
                  python_environment_executable=str(Path(sys.executable).resolve()),
                  command_line=kernel.GetCommandLineW(), process_start_utc=creation_utc.isoformat(),
                  priority="Idle", priority_class=priority, affinity_mask=chosen,
                  logical_cpu_index=chosen.bit_length() - 1,
                  benchmark_arguments=[JOB["entry"], "--config", JOB["config"], "--output", JOB["output"]])
    publish()
    print(json.dumps({"launcher": record}, ensure_ascii=False), flush=True)
    if not await_resources(kernel):
        raise SystemExit(3 if record["state"] == "stopped" else 2)
    if stop_wait_if_requested():
        raise SystemExit(3)
    os.chdir(str(Path(JOB["entry"]).parent.parent))
    sys.path.insert(0, str(Path(JOB["entry"]).parent))
    sys.argv = record["benchmark_arguments"]
    try:
        runpy.run_path(JOB["entry"], run_name="__main__")
        exit_code = 0
    except SystemExit as stopped:
        exit_code = stopped.code if isinstance(stopped.code, int) else (0 if stopped.code is None else 1)
        if stopped.code not in (0, None):
            record["error"] = str(stopped.code)
    if exit_code in (0, 3) and runner_stopped():
        exit_code = 3
except SystemExit as stopped:
    exit_code = stopped.code if isinstance(stopped.code, int) else (0 if stopped.code is None else 1)
    if stopped.code not in (0, None) and "error" not in record and record["state"] not in ("stopped", "blocked"):
        record["error"] = str(stopped.code)
except BaseException as error:
    record["error"] = repr(error)
    traceback.print_exc()
    exit_code = 1
finally:
    final_state = record["state"] if record["state"] in ("blocked", "stopped") else ("completed" if exit_code == 0 else "failed")
    record.update(state=final_state, exit_code=exit_code,
                  finished_at_utc=now(), elapsed_seconds=round(time.monotonic() - started, 3))
    atomic_json(JOB["status"], record)
    try:
        lock = Path(JOB["lock"])
        if json.loads(lock.read_text(encoding="utf-8"))["job_id"] == JOB["job_id"]:
            lock.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        traceback.print_exc()
sys.exit(exit_code)
'@
    $helper = $helper.Replace('__PAYLOAD__', $payloadBase64)
    [IO.File]::WriteAllText($helperPath, $helper, $utf8)
    # Restrict this child via its bootstrap; inherited environment in this
    # PowerShell session and the user's other applications stays unchanged.
    $spawned = Start-Process -FilePath $pythonPath -ArgumentList @('-u', ('"' + $helperPath + '"')) -WorkingDirectory $algorithmRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    $ready = $null
    for ($attempt = 0; $attempt -lt 24; $attempt++) {
        if (Test-Path -LiteralPath $statusPath -PathType Leaf) {
            try {
                $candidate = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
                if ($candidate.job_id -eq $jobId) { $ready = $candidate; break }
            } catch { }
        }
        if ($spawned.HasExited) { break }
        Start-Sleep -Milliseconds 250
    }
    if ($null -eq $ready) {
        [pscustomobject]@{ state = 'starting'; process_id = $spawned.Id; job_id = $jobId; status = $statusPath; stdout = $stdoutPath; stderr = $stderrPath; message = 'Bootstrap has not reported readiness yet. No other process was changed.' } | ConvertTo-Json -Depth 6
    } else { $ready | ConvertTo-Json -Depth 15 }
} catch {
    if ($null -eq $spawned -and (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
        try {
            $ownLock = Get-Content -LiteralPath $lockPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($ownLock.job_id -eq $jobId) { Remove-Item -LiteralPath $lockPath -Force }
        } catch { }
    }
    throw
} finally {
    if ($null -ne $guard) { $guard.Dispose() }
}
