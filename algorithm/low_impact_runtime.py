"""Standard-library-only resource controls; affect this process and no other PID."""
import ctypes
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

START_AVAILABLE_RAM_GIB = 1.5
PAUSE_AVAILABLE_RAM_GIB = 1.0
RESUME_AVAILABLE_RAM_GIB = 1.25
CRITICAL_AVAILABLE_RAM_GIB = .75
PROCESS_MEMORY_LIMIT_MIB = 768
_process_memory_job = None


class StopRequested(Exception):
    pass


def set_thread_environment():
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS']:
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONUTF8'] = '1'


def limit_own_process_memory():
    """Cap this process's private commit; retain the Job handle until exit."""
    global _process_memory_job
    if _process_memory_job is not None:
        return
    from ctypes import wintypes
    class BasicLimits(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                    ('flags', wintypes.DWORD), ('min_ws', ctypes.c_size_t), ('max_ws', ctypes.c_size_t),
                    ('active', wintypes.DWORD), ('affinity', ctypes.c_size_t),
                    ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]
    class ExtendedLimits(ctypes.Structure):
        _fields_ = [('basic', BasicLimits), ('io', ctypes.c_ulonglong * 6),
                    ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                    ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]
    if ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(BasicLimits) != 64 or ctypes.sizeof(ExtendedLimits) != 144:
        raise RuntimeError('Unexpected Win64 memory-limit structure layout')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    kernel.QueryInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        limits = ExtendedLimits()
        limits.basic.flags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY; this process only.
        limits.process_memory = PROCESS_MEMORY_LIMIT_MIB * 1024 ** 2
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
            raise ctypes.WinError(ctypes.get_last_error())
        actual = ExtendedLimits()
        if not kernel.QueryInformationJobObject(job, 9, ctypes.byref(actual), ctypes.sizeof(actual), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if not actual.basic.flags & 0x100 or actual.process_memory != limits.process_memory:
            raise RuntimeError('Could not verify own process memory cap')
        _process_memory_job = job
    except BaseException:
        kernel.CloseHandle(job)
        raise


def protect_current_process():
    set_thread_environment()
    if os.name != 'nt':
        raise RuntimeError('This launch policy currently requires Windows')
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
    kernel.GetPriorityClass.restype = wintypes.DWORD
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    handle = kernel.GetCurrentProcess()
    if not kernel.SetPriorityClass(handle, 0x40):  # IDLE_PRIORITY_CLASS
        raise ctypes.WinError(ctypes.get_last_error())
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not process.value:
        raise RuntimeError('No allowed logical CPU')
    # Respect a launcher restriction; otherwise choose one allowed logical CPU.
    mask = 1 << (process.value.bit_length() - 1)
    if not kernel.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)):
        raise ctypes.WinError(ctypes.get_last_error())
    actual, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(actual), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    if kernel.GetPriorityClass(handle) != 0x40 or actual.value != mask:
        raise RuntimeError('Could not verify own Idle priority and single-CPU affinity')
    limit_own_process_memory()  # before importing numerical libraries
    return dict(pid=os.getpid(), priority='Idle', affinity_mask=mask, logical_cpus=1,
                device='cpu', threads=1, process_memory_limit_mib=PROCESS_MEMORY_LIMIT_MIB,
                start_available_ram_gib=START_AVAILABLE_RAM_GIB,
                pause_available_ram_gib=PAUSE_AVAILABLE_RAM_GIB,
                resume_available_ram_gib=RESUME_AVAILABLE_RAM_GIB,
                critical_available_ram_gib=CRITICAL_AVAILABLE_RAM_GIB)


def process_creation_time(pid):
    """None = dead; permission/unknown errors fail closed, not a stale-lock guess."""
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return None
        raise ctypes.WinError(error)
    try:
        values = [ctypes.c_ulonglong() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *[ctypes.byref(v) for v in values]):
            raise ctypes.WinError(ctypes.get_last_error())
        exit_code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return str(values[0].value) if exit_code.value == 259 else None
    finally:
        kernel.CloseHandle(handle)


class RunLock:
    def __init__(self, output):
        self.path = Path(output) / '.runner.lock'
        self.owner = dict(pid=os.getpid(), creation_time=process_creation_time(os.getpid()))

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                old = json.loads(self.path.read_text('utf-8'))
                actual = process_creation_time(int(old['pid']))
                if actual is not None and actual == old['creation_time']:
                    raise RuntimeError(f'Output already owned by active PID {old["pid"]}')
                self.path.replace(self.path.with_name(f'.runner.stale.{time.time_ns()}.json'))
            else:
                with os.fdopen(fd, 'w', encoding='utf-8') as file:
                    json.dump(self.owner, file)
                return self
        raise RuntimeError('Could not acquire exclusive output lock')

    def __exit__(self, *args):
        if self.path.exists() and json.loads(self.path.read_text('utf-8')) == self.owner:
            self.path.unlink()


class ResourceGuard:
    def __init__(self, output, status_callback, interval=1.):
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in ['total', 'available', 'total_page',
                    'available_page', 'total_virtual', 'available_virtual', 'available_extended']]
        self.memory_type = Memory
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.output, self.callback, self.interval = Path(output), status_callback, interval
        self.last_poll = 0.
        self.previous = self.cpu_times()

    def cpu_times(self):
        values = [ctypes.c_ulonglong() for _ in range(3)]
        if not self.kernel.GetSystemTimes(*[ctypes.byref(v) for v in values]):
            raise ctypes.WinError(ctypes.get_last_error())
        return tuple(v.value for v in values)  # idle, kernel (incl idle), user

    def sample(self):
        memory = self.memory_type()
        memory.length = ctypes.sizeof(memory)
        if not self.kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise ctypes.WinError(ctypes.get_last_error())
        current = self.cpu_times()
        delta = [a - b for a, b in zip(current, self.previous)]
        self.previous = current
        total = delta[1] + delta[2]
        busy = 100 * (1 - delta[0] / total) if total > 0 else None
        return dict(cpu_percent=busy, available_ram_gib=memory.available / 1024 ** 3)

    def check(self, force=False):
        if (self.output / 'STOP_REQUESTED').exists():
            raise StopRequested('Graceful stop requested; completed candidates remain resumable')
        if not force and time.monotonic() - self.last_poll < self.interval:
            return
        self.last_poll = time.monotonic()
        sample = self.sample()
        self.check_critical(sample)
        if sample['cpu_percent'] is None or sample['cpu_percent'] > 85 or sample['available_ram_gib'] < PAUSE_AVAILABLE_RAM_GIB:
            while True:
                self.callback('waiting_for_resources', resources=sample)
                for _ in range(5):
                    if (self.output / 'STOP_REQUESTED').exists():
                        raise StopRequested('Stopped while waiting for resources')
                    time.sleep(1)
                sample = self.sample()
                self.check_critical(sample)
                if sample['cpu_percent'] is not None and sample['cpu_percent'] <= 70 and sample['available_ram_gib'] >= RESUME_AVAILABLE_RAM_GIB:
                    self.callback('running', resources=sample)
                    break

    def check_critical(self, sample):
        if sample['available_ram_gib'] < CRITICAL_AVAILABLE_RAM_GIB:
            raise StopRequested('Critical available RAM below 0.75 GiB; exiting to release memory. Completed candidates remain resumable.')

    def wait_for_start(self):
        """Recheck measured memory policy even when launched without the helper."""
        while True:
            if (self.output / 'STOP_REQUESTED').exists():
                raise StopRequested('Stopped before imports')
            time.sleep(1)
            sample = self.sample()
            if sample['cpu_percent'] is not None and sample['cpu_percent'] <= 70 and sample['available_ram_gib'] >= START_AVAILABLE_RAM_GIB:
                return
            self.callback('waiting_for_resources', resources=sample, stage='before_heavy_imports')
            for _ in range(30):
                if (self.output / 'STOP_REQUESTED').exists():
                    raise StopRequested('Stopped before imports')
                time.sleep(1)


def status_writer(output):
    """Small atomic status; no numeric libraries and no more than one write/5s."""
    output = Path(output)
    state = dict(pid=os.getpid())
    last_write = 0.
    def status(phase, **fields):
        nonlocal last_write
        previous = state.get('status')
        state.update(status=phase, **fields)
        now = time.monotonic()
        if previous != phase or now - last_write >= 5:
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            temporary = output / 'status.json.tmp'
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            temporary.replace(output / 'status.json')
            last_write = now
    return status
