"""What this machine has: the operating system, memory, disk, GPUs and their driver, and the environment
variables that change how AI tools behave. Everything here only reads: it runs `nvidia-smi` and the
like, reads files under /proc and /sys, and asks the operating system; it installs and changes nothing.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class GPU:
    vendor: str  # nvidia, amd, apple, intel, other
    name: str
    memory_mb: int | None = None  # dedicated memory; for Apple, the unified memory the GPU may use
    used_mb: int | None = None
    compute_capability: str | None = None  # NVIDIA, such as "8.6"
    driver: str | None = None


@dataclass
class System:
    os: str  # windows, linux, macos
    os_version: str
    machine: str  # the hardware: x86_64, arm64 ...
    wsl: bool = False
    rosetta: bool = False  # an x86_64 process on Apple Silicon
    ram_mb: int | None = None
    ram_free_mb: int | None = None
    cpu: str = ""
    cores: int | None = None
    gpus: list[GPU] = field(default_factory=list)
    driver_cuda: str | None = None  # the newest CUDA the NVIDIA driver supports ("CUDA Version" in nvidia-smi)
    nvidia_smi: str | None = None  # where nvidia-smi was found, or None
    gpu_seen_by_os: list[str] = field(default_factory=list)  # display adapters the OS lists, driver or not
    disk_free_mb: dict = field(default_factory=dict)  # cache folder -> free space where it lives
    env: dict = field(default_factory=dict)  # the variables that matter (see ENV_KEYS)

    def to_dict(self) -> dict:
        return asdict(self)


ENV_KEYS = ["CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "PYTORCH_CUDA_ALLOC_CONF", "HIP_VISIBLE_DEVICES",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy",
            "HF_ENDPOINT", "HF_HOME", "HF_HUB_OFFLINE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HUB_DISABLE_SSL_VERIFICATION",
            "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "UV_INDEX_URL",
            "UV_DEFAULT_INDEX", "PIP_PROXY", "PIP_CERT", "OLLAMA_HOST", "OLLAMA_MODELS", "VIRTUAL_ENV", "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV", "PYTHONPATH"]


def run(cmd: list[str], timeout: float = 15) -> str | None:
    """A command's output, or None if it is not there or fails."""
    exe = shutil.which(cmd[0])
    if not exe:
        return None
    try:
        out = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout, errors="replace",
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


# ---------------------------------------------------------------- NVIDIA

def parse_nvidia_query(text: str) -> list[GPU]:
    """`nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap --format=csv,noheader,nounits`"""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0]:
            continue

        def num(s):
            s = re.sub(r"[^\d.]", "", s)
            return int(float(s)) if s else None

        cap = parts[4] if len(parts) > 4 and re.match(r"^\d+\.\d+$", parts[4]) else None
        gpus.append(GPU("nvidia", parts[0], num(parts[1]), num(parts[2]), cap, parts[3] or None))
    return gpus


def parse_driver_cuda(text: str) -> str | None:
    """The "CUDA Version: 12.7" in nvidia-smi's banner: the newest CUDA this driver can run."""
    m = re.search(r"CUDA Version:\s*([\d.]+)", text or "")
    return m.group(1) if m else None


def nvidia() -> tuple[list[GPU], str | None, str | None]:
    smi = shutil.which("nvidia-smi")
    if not smi and sys.platform == "win32":
        guess = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        smi = str(guess) if guess.is_file() else None
    if not smi:
        return [], None, None
    query = run([smi, "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap", "--format=csv,noheader,nounits"])
    if query is None:  # drivers before 510 do not know compute_cap
        query = run([smi, "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"])
    banner = run([smi]) or ""
    return parse_nvidia_query(query or ""), parse_driver_cuda(banner), smi


# ---------------------------------------------------------------- what the OS lists, with or without a driver

def os_display_adapters() -> list[str]:
    if sys.platform == "win32":
        out = run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"])
        return [x.strip() for x in (out or "").splitlines() if x.strip()]
    if sys.platform.startswith("linux"):
        out = run(["lspci"])
        return [line.split(": ", 1)[-1].strip() for line in (out or "").splitlines()
                if re.search(r"VGA compatible|3D controller|Display controller", line)]
    return []


# ---------------------------------------------------------------- memory, CPU, Apple

class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memory() -> tuple[int | None, int | None]:
    try:
        if sys.platform == "win32":
            st = _MemoryStatus()
            st.dwLength = ctypes.sizeof(st)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullTotalPhys // 2**20, st.ullAvailPhys // 2**20
        elif sys.platform.startswith("linux"):
            info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
            kb = lambda k: int(info[k].split()[0]) if k in info else None  # noqa: E731
            total, avail = kb("MemTotal"), kb("MemAvailable")
            return (total // 1024 if total else None), (avail // 1024 if avail else None)
        elif sys.platform == "darwin":
            total = run(["sysctl", "-n", "hw.memsize"])
            return (int(total) // 2**20 if total and total.strip().isdigit() else None), None
    except (OSError, ValueError, AttributeError):
        pass
    return None, None


def cpu_name() -> str:
    if sys.platform == "darwin":
        return (run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip()
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "win32":
        out = run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"])
        if out and out.strip():
            return out.strip()
    return platform.processor()


def apple_gpu(ram_mb: int | None) -> GPU | None:
    """Apple Silicon: the GPU shares memory with the CPU; by default macOS lets it use about three
    quarters of it (two thirds on machines with 36 GB or less)."""
    if sys.platform != "darwin":
        return None
    brand = cpu_name()
    arm = "Apple" in brand or platform.machine() == "arm64" or rosetta()
    if not arm:
        return None
    usable = None
    if ram_mb:
        usable = int(ram_mb * (0.75 if ram_mb > 36 * 1024 else 0.66))
    return GPU("apple", brand or "Apple Silicon", usable)


def rosetta() -> bool:
    """An x86_64 Python on Apple Silicon runs under Rosetta: PyTorch then has no MPS."""
    if sys.platform != "darwin":
        return False
    out = run(["sysctl", "-n", "sysctl.proc_translated"])
    return (out or "").strip() == "1"


def wsl() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


# ---------------------------------------------------------------- disk

def cache_dirs() -> dict[str, Path]:
    """Where models are kept: the Hugging Face cache and Ollama's models folder."""
    home = Path.home()
    hf = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or home / ".cache" / "huggingface") / "hub")
    ollama = Path(os.environ.get("OLLAMA_MODELS") or home / ".ollama" / "models")
    return {"huggingface": hf, "ollama": ollama}


def free_space(path: Path) -> int | None:
    p = path
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        return shutil.disk_usage(p).free // 2**20
    except OSError:
        return None


def system() -> System:
    os_name = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux" if sys.platform.startswith("linux") else sys.platform)
    ram, ram_free = memory()
    gpus, driver_cuda, smi = nvidia()
    s = System(os=os_name, os_version=platform.version() if os_name != "macos" else platform.mac_ver()[0],
               machine=platform.machine().lower().replace("amd64", "x86_64"), wsl=wsl(), rosetta=rosetta(),
               ram_mb=ram, ram_free_mb=ram_free, cpu=cpu_name(), cores=os.cpu_count(), gpus=gpus,
               driver_cuda=driver_cuda, nvidia_smi=smi)
    a = apple_gpu(ram)
    if a:
        s.gpus.append(a)
        if s.rosetta:
            s.machine = "arm64"
    s.gpu_seen_by_os = os_display_adapters() if not gpus else []
    s.disk_free_mb = {name: free_space(path) for name, path in cache_dirs().items()}
    s.env = {k: redact(os.environ[k]) for k in ENV_KEYS if k in os.environ}
    return s


def redact(value: str) -> str:
    """A proxy or index URL with its password (or token) hidden: "http://ann:secret@proxy:8080" ->
    "http://ann:***@proxy:8080". The report may be pasted into an issue."""
    return re.sub(r"(://[^/:@\s]*):[^/@\s]*@", r"\1:***@", re.sub(r"(://)[^/:@\s]+@", r"\1***@", value)) if "@" in value else value
