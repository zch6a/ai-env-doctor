"""The Python environments that will run the user's AI code, and what PyTorch they have.

ai-env-doctor runs in its own environment (with uvx, an isolated one without torch), so it never imports
torch itself: it runs a small script *in the user's interpreter* and reads the JSON it prints. The
script imports torch (the only way to know whether CUDA initializes) and reads other packages' versions
from their metadata without importing them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

PACKAGES = ["pip", "torch", "torchvision", "torchaudio", "transformers", "accelerate", "bitsandbytes", "xformers", "flash-attn", "triton",
            "vllm", "llama-cpp-python", "onnxruntime", "onnxruntime-gpu", "tensorflow", "jax", "jaxlib", "mlx", "sentence-transformers",
            "diffusers", "peft", "huggingface-hub", "numpy", "nvidia-cudnn-cu12", "nvidia-cuda-runtime-cu12"]
# conda's names for PyTorch and the packages that choose its build (cpuonly pins the CPU one).
CONDA_TORCH = ["pytorch", "torchvision", "torchaudio", "cpuonly", "pytorch-cuda", "pytorch-gpu", "pytorch-cpu", "libtorch"]

PROBE = r'''
import json, os, platform, sys, warnings
out = {"python": sys.version.split()[0], "executable": sys.executable, "bits": 64 if sys.maxsize > 2**32 else 32,
       "machine": platform.machine(), "packages": {}, "torch": None}
try:
    from importlib import metadata
    for name in PACKAGES:
        try:
            out["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
except Exception as e:
    out["metadata_error"] = repr(e)
try:  # a conda environment: which torch packages conda itself installed (pip cannot remove those)
    import os
    meta = os.path.join(sys.prefix, "conda-meta")
    if os.path.isdir(meta):
        mine = {}
        for f in os.listdir(meta):
            parts = f[:-5].rsplit("-", 2) if f.endswith(".json") else []
            if len(parts) == 3 and parts[0] in CONDA_TORCH:
                try:
                    with open(os.path.join(meta, f), encoding="utf-8") as fh:
                        channel = str(json.load(fh).get("channel") or "")
                except Exception:
                    channel = ""
                # ".../pkgs/main/win-64" -> "main", "https://conda.anaconda.org/pytorch/noarch" -> "pytorch"
                segs = [s for s in channel.rstrip("/").split("/") if s]
                if segs and (segs[-1] == "noarch" or "-" in segs[-1] and segs[-1].split("-")[0] in ("win", "linux", "osx")):
                    segs = segs[:-1]
                mine[parts[0]] = {"version": parts[1], "build": parts[2], "channel": segs[-1] if segs else ""}
        out["conda"] = {"prefix": sys.prefix, "packages": mine}
except Exception as e:
    out["conda_error"] = repr(e)
try:  # PEP 668: a Python the system's package manager owns (Debian, Ubuntu, Homebrew); pip refuses to install into it
    import sysconfig
    out["externally_managed"] = sys.prefix == getattr(sys, "base_prefix", sys.prefix) and os.path.isfile(
        os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED"))
except Exception:
    pass
if True:  # torch is tried even without metadata: a source build or a copied folder may have none
    t = {}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import torch
            t["version"] = torch.__version__
            t["cuda"] = getattr(torch.version, "cuda", None)
            t["hip"] = getattr(torch.version, "hip", None)
            t["git"] = getattr(torch.version, "git_version", None)
            try:
                t["cuda_available"] = bool(torch.cuda.is_available())
            except Exception as e:
                t["cuda_available"] = False
                t["cuda_error"] = repr(e)
            try:
                t["arch_list"] = list(torch.cuda.get_arch_list()) if t.get("cuda") or t.get("hip") else []
            except Exception:
                t["arch_list"] = []
            t["devices"] = []
            if t["cuda_available"]:
                try:
                    for i in range(torch.cuda.device_count()):
                        p = torch.cuda.get_device_properties(i)  # where CUDA first starts, and can fail
                        t["devices"].append({"name": p.name, "memory_mb": p.total_memory // 2**20,
                                             "capability": "%d.%d" % (p.major, p.minor)})
                    # A kernel that runs on the GPU: copying memory alone would not show a missing kernel image.
                    (torch.ones(1, device="cuda") + 1).item()
                    t["cuda_works"] = True
                except Exception as e:
                    t["cuda_works"] = False
                    t["cuda_error"] = repr(e)[:600]
            try:
                t["cudnn"] = torch.backends.cudnn.version() if t["cuda_available"] else None
            except Exception:
                t["cudnn"] = None
            mps = getattr(torch.backends, "mps", None)
            t["mps_built"] = bool(mps and mps.is_built())
            t["mps_available"] = bool(mps and mps.is_available())
        t["warnings"] = [str(w.message)[:400] for w in caught]
    except ImportError as e:
        t = None if "No module named 'torch'" in str(e) else {"import_error": repr(e)[:600]}
    except Exception as e:
        t = {"import_error": repr(e)[:600]}
    out["torch"] = t
print("AI_ENV_DOCTOR_JSON" + json.dumps(out))
'''


@dataclass
class Interpreter:
    path: str
    why: str  # how it was found: venv, virtualenv, conda, path, given
    origin: str = ""  # the folder, environment or command name it was found by
    info: dict = field(default_factory=dict)  # what the probe printed
    error: str | None = None


def candidates(cwd: Path | None = None) -> list[tuple[str, str, str]]:
    """(path, why, origin) for each interpreter the user is likely to mean, most likely first, each once."""
    cwd = cwd or Path.cwd()
    out: list[tuple[str, str, str]] = []
    exe = "python.exe" if sys.platform == "win32" else "python"
    sub = "Scripts" if sys.platform == "win32" else "bin"
    for folder in (cwd, *cwd.parents[:3]):
        for name in (".venv", "venv", "env"):
            p = folder / name / sub / exe
            if p.is_file():
                out.append((str(p), "venv", str(folder / name)))
                break
        else:
            continue
        break
    if os.environ.get("VIRTUAL_ENV"):
        out.append((str(Path(os.environ["VIRTUAL_ENV"]) / sub / exe), "virtualenv", os.environ["VIRTUAL_ENV"]))
    if os.environ.get("CONDA_PREFIX"):
        conda_py = Path(os.environ["CONDA_PREFIX"]) / (exe if sys.platform == "win32" else f"bin/{exe}")
        out.append((str(conda_py), "conda", os.environ.get("CONDA_DEFAULT_ENV", "?")))
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            out.append((found, "path", name))
    if sys.platform == "win32" and shutil.which("py"):  # the Python launcher's default, often the Python people use
        launched = _py_launcher_default()
        if launched:
            out.append((launched, "path", "py"))
    seen, unique = set(), []
    for path, why, origin in out:
        # Not resolved: a venv's python is a link to its base interpreter, and both must be checked.
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen and Path(path).exists():
            seen.add(key)
            unique.append((path, why, origin))
    return unique


def _py_launcher_default() -> str | None:
    try:
        r = subprocess.run(["py", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True, timeout=20,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    path = r.stdout.strip()
    return path if r.returncode == 0 and path else None


def windows_store_stub(path: str, returncode: int | None = None, output: str = "") -> bool:
    """The `python` that Windows puts in WindowsApps when no Python is installed opens the Microsoft Store
    instead of running Python. Told by what it does (exit code 9009, or nothing but a message), not by
    where it is: a Python really installed from the Store lives there too."""
    return sys.platform == "win32" and "windowsapps" in path.lower() and (returncode == 9009 or "AI_ENV_DOCTOR_JSON" not in output)


def probe_env(environ) -> dict:
    """The user's environment for their interpreter, minus what the doctor's own launch put there.

    On Windows uv's launcher for a venv's python.exe (so `uvx`, and `uv run`) sets PYTHONHOME to its
    base Python (and says so in UV_INTERNAL__PYTHONHOME); CPython's own venv launcher sets
    __PYVENV_LAUNCHER__. Passed on, either makes every other Python believe it lives there: another
    sys.prefix, another standard library, no conda-meta, uv's EXTERNALLY-MANAGED marker. A PYTHONHOME
    the user set is theirs and stays."""
    ours = {os.path.normcase(p) for p in (environ.get("UV_INTERNAL__PYTHONHOME"), sys.base_prefix) if p}
    env = {k: v for k, v in environ.items() if k.upper() not in ("__PYVENV_LAUNCHER__", "UV_INTERNAL__PYTHONHOME")}
    for k in [k for k in env if k.upper() == "PYTHONHOME"]:
        if os.path.normcase(env[k].rstrip("\\/")) in {p.rstrip("\\/") for p in ours}:
            del env[k]
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def probe(path: str, why: str = "", origin: str = "", timeout: float = 90) -> Interpreter:
    it = Interpreter(path, why, origin)
    script = f"PACKAGES = {PACKAGES!r}\nCONDA_TORCH = {CONDA_TORCH!r}\n{PROBE}"
    try:
        # From a neutral folder: with -c, the current folder comes first on sys.path, and a project's own
        # json.py (or torch/ folder) would be imported instead of the real one.
        r = subprocess.run([path, "-c", script], capture_output=True, text=True, timeout=timeout, errors="replace",
                           cwd=tempfile.gettempdir(), env=probe_env(os.environ),
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        it.error = f"timed out after {timeout:g} s (importing torch can hang when a driver is stuck)"
        return it
    except OSError as e:
        it.error = str(e)
        return it
    if windows_store_stub(path, r.returncode, r.stdout):
        it.error = "store-stub"
        return it
    at = r.stdout.rfind("AI_ENV_DOCTOR_JSON")  # anywhere: something may print before it without a newline
    if at < 0:
        it.error = (r.stderr or r.stdout).strip()[-600:] or f"exited with {r.returncode}"
        return it
    try:
        it.info = json.loads(r.stdout[at + len("AI_ENV_DOCTOR_JSON"):].splitlines()[0])
    except ValueError as e:
        it.error = f"unreadable probe output ({e})"
    return it


def interpreters(explicit: list[str] | None = None, cwd: Path | None = None) -> list[Interpreter]:
    found = [(p, "given", p) for p in explicit] if explicit else candidates(cwd)
    return [probe(p, why, origin) for p, why, origin in found]
