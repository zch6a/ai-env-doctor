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
from dataclasses import dataclass, field
from pathlib import Path

PACKAGES = ["torch", "torchvision", "torchaudio", "transformers", "accelerate", "bitsandbytes", "xformers", "flash-attn", "triton",
            "vllm", "llama-cpp-python", "onnxruntime", "onnxruntime-gpu", "tensorflow", "jax", "jaxlib", "mlx", "sentence-transformers",
            "diffusers", "peft", "huggingface-hub", "numpy", "nvidia-cudnn-cu12", "nvidia-cuda-runtime-cu12"]

PROBE = r'''
import json, platform, sys, warnings
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
                for i in range(torch.cuda.device_count()):
                    p = torch.cuda.get_device_properties(i)
                    t["devices"].append({"name": p.name, "memory_mb": p.total_memory // 2**20,
                                         "capability": "%d.%d" % (p.major, p.minor)})
                try:
                    torch.zeros(1).cuda()
                    t["cuda_works"] = True
                except Exception as e:
                    t["cuda_works"] = False
                    t["cuda_error"] = repr(e)
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
    seen, unique = set(), []
    for path, why, origin in out:
        try:
            key = os.path.normcase(str(Path(path).resolve()))
        except OSError:
            key = path
        if key not in seen and Path(path).exists():
            seen.add(key)
            unique.append((path, why, origin))
    return unique


def windows_store_stub(path: str) -> bool:
    """The `python` that Windows puts in WindowsApps opens the Microsoft Store instead of running Python."""
    return sys.platform == "win32" and "windowsapps" in path.lower()


def probe(path: str, why: str = "", origin: str = "", timeout: float = 90) -> Interpreter:
    it = Interpreter(path, why, origin)
    if windows_store_stub(path):
        it.error = "store-stub"
        return it
    script = f"PACKAGES = {PACKAGES!r}\n{PROBE}"
    try:
        r = subprocess.run([path, "-c", script], capture_output=True, text=True, timeout=timeout, errors="replace",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"}, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        it.error = f"timed out after {timeout:g} s (importing torch can hang when a driver is stuck)"
        return it
    except OSError as e:
        it.error = str(e)
        return it
    line = next((x for x in r.stdout.splitlines() if x.startswith("AI_ENV_DOCTOR_JSON")), None)
    if line is None:
        it.error = (r.stderr or r.stdout).strip()[-600:] or f"exited with {r.returncode}"
        return it
    try:
        it.info = json.loads(line[len("AI_ENV_DOCTOR_JSON"):])
    except ValueError as e:
        it.error = f"unreadable probe output ({e})"
    return it


def interpreters(explicit: list[str] | None = None, cwd: Path | None = None) -> list[Interpreter]:
    found = [(p, "given", p) for p in explicit] if explicit else candidates(cwd)
    return [probe(p, why, origin) for p, why, origin in found]
