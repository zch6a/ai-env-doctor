"""Regression tests for the second independent review (POLISH_LOG, round 4). Each one failed before its fix."""

import json
import shutil
import subprocess
import sys

import pytest
from test_rules import INDEX, ids, py, rtx3060

from ai_env_doctor import cli, probe, pytorch
from ai_env_doctor.probe import GPU, System, env_snapshot, redact
from ai_env_doctor.rules import diagnose, install_cmd

CPU = {"version": "2.5.1+cpu", "cuda": None, "cuda_available": False}
WORKS = {"version": "2.14.1+cu126", "cuda": "12.6", "cuda_available": True, "cuda_works": True, "arch_list": ["sm_86"],
         "devices": [{"name": "NVIDIA GeForce RTX 3090"}]}
AWKWARD = [r"C:\Users\O'Brien\proj\.venv\Scripts\python.exe", r"D:\R&D\.venv\Scripts\python.exe",
           r"C:\Users\x\proj(1)\.venv\Scripts\python.exe", r"C:\Users\a b\$work\.venv\Scripts\python.exe"]
AWKWARD_POSIX = ["/home/o'brien/.venv/bin/python", "/srv/r&d/.venv/bin/python", "/home/u/my $proj/.venv/bin/python"]


def powershell_words(line: str) -> list[str] | None:
    """How PowerShell's own parser splits `line` (None: it does not parse)."""
    ps = shutil.which("powershell") or shutil.which("pwsh")
    script = ("$e=$null; $t=$null; $a=[System.Management.Automation.Language.Parser]::ParseInput($env:LINE,[ref]$t,[ref]$e);"
              "if ($e.Count) { 'ERROR' } else { $a.EndBlock.Statements[0].PipelineElements[0].CommandElements | "
              "ForEach-Object { if ($_ -is [System.Management.Automation.Language.ExpandableStringExpressionAst]) { 'EXPANDS:' + $_.Value } "
              "elseif ($_.Value -ne $null) { $_.Value } else { $_.Extent.Text } } }")  # "...$work..." would lose $work
    r = subprocess.run([ps, "-NoProfile", "-Command", script], capture_output=True, text=True, encoding="utf-8",
                       env={**__import__("os").environ, "LINE": line})
    out = r.stdout.strip().splitlines()
    return None if out == ["ERROR"] else out


@pytest.mark.skipif(not (shutil.which("powershell") or shutil.which("pwsh")), reason="needs PowerShell")
@pytest.mark.parametrize("path", AWKWARD)
def test_powershell_reads_the_interpreter_path_as_one_word(path):
    line = install_cmd(py(CPU, packages={"pip": "24", "torch": "2.5.1"}, path=path), "cu126", "windows").splitlines()[0]
    words = powershell_words(line)
    assert words is not None, line
    assert path in words, (line, words)


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("bash"), reason="needs a POSIX bash")
@pytest.mark.parametrize("path", AWKWARD_POSIX)
def test_bash_reads_the_interpreter_path_as_one_word(path):
    line = install_cmd(py(CPU, packages={"pip": "24", "torch": "2.5.1"}, path=path), "cu126", "linux").splitlines()[0]
    r = subprocess.run(["bash", "-c", 'set -- ' + line + '; printf "%s\\n" "$1"'], capture_output=True, text=True)
    assert r.stdout.strip() == path, (line, r.stdout, r.stderr)


def test_a_driver_without_nvidia_smi_is_not_a_missing_driver():
    linux = System("linux", "6", "x86_64", ram_mb=32768, gpu_seen_by_os=["NVIDIA Corporation GA102 [GeForce RTX 3090]"])
    f = diagnose(linux, [py(WORKS)])
    assert "driver-missing" not in ids(f) and "nvidia-smi-missing" in ids(f, "warning") and "torch-cuda" in ids(f, "ok")
    loaded = System("linux", "6", "x86_64", ram_mb=32768, nvidia_kernel_driver="575.51",
                    gpu_seen_by_os=["NVIDIA Corporation GA102 [GeForce RTX 3090]"])
    f = diagnose(loaded, [py(CPU)])
    assert "nvidia-smi-missing" in ids(f) and "torch-cpu" in ids(f, "error"), "the CPU build is still the problem"
    windows = rtx3060(gpus=[], nvidia_smi=None, driver_cuda=None, gpu_seen_by_os=["NVIDIA GeForce GTX 1060 6GB"])
    assert "driver-missing" not in ids(diagnose(windows, [py(CPU)]))
    really_missing = System("linux", "6", "x86_64", ram_mb=32768, gpu_seen_by_os=["NVIDIA Corporation GA102 [GeForce RTX 3090]"])
    assert "driver-missing" in ids(diagnose(really_missing, [py(CPU)]), "error")


def test_proxy_credentials_are_hidden_with_or_without_a_scheme():
    env = env_snapshot({"HTTPS_PROXY": "ann:s3cret@proxy.corp:8080", "HTTP_PROXY": "http://ann:pa/ss@proxy:1"}, "linux")
    dumped = json.dumps(env)
    assert "s3cret" not in dumped and "pa/ss" not in dumped and "ann" not in dumped
    assert redact("https://pypi.example/simple/pkg@1.0") == "https://pypi.example/simple/pkg@1.0", "an index path keeps its @"


def test_a_failing_gpu_environment_is_not_offered_as_the_working_one():
    s = rtx3060(gpus=[GPU("nvidia", "RTX 5090", 32607, 0, "12.0", "575")], driver_cuda="12.9")
    broken = {"version": "2.5.1+cu124", "cuda": "12.4", "cuda_available": True, "cuda_works": False, "arch_list": ["sm_90"]}
    f = diagnose(s, [py(broken), py(CPU, path="C:/b/python.exe")], {"torch_index": INDEX})
    assert "environments-disagree" not in ids(f)


def test_containers_wsl1_and_unknown_drivers_get_their_own_fixes():
    box = System("linux", "6", "x86_64", ram_mb=8192, container=True, gpu_seen_by_os=["NVIDIA GPU (PCI 0000:01:00.0)"])
    f = diagnose(box, [py(CPU)])
    assert "container-no-gpu" in ids(f, "error") and "driver-missing" not in ids(f)
    one = System("linux", "4.4", "x86_64", wsl=True, wsl1=True, ram_mb=8192, env={"WSL_DISTRO_NAME": "Ubuntu"})
    f = next(x for x in diagnose(one, [py(CPU)]) if x.id == "wsl1")
    assert "wsl --set-version Ubuntu 2" in f.fix and "PowerShell" in f.fix
    assert probe.wsl1("Linux version 4.4.0-19041-Microsoft") and not probe.wsl1("Linux version 5.15.153.1-microsoft-standard-WSL2")
    blackwell = rtx3060(gpus=[GPU("nvidia", "RTX 5090", 32607, 0, "12.0", "575")], driver_cuda=None)
    f = diagnose(blackwell, [py(None)], {"torch_index": INDEX})
    assert "driver-too-old-for-gpu" not in ids(f) and "newest-needs-driver" not in ids(f)
    assert pytorch.advise(None, "12.0").tag == "cu128" and pytorch.advise(None, "8.6").tag == "cu126"


def test_amd_advice_fits_the_gpu_and_the_system():
    laptop = System("windows", "10", "x86_64", ram_mb=16384, gpu_seen_by_os=["AMD Radeon(TM) Graphics"])
    f = diagnose(laptop, [py(CPU)])
    assert "torch-cpu-amd-windows" not in ids(f) and "gpu-amd" not in ids(f), "integrated graphics are no ROCm target"
    wsl = System("linux", "5.15", "x86_64", wsl=True, wsl_gpu=True, ram_mb=32768, gpu_seen_by_os=["AMD Radeon RX 7900 XTX"])
    f = next(x for x in diagnose(wsl, [py(CPU)]) if x.id.startswith("torch-cpu-amd"))
    assert f.id == "torch-cpu-amd-wsl" and "/wsl/" in f.fix
    cuda_on_amd = System("linux", "6", "x86_64", ram_mb=32768, gpu_seen_by_os=["Navi 31 [Radeon RX 7900 XTX]"])
    built = {"version": "2.14.1+cu126", "cuda": "12.6", "cuda_available": False}
    f = next(x for x in diagnose(cuda_on_amd, [py(built)]) if x.where)
    assert f.id == "torch-cuda-on-amd" and "ROCm" in f.fix


def test_an_intel_macs_amd_gpu_through_metal():
    mac = System("macos", "13", "x86_64", ram_mb=16384)
    mps = {"version": "2.2.2", "cuda": None, "cuda_available": False, "mps_built": True, "mps_available": True}
    f = diagnose(mac, [py(mps)])
    assert "gpu-mac-metal" in ids(f, "ok") and "torch-mps-metal" in ids(f, "ok") and "gpu-none" not in ids(f)


def test_pip_in_a_system_python_and_conda_only_for_torchvision():
    system_py = py(CPU, packages={"pip": "24", "torch": "2.5.1"}, path="/usr/bin/python3", externally_managed=True)
    f = diagnose(rtx3060(os="linux"), [system_py], {"torch_index": INDEX})
    fix = next(x for x in f if x.id == "torch-cpu").fix
    assert fix.startswith("/usr/bin/python3 -m venv ~/.venvs/ai") and "externally-managed" in ids(f, "info")
    vision = py(CPU, packages={"pip": "24", "torch": "2.5.1", "torchvision": "0.20.1"}, path="/opt/c/envs/ml/bin/python",
                conda={"prefix": "/opt/c/envs/ml", "packages": {"torchvision": {"version": "0.20.1", "build": "py_cpu", "channel": "pytorch"}}})
    fix = next(x for x in diagnose(rtx3060(os="linux"), [vision], {"torch_index": INDEX}) if x.id == "torch-cpu").fix
    assert fix.startswith("conda remove -p /opt/c/envs/ml torchvision && /opt/c/envs/ml/bin/python -m pip uninstall -y torch && ")


def test_cuda_visible_devices_past_the_last_gpu():
    f = diagnose(rtx3060(env={"CUDA_VISIBLE_DEVICES": "1"}), [])
    assert "cuda-visible-devices-range" in ids(f, "warning")
    assert "cuda-visible-devices-range" not in ids(diagnose(rtx3060(env={"CUDA_VISIBLE_DEVICES": "0"}), []))


def test_the_doctors_own_python_home_is_not_passed_to_the_users_python():
    """Found by running it on a real Miniconda: uv's launcher had set PYTHONHOME to uv's Python, so the
    Miniconda Python probed reported uv's prefix, uv's EXTERNALLY-MANAGED marker, and no conda-meta."""
    from ai_env_doctor import pyprobe
    uv_home = r"C:\Users\me\AppData\Roaming\uv\python\cpython-3.14-windows-x86_64-none"
    env = pyprobe.probe_env({"PYTHONHOME": uv_home, "UV_INTERNAL__PYTHONHOME": uv_home, "__PYVENV_LAUNCHER__": "x", "PATH": "p"})
    assert "PYTHONHOME" not in env and "__PYVENV_LAUNCHER__" not in env and env["PATH"] == "p"
    assert pyprobe.probe_env({"PYTHONHOME": r"D:\my\python"})["PYTHONHOME"] == r"D:\my\python", "a PYTHONHOME the user set stays"


def test_a_python_home_leak_reaches_no_real_child(monkeypatch, tmp_path):
    from ai_env_doctor import pyprobe
    monkeypatch.setenv("UV_INTERNAL__PYTHONHOME", str(tmp_path))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path))  # a home with no standard library: Python would not even start
    it = pyprobe.probe(sys.executable)
    assert it.error is None and it.info["python"]


@pytest.mark.skipif(not (shutil.which("powershell") or shutil.which("pwsh")), reason="needs PowerShell")
def test_the_venv_fix_parses_in_powershell():
    managed = py(CPU, packages={"pip": "24", "torch": "2.5.1"}, path=r"C:\Users\me\AppData\Roaming\uv\python\cpython-3.12\python.exe",
                 externally_managed=True)
    first, second = install_cmd(managed, "cu126", "windows").splitlines()
    assert first.endswith(r"-m venv $HOME\.venvs\ai") and powershell_words(first) is not None
    assert powershell_words(second) is not None and second.startswith('& "$HOME')


def test_the_json_says_its_schema(monkeypatch, capsys):
    monkeypatch.setattr(probe, "system", lambda: rtx3060())
    monkeypatch.setattr(cli.pyprobe, "interpreters", lambda explicit=None: [])
    cli.main(["--offline", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == 1 and out["models"]["as_of"] == "2026-11"
