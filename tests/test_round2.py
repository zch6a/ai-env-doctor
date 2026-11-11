"""Round 2 (POLISH_LOG): conda environments, a dated model list, WSL and machines without lspci."""

import contextlib
import io
import json
from datetime import date

from test_rules import INDEX, ids, py, rtx3060

from ai_env_doctor import cli, models, probe, pyprobe
from ai_env_doctor.probe import System
from ai_env_doctor.rules import diagnose

CPU_BUILD = {"version": "2.5.1", "cuda": None, "cuda_available": False}
PREFIX = r"C:\Users\me\miniconda3\envs\ml"


def conda_env(packages, pip=True):
    conda = {"prefix": PREFIX, "packages": {name: {"version": "2.5.1", "build": "py3.12_cpu_0", "channel": "pytorch"} for name in packages}}
    pkgs = {"torch": "2.5.1", "torchvision": "0.20.1", **({"pip": "24.2"} if pip else {})}
    return py(CPU_BUILD, packages=pkgs, path=PREFIX + r"\python.exe", why="conda", conda=conda)


def test_conda_installed_pytorch_is_removed_with_conda_then_installed_with_pip():
    env = conda_env(["pytorch", "torchvision", "cpuonly", "pytorch-mutex"])
    f = diagnose(rtx3060(), [env], {"torch_index": INDEX})
    cpu = next(x for x in f if x.id == "torch-cpu")
    assert cpu.fix.splitlines() == [
        rf"conda remove -p {PREFIX} pytorch torchvision cpuonly",  # pip cannot uninstall these; cpuonly pins the CPU build
        rf"{PREFIX}\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126"]
    note = next(x for x in f if x.id == "torch-from-conda-cpuonly")
    assert note.level == "info" and "pytorch" in note.title and "cpuonly" in note.detail


def test_torch_from_pip_in_a_conda_env_is_uninstalled_with_pip():
    env = conda_env(["cpuonly"])  # cpuonly left over, torch itself from pip
    fix = next(x for x in diagnose(rtx3060(), [env], {"torch_index": INDEX}) if x.id == "torch-cpu").fix
    assert fix.splitlines()[0] == rf"conda remove -p {PREFIX} cpuonly"
    assert fix.splitlines()[1] == rf"{PREFIX}\python.exe -m pip uninstall -y torch torchvision", "else pip says 'already satisfied'"
    plain = conda_env([])  # a conda env, nothing torch from conda: the ordinary pip fix, no note
    f = diagnose(rtx3060(), [plain], {"torch_index": INDEX})
    assert "conda remove" not in next(x for x in f if x.id == "torch-cpu").fix and "torch-from-conda" not in ids(f)


def test_a_conda_prefix_with_spaces_is_quoted():
    env = conda_env(["pytorch"])
    env.info["conda"]["prefix"] = r"C:\Program Files\conda\envs\ml"
    fix = next(x for x in diagnose(rtx3060(), [env], {"torch_index": INDEX}) if x.id == "torch-cpu").fix
    assert fix.startswith('conda remove -p "C:\\Program Files\\conda\\envs\\ml" pytorch')


def test_the_probe_reads_what_conda_installed(tmp_path, monkeypatch):
    meta = tmp_path / "conda-meta"
    meta.mkdir()
    (meta / "pytorch-2.5.1-py3.12_cpu_0.json").write_text(json.dumps({"channel": "https://conda.anaconda.org/pytorch/win-64"}))
    (meta / "cpuonly-2.0-0.json").write_text("{}")
    (meta / "numpy-2.1.0-py312_0.json").write_text("{}")
    (meta / "pytorch-lightning-2.4.0-pyhd8ed1ab_0.json").write_text("{}")  # not PyTorch itself
    monkeypatch.setattr("sys.prefix", str(tmp_path))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(f"PACKAGES = []\nCONDA_TORCH = {pyprobe.CONDA_TORCH!r}\n{pyprobe.PROBE}", {})
    info = json.loads(out.getvalue().split("AI_ENV_DOCTOR_JSON", 1)[1])
    assert info["conda"]["packages"] == {"pytorch": {"version": "2.5.1", "build": "py3.12_cpu_0", "channel": "win-64"},
                                         "cpuonly": {"version": "2.0", "build": "0", "channel": ""}}


def test_the_model_list_says_when_it_is_old(capsys, monkeypatch):
    assert not models.catalog_stale(date(2027, 3, 1))
    assert models.catalog_stale(date(2027, 9, 1))
    monkeypatch.setattr(models, "catalog_stale", lambda today=None: True)
    monkeypatch.setattr(probe, "system", lambda: rtx3060(gpus=[], nvidia_smi=None, driver_cuda=None))
    monkeypatch.setattr(pyprobe, "interpreters", lambda explicit=None: [])
    cli.main(["--offline", "--lang", "en"])
    assert "This list is from 2026-11" in capsys.readouterr().out


def test_wsl_without_nvidia_is_not_called_a_missing_nvidia_driver():
    intel_laptop = System("linux", "6", "x86_64", wsl=True, wsl_gpu=True, ram_mb=16384, gpu_seen_by_os=["Intel(R) Iris(R) Xe Graphics"])
    f = diagnose(intel_laptop, [py(CPU_BUILD)])
    assert "wsl-no-driver" not in ids(f) and "wsl-gpu-unknown" not in ids(f) and "gpu-none" in ids(f)
    unknown = System("linux", "6", "x86_64", wsl=True, wsl_gpu=True, ram_mb=16384)  # powershell.exe could not be run
    f = diagnose(unknown, [py(CPU_BUILD)])
    assert "wsl-gpu-unknown" in ids(f, "info") and not ids(f, "error")
    nvidia = System("linux", "6", "x86_64", wsl=True, wsl_gpu=True, ram_mb=16384, gpu_seen_by_os=["NVIDIA GeForce RTX 4070"])
    assert "wsl-no-driver" in ids(diagnose(nvidia, [py(CPU_BUILD)]), "error")


def test_gpus_are_found_without_lspci(tmp_path):
    # Real names are 0000:01:00.0; Windows, where the tests also run, allows no colons in file names.
    for name, cls, vendor in [("0000-01-00.0", "0x030000", "0x10de"), ("0000-00-02.0", "0x030000", "0x8086"),
                              ("0000-00-1f.3", "0x040300", "0x8086"), ("0000-03-00.0", "0x038000", "0x1002")]:
        d = tmp_path / name
        d.mkdir()
        (d / "class").write_text(cls + "\n")
        (d / "vendor").write_text(vendor + "\n")
    assert probe.pci_display_devices(tmp_path) == ["Intel GPU (PCI 0000-00-02.0)", "NVIDIA GPU (PCI 0000-01-00.0)", "AMD GPU (PCI 0000-03-00.0)"]
    assert probe.pci_display_devices(tmp_path / "missing") == []


def test_the_language_follows_posix_precedence(monkeypatch):
    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")
    assert cli.default_lang() == "zh"
    monkeypatch.setenv("LC_ALL", "C")
    assert cli.default_lang() == "en", "LC_ALL overrides LANG"


def test_continuation_lines_line_up_under_a_chinese_label():
    lines = [ln for ln in _report_with_two_line_fix("zh").splitlines() if "uv pip" in ln]
    assert len(lines) == 2
    first, second = lines
    assert cli._width(first[: first.index("uv pip")]) == len(second) - len(second.lstrip()) == 12


def _report_with_two_line_fix(lang):
    f = diagnose(rtx3060(), [py(CPU_BUILD, packages={"torch": "2.5.1"})], {"torch_index": INDEX}, lang)
    return cli.report(rtx3060(), [x for x in f if x.id == "torch-cpu"], models.plan(rtx3060()), lang, False, True)
