import sys

import pytest
from faketorch import make_torch

from ai_env_doctor import pyprobe
from ai_env_doctor.probe import parse_driver_cuda, parse_nvidia_query, redact

SMI_QUERY = "NVIDIA GeForce RTX 3060 Laptop GPU, 6144, 1712, 566.36, 8.6\nNVIDIA RTX A6000, 49140, 3, 570.12, 8.6\n"
SMI_BANNER = "| NVIDIA-SMI 566.36                 Driver Version: 566.36         CUDA Version: 12.7     |"


def test_nvidia_smi_is_read():
    gpus = parse_nvidia_query(SMI_QUERY)
    assert [(g.name, g.memory_mb, g.compute_capability, g.driver) for g in gpus] == [
        ("NVIDIA GeForce RTX 3060 Laptop GPU", 6144, "8.6", "566.36"), ("NVIDIA RTX A6000", 49140, "8.6", "570.12")]
    assert parse_driver_cuda(SMI_BANNER) == "12.7"
    assert parse_nvidia_query("") == [] and parse_driver_cuda("no banner") is None
    old = parse_nvidia_query("Tesla K80, 11441 MiB, 0 MiB, 470.82\n")  # an old driver, no compute_cap, with units
    assert old[0].memory_mb == 11441 and old[0].compute_capability is None


def test_passwords_in_proxy_urls_are_hidden():
    assert redact("http://ann:s3cret@proxy.example:8080") == "http://***@proxy.example:8080"
    assert redact("https://TOKEN123@pypi.example/simple") == "https://***@pypi.example/simple"
    assert redact("https://TOKEN_abc123:@pypi.fury.io/user/") == "https://***@pypi.fury.io/user/", "a token as the user name"
    assert redact("http://ann:p@ss@proxy:8080") == "http://***@proxy:8080", "an @ inside the password"
    assert redact("https://host/simple?token=SECRET&x=1") == "https://host/simple?token=***&x=***"
    assert redact("http://proxy.example:8080") == "http://proxy.example:8080"


def run_probe(tmp_path, monkeypatch, **fake):
    folder = make_torch(tmp_path / "site", **fake)
    monkeypatch.setenv("PYTHONPATH", str(folder))
    return pyprobe.probe(sys.executable, "given", "test").info["torch"]


def test_a_working_cuda_build_is_seen_as_working(tmp_path, monkeypatch):
    t = run_probe(tmp_path, monkeypatch)
    assert t["version"] == "2.14.1+cu126" and t["cuda"] == "12.6" and t["cuda_available"] and t["cuda_works"]
    assert t["devices"][0]["capability"] == "8.6" and "sm_86" in t["arch_list"]


def test_a_cpu_build_and_a_failing_cuda_are_told_apart(tmp_path, monkeypatch):
    cpu = run_probe(tmp_path / "a", monkeypatch, version="2.15.0+cpu", cuda=None, available=False)
    assert cpu["cuda"] is None and not cpu["cuda_available"]
    broken = run_probe(tmp_path / "b", monkeypatch, available=False, warn="CUDA driver version is insufficient for CUDA runtime version")
    assert broken["cuda"] == "12.6" and not broken["cuda_available"] and "insufficient" in broken["warnings"][0]
    nokernel = run_probe(tmp_path / "c", monkeypatch, cuda_fails=True)
    assert nokernel["cuda_available"] and nokernel["cuda_works"] is False and "no kernel image" in nokernel["cuda_error"]


def test_no_torch_and_a_torch_that_fails_to_import(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "empty"))
    assert pyprobe.probe(sys.executable).info["torch"] is None, "the test venv has no torch"
    t = run_probe(tmp_path, monkeypatch, import_error="[WinError 126] fbgemm.dll not found")
    assert "fbgemm.dll" in t["import_error"]


def test_packages_are_read_without_importing_them():
    info = pyprobe.probe(sys.executable).info
    assert "pytest" not in info["packages"] and info["bits"] == 64 and info["python"].count(".") == 2


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_the_store_placeholder_is_told_by_what_it_does():
    stub = r"C:\Users\x\AppData\Local\Microsoft\WindowsApps\python.exe"
    assert pyprobe.windows_store_stub(stub, 9009, "")
    assert not pyprobe.windows_store_stub(stub, 0, 'AI_ENV_DOCTOR_JSON{"python": "3.12.1"}'), "a real Store Python is checked"
    assert not pyprobe.windows_store_stub(r"C:\Python312\python.exe", 9009, "")


def test_the_project_venv_comes_first(tmp_path, monkeypatch):
    sub = "Scripts" if sys.platform == "win32" else "bin"
    exe = tmp_path / "proj" / ".venv" / sub / ("python.exe" if sys.platform == "win32" else "python")
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    (tmp_path / "proj" / "src").mkdir()
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    found = pyprobe.candidates(tmp_path / "proj" / "src")
    assert found[0][0] == str(exe) and found[0][1] == "venv"


def test_a_python_that_does_not_run_is_reported(tmp_path):
    bad = tmp_path / ("python.exe" if sys.platform == "win32" else "python")
    bad.write_bytes(b"not a program")
    it = pyprobe.probe(str(bad))
    assert it.error and not it.info
