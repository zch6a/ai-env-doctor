"""Regression tests for the first independent review (POLISH_LOG, round 1). Each one failed before its fix."""

import os
import ssl
import sys
import urllib.error

import pytest
from faketorch import make_torch
from test_rules import INDEX, ids, py, rtx3060

from ai_env_doctor import cli, models, net, probe, pyprobe, pytorch
from ai_env_doctor.probe import GPU, System, env_snapshot, parse_video_controllers
from ai_env_doctor.rules import diagnose, install_cmd

CPU_BUILD = {"version": "2.15.0+cpu", "cuda": None, "cuda_available": False}


def test_the_fix_runs_where_there_is_no_pip_and_where_the_path_has_spaces():
    uv_env = py(CPU_BUILD, packages={"torch": "2.15.0+cpu"})  # made by uv: no pip
    cmd = install_cmd(uv_env, "cu126", "windows")
    assert cmd.splitlines() == ["uv pip uninstall --python C:/p/.venv/Scripts/python.exe torch",
                                "uv pip install --python C:/p/.venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu126"]
    spaced = py(CPU_BUILD, packages={"pip": "25", "torch": "x", "torchvision": "y"}, path=r"C:\Program Files\Python312\python.exe")
    cmd = install_cmd(spaced, "cu126", "windows")
    assert cmd.startswith("& 'C:\\Program Files\\Python312\\python.exe' -m pip uninstall -y torch torchvision"), "PowerShell needs &"
    assert "--force-reinstall" not in cmd, "it would also reinstall numpy and every other dependency at its newest"
    assert install_cmd(spaced, "cu126", "linux").startswith("'C:\\Program Files")


def test_data_centre_blackwell_and_a_mixed_machine_get_a_build_for_the_newest_gpu():
    assert pytorch.advise("12.9", "10.0", pytorch.builds(INDEX, "3.12.1", "win_amd64")).tag == "cu128"  # B200 / GB200
    mixed = rtx3060(gpus=[GPU("nvidia", "RTX 3090", 24576, 0, "8.6", "575"), GPU("nvidia", "RTX 5090", 32607, 0, "12.0", "575")],
                    driver_cuda="12.9")
    f = diagnose(mixed, [py(CPU_BUILD)], {"torch_index": INDEX})
    assert next(x for x in f if x.id == "torch-cpu").fix.endswith("/cu128")
    old = {"version": "2.5.1+cu124", "cuda": "12.4", "cuda_available": True, "cuda_works": True, "arch_list": ["sm_80", "sm_86", "sm_90"]}
    arch = next(x for x in diagnose(mixed, [py(old)], {"torch_index": INDEX}) if x.id == "arch-missing")
    assert "RTX 5090" in arch.title, "the GPU that has no code is named, not the first one"


@pytest.mark.skipif(sys.platform == "win32", reason="symbolic links need privileges on Windows")
def test_a_venv_and_its_base_python_are_both_checked(tmp_path, monkeypatch):
    base = tmp_path / "base" / "bin" / "python"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"")
    venv = tmp_path / "proj" / ".venv" / "bin" / "python"
    venv.parent.mkdir(parents=True)
    os.symlink(base, venv)
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "other"))
    (tmp_path / "other" / "bin").mkdir(parents=True)
    os.symlink(base, tmp_path / "other" / "bin" / "python")
    found = [p for p, _, _ in pyprobe.candidates(tmp_path / "proj")]
    assert str(venv) in found and str(tmp_path / "other" / "bin" / "python") in found


def test_nothing_is_fine_when_no_python_could_be_checked():
    stubs = [pyprobe.Interpreter(r"C:\x\WindowsApps\python.exe", "path", "python", error="store-stub")]
    f = diagnose(rtx3060(), stubs)
    assert "no-python" in ids(f, "error")


def test_no_contradictions_about_the_gpu_and_rocm_is_understood():
    cuda_build = {"version": "2.14.1+cu126", "cuda": "12.6", "cuda_available": False}
    no_driver = rtx3060(gpus=[], nvidia_smi=None, driver_cuda=None, gpu_seen_by_os=["NVIDIA GPU (Microsoft Basic Display Adapter)"])
    f = diagnose(no_driver, [py(cuda_build)])
    assert "driver-missing" in ids(f) and "torch-cuda-no-gpu" not in ids(f)
    amd = System("linux", "6", "x86_64", ram_mb=32768, gpu_seen_by_os=["AMD Radeon RX 7900 XTX"])
    rocm = {"version": "2.5.1+rocm6.2", "cuda": None, "hip": "6.2", "cuda_available": True, "devices": [{"name": "RX 7900 XTX"}]}
    assert "torch-rocm" in ids(diagnose(amd, [py(rocm)]), "ok")
    assert "rocm-unavailable" in ids(diagnose(amd, [py({**rocm, "cuda_available": False})]))


def test_a_cuda_failure_at_start_up_is_not_a_broken_import(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(make_torch(tmp_path, props_fail=True)))
    t = pyprobe.probe(sys.executable).info["torch"]
    assert "import_error" not in t and t["cuda_works"] is False and "unknown error" in t["cuda_error"]


def test_a_blocked_host_is_not_called_a_certificate_problem():
    assert net.classify(urllib.error.URLError(TimeoutError("_ssl.c:989: The handshake operation timed out"))) == "timeout"
    assert net.classify(urllib.error.URLError(ssl.SSLEOFError(8, "EOF occurred in violation of protocol"))) == "reset"
    verify = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate")
    assert net.classify(urllib.error.URLError(verify)) == "ssl"


def test_after_a_failure_on_a_newer_build_one_no_newer_than_the_driver_is_offered():
    index = INDEX + '\n<a href="/whl/cu121/torch-2.5.1%2Bcu121-cp312-cp312-win_amd64.whl">x</a>'
    failed = {"version": "2.11.0+cu128", "cuda": "12.8", "cuda_available": False}
    f = diagnose(rtx3060(driver_cuda="12.2"), [py(failed)], {"torch_index": index})
    assert next(x for x in f if x.id == "cuda-unavailable-minor").fix.endswith("/cu121")


def test_linux_builds_are_matched_by_architecture():
    html = ('<a href="torch-2.9.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl">a</a>'
            '<a href="torch-2.4.0%2Bcu121-cp312-cp312-linux_x86_64.whl">b</a>')
    assert pytorch.builds(html, "3.12.1", "linux_x86_64") == {(2, 4, 0): {"cu121"}}
    assert pytorch.builds(html, "3.12.1", "linux_aarch64") == {(2, 9, 0): {"cu130"}}


def test_windows_reads_each_variable_once():
    env = {"HTTP_PROXY": "http://p:1", "http_proxy": "http://p:1", "HTTPS_PROXY": "http://p:1"}
    assert sorted(env_snapshot(env, "win32")) == ["HTTPS_PROXY", "HTTP_PROXY"]
    assert sorted(env_snapshot(env, "linux")) == ["HTTPS_PROXY", "HTTP_PROXY", "http_proxy"]


def test_an_nvidia_gpu_without_its_driver_is_recognized_on_windows():
    names = parse_video_controllers("Microsoft Basic Display Adapter|PCI\\VEN_10DE&DEV_2560&SUBSYS\nIntel(R) UHD Graphics|PCI\\VEN_8086&DEV_46A6\n")
    assert names[0].startswith("NVIDIA GPU (") and names[1] == "Intel(R) UHD Graphics"


def test_the_doctor_failing_exits_2(monkeypatch, capsys):
    def boom():
        raise RuntimeError("probe broke")

    monkeypatch.setattr(probe, "system", boom)
    assert cli.main(["--offline"]) == 2 and "could not finish" in capsys.readouterr().err


def test_a_broken_pytorch_is_not_counted_as_cpu_only():
    ok = {"version": "2.14.1+cu126", "cuda": "12.6", "cuda_available": True, "cuda_works": True, "arch_list": ["sm_86"], "devices": [{}]}
    f = diagnose(rtx3060(), [py(ok), py({"import_error": "OSError"}, path="C:/b/python.exe")])
    assert "environments-disagree" not in ids(f)


def test_the_probe_ignores_the_project_folder_and_noise(tmp_path, monkeypatch):
    work = tmp_path / "proj"
    work.mkdir()
    (work / "json.py").write_text("raise SystemExit('the project json.py ran')", encoding="utf-8")
    monkeypatch.chdir(work)
    monkeypatch.setenv("PYTHONPATH", str(make_torch(tmp_path / "site", noise=True)))
    it = pyprobe.probe(sys.executable)
    assert it.error is None and it.info["torch"]["version"] == "2.14.1+cu126"


def test_a_gpu_with_shared_memory_plans_from_ram():
    gb10 = rtx3060(gpus=[GPU("nvidia", "NVIDIA GB10", None, None, "12.1", "580")], ram_mb=128 * 1024)
    fit = models.plan(gb10)
    assert fit.device == "gpu" and fit.memory_gb > 90
