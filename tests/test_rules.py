from ai_env_doctor import models, pytorch
from ai_env_doctor.net import Reach
from ai_env_doctor.probe import GPU, System
from ai_env_doctor.pyprobe import Interpreter
from ai_env_doctor.rules import arch_supported, diagnose

# A fragment of download.pytorch.org/whl/torch/ (late 2026)
INDEX = "\n".join(f'<a href="/whl/{tag}/torch-{v}%2B{tag}-cp312-cp312-win_amd64.whl">x</a>'
                  for v, tags in [("2.15.0", ["cu132", "cu134"]), ("2.14.1", ["cu126", "cu130", "cu132"]), ("2.11.0", ["cu126", "cu128"])]
                  for tag in tags)


def rtx3060(**kw):
    base = dict(os="windows", os_version="10", machine="x86_64", ram_mb=16 * 1024, cpu="i9", gpus=[GPU("nvidia", "RTX 3060 Laptop", 6144, 500, "8.6", "566.36")],
                driver_cuda="12.7", nvidia_smi="nvidia-smi")
    base.update(kw)
    return System(**base)


def py(torch=None, packages=None, path="C:/p/.venv/Scripts/python.exe", why="venv", **info):
    return Interpreter(path, why, "C:/p/.venv", {"python": "3.12.1", "bits": 64, "packages": packages or {}, "torch": torch, **info})


def ids(findings, level=None):
    return [f.id for f in findings if level is None or f.level == level]


def test_the_cpu_build_on_an_nvidia_machine_gets_the_right_index():
    f = diagnose(rtx3060(), [py({"version": "2.15.0+cpu", "cuda": None, "cuda_available": False})], {"torch_index": INDEX})
    cpu = next(x for x in f if x.id == "torch-cpu")
    assert cpu.level == "error" and cpu.fix.endswith("--index-url https://download.pytorch.org/whl/cu126")
    note = next(x for x in f if x.id == "newest-needs-driver")
    assert "2.15.0" in note.title and "CUDA 13.2" in note.title and "2.14.1" in note.detail and "12.6" in note.detail


def test_a_wheel_for_a_newer_major_cuda_than_the_driver():
    f = diagnose(rtx3060(), [py({"version": "2.15.0+cu132", "cuda": "13.2", "cuda_available": False})], {"torch_index": INDEX})
    assert "wheel-major-newer" in ids(f, "error")


def test_a_blackwell_gpu_with_a_build_that_has_no_code_for_it():
    s = rtx3060(gpus=[GPU("nvidia", "RTX 5090", 32607, 0, "12.0", "575.0")], driver_cuda="12.9")
    t = {"version": "2.5.1+cu124", "cuda": "12.4", "cuda_available": True, "cuda_works": False, "arch_list": ["sm_50", "sm_80", "sm_86", "sm_90"]}
    f = diagnose(s, [py(t)], {"torch_index": INDEX})
    arch = next(x for x in f if x.id == "arch-missing")
    assert arch.fix.endswith("/cu128"), arch.fix


def test_arch_rules():
    assert arch_supported("8.6", ["sm_80"]), "same major: sm_80 code runs on 8.6"
    assert not arch_supported("12.0", ["sm_80", "sm_90"])
    assert arch_supported("12.0", ["sm_90", "compute_90"]), "PTX can be compiled for a newer GPU"
    assert arch_supported("8.9", ["sm_86", "sm_90"])


def test_cuda_that_does_not_start_and_a_working_one():
    t = {"version": "2.14.1+cu126", "cuda": "12.6", "cuda_available": False, "warnings": ["CUDA initialization: driver too old"]}
    f = diagnose(rtx3060(), [py(t)])
    bad = next(x for x in f if x.id == "cuda-unavailable")
    assert "driver too old" in bad.detail
    ok = {"version": "2.14.1+cu126", "cuda": "12.6", "cuda_available": True, "cuda_works": True, "arch_list": ["sm_86"],
          "devices": [{"name": "RTX 3060 Laptop", "memory_mb": 6144, "capability": "8.6"}]}
    f = diagnose(rtx3060(), [py(ok)])
    assert "torch-cuda" in ids(f, "ok") and not ids(f, "error")
    newer = {**ok, "version": "2.11.0+cu128", "cuda": "12.8"}
    assert "wheel-minor-newer" in ids(diagnose(rtx3060(), [py(newer)]), "info"), "works through minor version compatibility"


def test_two_environments_that_disagree():
    ok = {"version": "2.14.1+cu126", "cuda": "12.6", "cuda_available": True, "cuda_works": True, "arch_list": ["sm_86"], "devices": [{}]}
    cpu = {"version": "2.15.0+cpu", "cuda": None, "cuda_available": False}
    f = diagnose(rtx3060(), [py(ok), py(cpu, path="C:/Python312/python.exe", why="path")])
    assert "environments-disagree" in ids(f) and "torch-cpu" in ids(f)


def test_driver_problems():
    assert "driver-missing" in ids(diagnose(rtx3060(gpus=[], nvidia_smi=None, driver_cuda=None, gpu_seen_by_os=["NVIDIA GeForce RTX 4070"]), []))
    assert "wsl-no-driver" in ids(diagnose(rtx3060(gpus=[], nvidia_smi=None, driver_cuda=None, wsl=True, os="linux",
                                                    gpu_seen_by_os=["NVIDIA Corporation GA106"]), []))
    assert "nvidia-smi-failed" in ids(diagnose(rtx3060(gpus=[]), []))
    assert "driver-old" in ids(diagnose(rtx3060(driver_cuda="11.4"), []))
    assert "gpu-none" in ids(diagnose(rtx3060(gpus=[], nvidia_smi=None, driver_cuda=None), []))


def test_apple_silicon_and_rosetta():
    mac = System("macos", "14.5", "arm64", ram_mb=16 * 1024, cpu="Apple M3", gpus=[GPU("apple", "Apple M3", 10813)])
    good = {"version": "2.15.0", "cuda": None, "mps_built": True, "mps_available": True}
    assert "torch-mps" in ids(diagnose(mac, [py(good)]), "ok")
    mac.rosetta = True  # the doctor itself under Rosetta says nothing about an arm64 environment
    arm = diagnose(mac, [py(good, machine="arm64")])
    assert "torch-mps" in ids(arm, "ok") and "rosetta" not in ids(arm)
    intel = diagnose(mac, [py({**good, "mps_available": False}, machine="x86_64")])
    assert "rosetta" in ids(intel, "error") and "mps-unavailable" not in ids(intel), "said once, as the cause"


def test_python_problems():
    stubs = [Interpreter(r"C:\Users\x\AppData\Local\Microsoft\WindowsApps\python.exe", "path", "python", error="store-stub"),
             Interpreter(r"C:\Users\x\AppData\Local\Microsoft\WindowsApps\python3.exe", "path", "python3", error="store-stub")]
    f = diagnose(rtx3060(), stubs)
    assert ids(f).count("store-stub") == 1, "said once for both"
    assert "python-32bit" in ids(diagnose(rtx3060(), [py(None, bits=32)]))
    assert "no-python" in ids(diagnose(rtx3060(), []))
    imp = diagnose(rtx3060(), [py({"import_error": "OSError('[WinError 126] fbgemm.dll')"})])
    assert "torch-import" in ids(imp, "error") and "vc_redist" in next(x.fix for x in imp if x.id == "torch-import")


def test_network_findings():
    reach = [Reach("pypi", "https://pypi.org/simple/pip/", "Python packages", True, 200, 0.3),
             Reach("pytorch", "https://download.pytorch.org/whl/", "PyTorch", False, None, 8.0, "timed out", "timeout"),
             Reach("huggingface", "https://huggingface.co/api/models?limit=1", "models", False, None, 8.0, "timed out", "timeout"),
             Reach("hf-mirror", "https://hf-mirror.com/api/models?limit=1", "mirror", True, 200, 0.5)]
    f = diagnose(rtx3060(), [], {"proxy_problems": [("proxy-localhost", "HTTPS_PROXY")], "reach": reach})
    assert {"net-timeout", "hf-blocked-mirror-ok", "proxy-localhost"} <= set(ids(f))
    assert next(x for x in f if x.id == "hf-blocked-mirror-ok").fix.startswith("Set HF_ENDPOINT=https://hf-mirror.com")


def test_chinese_findings_and_safe_formatting():
    f = diagnose(rtx3060(), [py({"version": "2.15.0+cpu", "cuda": None, "cuda_available": False})], lang="zh")
    assert "纯 CPU 版" in next(x.title for x in f if x.id == "torch-cpu")


def test_the_index_is_read_for_this_python_and_platform():
    b = pytorch.builds(INDEX, "3.12.1", "win_amd64")
    assert b[(2, 15, 0)] == {"cu132", "cu134"} and pytorch.builds(INDEX, "3.11.9", "win_amd64") == {}
    a = pytorch.advise("12.7", "8.6", b)
    assert (a.tag, a.torch, a.newest_needs) == ("cu126", "2.14.1", "CUDA 13.2")
    assert pytorch.advise("13.4", "8.6", b).tag == "cu134"
    assert pytorch.advise("12.2", "8.9", b).tag == "cu126", "minor version compatibility: a 12.x build on a 12.2 driver"
    assert pytorch.advise("12.9", "12.0", b).tag == "cu128", "Blackwell needs 12.8 or later"
    assert pytorch.advise("12.7", "12.0").driver_too_old
    assert pytorch.advise("12.7", "8.6").tag == "cu126", "offline: the table"


def test_models_that_fit():
    assert models.plan(rtx3060()).full["chat"].size_gb <= 4.1
    big = models.plan(rtx3060(gpus=[GPU("nvidia", "RTX 4090", 24564, 0, "8.9", "570")]))
    assert big.full["chat"].size_gb >= 17 and big.full["code"].tag == "qwen2.5-coder:32b"
    cpu = models.plan(rtx3060(gpus=[], ram_mb=8 * 1024))
    assert cpu.device == "cpu" and cpu.full["chat"].size_gb < 3 and not cpu.partial
    split = models.plan(rtx3060())
    assert split.partial["chat"].size_gb > split.full["chat"].size_gb, "the next size up, on GPU and CPU"
