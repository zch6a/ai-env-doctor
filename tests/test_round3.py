"""Round 3 (POLISH_LOG): the model list checked against Ollama's library, AMD's PyTorch on Windows."""

from test_rules import ids, py, rtx3060

from ai_env_doctor import models
from ai_env_doctor.probe import GPU, System
from ai_env_doctor.rules import diagnose

CPU_BUILD = {"version": "2.9.1+cpu", "cuda": None, "cuda_available": False}


def test_the_list_has_something_for_every_use_and_every_machine():
    tags = [m.tag for m in models.CATALOG]
    assert len(tags) == len(set(tags))
    small = models.plan(System("windows", "10", "x86_64", ram_mb=8192))  # an 8 GB laptop, no GPU
    assert set(small.full) == set(models.USES), "every use has a model even there"


def test_the_largest_that_fits_is_a_current_model():
    fit = models.plan(rtx3060())  # 6 GB
    assert fit.full["chat"].tag == "qwen3.5:4b" and fit.partial["chat"].tag == "qwen3.5:9b"
    assert fit.full["vision"].tag == "qwen3-vl:4b"
    big = models.plan(rtx3060(gpus=[GPU("nvidia", "RTX 4090", 24564, 0, "8.9", "570")]))
    assert big.full["code"].tag == "qwen3-coder:30b"


def test_a_cpu_build_on_an_amd_machine_points_to_a_rocm_build():
    win = System("windows", "10", "x86_64", ram_mb=32768, gpu_seen_by_os=["AMD Radeon RX 9070 XT"])
    f = next(x for x in diagnose(win, [py(CPU_BUILD)]) if x.where)
    assert f.id == "torch-cpu-amd-windows" and "radeon-ryzen" in f.fix and "RX 9070 XT" in f.title
    linux = System("linux", "6", "x86_64", ram_mb=32768, gpu_seen_by_os=["Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX]"])
    assert "torch-cpu-amd" in ids(diagnose(linux, [py(CPU_BUILD)]), "info")
    intel = System("windows", "10", "x86_64", ram_mb=16384, gpu_seen_by_os=["Intel(R) UHD Graphics"])
    assert "torch-cpu-ok" in ids(diagnose(intel, [py(CPU_BUILD)]))


def test_amds_windows_build_that_works_is_recognized():
    win = System("windows", "10", "x86_64", ram_mb=32768, gpu_seen_by_os=["AMD Radeon RX 9070 XT"])
    rocm = {"version": "2.9.1+rocmsdk20251116", "cuda": None, "hip": "7.1.52802", "cuda_available": True, "devices": [{"name": "AMD Radeon RX 9070 XT"}]}
    f = diagnose(win, [py(rocm)])
    assert "torch-rocm" in ids(f, "ok") and not ids(f, "error")
