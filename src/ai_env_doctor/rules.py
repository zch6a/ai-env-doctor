"""From what was found to what is wrong and how to fix it. Each rule looks at the facts (system, Python
environments, network) and gives findings: a level, what is the matter, the evidence, and one fix,
usually a command to copy. Nothing here runs anything.

Which PyTorch build to recommend is pytorch.py's business: from PyTorch's own index when the network
is checked, from a table otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import pytorch
from .i18n import msg

LEVELS = ("error", "warning", "info", "ok")


@dataclass
class Finding:
    level: str  # error: AI work will fail; warning: may fail or be slow; info: worth knowing; ok: checked and fine
    id: str
    title: str
    detail: str = ""
    fix: str = ""
    where: str = ""  # which interpreter, if it is about one
    facts: dict = field(default_factory=dict)


def version_tuple(v: str | None) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:3])


def _quote(path: str, os_name: str) -> str:
    return f'"{path}"' if " " in path else path


def install_cmd(it, tag: str | None, os_name: str, reinstall: bool = True) -> str:
    """The commands that put PyTorch's `tag` build (None: the default one) into this environment.

    Uninstall first, then install: `--force-reinstall` would also reinstall every dependency at its
    newest version (numpy 1 replaced by 2, breaking whatever needed 1). Only the torch packages that are
    there are reinstalled. An environment made by uv has no pip: then the same with `uv pip`. On Windows
    a quoted path needs `&` in front for PowerShell to run it."""
    pkgs_present = (it.info or {}).get("packages") or {}
    pkgs = ["torch"] + [p for p in ("torchvision", "torchaudio") if p in pkgs_present]
    names = " ".join(pkgs)
    index = f" --index-url https://download.pytorch.org/whl/{tag}" if tag else ""
    q = _quote(it.path, os_name)
    conda = conda_torch(it)
    if conda:
        # Installed by conda: pip cannot remove it ("RECORD file not found ... installed by conda"), and
        # cpuonly would keep pulling the CPU build back. conda removes them (it lists what and asks), then
        # pip installs from PyTorch's index, as PyTorch itself now recommends inside conda.
        prefix = _quote(it.info["conda"]["prefix"], os_name)
        pkgs = ["torch"] + [p for p in ("torchvision", "torchaudio") if p in pkgs_present or p in conda]
        names = " ".join(pkgs)
        lines = [f"conda remove -p {prefix} {' '.join(conda)}"] if reinstall else []
        by_pip = [p for p in ("torch", "torchvision", "torchaudio") if p in pkgs_present
                  and {"torch": "pytorch"}.get(p, p) not in conda]  # e.g. cpuonly from conda, torch from pip
        run = f"& {q}" if os_name == "windows" and q.startswith('"') else q
        if reinstall and by_pip:
            lines.append(f"{run} -m pip uninstall -y {' '.join(by_pip)}" if "pip" in pkgs_present
                         else f"uv pip uninstall --python {q} {' '.join(by_pip)}")
        lines.append(f"{run} -m pip install {names}{index}" if "pip" in pkgs_present else f"uv pip install --python {q} {names}{index}")
        return "\n".join(lines)
    if "pip" in pkgs_present:
        run = f"& {q}" if os_name == "windows" and q.startswith('"') else q
        lines = [f"{run} -m pip uninstall -y {names}"] if reinstall else []
        lines.append(f"{run} -m pip install {names}{index}")
    else:
        lines = [f"uv pip uninstall --python {q} {names}"] if reinstall else []
        lines.append(f"uv pip install --python {q} {names}{index}")
    return "\n".join(lines)


def conda_torch(it) -> list[str]:
    """The torch packages conda installed in this environment, in the order to name them to `conda remove`;
    empty when PyTorch there came from pip (or there is no conda)."""
    mine = (((it.info or {}).get("conda") or {}).get("packages")) or {}
    if not ({"pytorch", "libtorch", "cpuonly"} & set(mine)):
        return []
    order = ("pytorch", "torchvision", "torchaudio", "cpuonly", "pytorch-cuda", "pytorch-gpu", "pytorch-cpu", "libtorch")
    return [p for p in order if p in mine]


class Diagnosis:
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.findings: list[Finding] = []

    def add(self, level: str, id_: str, where: str = "", **facts) -> None:
        title, detail, fix = msg(self.lang, id_, **facts)
        self.findings.append(Finding(level, id_, title, detail, fix, where, facts))


def diagnose(system, interpreters, net: dict | None = None, lang: str = "en") -> list[Finding]:
    d = Diagnosis(lang)
    nvidia = [g for g in system.gpus if g.vendor == "nvidia"]
    apple = [g for g in system.gpus if g.vendor == "apple"]
    listed_nvidia = [n for n in system.gpu_seen_by_os if "nvidia" in n.lower()]
    listed_amd = [n for n in system.gpu_seen_by_os if re.search(r"\b(amd|radeon)\b", n, re.I)]
    # The newest GPU decides the build: a machine with an RTX 3090 and an RTX 5090 needs one for the 5090.
    caps = [g.compute_capability for g in nvidia if g.compute_capability]
    cap = max(caps, key=version_tuple) if caps else None
    top = next((g for g in nvidia if g.compute_capability == cap), nvidia[0] if nvidia else None)
    index = (net or {}).get("torch_index")
    plat = pytorch.platform_tag(system.os, system.machine)
    newest_noted = False

    # ---- the GPU and its driver
    if nvidia:
        for g in nvidia:
            d.add("ok", "gpu-nvidia", name=g.name, memory=_gb(g.memory_mb), driver=g.driver or "?", cuda=system.driver_cuda or "?")
        if system.driver_cuda and version_tuple(system.driver_cuda) < (12, 0):
            d.add("warning", "driver-old", cuda=system.driver_cuda)
    elif system.nvidia_smi:
        d.add("error", "nvidia-smi-failed", path=system.nvidia_smi)
    elif listed_nvidia:
        d.add("error", "wsl-no-driver" if system.wsl else "driver-missing", name=listed_nvidia[0])
    elif apple:
        d.add("ok", "gpu-apple", name=apple[0].name, memory=_gb(apple[0].memory_mb))
    elif listed_amd:
        d.add("info", "gpu-amd", name=listed_amd[0], os=system.os)
    elif getattr(system, "wsl_gpu", False) and not system.gpu_seen_by_os:
        d.add("info", "wsl-gpu-unknown", ram=_gb(system.ram_mb))  # a GPU, but Windows could not be asked whose
    else:
        d.add("info", "gpu-none", ram=_gb(system.ram_mb))
    nvidia_expected = bool(nvidia or listed_nvidia or system.nvidia_smi)
    cvd = system.env.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and (cvd.strip() in ("", "-1", "none", "NoDevFiles") or not re.match(r"^[\w\-,\s]+$", cvd)):
        d.add("warning", "cuda-visible-devices", value=cvd)

    # ---- each Python environment
    usable = [it for it in interpreters if it.info]
    if not usable:
        d.add("error", "no-python")  # none found, or none that runs: nothing about PyTorch can be said
    stubs = [it.path for it in interpreters if it.error == "store-stub"]
    if stubs:
        d.add("warning", "store-stub", stubs[0], path="; ".join(stubs))
    for it in interpreters:
        where = it.path
        if it.error == "store-stub":
            continue
        if it.error:
            d.add("warning", "python-failed", where, path=it.path, error=it.error[:300])
            continue
        info = it.info
        if info.get("bits") == 32:
            d.add("error", "python-32bit", where, path=it.path)
        if apple and system.os == "macos" and info.get("machine") == "x86_64":
            d.add("error", "rosetta", where, path=it.path)  # this environment is an Intel build on Apple Silicon
            continue
        t = info.get("torch")
        advice = None
        if nvidia:
            available = pytorch.builds(index, info.get("python", ""), plat) if index and plat else None
            advice = pytorch.advise(system.driver_cuda, cap, available)
            if advice.newest_needs and not newest_noted:
                newest_noted = True
                cv = pytorch.tag_version(advice.tag)
                d.add("info", "newest-needs-driver", newest=advice.newest, needs=advice.newest_needs, driver=system.driver_cuda,
                      best=advice.torch, tag=advice.tag, tag_cuda=f"{cv[0]}.{cv[1]}")
            if advice.driver_too_old:
                d.add("error", "driver-too-old-for-gpu", where, driver=system.driver_cuda, gpu=top.name)
        if t is None:
            fix = install_cmd(it, advice.tag if advice else None, system.os, reinstall=False) if nvidia else ""
            d.add("info", "torch-missing", where, path=it.path, why=why_text(lang, it), python=info.get("python"), fix_cmd=fix)
            continue
        if t.get("import_error"):
            d.add("error", "torch-import", where, path=it.path, error=t["import_error"][:400])
            continue
        ver, built = t.get("version", "?"), t.get("cuda")
        if apple and system.os == "macos":
            if not t.get("mps_built"):
                d.add("error", "mps-not-built", where, version=ver, path=it.path, fix_cmd=install_cmd(it, None, system.os))
            elif not t.get("mps_available"):
                d.add("warning", "mps-unavailable", where, version=ver)
            else:
                d.add("ok", "torch-mps", where, version=ver, path=it.path)
            continue
        if nvidia:
            tag = advice.tag
            if t.get("hip"):
                d.add("error", "torch-rocm-on-nvidia", where, version=ver, fix_cmd=install_cmd(it, tag, system.os))
                continue
            if not built:
                d.add("error", "torch-cpu", where, version=ver, gpu=top.name, path=it.path, fix_cmd=install_cmd(it, tag, system.os))
                continue
            drv = version_tuple(system.driver_cuda)
            wheel = version_tuple(built)
            if drv and wheel[:1] > drv[:1]:
                d.add("error", "wheel-major-newer", where, version=ver, wheel=built, driver=system.driver_cuda,
                      fix_cmd=install_cmd(it, tag, system.os))
                continue
            archs = t.get("arch_list") or []
            lacking = next((g for g in nvidia if g.compute_capability and archs and not arch_supported(g.compute_capability, archs)), None)
            if lacking:
                d.add("error", "arch-missing", where, version=ver, cap=lacking.compute_capability, archs=" ".join(archs), gpu=lacking.name,
                      fix_cmd=install_cmd(it, tag, system.os))
                continue
            if not t.get("cuda_available") or t.get("cuda_works") is False:
                reason = (t.get("cuda_error") or " ".join(t.get("warnings") or []) or "").strip()[:400]
                newer_minor = bool(drv) and wheel > drv
                if newer_minor:  # it failed on a build newer than the driver: offer one no newer, even an older PyTorch
                    strict = pytorch.advise(system.driver_cuda, cap, available, strict=True)
                    tag = strict.tag
                d.add("error", "cuda-unavailable-minor" if newer_minor else "cuda-unavailable", where, version=ver, wheel=built,
                      driver=system.driver_cuda or "?", reason=reason or "-", fix_cmd=install_cmd(it, tag, system.os))
                continue
            dev = (t.get("devices") or [{}])[0]
            d.add("ok", "torch-cuda", where, version=ver, wheel=built, gpu=dev.get("name", "?"), path=it.path)
            if drv and wheel > drv:
                d.add("info", "wheel-minor-newer", where, wheel=built, driver=system.driver_cuda)
            continue
        if t.get("hip"):
            if t.get("cuda_available"):
                d.add("ok", "torch-rocm", where, version=ver, gpu=((t.get("devices") or [{}])[0]).get("name", "?"))
            else:
                d.add("warning", "rocm-unavailable", where, version=ver, reason=(t.get("cuda_error") or " ".join(t.get("warnings") or []))[:300] or "-")
        elif built and not nvidia_expected:
            d.add("info", "torch-cuda-no-gpu", where, version=ver, wheel=built)
        elif not built:
            d.add("info", "torch-cpu-ok", where, version=ver, path=it.path)
    for it in usable:  # why a fix starts with `conda remove`, said once per environment
        conda = conda_torch(it)
        if conda and any(f.where == it.path and "conda remove" in f.fix for f in d.findings):
            channel = next((it.info["conda"]["packages"][p].get("channel") for p in conda if it.info["conda"]["packages"][p].get("channel")), "")
            d.add("info", "torch-from-conda-cpuonly" if "cpuonly" in conda else "torch-from-conda", it.path, path=it.path,
                  packages=" ".join(conda), channel=channel or "?")
    _environments_disagree(d, usable)
    _packages(d, usable, system)

    # ---- the network
    if net is not None:
        _network(d, net)

    # ---- disk
    for name, free in (system.disk_free_mb or {}).items():
        if free is not None and free < 20 * 1024:
            d.add("warning", "disk-low", cache={"huggingface": "Hugging Face", "ollama": "Ollama"}.get(name, name), free=_gb(free))
    order = {lv: i for i, lv in enumerate(LEVELS)}
    return sorted(d.findings, key=lambda f: order[f.level])


def why_text(lang: str, it) -> str:
    from .i18n import WHY

    return WHY.get(lang, WHY["en"]).get(it.why, "{origin}").format(origin=it.origin)


def arch_supported(cap: str, archs: list[str]) -> bool:
    """Can a wheel built for `archs` (sm_86, compute_90 ...) run on a GPU of capability `cap`? Exact
    machine code for it, or PTX for an earlier capability of the same major version, which the driver
    can compile for it."""
    major, minor = version_tuple(cap)[:2] if len(version_tuple(cap)) >= 2 else (0, 0)
    target = major * 10 + minor
    for a in archs:
        m = re.match(r"(sm|compute)_(\d+)", a)
        if not m:
            continue
        n = int(m.group(2))
        if m.group(1) == "sm" and n == target:
            return True
        if m.group(1) == "sm" and n // 10 == major and n <= target:
            return True  # same major: binary compatible upwards in minor versions
        if m.group(1) == "compute" and n <= target:
            return True
    return False


def _environments_disagree(d: Diagnosis, usable) -> None:
    working = [it for it in usable if it.info.get("torch") and not it.info["torch"].get("import_error")]
    gpu = [it for it in working if it.info["torch"].get("cuda_available")]
    cpu = [it for it in working if not it.info["torch"].get("cuda_available")]
    if gpu and cpu:
        d.add("info", "environments-disagree", gpu=gpu[0].path, cpu=cpu[0].path)


def _packages(d: Diagnosis, usable, system) -> None:
    for it in usable:
        pkgs = it.info.get("packages") or {}
        bnb = pkgs.get("bitsandbytes")
        if bnb and system.os == "windows" and version_tuple(bnb) < (0, 43):
            d.add("warning", "bnb-windows-old", it.path, version=bnb)
        for name in ("xformers", "flash-attn", "torchvision", "torchaudio"):
            if name in pkgs and "torch" in pkgs and "+" in pkgs["torch"] and "+" in pkgs[name]:
                a, b = pkgs["torch"].split("+")[1], pkgs[name].split("+")[1]
                if a != b and name in ("torchvision", "torchaudio"):
                    d.add("warning", "companion-build-differs", it.path, name=name, torch=pkgs["torch"], other=pkgs[name])


def _network(d: Diagnosis, net: dict) -> None:
    for problem, detail in net.get("proxy_problems", []):
        level = "error" if problem in ("proxy-no-host", "ca-missing") else "warning" if problem != "hf-offline" else "info"
        d.add(level, problem, detail=detail)
    reach = {r.name: r for r in net.get("reach", [])}
    if not reach:
        return
    mirror = reach.get("hf-mirror")
    for r in reach.values():
        if r.ok:
            continue
        if r.name == "hf-mirror":
            continue
        if r.name == "huggingface" and mirror and mirror.ok:
            d.add("warning", "hf-blocked-mirror-ok", kind=r.kind or "?", error=r.error or "")
            continue
        d.add("error" if r.name in ("pypi", "pytorch") else "warning", f"net-{r.kind or 'other'}", host=r.url.split("/")[2],
              purpose=r.purpose, error=r.error or "")
    if all(r.ok for r in reach.values() if r.name != "hf-mirror"):
        slow = [r for r in reach.values() if r.ok and r.seconds and r.seconds > 3]
        d.add("ok", "net-ok", hosts=", ".join(r.url.split("/")[2] for r in reach.values() if r.ok))
        for r in slow:
            d.add("info", "net-slow", host=r.url.split("/")[2], seconds=f"{r.seconds:.1f}")
    ollama = net.get("ollama")
    if ollama:
        d.add("ok", "ollama-running", url=ollama["url"], version=ollama.get("version") or "?")


def _gb(mb: int | None) -> str:
    return f"{mb / 1024:.1f} GB" if mb else "?"
