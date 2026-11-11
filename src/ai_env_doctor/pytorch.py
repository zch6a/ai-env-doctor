"""Which PyTorch build to install for this driver and GPU.

When the network is checked, the answer comes from PyTorch's own index (download.pytorch.org/whl/torch/,
1.3 MB, read once): it lists every build that exists, so the advice is never older than the index. In
November 2026, for example, PyTorch 2.15 was built only for CUDA 13.2 and 13.4, so on a driver for
CUDA 12.7 the newest PyTorch that can use the GPU is 2.14, for CUDA 12.6. Offline, a table of the CUDA
builds PyTorch has made is used instead.

A build for CUDA 12.x runs on any driver for 12.0 or later (NVIDIA's minor version compatibility);
13.x needs a driver for 13. Still, a build for a CUDA no newer than the driver's is the safe choice,
and the one recommended. Blackwell GPUs need CUDA 12.8 or later: compute capability 10.x (B200,
GB200), 11.x and 12.x (RTX 50 series).
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass

INDEX = "https://download.pytorch.org/whl/torch/"
# Offline: the CUDA builds PyTorch still makes (late 2026); 11.8 for drivers too old for CUDA 12.
MAINTAINED = ["cu126", "cu128", "cu130", "cu132", "cu134"]


def tag_version(tag: str) -> tuple[int, int]:
    """"cu126" -> (12, 6); "cu118" -> (11, 8)."""
    digits = tag[2:]
    return (int(digits[:-1]), int(digits[-1])) if digits.isdigit() and len(digits) >= 2 else (0, 0)


def version(v: str | None) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:3])


def fetch_index(timeout: float = 20) -> str | None:
    try:
        req = urllib.request.Request(INDEX, headers={"User-Agent": "ai-env-doctor"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def platform_tag(os_name: str, machine: str) -> str | None:
    if os_name == "windows":
        return "win_amd64" if machine in ("x86_64", "amd64") else None
    if os_name == "linux":
        return "linux_x86_64" if machine in ("x86_64", "amd64") else "linux_aarch64"
    return None  # macOS builds are not on the CUDA index


def _platform_matches(wheel_platform: str, plat: str) -> bool:
    """"win_amd64" exactly; on Linux the architecture at the end of any manylinux or linux tag
    (manylinux_2_28_x86_64, linux_x86_64 ...), never another architecture's."""
    if plat == "win_amd64":
        return wheel_platform == "win_amd64"
    arch = plat.split("_", 1)[1]
    return "linux" in wheel_platform and wheel_platform.endswith("_" + arch)


def builds(index_html: str, python: str, plat: str) -> dict[tuple[int, ...], set[str]]:
    """{torch version: {cuda tags}} for this Python and platform."""
    cp = "cp" + "".join(python.split(".")[:2])
    out: dict[tuple[int, ...], set[str]] = {}
    for m in re.finditer(rf"torch-(\d+\.\d+\.\d+)%2B(cu\d+)-{cp}-{cp}t?-([\w.]+)\.whl", index_html):
        if any(_platform_matches(part, plat) for part in m.group(3).split(".")):
            out.setdefault(version(m.group(1)), set()).add(m.group(2))
    return out


@dataclass
class Advice:
    tag: str  # the CUDA build to install, "cu126"
    torch: str | None  # the newest PyTorch available for it, when known from the index
    newest: str | None  # the newest PyTorch of all, when known
    newest_needs: str | None  # the lowest CUDA the newest PyTorch is built for, when the driver cannot run it
    driver_too_old: bool = False  # no build runs on this driver (and GPU): the driver must be updated first


def advise(driver_cuda: str | None, capability: str | None, available: dict[tuple[int, ...], set[str]] | None = None,
           strict: bool = False) -> Advice:
    """`strict`: only a build for a CUDA no newer than the driver's (after one newer failed to start)."""
    drv = version(driver_cuda)[:2] or (12, 0)
    floor = (12, 8) if version(capability)[:1] >= (10,) else (0, 0)  # Blackwell: 10.x, 11.x, 12.x

    def pick(tags) -> str | None:
        """The build for a CUDA no newer than the driver's, the newest such; else the lowest of the
        driver's major version above it (NVIDIA's minor version compatibility runs it); else none."""
        if floor > drv:
            return None  # the driver cannot run this GPU at all (Blackwell needs a driver for 12.8)
        tags = [t for t in tags if tag_version(t) >= floor]
        safe = [t for t in tags if tag_version(t) <= drv]
        if safe:
            return max(safe, key=tag_version)
        compatible = [t for t in tags if tag_version(t)[0] == drv[0] and drv >= (12, 0) and not strict]
        return min(compatible, key=tag_version) if compatible else None

    if available:
        newest = max(available)
        for ver in sorted(available, reverse=True):
            tag = pick(available[ver])
            if tag:
                low = tag_version(min(available[newest], key=tag_version))
                needs = None if ver == newest else f"CUDA {low[0]}.{low[1]}"
                return Advice(tag, ".".join(map(str, ver)), ".".join(map(str, newest)), needs)
    tag = pick(MAINTAINED if not strict else ["cu118", "cu121", "cu124", *MAINTAINED])  # strict: an older PyTorch, if need be
    if tag:
        return Advice(tag, None, None, None)
    # The driver is older than any build that runs on this GPU: the driver comes first.
    if floor > (0, 0):
        return Advice("cu128", None, None, None, True)
    return Advice("cu118" if drv[:1] == (11,) else "cu126", None, None, None, drv < (11, 8))
