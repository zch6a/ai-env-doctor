"""A fake `torch` package, written to a folder and put on PYTHONPATH, so that the probe can be tested
against every kind of PyTorch build without installing one."""

from __future__ import annotations

from pathlib import Path

TEMPLATE = '''
__version__ = {version!r}

class _Version:
    cuda = {cuda!r}
    hip = None
    git_version = "fake"

version = _Version()

class _Props:
    def __init__(self, name, mem, major, minor):
        self.name, self.total_memory, self.major, self.minor = name, mem, major, minor

class _Cuda:
    @staticmethod
    def is_available():
        {available_body}
    @staticmethod
    def get_arch_list():
        return {archs!r}
    @staticmethod
    def device_count():
        return 1
    @staticmethod
    def get_device_properties(i):
        return _Props({gpu!r}, 6 * 2**30, 8, 6)

cuda = _Cuda()

class _Tensor:
    def cuda(self):
        return self
    def __add__(self, other):
        {tensor_cuda_body}
        return self
    def item(self):
        return 2.0

def zeros(*a, **kw):
    return _Tensor()

def ones(*a, **kw):
    return _Tensor()

class _Cudnn:
    @staticmethod
    def version():
        return 91000

class _Mps:
    @staticmethod
    def is_built():
        return {mps!r}
    @staticmethod
    def is_available():
        return {mps!r}

class _Backends:
    cudnn = _Cudnn()
    mps = _Mps()

backends = _Backends()
{extra}
'''


def make_torch(folder: Path, *, version="2.14.1+cu126", cuda="12.6", available=True, archs=("sm_75", "sm_80", "sm_86", "sm_90"),
               gpu="NVIDIA GeForce RTX 3060 Laptop GPU", mps=False, cuda_fails=False, warn=None, import_error=None,
               props_fail=False, noise=False) -> Path:
    pkg = folder / "torch"
    pkg.mkdir(parents=True, exist_ok=True)
    if import_error:
        (pkg / "__init__.py").write_text(f"raise OSError({import_error!r})\n", encoding="utf-8")
        return folder
    body = "return True" if available else ("import warnings; warnings.warn(%r); return False" % (warn or "CUDA driver version is insufficient"))
    extra = ""
    if props_fail:  # CUDA starts here, and can fail ("CUDA error: unknown error" after a suspend)
        extra += "\ndef _fail(i):\n    raise RuntimeError('CUDA error: unknown error')\ncuda.get_device_properties = _fail\n"
    if noise:  # something printing at import, with no newline
        extra += "\nimport sys\nsys.stdout.write('loading...')\n"
    (pkg / "__init__.py").write_text(TEMPLATE.format(version=version, cuda=cuda, available_body=body, archs=list(archs), gpu=gpu, mps=mps,
                                                     tensor_cuda_body="raise RuntimeError('no kernel image is available')" if cuda_fails else "pass",
                                                     extra=extra), encoding="utf-8")
    return folder
