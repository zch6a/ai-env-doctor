<div align="center">

# ai-env-doctor

**Why does PyTorch not use my GPU?** One command checks the GPU, the driver, the PyTorch you actually
installed, your proxies and mirrors, and says what is wrong, with the exact command that fixes it. Then
it lists the local models this machine can run.

<a href="https://github.com/zch6a/ai-env-doctor/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/zch6a/ai-env-doctor/actions/workflows/ci.yml/badge.svg"></a>
<img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white">
<img alt="No dependencies" src="https://img.shields.io/badge/dependencies-none-2ea44f">
<img alt="Read-only" src="https://img.shields.io/badge/changes%20nothing-read--only-2ea44f">
<a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-blue"></a>

**English** · [简体中文](README.zh-CN.md)

</div>

---

```bash
uvx --from git+https://github.com/zch6a/ai-env-doctor ai-env-doctor
```

On a laptop where `uv pip install torch` had installed PyTorch the default way:

```text
Findings:
  ✗ PyTorch 2.15.0+cpu in C:\proj\.venv\Scripts\python.exe is the CPU-only build: it will not use the NVIDIA GeForce RTX 3060 Laptop GPU
      This is the most common reason for torch.cuda.is_available() to be False: `pip install torch` without PyTorch's own
      index gives the CPU build on Windows (on Linux, PyPI's torch is a CUDA build: there the CPU one comes from PyTorch's CPU
      index or a lock file that chose it).
      fix: uv pip uninstall --python C:\proj\.venv\Scripts\python.exe torch
           uv pip install --python C:\proj\.venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu126
  · The newest PyTorch (2.15.0) needs a driver for CUDA 13.2; this one supports CUDA 12.7
      The newest PyTorch this driver can run is 2.14.1, for CUDA 12.6 (cu126); the commands here install that.
      fix: To use PyTorch 2.15.0, update the driver: https://www.nvidia.com/drivers
  ✓ NVIDIA GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6.0 GB), driver 566.36, supports CUDA up to 12.7
  ✓ The network reaches pypi.org, download.pytorch.org, huggingface.co, hf-mirror.com, registry.ollama.ai

Models this machine can run (Ollama) — on the GPU (5.5 GB):
  chat                 fits: qwen3.5:4b (3.3 GB);  split GPU/CPU, slower: qwen3.5:9b (6.6 GB)
  code                 fits: qwen2.5-coder:3b (1.9 GB);  split GPU/CPU, slower: qwen2.5-coder:7b (4.7 GB)
  ...
```

Running those two commands gave PyTorch 2.14.1+cu126, which ran on the GPU, and ai-env-doctor then
said `✓ PyTorch 2.14.1+cu126 uses the GPU`. Note the second finding: the obvious fix, "install the
CUDA build", would have installed PyTorch 2.15, which is built only for CUDA 13 and does not run on
this driver either. ai-env-doctor reads PyTorch's own index to know which builds exist, and picks the
newest one this driver can run. The fix uses `uv pip` here because the environment was made by uv and
has no pip; elsewhere it is `python -m pip`.

## What it checks

- **GPU and driver**: NVIDIA (through `nvidia-smi`), Apple Silicon, AMD and others (as the operating
  system lists them; in WSL, as Windows lists them). A GPU with no driver, a driver that is loaded
  wrongly after an update, a driver without `nvidia-smi`, WSL without the Windows driver (and WSL 1,
  which has no GPU), a container started without the GPU, a driver too old for current PyTorch.
- **The PyTorch you will actually run**, in the project's `.venv`, the active virtualenv or conda
  environment, and the `python` on PATH (or `--python PATH`). It asks each one, in its own process, so
  it sees what your code sees:
  - the CPU-only build on a machine with an NVIDIA GPU, the most common problem by far;
  - a build for a newer CUDA than the driver supports (CUDA 13 builds on a CUDA 12 driver);
  - a build without code for the GPU (RTX 50-series need CUDA 12.8 or later);
  - CUDA that will not start, with what PyTorch said; an AMD build on an NVIDIA GPU;
  - on an AMD GPU, whether PyTorch is a ROCm build that sees it, and where the right build comes from
    (PyTorch's index on Linux, AMD's own on Windows and in WSL);
  - Apple GPU (MPS) support, and Python running under Rosetta on Apple Silicon;
  - two environments that disagree (the GPU works in one, not in the one your editor uses);
  - in a conda environment, a PyTorch that conda installed (pip cannot remove it) and `cpuonly`, which
    keeps bringing the CPU build back: the fix removes them with conda first;
  - a system Python that pip may not install into (PEP 668: the fix makes a virtual environment);
  - a 32-bit Python, the Microsoft Store `python` placeholder, a PyTorch that fails to import (with
    the missing DLL), torchvision built for another PyTorch, bitsandbytes too old for GPUs on Windows.
- **Proxies and certificates**: an HTTP proxy without an HTTPS one (every download goes around it),
  a proxy without NO_PROXY for localhost (local servers such as Ollama break), SOCKS proxies, a
  certificate bundle that does not exist, certificate checks turned off.
- **The network**: whether pypi.org, download.pytorch.org, huggingface.co, hf-mirror.com and Ollama's
  registry answer, through your proxy; DNS, timeouts, refused connections and certificate errors told
  apart; the mirror suggested when Hugging Face is blocked but the mirror is not.
- **Room for models**: free space where Hugging Face and Ollama keep them; whether Ollama is running
  on this machine (one request to localhost, never through a proxy).
- **Models this machine can run**: for chat, code, reasoning, images and embeddings, the largest
  well-known model that fits in GPU memory (or in RAM, without a GPU), and the next size up that runs
  split between GPU and CPU. The list is dated (November 2026), and the report says when it has grown
  old. For a full catalogue with speed estimates, see [llmfit](https://github.com/AlexsJones/llmfit).

## Safe to run

- **It changes nothing.** It installs, removes and writes nothing; the fixes are printed for you to
  run. What it executes: the system's own tools that report hardware (`nvidia-smi`, `lspci`,
  PowerShell's `Get-CimInstance`, `sysctl`), and each Python it checks (your project's `.venv`, the
  active environment, the `python` on PATH), with a short script that imports torch, from a neutral
  folder so that your project's own files are not picked up.
- **It sends nothing about you.** The network check is one HEAD request to each of the five hosts
  above, through your proxy as pip would make it, and one GET of PyTorch's public index; nothing about
  the machine goes with them. `--offline` skips all of it.
- **Passwords in proxy URLs are hidden** in the report and in `--json` (and user names and tokens in
  them). Paths are not, and they contain your user name: look before you paste.
- No dependencies: it is plain Python 3.10+. With `uvx`, the first run fetches and builds it (about
  15 seconds here); later runs start at once.

## Install and use

```bash
uvx --from git+https://github.com/zch6a/ai-env-doctor ai-env-doctor      # run once, nothing installed
uv tool install git+https://github.com/zch6a/ai-env-doctor               # or install the command
pip install "ai-env-doctor @ git+https://github.com/zch6a/ai-env-doctor"
```

```bash
ai-env-doctor                    # check everything (the report follows the system language)
ai-env-doctor --lang zh          # in Chinese; --lang en for English
ai-env-doctor --python PATH      # check this interpreter (can be given more than once)
ai-env-doctor --offline          # no network checks
ai-env-doctor --json             # everything found, for scripts and bug reports
```

Run it in your project's folder: it checks the `.venv` there first. The exit code is 1 when something
needs fixing, 0 when not, 2 when the doctor itself failed. What `--json` contains, field by field:
[docs/json.md](docs/json.md).

## Status

1.0. Tested with fake PyTorch builds of every kind (CPU, CUDA that works, CUDA that fails at start-up
or in a kernel, missing GPU code, a broken import) run through the real probe, synthetic machines for
every rule, the fix commands parsed by PowerShell and bash, and local stand-ins for the network
checks. Reviewed twice by someone who had not written it (see [POLISH_LOG.md](POLISH_LOG.md)). By hand
on Windows 11 with an RTX 3060 Laptop GPU (driver for CUDA 12.7), end to end: the CPU build
diagnosed, the printed commands run, the GPU then working; and run with `uvx` from GitHub. CI runs it
on Linux (no GPU) and macOS (Apple M1). WSL, containers, conda with a conda-installed PyTorch, and
AMD GPUs are covered by the tests only: reports from those machines are welcome.

## Develop

```bash
uv venv && uv pip install -e . pytest ruff
uv run pytest
```

## License

[MIT](LICENSE)
