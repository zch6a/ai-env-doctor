# Polish log

## Development

**What it is for.** When PyTorch does not use the GPU, the cause is nearly always one of a dozen things: a CPU-only wheel, a wheel for a newer CUDA than the driver, a GPU too new for the wheel, the wrong Python, a proxy that breaks downloads. Finding which takes an hour of forum reading. The existing tools each cover a part:
- `torch.utils.collect_env` prints facts and diagnoses nothing;
- ai-infradr (2026) covers Linux with NVIDIA only, with no network checks;
- llmfit and whichllm say which models fit, and nothing about why the GPU does not work.

ai-env-doctor checks the whole chain on Windows, macOS and Linux, and gives one fix per problem: driver → PyTorch build → Python environment → proxy and network → room for models → which models fit.

**Design**
- **No dependencies**, standard library only, so `uvx` starts it in a second or two.
- **It never imports torch itself.** Under `uvx` it runs in an isolated environment, so it runs a small script in each of the user's interpreters (the project's `.venv`, the active virtualenv or conda environment, `python` on PATH) and reads the JSON that script prints. That is the only way to see what the user's code sees. Package versions are read from metadata, without importing the packages.
- **Read-only.** Fixes are printed, never run. Passwords in proxy URLs are redacted in the report and in `--json`.
- **Network**: one HEAD request to each of five public hosts, through the configured proxy, plus one GET of PyTorch's index; `--offline` skips them. Failures are classified as DNS, timeout, refused, SSL, or proxy (407); the Hugging Face mirror is suggested only when huggingface.co fails and the mirror answers.
- **Which PyTorch build to recommend comes from PyTorch's own index** (download.pytorch.org/whl/torch/, 1.3 MB). This turned out to matter: in November 2026 PyTorch 2.15 is built only for CUDA 13.2 and 13.4. On a driver for CUDA 12.7 the naive fix ("install the CUDA build") would install a PyTorch that cannot use the GPU either. The tool picks, in order:
  1. the newest PyTorch that has a build for a CUDA no newer than the driver's;
  2. otherwise one for the driver's major CUDA version, which NVIDIA's minor version compatibility runs;
  3. otherwise it says that the driver must be updated.

  It also says what the newest PyTorch would need. Blackwell GPUs need CUDA 12.8 or later, and a driver too old for them is said so. Offline, a table of the CUDA builds PyTorch still makes is used.
- **Findings**: error, warning, info or ok, each with a title, the evidence and one fix, in English and Chinese. The language follows the system, or `--lang`.
- **Models**: a short, dated list of widely used Ollama models by use, rather than a catalogue (llmfit has one). A model "fits" when its download plus about 1 GB and 10% fits in GPU memory (Apple: about two thirds to three quarters of RAM); the next size up is listed as split between GPU and CPU.

**Tried for real on this laptop** (Windows 11, RTX 3060 Laptop 6 GB, driver 566.36 for CUDA 12.7)
- A plain `uv pip install torch` gave 2.15.0+cpu. The doctor reported the CPU-only build, gave the command for cu126, and noted that 2.15 needs a CUDA 13.2 driver.
- **Found while building**:
  - the Microsoft Store `python` placeholder was reported twice (python and python3), and is now said once;
  - where an interpreter was found was in English inside the Chinese report;
  - the Blackwell advice suggested cu126, which has no code for Blackwell;
  - the offline table recommended cu121 for a 12.2 driver, which only has PyTorch 2.5, instead of cu126 through minor version compatibility;
  - a Blackwell GPU on a driver older than 12.8 got a build instead of "update the driver".

**Tests**: 28. A fake `torch` package, put on PYTHONPATH, plays every kind of build (CPU, CUDA working, CUDA failing to start, no kernel image, a broken import) through the real probe in a real subprocess. Synthetic machines exercise every rule, a fixture of PyTorch's index drives the advice, and local HTTP servers stand in for the reachability and Ollama checks.

**Open for the polishing rounds**
- The CUDA path checked by hand end to end: the cu126 build was being installed into the scratch environment as this was written.
- macOS, Linux and WSL by hand, not only through CI.
- AMD/ROCm in more depth.
- An independent review.

## Round 1 — 2026-11-12

**Independent review.** A reviewer read the whole tool without my notes and tried to break each rule. What it found, each fixed with a regression test in `tests/test_review.py` that failed before the fix:
- **Fixes that did not run.** In an environment made by uv there is no pip, so `python -m pip ...` failed; the fix now uses `uv pip ... --python PATH` there. A path with spaces needs `& "..."` in PowerShell. `--force-reinstall` also reinstalled numpy and every other dependency at its newest; the fix is now an uninstall of the torch packages that are present, then an install of the same ones from PyTorch's index.
- **The wrong build advised.** Data-centre Blackwell (B200/GB200, compute capability 10.x) was not treated as Blackwell. On a machine with two GPUs only the first counted; the newest one now sets the floor, and the GPU that lacks code is named. After a failure on a build newer than the driver, the strict advice (no newer than the driver) is used. Linux wheels were not matched by architecture (aarch64 builds were offered on x86_64).
- **Contradictions and false all-clears.** With no usable Python at all (only the Store placeholder), the report said nothing was wrong; that is now an error. "CUDA build but no GPU" appeared next to "the GPU has no driver". A PyTorch that failed to import counted as CPU-only in "environments disagree". ROCm builds were reported as CPU builds.
- **Misreported failures.** CUDA failing in `get_device_properties` was reported as a broken import. A timeout or a reset during the TLS handshake was called a certificate problem; only a failed verification is now. The probe also runs a one-element kernel, since `is_available()` can be True on a GPU the build has no code for.
- **The probe could be fooled.** It ran in the project folder, so a project `json.py` replaced the standard one; it now runs from the temp folder. Output printed by `sitecustomize` or by torch before the marker broke parsing; the marker is now searched from the end.
- **Detection.** An NVIDIA GPU without its driver shows on Windows as "Microsoft Basic Display Adapter"; it is now recognized by its PCI vendor id. WSL is recognized, with `/usr/lib/wsl/lib/nvidia-smi`. Windows read each environment variable twice (names are case-insensitive there). GPUs with shared memory (GB10) plan models from RAM. Rosetta is judged per interpreter, not from the doctor's own process. A virtualenv whose `python` is a symlink to the base Python was deduplicated away.
- **Privacy.** Redaction missed user names without passwords (`http://user@host`) and query-string tokens; both are hidden now.
- An internal failure now exits 2 (1 means "something to fix"), with the traceback under `AI_ENV_DOCTOR_DEBUG=1`.

**End to end, by hand** (RTX 3060 Laptop, driver 566.36 for CUDA 12.7). A fresh uv environment with a plain `uv pip install torch` got 2.15.0+cpu. The doctor printed the two-line fix; running exactly those lines gave 2.14.1+cu126, `torch.cuda.is_available()` True, and a kernel on the GPU returned 2.0. The doctor then said `✓ PyTorch 2.14.1+cu126 uses the GPU`, 0 to fix. CI ran it on Linux without a GPU and on macOS (M1); both reports were right for those machines.

**Docs.** The READMEs show the real two-line fix, and say exactly what "changes nothing" means: it does run each Python it checks, with a short import script, from a neutral folder.

**Tests**: 42 (+14), 1 skipped on Windows (symlinks). ruff clean.

**Open**
- WSL and AMD/ROCm by hand.
- The model list is dated (November 2026); a way to tell when it is stale.
- The fix for a conda environment (`conda install` versus pip into conda).
