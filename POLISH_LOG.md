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

## Round 2 — 2026-11-12

From the open list of round 1, then the whole tool read and run again.

**conda environments.** The fix was wrong there. `pip uninstall torch` fails on a PyTorch that conda installed ("RECORD file not found ... installed by conda"), and the `cpuonly` package makes conda pick the CPU build again at every update. The probe now reads `conda-meta` (only the torch packages' file names, and their channel) and the fix becomes `conda remove -p PREFIX <the torch packages conda installed>` followed by the pip install from PyTorch's index, which is what PyTorch recommends inside conda since it stopped its own conda channel. Without `-y`: conda lists what it will remove, packages that depend on PyTorch included, and asks. When torch came from pip but `cpuonly` from conda, pip still uninstalls torch first, or the install would say "already satisfied". An info finding says why the fix starts with conda, and mentions `conda install -c conda-forge pytorch-gpu` for those who would rather stay with conda packages (checked: conda-forge publishes it, Windows included).

**A dated model list.** `models.CATALOG_AS_OF` (2026-11); after 270 days the report says that newer models of the same sizes have probably come out, and where to look.

**WSL and AMD, read again; two real bugs.**
- In WSL, `/dev/dxg` was taken to mean an NVIDIA GPU without its driver. It exists for any GPU Windows has: every Intel-only laptop running WSL got a false "WSL cannot see the NVIDIA GPU" error. Now WSL asks Windows which GPUs there are (`powershell.exe` through interop, the same query as on Windows); only an NVIDIA GPU there gives the error, and when Windows cannot be asked, an info says there is a GPU of unknown make, with what to do if it is NVIDIA.
- On Linux without `lspci` (containers, minimal installs) no GPU was seen at all, so a GPU without its driver went unnoticed. Now `/sys/bus/pci/devices` is read (class 0x03, vendor 10de / 1002 / 8086).

**Experience, from running it here in both languages.**
- The Chinese model table was misaligned: padding counted characters, and CJK characters take two columns. The same for the continuation lines of a two-line fix under "解决:". Both are padded by display width now.
- The disk warning said "huggingface"; now "Hugging Face".
- `--help` still said "only reads"; it now says what it runs.
- The report language: `LANG=zh_CN` won over `LC_ALL=C`. Now the POSIX order (LC_ALL, LC_MESSAGES, LANG; the first one set decides).

**Tests**: 51 (+9), 1 skipped on Windows. The conda reading is tested by running the real probe script against a fake `conda-meta` (including `pytorch-lightning`, which must not count as PyTorch).

**Open**
- WSL and conda by hand: neither is installed on this laptop, and I will not install them on the user's machine for a test.
- The model list's contents: compare with what Ollama's library offers now, model by model.
- ROCm on Windows (AMD's 2025 PyTorch builds for Radeon on Windows) is not recognized as such.

## Round 3 — 2026-11-12

**The model list, checked against Ollama's library** (ollama.com/library, read in the browser, November 2026). Every model in it still exists, but newer families had come out at most sizes in the past months: Gemma 4, Qwen 3.5 (which also reads images), Qwen3-VL, Qwen3-Coder, Qwen3-Embedding, EmbeddingGemma. Since the report recommends "the largest that fits", an older model of the same class that is slightly larger wins over a better new one (gemma3:4b at 3.4 GB over qwen3.5:4b at 3.3). So the list now keeps, per use, the newest widely used family at each size, with the sizes of the default (q4_K_M) downloads read from each model page:
- chat: Qwen 3.5 from 0.8B to 122B, Gemma 4 12B / 26B / 31B, gpt-oss 20B / 120B. Gemma 4's E2B and E4B are left out: their downloads (4.6 and 6.6 GB) are those of much larger models, and by size they would push out better ones.
- code: qwen2.5-coder 1.5B / 3B / 7B / 14B, then qwen3-coder 30B. The 3B was missing: on a 6 GB GPU the best code model that fit was the 1.5B.
- reasoning: Qwen3 1.7B to 32B, which think before answering (better than the DeepSeek-R1 distills at each size).
- images: Qwen3-VL 2B to 32B. Embeddings: nomic-embed-text, EmbeddingGemma, bge-m3, qwen3-embedding 4B.

On this laptop (6 GB): chat qwen3.5:4b, split qwen3.5:9b; code qwen2.5-coder:3b; images qwen3-vl:4b. The note under the table said "embeddings 16-bit", no longer true; it now says the sizes are the default downloads and that the newest models need a recent Ollama.

**AMD on Windows.** AMD now publishes its own PyTorch for recent Radeon GPUs on Windows (RX 7000/9000, Ryzen AI Max; repo.radeon.com); PyTorch's index has none. A ROCm build that works was already recognized on any OS (by `torch.version.hip`). What was missing: a CPU build on an AMD machine was reported as fine, with no hint that a GPU build exists. Now an info points to PyTorch's index (choose ROCm) on Linux and to AMD's Windows instructions on Windows. AMD's Radeon/Ryzen docs say they cover releases through ROCm 7.2.1 and newer docs are unified, so the link goes to the page that exists now, and the message does not name a Python version (it changes with AMD's releases). The AMD GPU finding no longer says "this tool does not check ROCm"; it says what it checks.

**CI**: actions/checkout v4 → v7 and setup-uv v6 → v10, both on Node 24 (GitHub warned that Node 20 actions are deprecated). The inputs used (`python-version`, `activate-environment`) are unchanged in v10. The first push failed on every runner: setup-uv no longer publishes a moving `v10` tag, only exact releases, so it is now pinned to the v10.3.3 commit (safer for a third-party action anyway). CI then passed on all six jobs with no Node 20 warning.

**Tests**: 55 (+4): the list has a model for every use on an 8 GB laptop without a GPU, the picks on a 6 GB and a 24 GB GPU, the AMD findings on Windows and Linux, AMD's Windows ROCm build recognized as working.

**Open**
- A reviewer who has not seen the code, again (the last review was round 1).
- `--json` is not documented field by field; it is meant for bug reports, but scripts will use it.

## Round 4 — 2026-11-12

**Second independent review.** A reviewer who had not seen the code read all of it and reproduced each finding with synthetic machines (scripts outside the repository, nothing installed, no network). 15 findings; all fixed, each with a regression test in `tests/test_review2.py` that fails on the code before this round (checked by running the new tests with `src/` stashed).

- **Fix commands broke on ordinary paths.** Only paths with spaces were quoted, and with double quotes. `C:\Users\O'Brien\...` and `D:\R&D\...` did not parse in PowerShell; `proj(1)` ran `C:\Users\x\proj`; `"...\$work\..."` dropped `$work`; in bash, `/srv/r&d/...` put `/srv/r` in the background and ran a relative `d/.venv/bin/python`. Now PowerShell gets single quotes (only `'` is special, doubled) after `&`, POSIX shells get `shlex.quote`, for the interpreter, `--python` and the conda prefix. The tests run the generated commands through PowerShell's own parser (on Windows) and `bash` (in CI on Linux and macOS) and check that the path comes out as one word.
- **A working driver was called missing.** Without `nvidia-smi` on PATH (a separate package on Debian; `Program Files\NVIDIA Corporation\NVSMI` with old Windows drivers), an NVIDIA GPU in `lspci` or a properly named one in Windows gave "install the driver", and a CUDA PyTorch that worked got no finding at all. Now the NVSMI folder is searched; the driver counts as present when the Linux kernel module is loaded (`/proc/driver/nvidia/version`), when Windows names the GPU (not "Microsoft Basic Display Adapter"), or when any environment runs CUDA; that is a warning that only nvidia-smi is missing, and a working CUDA PyTorch is reported as working. A CPU build there is still the error it is.
- **A proxy password leaked into `--json`.** `HTTPS_PROXY=ann:s3cret@proxy:8080` (no scheme, which pip accepts) and passwords containing `/` were not redacted. Proxy variables are now redacted up to the last `@`, with or without a scheme; index URLs only up to the host, since their paths may contain `@`.
- "Environments disagree" counted a PyTorch whose GPU test kernel failed as the one that works, and pointed the user at it.
- **Containers** (Docker, Podman: `/.dockerenv`, `/run/.containerenv`, cgroups) see the host's GPU in sysfs and were told to install a driver; now they are told to start the container with `--gpus all` and never to install one inside. **WSL 1** (no GPU at all) is told to convert to WSL 2, with the distribution's name; the `wsl` commands now say to run them in Windows PowerShell, where `wsl` exists.
- **AMD**: Ryzen laptops' integrated graphics ("AMD Radeon(TM) Graphics", 780M, Cezanne ...) were treated as ROCm targets; WSL with a Radeon got native-Linux advice (install amdgpu; in WSL it must not be); a CUDA PyTorch on an AMD-only machine got no hint about ROCm. An Intel Mac's AMD GPU, which PyTorch uses through Metal, was reported as "no GPU".
- **conda**: the channel showed as "win-64" (the platform segment of the URL; the test had locked that in); torchvision installed by conda alone did not switch to the conda fix; and `conda remove` (which asks before removing) was a separate line, so a pasted next line could be taken as its answer and the install skipped. Now the removal and what follows are one line (`&&`; `; if ($?) { ... }` in Windows PowerShell 5.1).
- **PEP 668**: a system Python (Debian, Ubuntu, Homebrew) got a `pip install` that fails with "externally-managed-environment". Now the probe reads the marker and the fix makes a virtual environment instead.
- An unknown driver CUDA version (nvidia-smi failed or said N/A) was taken as 12.0: "the driver (CUDA None) is too old" on an RTX 5090. Unknown now means a build most drivers run, and no driver verdict.
- `CUDA_VISIBLE_DEVICES=1` on a one-GPU machine hides it; now said.

**Found by running it for real, not by the review.** Running the doctor against the Miniconda Python on this laptop (only the probe script, which reads) reported it as externally managed, with no conda information and someone else's pip version. The cause: uv's launcher for a venv's `python.exe` sets `PYTHONHOME` to uv's base Python, and the doctor passed its environment on to every Python it probed, so each of them took uv's Python as its home. On Windows that is every run under `uvx` or `uv run`. Round 1's end-to-end check did not show it because that environment was made from the same uv Python. The probe now drops a `PYTHONHOME` that the doctor's own launch set (and `__PYVENV_LAUNCHER__`), and keeps one the user set; a test starts a real child Python with a broken `PYTHONHOME` that would not even start before. Then, since a Python installed by `uv python install` really is marked externally managed, the venv fix was made to work in PowerShell too (`$HOME\.venvs\ai\Scripts\python.exe`, run with `&`).

**`--json` documented**: `docs/json.md`, field by field, with `schema: 1` now in the output and `models.as_of`. The READMEs link it, and now say that paths are not redacted and contain the user name.

**Tests**: 71 (+16), 4 skipped on Windows (bash and symlinks; CI runs them).

**Open**
- The conda fix end to end on a real conda environment with a CPU PyTorch (Miniconda is here, but installing PyTorch into it is a change to the user's machine I will not make for a test).
- The review's findings on Linux, macOS and WSL are tested with synthetic machines only.

## Round 5 — 2026-11-12: 1.0.0

**Run from GitHub, as a user would**: `uvx --from git+https://github.com/zch6a/ai-env-doctor ai-env-doctor`, pointed at the Miniconda Python on this laptop. It fetched and built the commit in about 15 seconds and reported correctly: the conda environment seen, Miniconda's own pip version, not marked PEP 668. So the round 4 PYTHONHOME fix holds under `uvx`, the case it was for.

**Every claim in the READMEs checked against the code.** Wrong ones, fixed:
- "uv may pick [the CPU build] anywhere": not so. On Linux, PyPI's torch is a CUDA build; there a CPU build comes from PyTorch's CPU index or a lock file. Fixed in the finding's text and in the example.
- "that is all it executes" (of the probe script): it also runs `nvidia-smi`, `lspci`, PowerShell's `Get-CimInstance`, `sysctl` and the `py` launcher. The README and `--help` now list them.
- "`uvx` starts it in a second or two": only once it is cached; the first run fetches and builds it (about 15 s here).
- "What it checks" did not mention what rounds 2–4 added (nvidia-smi missing, WSL 1, containers, PEP 668, AMD in WSL); the local Ollama check was not said to be a request to localhost only.

**Every message read again.** Two said too much: "pip misreads a proxy without http://" (pip, through requests, assumes http://), and "the build advised is for CUDA 12.6" when nvidia-smi is missing (Blackwell gets 12.8). The eight entries added in round 4 had been written by a script on single lines of up to 443 characters; they are now in the file's style, and a check confirmed that only the four intended strings changed.

**1.0.0**: version set, "in development" removed, the status says what was tested by hand, what by CI and what only by tests (WSL, containers, conda with a conda-installed PyTorch, AMD), and asks for reports from those machines. Tag `v1.0.0`.

**Tests**: 71 on Windows (74 on Linux and macOS, where the bash checks run).

## Agent Skill and presentation — 2026-11-12: 1.1.0

- **Agent Skill + Claude Code plugin** (`plugin/skills/ai-env-doctor/SKILL.md`, a marketplace in the repository itself; `claude plugin validate` passes for both). For an assistant the questions are "why is `torch.cuda.is_available()` False?", "pip install torch hangs" and "which local model fits?". The skill has it run the doctor in the user's project with `--json`, decide on `findings[].id` and `level` (the titles are translated and reworded: the JSON reference already says so), explain most severe first, and use the printed fix rather than a plain `pip install torch` (the fix already picks the newest build the driver can run). The fix commands uninstall and install packages: the assistant shows them and asks first; driver updates are the user's. `--offline` when the user has not agreed to the five network requests. Paths in the JSON contain the user's name: say so before it is posted.
- **`tests/test_skill.py`**: the frontmatter (portable fields, limits), plugin and marketplace against `__version__`, every command in the skill's bash blocks run through `cli.main` on a synthetic machine (an RTX 3060 with the CPU build; the probes replaced, so never the machine running the tests, and always offline), checking the exit code the skill states, every finding field it names, `torch-cpu` with its fix, and the `models` fields; `--python` reaching the interpreter search; and every finding id the skill names existing in the source.
- **README** (both languages): an "Ask your AI assistant" section. **Social preview** (`docs/social-preview.png`, from `scripts/social-card.html`): the README's real report, shortened. **CHANGELOG.md** (there was none). **Launch kit**: `launch/ai-env-doctor.md` outside the repository (drafts; posting is the user's).

79 tests (was 75; 4 of them skip by platform).

## Summary

ai-env-doctor answers "why does PyTorch not use my GPU?" with one command on Windows, macOS and Linux: it checks the driver, the PyTorch in each environment the user will actually run (by asking that environment), proxies, the network and disk space, and prints one fix per problem, as commands to copy, in English or Chinese; then the local models that fit.

What mattered most, in hindsight:
- **Asking the user's interpreter, not guessing.** It never imports torch itself; a probe script runs inside each Python found. Round 4 showed how easy this is to get wrong: the doctor's own launcher environment (uv's PYTHONHOME) silently changed what every probed Python saw.
- **Reading PyTorch's own index** to advise a build. In November 2026 the obvious fix ("install the CUDA build") installs PyTorch 2.15, built only for CUDA 13, which a CUDA 12 driver cannot run.
- **Fix commands that survive being pasted**: uv environments without pip, PowerShell's quoting rules, paths with `'`, `&`, `(` or `$`, conda's confirmation prompt, PEP 668 Pythons. Most of the bugs the two reviews found were here, and the tests now run the commands through the real PowerShell parser and bash.
- **Not crying wolf**: no "install the driver" when only nvidia-smi is missing, in a container, or on integrated graphics; WSL's /dev/dxg is not an NVIDIA GPU.

Rounds: development (28 tests) → 1 first review, end to end on an RTX 3060 (42) → 2 conda, WSL, dated model list (51) → 3 the model list checked against Ollama's library, AMD on Windows, CI on Node 24 (55) → 4 second review, the PYTHONHOME leak, `--json` documented (71) → 5 claims checked, 1.0.0.

Still open: WSL, containers, AMD GPUs and conda with a conda-installed PyTorch have not been tried by hand; the model list needs checking against Ollama's library every few months (the report says when it is old).

### 2026-11-14 — Chinese README: AI tone

2 places changed: the five-item 、 list in the tagline (now grouped into three), and the only decorative —— (now a comma, with the following semicolon split into a full stop).
The checks follow the findings of lieflat-less-ai-tone (shiujan, CC BY-NC 4.0); its findings were applied, none of its text was copied.
Left as they were: the colon-led lists and the exhaustive lists of what is checked and what is untested, the sample output, tables and code blocks.
