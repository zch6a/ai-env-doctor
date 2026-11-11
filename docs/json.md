# `ai-env-doctor --json`

Everything the doctor found, as one JSON object on standard output, for scripts and bug reports. The
exit code is the same as without `--json`: 0 nothing to fix, 1 something to fix, 2 the doctor failed.

`schema` is 1. Fields may be added within a schema version; a field is removed or changes meaning
only with a new one. Decide on `findings[].id` and `findings[].level`, not on the titles, which are
translated (`--lang`) and reworded between releases.

**Privacy.** Passwords, user names and query-string tokens in proxy and index URLs are replaced with
`***`. Paths are not: they contain your user name (for example `C:\Users\<you>\...`). Look over the
output before you post it.

## Top level

| field | |
|---|---|
| `schema` | 1 |
| `version` | the ai-env-doctor version |
| `system` | the machine, below |
| `pythons` | each Python environment checked, below |
| `network` | proxies and reachability, below |
| `findings` | what is wrong and what is fine, most severe first, below |
| `models` | the local models that fit, below |

## `system`

| field | |
|---|---|
| `os` | `windows`, `linux`, `macos` |
| `os_version` | as the OS reports it |
| `machine` | the hardware: `x86_64`, `arm64`, `aarch64` (`arm64` also under Rosetta) |
| `wsl` | running inside WSL |
| `wsl_gpu` | WSL can reach a GPU through `/dev/dxg` (of any make) |
| `rosetta` | the doctor itself runs as an Intel process on Apple Silicon |
| `ram_mb`, `ram_free_mb` | system memory |
| `cpu`, `cores` | |
| `gpus` | GPUs that can be used: `vendor` (`nvidia`, `apple`), `name`, `memory_mb` (Apple: what the GPU may use of the unified memory; `null` for NVIDIA GPUs with shared memory), `used_mb`, `compute_capability` (NVIDIA, such as `"8.6"`), `driver` |
| `driver_cuda` | the newest CUDA the NVIDIA driver supports, such as `"12.7"`; `null` without one |
| `nvidia_smi` | the path of `nvidia-smi`, if found |
| `gpu_seen_by_os` | display adapters the OS lists, with or without a driver; filled only when `gpus` is empty |
| `disk_free_mb` | free space where models are kept: `huggingface`, `ollama` (`null` when it cannot be read) |
| `env` | the environment variables that matter here (CUDA, proxies, Hugging Face, pip/uv indexes, Ollama, virtualenv/conda); redacted |

## `pythons[]`

| field | |
|---|---|
| `path` | the interpreter |
| `why` | how it was found: `venv` (the project's `.venv`), `virtualenv`, `conda`, `path`, `given` (`--python`) |
| `origin` | the folder, environment or command it was found by |
| `error` | why it could not be checked (`store-stub`: the Microsoft Store placeholder); `null` when it was |
| `info` | what the probe printed from inside it, below; `{}` on error |

`info`: `python` (version), `executable`, `bits`, `machine` (of that process: `x86_64` under Rosetta),
`packages` (name → version, for pip, torch and the usual AI packages; read from metadata, not
imported), `conda` (in a conda environment: `prefix`, and `packages`, the PyTorch packages conda
itself installed, each with `version`, `build`, `channel`), and `torch`:

- `null`: PyTorch is not installed there.
- `{"import_error": "..."}`: it is, but `import torch` failed.
- otherwise: `version`, `cuda` (the CUDA it was built for, `null` for CPU and ROCm builds), `hip`
  (ROCm builds), `cuda_available` (`torch.cuda.is_available()`), `cuda_works` (a small kernel ran on
  the GPU; absent when CUDA is not available), `cuda_error`, `arch_list` (the GPU architectures it
  has code for), `devices` (`name`, `memory_mb`, `capability`), `cudnn`, `mps_built`,
  `mps_available`, `warnings` (what importing it warned).

## `network`

| field | |
|---|---|
| `proxy_problems` | `[id, detail]` pairs, the same ids as the findings |
| `reach` | each host checked: `name`, `url`, `purpose`, `ok`, `status`, `seconds`, `error`, `kind` (`dns`, `timeout`, `refused`, `reset`, `ssl`, `http`, `proxy`); empty with `--offline` |
| `ollama` | the local Ollama server, `{"url", "version"}`, or `null` |
| `pytorch_index_read` | whether PyTorch's index was read to choose a build (otherwise a built-in table was used) |

## `findings[]`

| field | |
|---|---|
| `level` | `error` (AI work will fail), `warning` (may fail or be slow), `info`, `ok` |
| `id` | stable, such as `torch-cpu`, `wheel-major-newer`, `driver-missing`, `net-timeout` |
| `title`, `detail` | in the report's language |
| `fix` | the command or step that fixes it; may be several lines, to run in order; empty when there is nothing to do |
| `where` | the interpreter it is about, if any |
| `facts` | the values the message was made from (versions, paths, the CUDA tag chosen, ...) |

## `models`

| field | |
|---|---|
| `device` | where models would run: `gpu`, `apple`, `cpu` |
| `memory_gb` | the memory they may use there |
| `fits` | use (`chat`, `code`, `reasoning`, `vision`, `embedding`) → the largest model in the list that fits whole |
| `split` | use → the next size up, which runs split between GPU and CPU |
| `as_of` | the month the model list was checked against Ollama's library |
