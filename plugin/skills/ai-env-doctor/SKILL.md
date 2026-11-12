---
name: ai-env-doctor
description: Diagnose why PyTorch does not use the GPU, CUDA and driver mismatches, broken AI Python environments, proxy and mirror problems that break pip or Hugging Face downloads, and which local models this machine can run, with ai-env-doctor. Use when torch.cuda.is_available() is False, on CUDA or driver errors, when installing PyTorch, when downloads from pypi.org, download.pytorch.org or huggingface.co fail or hang, or when the user asks which local LLM fits their GPU.
license: MIT
compatibility: Needs Python 3.10+ and uv (uvx) or pip. Runs ai-env-doctor from GitHub (git+https://github.com/zch6a/ai-env-doctor) unless the ai-env-doctor command is installed. Read-only; the network check contacts five public hosts and can be turned off with --offline.
metadata:
  homepage: https://github.com/zch6a/ai-env-doctor
---

# Diagnosing an AI Python environment with ai-env-doctor

ai-env-doctor checks the GPU, the driver, the PyTorch each Python environment actually has, proxies, mirrors and
the network, and says what is wrong with the exact command that fixes it. It changes nothing.

## Run it

From the user's project folder (it checks the project's `.venv` first, then the active environment and the `python`
on PATH):

```bash
uvx --from git+https://github.com/zch6a/ai-env-doctor ai-env-doctor --json
```

Use `ai-env-doctor` instead of the `uvx --from …` prefix if the command is installed. Exit code: 0 nothing to fix,
1 something to fix (not a failure of the tool), 2 the doctor itself failed.

To check one interpreter, such as the one the user's editor uses:

```bash
uvx --from git+https://github.com/zch6a/ai-env-doctor ai-env-doctor --json --python .venv/bin/python
```

Without network checks (no requests to pypi.org, download.pytorch.org, huggingface.co, hf-mirror.com or Ollama's
registry), for example when the user has not agreed to them or the machine is offline:

```bash
uvx --from git+https://github.com/zch6a/ai-env-doctor ai-env-doctor --json --offline
```

## Read the result

Decide on `findings[].id` and `findings[].level` (`error`, `warning`, `info`, `ok`), never on the titles, which
are translated and reworded. Each finding has `title`, `detail`, `fix` (commands to run in order, empty when there
is nothing to do), `where` (the interpreter it is about) and `facts`. The field reference is
https://github.com/zch6a/ai-env-doctor/blob/main/docs/json.md.

Tell the user, most severe first, what is wrong and why, in plain words. `torch-cpu` (the CPU-only build on a
machine with an NVIDIA GPU) is the most common cause of `torch.cuda.is_available()` being False. When it says the
newest PyTorch needs a newer driver, the fix it prints already picks the newest build this driver can run: use that,
not a plain `pip install torch`.

`models` lists, per use (`chat`, `code`, `reasoning`, `vision`, `embedding`), the largest well-known model that fits
(`fits`) and the next size up that runs split between GPU and CPU (`split`), for the memory in `memory_gb`.

## Fixing

The `fix` commands uninstall and install packages, or change the environment. Show them and ask before running any;
run them in the order given, against the interpreter in `where`. Driver updates are for the user to do. Afterwards
run the doctor again to confirm.

## Privacy

Passwords, user names and tokens in proxy and index URLs are already replaced with `***`. Paths are not: they
contain the user's name. Before the user posts the output in an issue or a chat, say so.
