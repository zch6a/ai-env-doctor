"""Which well-known local models this machine can run, with Ollama.

The list is short on purpose (a few dozen widely used models, by use), with approximate download sizes
of Ollama's default 4-bit builds as of late 2026; check `ollama show` for the exact numbers. For a full
catalogue with speed estimates, see llmfit (https://github.com/AlexsJones/llmfit).

A model "fits" when its weights plus room for a 4K-token context and the runtime (about 1 GB and 10%)
fit in GPU memory; with less, Ollama puts part of it on the CPU, which works but is several times
slower; with no GPU, it runs on the CPU from system memory.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    tag: str  # `ollama run <tag>`
    size_gb: float  # download, Ollama's default quantization
    use: str  # chat, code, reasoning, vision, embedding
    note: str = ""


CATALOG = [
    Model("qwen3:0.6b", 0.5, "chat", "Chinese and English"), Model("llama3.2:1b", 1.3, "chat"), Model("gemma3:1b", 0.8, "chat"),
    Model("qwen3:1.7b", 1.4, "chat", "Chinese and English"), Model("llama3.2:3b", 2.0, "chat"), Model("qwen3:4b", 2.5, "chat", "Chinese and English"),
    Model("phi4-mini", 2.5, "chat"), Model("gemma3:4b", 3.3, "chat", "also reads images"), Model("mistral:7b", 4.1, "chat"),
    Model("llama3.1:8b", 4.9, "chat"), Model("qwen3:8b", 5.2, "chat", "Chinese and English"), Model("gemma3:12b", 8.1, "chat", "also reads images"),
    Model("qwen3:14b", 9.3, "chat", "Chinese and English"), Model("phi4:14b", 9.1, "chat"), Model("mistral-small:24b", 14.0, "chat"),
    Model("gpt-oss:20b", 14.0, "chat", "OpenAI's open model; reasoning"), Model("gemma3:27b", 17.0, "chat", "also reads images"),
    Model("qwen3:30b", 19.0, "chat", "mixture of experts: fast for its size"), Model("qwen3:32b", 20.0, "chat", "Chinese and English"),
    Model("llama3.1:70b", 43.0, "chat"), Model("gpt-oss:120b", 65.0, "chat"),
    Model("qwen2.5-coder:1.5b", 1.0, "code"), Model("qwen2.5-coder:7b", 4.7, "code"), Model("qwen2.5-coder:14b", 9.0, "code"),
    Model("qwen2.5-coder:32b", 20.0, "code"),
    Model("deepseek-r1:1.5b", 1.1, "reasoning"), Model("deepseek-r1:7b", 4.7, "reasoning"), Model("deepseek-r1:8b", 5.2, "reasoning"),
    Model("deepseek-r1:14b", 9.0, "reasoning"), Model("deepseek-r1:32b", 20.0, "reasoning"), Model("deepseek-r1:70b", 43.0, "reasoning"),
    Model("qwen2.5vl:3b", 3.2, "vision"), Model("qwen2.5vl:7b", 6.0, "vision"), Model("llava:7b", 4.7, "vision"),
    Model("nomic-embed-text", 0.3, "embedding"), Model("bge-m3", 1.2, "embedding", "100+ languages"),
]
USES = ("chat", "code", "reasoning", "vision", "embedding")


def need_gb(m: Model) -> float:
    return m.size_gb * 1.1 + 1.0


@dataclass
class Fit:
    device: str  # "gpu", "apple", "cpu"
    memory_gb: float  # what the model may use there
    spill_gb: float  # system memory that can take what does not fit (GPU only)
    full: dict  # use -> the largest model that fits whole
    partial: dict  # use -> a larger one that runs split between GPU and CPU
    note: str = ""


def plan(system) -> Fit:
    nvidia = [g for g in system.gpus if g.vendor == "nvidia" and g.memory_mb]
    apple = [g for g in system.gpus if g.vendor == "apple" and g.memory_mb]
    ram_gb = (system.ram_mb or 0) / 1024
    if nvidia:
        g = max(nvidia, key=lambda g: g.memory_mb)
        memory = g.memory_mb / 1024 - 0.5  # the desktop and other programs keep some
        device, spill = "gpu", max(0.0, ram_gb * 0.6 - 2)
    elif apple:
        memory, device, spill = apple[0].memory_mb / 1024, "apple", 0.0
    else:
        memory, device, spill = max(0.0, ram_gb * 0.7 - 2), "cpu", 0.0
    full, partial = {}, {}
    for use in USES:
        fits = [m for m in CATALOG if m.use == use and need_gb(m) <= memory]
        if fits:
            full[use] = max(fits, key=lambda m: m.size_gb)
        if spill:
            bigger = [m for m in CATALOG if m.use == use and memory < need_gb(m) <= memory + spill]
            if bigger:  # the next size up, not the largest: split across GPU and CPU, a huge model crawls
                partial[use] = min(bigger, key=lambda m: m.size_gb)
    note = "cpu" if device == "cpu" else ""
    return Fit(device, memory, spill, full, partial, note)
