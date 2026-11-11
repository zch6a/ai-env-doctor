"""Which well-known local models this machine can run, with Ollama.

The list is short on purpose (a few dozen widely used models, by use), with approximate download sizes
of Ollama's default builds as of November 2026 (mostly 4-bit); check `ollama show` for the exact numbers. For a full
catalogue with speed estimates, see llmfit (https://github.com/AlexsJones/llmfit).

A model "fits" when its weights plus room for a 4K-token context and the runtime (about 1 GB and 10%)
fit in GPU memory; with less, Ollama puts part of it on the CPU, which works but is several times
slower; with no GPU, it runs on the CPU from system memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

CATALOG_AS_OF = date(2026, 11, 1)  # update with the list
STALE_AFTER_DAYS = 270  # new model families come out every few months


def catalog_stale(today: date | None = None) -> bool:
    """True once the list is old enough that newer, better models of the same size probably exist."""
    return ((today or date.today()) - CATALOG_AS_OF).days > STALE_AFTER_DAYS


@dataclass(frozen=True)
class Model:
    tag: str  # `ollama run <tag>`
    size_gb: float  # download, Ollama's default quantization
    use: str  # chat, code, reasoning, vision, embedding
    note: str = ""


# Checked against ollama.com/library in November 2026: per use, the newest widely used family at each
# size, so that "the largest that fits" is also a good choice (an older, larger model of the same
# class is left out). Gemma 4's E2B/E4B are left out too: their downloads (4.6 and 6.6 GB) are those of
# much larger models, so by size they would crowd out better ones.
CATALOG = [
    Model("qwen3.5:0.8b", 1.2, "chat", "also reads images"), Model("qwen3.5:2b", 2.7, "chat", "also reads images"),
    Model("qwen3.5:4b", 3.3, "chat", "also reads images"), Model("qwen3.5:9b", 6.6, "chat", "also reads images"),
    Model("gemma4:12b", 7.7, "chat", "also reads images"), Model("gpt-oss:20b", 14.0, "chat", "OpenAI's open model"),
    Model("gemma4:26b", 16.0, "chat", "mixture of experts: fast for its size"), Model("qwen3.5:27b", 17.0, "chat", "also reads images"),
    Model("gemma4:31b", 19.0, "chat", "also reads images"), Model("qwen3.5:35b", 22.0, "chat", "mixture of experts: fast for its size"),
    Model("gpt-oss:120b", 65.0, "chat", "mixture of experts"), Model("qwen3.5:122b", 81.0, "chat", "mixture of experts"),
    Model("qwen2.5-coder:1.5b", 1.0, "code"), Model("qwen2.5-coder:3b", 1.9, "code"), Model("qwen2.5-coder:7b", 4.7, "code"), Model("qwen2.5-coder:14b", 9.0, "code"),
    Model("qwen3-coder:30b", 19.0, "code", "mixture of experts: fast for its size"),
    Model("qwen3:1.7b", 1.4, "reasoning", "thinks before it answers"), Model("qwen3:4b", 2.5, "reasoning", "thinks before it answers"),
    Model("qwen3:8b", 5.2, "reasoning", "thinks before it answers"), Model("qwen3:14b", 9.3, "reasoning", "thinks before it answers"),
    Model("qwen3:30b", 19.0, "reasoning", "mixture of experts: fast for its size"), Model("qwen3:32b", 20.0, "reasoning"),
    Model("qwen3-vl:2b", 1.9, "vision"), Model("qwen3-vl:4b", 3.3, "vision"), Model("qwen3-vl:8b", 6.2, "vision"),
    Model("qwen3-vl:30b", 20.0, "vision", "mixture of experts: fast for its size"), Model("qwen3-vl:32b", 21.0, "vision"),
    Model("nomic-embed-text", 0.3, "embedding"), Model("embeddinggemma", 0.6, "embedding", "100+ languages"),
    Model("bge-m3", 1.2, "embedding", "100+ languages"), Model("qwen3-embedding:4b", 2.5, "embedding", "100+ languages"),
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
    unified = [g for g in system.gpus if g.vendor == "nvidia" and not g.memory_mb]  # GB10 and the like: nvidia-smi says [N/A]
    apple = [g for g in system.gpus if g.vendor == "apple" and g.memory_mb]
    ram_gb = (system.ram_mb or 0) / 1024
    if nvidia:
        g = max(nvidia, key=lambda g: g.memory_mb)
        memory = g.memory_mb / 1024 - 0.5  # the desktop and other programs keep some
        device, spill = "gpu", max(0.0, ram_gb * 0.6 - 2)
    elif apple:
        memory, device, spill = apple[0].memory_mb / 1024, "apple", 0.0
    elif unified and ram_gb:
        memory, device, spill = ram_gb * 0.75, "gpu", 0.0
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
