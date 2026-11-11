"""ai-env-doctor: why does my GPU not work for AI, and what can this machine run?

    ai-env-doctor                 check everything and say what to fix
    ai-env-doctor --lang zh       in Chinese
    ai-env-doctor --python PATH   check that Python environment (default: the ones you are likely to use)
    ai-env-doctor --offline       no network checks
    ai-env-doctor --json          everything found, for scripts and bug reports
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

from . import __version__, models, net, probe, pyprobe
from .rules import diagnose

MARK = {"error": "✗", "warning": "!", "info": "·", "ok": "✓"}
ASCII = {"error": "X", "warning": "!", "info": "-", "ok": "+"}
COLOR = {"error": "\033[1;31m", "warning": "\033[1;33m", "info": "\033[2m", "ok": "\033[32m"}
TEXT = {
    "en": {"title": "ai-env-doctor {v}", "machine": "Machine", "findings": "Findings", "models": "Models this machine can run (Ollama)",
           "whole": "fits", "split": "split GPU/CPU, slower", "none": "none", "fix": "fix", "summary": "{e} to fix, {w} to look at",
           "all_good": "Nothing to fix.", "net_skipped": "Network not checked (--offline).",
           "contacting": "Contacting {hosts} (one HEAD request each; --offline to skip)",
           "device": {"gpu": "on the GPU ({mem:.1f} GB)", "apple": "on the Apple GPU ({mem:.1f} GB)", "cpu": "on the CPU ({mem:.1f} GB of memory; slow)"},
           "uses": {"chat": "chat", "code": "code", "reasoning": "reasoning", "vision": "images", "embedding": "embeddings (search)"},
           "sizes": "Sizes are Ollama's default 4-bit builds, approximate; `ollama run <name>` to try one."},
    "zh": {"title": "ai-env-doctor {v}", "machine": "本机", "findings": "检查结果", "models": "这台电脑能跑的模型（Ollama）",
           "whole": "能完整放下", "split": "显卡和 CPU 分担，较慢", "none": "无", "fix": "解决", "summary": "{e} 项需要解决，{w} 项值得看看",
           "all_good": "没有需要解决的问题。", "net_skipped": "未检查网络（--offline）。",
           "contacting": "正在连接 {hosts}（每个只发一个 HEAD 请求；加 --offline 可跳过）",
           "device": {"gpu": "在显卡上（{mem:.1f} GB）", "apple": "在 Apple 显卡上（{mem:.1f} GB）", "cpu": "在 CPU 上（内存 {mem:.1f} GB，较慢）"},
           "uses": {"chat": "对话", "code": "写代码", "reasoning": "推理", "vision": "看图", "embedding": "向量（搜索）"},
           "sizes": "大小是 Ollama 默认的 4 位量化版本，为近似值；用 `ollama run <名称>` 试用。"},
}


def _can_print(s: str) -> bool:
    try:
        s.encode(sys.stdout.encoding or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def report(system, findings, fit, lang: str, color: bool, net_checked: bool) -> str:
    t = TEXT[lang]
    marks = MARK if _can_print("✓✗·") else ASCII
    out = [t["title"].format(v=__version__), ""]
    g = system.gpus[0].name if system.gpus else "-"
    out.append(f"{t['machine']}: {system.os} {system.machine}{' (WSL)' if system.wsl else ''}, {system.cpu or '?'}, "
               f"{(system.ram_mb or 0) / 1024:.0f} GB RAM, GPU: {g}")
    out.append("")
    out.append(t["findings"] + ":")
    for f in findings:
        mark = marks[f.level]
        head = f"{COLOR[f.level]}{mark}\033[0m" if color else mark
        out.append(f"  {head} {f.title}")
        for line in (f.detail or "").splitlines():
            if line.strip():
                out.append(f"      {line}")
        if f.fix:
            out.append(f"      {t['fix']}: {f.fix}")
    if not net_checked:
        out.append(f"  {marks['info']} {t['net_skipped']}")
    errors = sum(f.level == "error" for f in findings)
    warnings = sum(f.level == "warning" for f in findings)
    out.append("")
    out.append(t["summary"].format(e=errors, w=warnings) if errors or warnings else t["all_good"])
    out.append("")
    out.append(f"{t['models']} — {t['device'][fit.device].format(mem=fit.memory_gb)}:")
    for use in models.USES:
        whole = fit.full.get(use)
        split = fit.partial.get(use)
        line = f"  {t['uses'][use]:<20} {t['whole']}: {whole.tag + f' ({whole.size_gb:g} GB)' if whole else t['none']}"
        if split:
            line += f";  {t['split']}: {split.tag} ({split.size_gb:g} GB)"
        out.append(line)
    out.append("  " + t["sizes"])
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Exit codes: 0 nothing to fix, 1 something to fix (an error), 2 the doctor itself failed."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace", **({} if sys.stdout.isatty() else {"encoding": "utf-8"}))
    p = argparse.ArgumentParser(prog="ai-env-doctor", description="Why does my GPU not work for AI, and what can this machine run?",
                                epilog="Only reads: it installs and changes nothing. Network checks send one HEAD request to each of "
                                       + ", ".join(h[1].split("/")[2] for h in net.HOSTS) + " (--offline to skip).")
    p.add_argument("--version", action="version", version=f"ai-env-doctor {__version__}")
    p.add_argument("--lang", choices=["en", "zh"], default=None, help="language of the report (default: from the system, else en)")
    p.add_argument("--python", action="append", metavar="PATH", help="a Python interpreter to check (can be given several times)")
    p.add_argument("--offline", action="store_true", help="no network checks")
    p.add_argument("--json", action="store_true", help="everything found, as JSON (passwords in proxy URLs are hidden)")
    args = p.parse_args(argv)
    lang = args.lang or default_lang()
    try:
        system = probe.system()
        interpreters = pyprobe.interpreters(args.python)
        net_facts = None
        if not args.offline:
            if not args.json:
                print(TEXT[lang]["contacting"].format(hosts=", ".join(h[1].split("/")[2] for h in net.HOSTS)), file=sys.stderr)
            from concurrent.futures import ThreadPoolExecutor

            from . import pytorch

            with ThreadPoolExecutor(max_workers=2) as pool:
                index = pool.submit(pytorch.fetch_index) if system.gpus and any(g.vendor == "nvidia" for g in system.gpus) else None
                reach = net.reachability()
                torch_index = index.result() if index else None
            net_facts = {"proxy_problems": net.proxy_problems(system.env), "reach": reach, "ollama": net.local_ollama(dict(os.environ)),
                         "torch_index": torch_index}
        else:
            net_facts = {"proxy_problems": net.proxy_problems(system.env), "reach": [], "ollama": None}
        findings = diagnose(system, interpreters, net_facts, lang)
        fit = models.plan(system)
    except KeyboardInterrupt:
        return 130
    if args.json:
        print(json.dumps({"version": __version__, "system": system.to_dict(),
                          "pythons": [asdict(i) for i in interpreters],
                          "network": {"proxy_problems": net_facts["proxy_problems"], "reach": [asdict(r) for r in net_facts["reach"]],
                                      "ollama": net_facts["ollama"], "pytorch_index_read": bool(net_facts.get("torch_index"))},
                          "findings": [asdict(f) for f in findings],
                          "models": {"device": fit.device, "memory_gb": round(fit.memory_gb, 1),
                                     "fits": {u: m.tag for u, m in fit.full.items()}, "split": {u: m.tag for u, m in fit.partial.items()}}},
                         ensure_ascii=False, indent=2, default=str))
    else:
        print(report(system, findings, fit, lang, sys.stdout.isatty() and not os.environ.get("NO_COLOR"), not args.offline))
    return 1 if any(f.level == "error" for f in findings) else 0


def default_lang() -> str:
    import locale

    for value in (os.environ.get("LANG"), os.environ.get("LC_ALL"), (locale.getlocale()[0] or "")):
        if value and value.lower().startswith(("zh", "chinese")):
            return "zh"
    return "en"


if __name__ == "__main__":
    sys.exit(main())
