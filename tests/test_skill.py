"""The Agent Skill (plugin/skills/ai-env-doctor/SKILL.md) and the Claude Code plugin around it: the format, and that
every command the skill gives works and returns the fields it tells the assistant to read. Run on a synthetic
machine (the probes replaced), never on the one running the tests, and offline."""

import json
import re
import shlex
from pathlib import Path

import pytest

from ai_env_doctor import __version__, cli, probe, pyprobe
from ai_env_doctor.probe import GPU, System

ROOT = Path(__file__).resolve().parent.parent
SKILL = (ROOT / "plugin" / "skills" / "ai-env-doctor" / "SKILL.md").read_text(encoding="utf-8").replace("\r\n", "\n")
PREFIX = "uvx --from git+https://github.com/zch6a/ai-env-doctor ai-env-doctor"


def frontmatter() -> dict[str, str]:
    m = re.match(r"---\n(.*?)\n---\n", SKILL, re.S)
    assert m, "the frontmatter starts on the first line"
    out = {}
    for line in m.group(1).split("\n"):
        kv = re.match(r"([a-z-]+):\s*(.*)$", line)
        if kv:
            out[kv.group(1)] = kv.group(2)
    return out


def test_frontmatter_has_portable_fields_within_the_limits():
    fm = frontmatter()
    assert fm["name"] == "ai-env-doctor"
    assert set(fm) <= {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
    assert 100 < len(fm["description"]) <= 1536 and not re.search(r"[<>]", fm["description"])
    assert len(fm.get("compatibility", "")) <= 500


def test_plugin_and_marketplace_agree_and_follow_the_version():
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    plugin = json.loads((ROOT / "plugin" / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert market["plugins"][0]["name"] == plugin["name"]
    assert (ROOT / market["plugins"][0]["source"] / ".claude-plugin" / "plugin.json").exists()
    assert plugin["version"] == __version__


def commands() -> list[list[str]]:
    lines = [line for block in re.findall(r"```bash\n(.*?)```", SKILL, re.S) for line in block.split("\n")]
    return [shlex.split(line[len(PREFIX):]) for line in lines if line.startswith(PREFIX)]


@pytest.fixture
def machine(monkeypatch):
    s = System("windows", "10", "x86_64", ram_mb=16384, cpu="i9", gpus=[GPU("nvidia", "RTX 3060 Laptop", 6144, 0, "8.6", "566.36")],
               driver_cuda="12.7", nvidia_smi="nvidia-smi", env={})
    monkeypatch.setattr(probe, "system", lambda: s)
    asked = []
    cpu = pyprobe.Interpreter("C:/p/.venv/Scripts/python.exe", "venv", "C:/p/.venv",
                              {"python": "3.12.1", "bits": 64, "packages": {"torch": "2.15.0+cpu"}, "torch": {"version": "2.15.0+cpu", "cuda": None}})

    def interpreters(explicit=None, cwd=None):
        asked.append(explicit)
        return [cpu]

    monkeypatch.setattr(pyprobe, "interpreters", interpreters)
    return asked


def test_every_command_the_skill_gives_works(machine, capsys):
    cmds = commands()
    assert [c for c in cmds if "--json" not in c] == [], "every command asks for JSON, which the skill reads"
    assert len(cmds) == 3
    for args in cmds:
        # The network check is tested on its own (tests/test_net_cli.py, against local stand-ins); here, never online.
        run = args if "--offline" in args else [*args, "--offline"]
        code = cli.main(run)
        out = capsys.readouterr().out
        assert code == 1, "a CPU build on an NVIDIA machine: something to fix, exit 1 as the skill says"
        data = json.loads(out)
        findings = data["findings"]
        for f in findings:
            assert {"id", "level", "title", "detail", "fix", "where", "facts"} <= set(f), f
            assert f["level"] in {"error", "warning", "info", "ok"}
        cpu = next(f for f in findings if f["id"] == "torch-cpu")
        assert cpu["level"] == "error" and "download.pytorch.org/whl/cu126" in cpu["fix"]
        models = data["models"]
        assert {"device", "memory_gb", "fits", "split"} <= set(models)
        assert set(models["fits"]) <= {"chat", "code", "reasoning", "vision", "embedding"}
    assert [".venv/bin/python"] in machine, "--python reaches the interpreter search"


def test_the_ids_the_skill_names_exist():
    source = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src" / "ai_env_doctor").glob("*.py"))
    for finding_id in re.findall(r"`([a-z]+(?:-[a-z]+)+)`", SKILL):
        if finding_id.startswith(("ai-env", "torch.")):
            continue
        assert f'"{finding_id}"' in source, finding_id
