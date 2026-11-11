import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ai_env_doctor import cli, net, probe, pyprobe
from ai_env_doctor.probe import GPU, System


class Server:
    def __init__(self, status=200, body=b"{}"):
        me = self
        self.requests = []

        class H(BaseHTTPRequestHandler):
            def _answer(self):
                me.requests.append((self.command, self.path))
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            do_HEAD = do_GET = _answer

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def test_proxy_settings_that_break_things():
    found = dict(net.proxy_problems({"HTTP_PROXY": "http://proxy:8080"}))
    assert "proxy-https-missing" in found and "proxy-localhost" in found
    assert "proxy-no-scheme" in dict(net.proxy_problems({"HTTPS_PROXY": "proxy:8080", "NO_PROXY": "localhost"}))
    assert "proxy-socks" in dict(net.proxy_problems({"ALL_PROXY": "socks5://127.0.0.1:1080", "NO_PROXY": "localhost,127.0.0.1"}))
    assert net.proxy_problems({"HTTPS_PROXY": "http://p:1", "HTTP_PROXY": "http://p:1", "NO_PROXY": "localhost,127.0.0.1"}) == []
    assert "ca-missing" in dict(net.proxy_problems({"REQUESTS_CA_BUNDLE": "/no/such/ca.pem"}))
    assert "hf-ssl-off" in dict(net.proxy_problems({"HF_HUB_DISABLE_SSL_VERIFICATION": "1"}))


def test_reachability_counts_any_answer_as_reached(monkeypatch):
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(v, raising=False)
    ok, refused = Server(200), Server(405)
    try:
        assert net.check("a", ok.url + "/x", "t").ok and ok.requests == [("HEAD", "/x")], "one HEAD request, nothing else"
        r = net.check("b", refused.url, "t")
        assert r.ok and r.status == 405, "the server answered: the network works"
    finally:
        ok.close()
        refused.close()
    gone = net.check("c", "http://127.0.0.1:9/", "t", timeout=3)
    assert not gone.ok and gone.kind in ("refused", "timeout")


def test_a_local_ollama_is_found_without_the_proxy(monkeypatch):
    ollama = Server(200, b'{"version": "0.12.3"}')
    try:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")  # a proxy that does not exist: going through it would fail
        found = net.local_ollama({"OLLAMA_HOST": ollama.url})
        assert found == {"url": ollama.url, "version": "0.12.3"}
        assert net.local_ollama({"OLLAMA_HOST": "http://10.1.2.3:11434"}) is None, "only this machine"
    finally:
        ollama.close()


@pytest.fixture
def machine(monkeypatch):
    s = System("windows", "10", "x86_64", ram_mb=16384, cpu="i9", gpus=[GPU("nvidia", "RTX 3060 Laptop", 6144, 0, "8.6", "566.36")],
               driver_cuda="12.7", nvidia_smi="nvidia-smi", env={"HTTPS_PROXY": "http://ann:***@proxy:8080"})
    monkeypatch.setattr(probe, "system", lambda: s)
    cpu = pyprobe.Interpreter("C:/p/.venv/Scripts/python.exe", "venv", "C:/p/.venv",
                              {"python": "3.12.1", "bits": 64, "packages": {"torch": "2.15.0+cpu"}, "torch": {"version": "2.15.0+cpu", "cuda": None}})
    monkeypatch.setattr(pyprobe, "interpreters", lambda explicit=None, cwd=None: [cpu])
    return s


def test_the_report_names_the_problem_and_the_fix(machine, capsys):
    code = cli.main(["--offline", "--lang", "en"])
    out = capsys.readouterr().out
    assert code == 1, "an error: exit 1"
    assert "CPU-only build" in out and "--index-url https://download.pytorch.org/whl/cu126" in out
    assert "Network not checked" in out and "Models this machine can run" in out


def test_json_for_bug_reports_hides_passwords(machine, capsys):
    cli.main(["--offline", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["system"]["env"]["HTTPS_PROXY"] == "http://ann:***@proxy:8080"
    assert any(f["id"] == "torch-cpu" for f in data["findings"]) and data["models"]["device"] == "gpu"


def test_chinese_report(machine, capsys):
    cli.main(["--offline", "--lang", "zh"])
    assert "纯 CPU 版" in capsys.readouterr().out
