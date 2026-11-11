"""The network as AI tools see it: proxy settings, certificate bundles, and whether the places models
and packages come from can be reached.

Reachability is checked with one HEAD request to each of a few public hosts (listed in HOSTS), through
the proxy the environment configures, as pip and huggingface_hub would. Nothing else is sent: no
system details, no file names. `--offline` skips it.
"""

from __future__ import annotations

import os
import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit

# (name, URL, what it is for)
HOSTS = [
    ("pypi", "https://pypi.org/simple/pip/", "Python packages (pip, uv)"),
    ("pytorch", "https://download.pytorch.org/whl/", "PyTorch's CUDA builds"),
    ("huggingface", "https://huggingface.co/api/models?limit=1", "models from Hugging Face"),
    ("hf-mirror", "https://hf-mirror.com/api/models?limit=1", "a Hugging Face mirror many use in China"),
    ("ollama-registry", "https://registry.ollama.ai/v2/", "models for Ollama"),
]
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
LOCAL = ("localhost", "127.0.0.1", "::1")


@dataclass
class Reach:
    name: str
    url: str
    purpose: str
    ok: bool
    status: int | None = None
    seconds: float | None = None
    error: str | None = None
    kind: str | None = None  # dns, timeout, refused, ssl, http, proxy


def proxies(env: dict) -> dict[str, str]:
    """The proxy variables that are set, by name."""
    return {k: env[k] for k in PROXY_KEYS if env.get(k)}


def no_proxy_covers_localhost(env: dict) -> bool:
    value = env.get("NO_PROXY") or env.get("no_proxy") or ""
    entries = [x.strip().lower() for x in value.split(",") if x.strip()]
    return "*" in entries or any(x in entries for x in LOCAL)


def proxy_problems(env: dict) -> list[tuple[str, str]]:
    """(problem id, detail) for the proxy settings themselves, before any request."""
    found = []
    set_ = proxies(env)
    for key, value in set_.items():
        parts = urlsplit(value if "://" in value else "http://" + value)
        if "://" not in value:
            found.append(("proxy-no-scheme", key))
        elif parts.scheme not in ("http", "https", "socks5", "socks5h", "socks4"):
            found.append(("proxy-bad-scheme", f"{key}={parts.scheme}"))
        if not parts.hostname:
            found.append(("proxy-no-host", key))
        if parts.scheme.startswith("socks"):
            found.append(("proxy-socks", key))
    names = {k.lower() for k in set_}
    if "http_proxy" in names and not names & {"https_proxy", "all_proxy"}:
        # Every package and model download is HTTPS: with only an HTTP proxy, they all go around it.
        found.append(("proxy-https-missing", ",".join(k for k in set_ if k.lower() == "http_proxy")))
    if set_ and not no_proxy_covers_localhost(env):
        found.append(("proxy-localhost", ",".join(sorted(set_))))
    for key in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE", "PIP_CERT"):
        if env.get(key) and not os.path.isfile(env[key]):
            found.append(("ca-missing", f"{key}={env[key]}"))
    if (env.get("HF_HUB_DISABLE_SSL_VERIFICATION") or "").lower() in ("1", "true", "yes"):
        found.append(("hf-ssl-off", "HF_HUB_DISABLE_SSL_VERIFICATION"))
    if (env.get("HF_HUB_OFFLINE") or "").lower() in ("1", "true", "yes"):
        found.append(("hf-offline", "HF_HUB_OFFLINE"))
    return found


def classify(e: BaseException) -> str:
    text = repr(e).lower()
    reason = getattr(e, "reason", e)
    if isinstance(reason, ssl.SSLError) or "certificate" in text or "ssl" in text:
        return "ssl"
    if isinstance(reason, socket.gaierror) or "getaddrinfo" in text or "name or service not known" in text or "nodename nor servname" in text:
        return "dns"
    if isinstance(reason, TimeoutError | socket.timeout) or "timed out" in text:
        return "timeout"
    if isinstance(reason, ConnectionRefusedError) or "refused" in text or "10061" in text:
        return "refused"
    if "tunnel" in text or "proxy" in text or "407" in text:
        return "proxy"
    return "other"


def check(name: str, url: str, purpose: str, timeout: float = 8) -> Reach:
    """One HEAD request, through the environment's proxy (urllib reads the same variables as pip)."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "ai-env-doctor"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return Reach(name, url, purpose, True, r.status, time.perf_counter() - t0)
    except urllib.error.HTTPError as e:
        # The server answered: the network works, whatever it thinks of a HEAD request.
        ok = e.code < 500 and e.code != 407
        return Reach(name, url, purpose, ok, e.code, time.perf_counter() - t0, None if ok else f"HTTP {e.code}",
                     None if ok else ("proxy" if e.code == 407 else "http"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        return Reach(name, url, purpose, False, None, time.perf_counter() - t0, str(getattr(e, "reason", e))[:200], classify(e))


def reachability(hosts=HOSTS, timeout: float = 8) -> list[Reach]:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        return list(pool.map(lambda h: check(*h, timeout=timeout), hosts))


def local_ollama(env: dict, timeout: float = 2) -> dict | None:
    """Is an Ollama server running here? Asked directly, never through a proxy."""
    host = env.get("OLLAMA_HOST") or "127.0.0.1:11434"
    if "://" not in host:
        host = "http://" + host
    host = host.replace("//0.0.0.0", "//127.0.0.1")
    if urlsplit(host).hostname not in LOCAL:
        return None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(host.rstrip("/") + "/api/version", timeout=timeout) as r:
            import json

            return {"url": host, "version": json.loads(r.read().decode()).get("version")}
    except (urllib.error.URLError, OSError, ValueError):
        return None
