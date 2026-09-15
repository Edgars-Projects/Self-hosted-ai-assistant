"""Boot-time installers for Ollama, SearxNG, Crawl4AI and the browser.

Each installer is idempotent and records failures with :func:`boot_issue`
instead of crashing, so the assistant can report them once it is running.
"""

from __future__ import annotations

import glob
import os
import platform
import shutil
import subprocess
import sys
import time
from types import ModuleType

from .config import (
    CRAWL_LIB,
    ENABLE_VOICE,
    JOB_DIR,
    OLLAMA_URL,
    SEARX_LIB,
    SEARX_PORT,
    SEARX_URL,
    STATE_DIR,
)

# ============================================================= bootstrap
# Anything that fails during boot lands here. The model is not running
# yet at that point, so the failures are handed to it afterwards.
BOOT_ISSUES: list[dict[str, str]] = []


def boot_issue(component: str, detail: object) -> None:
    """Record a non-fatal startup failure to report to the owner later."""
    BOOT_ISSUES.append({"component": component, "detail": str(detail)[-600:]})
    print(f"[boot issue] {component}: {str(detail)[:200]}")


CLEAN_ENV_GLOBAL = {
    k: v
    for k, v in os.environ.items()
    if k
    not in (
        "TELEGRAM_TOKEN",
        "KAGGLE_USER_SECRETS_TOKEN",
        "KAGGLE_KEY",
        "KAGGLE_USERNAME",
        "KAGGLE_DATA_PROXY_TOKEN",
    )
}


def sh(cmd: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command, optionally silencing its output."""
    return subprocess.run(
        cmd,
        shell=True,
        check=check,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )


def ensure_pip_deps() -> None:
    """Install any missing core Python packages."""
    need = []
    pairs = [
        ("requests", "requests"),
        ("telegram", "python-telegram-bot"),
        ("apscheduler", "APScheduler"),
        ("bs4", "beautifulsoup4"),
        ("numpy", "numpy"),
        ("pypdf", "pypdf"),
        ("nest_asyncio", "nest_asyncio"),
    ]
    if ENABLE_VOICE:
        pairs += [("faster_whisper", "faster-whisper"), ("edge_tts", "edge-tts")]
    pairs += [
        ("yt_dlp", "yt-dlp"),
        ("reportlab", "reportlab"),
        ("openpyxl", "openpyxl"),
    ]
    for mod, pkg in pairs:
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    if need:
        print("installing:", " ".join(need))
        sh(f"{sys.executable} -m pip install -q " + " ".join(need), check=False)


def find_ollama() -> str | None:
    """Locate an executable ``ollama`` binary in common install locations."""
    cands = [
        shutil.which("ollama"),
        "/usr/local/bin/ollama",
        "/usr/bin/ollama",
        "/opt/homebrew/bin/ollama",
        os.path.expanduser("~/.local/bin/ollama"),
    ]
    for p in cands:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    for pat in ("/usr/local/*/ollama", "/usr/*/ollama", "/opt/*/ollama"):
        for p in glob.glob(pat):
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    return None


def install_ollama() -> str:
    """Install Ollama via the official script, falling back to the tarball."""
    arch = platform.machine().lower()
    arch = "arm64" if arch in ("aarch64", "arm64") else "amd64"
    sh(
        "(apt-get update -qq && apt-get install -y -qq zstd curl ffmpeg) "
        "|| (yum install -y zstd curl ffmpeg) || true",
        check=False,
        quiet=True,
    )
    print("installing ollama")
    sh("curl -fsSL https://ollama.com/install.sh | sh", check=False)
    found = find_ollama()
    if found:
        return found
    url = f"https://ollama.com/download/ollama-linux-{arch}.tgz"
    if sh(f"curl -fsSL {url} -o /tmp/ollama.tgz", check=False).returncode == 0:
        sh("tar -C /usr/local -xzf /tmp/ollama.tgz", check=False)
        found = find_ollama()
        if found:
            return found
    raise RuntimeError("could not install ollama")


def iso_install(target: str, spec: str, editable: str | None = None) -> bool:
    """Install into a private directory. PYTHONPATH later makes it win
    over the host's version of the same package."""
    os.makedirs(target, exist_ok=True)
    what = f"-e {editable}" if editable else spec
    r = sh(
        f'{sys.executable} -m pip install -q --target "{target}" '
        f"--break-system-packages --upgrade {what}",
        check=False,
    )
    return r.returncode == 0


def iso_env(target: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for running an isolated service."""
    e = dict(CLEAN_ENV_GLOBAL)
    path = target + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    e["PYTHONPATH"] = path
    e["PYTHONNOUSERSITE"] = "1"
    if extra:
        e.update(extra)
    return e


def ensure_searx() -> bool:
    """SearxNG as a local service, isolated via --target. Replaces paid
    search APIs entirely: aggregates Google/Bing/DDG/Brave, no key."""
    root = "/opt/searxng"
    if not os.path.isdir(os.path.join(root, "searx")):
        print("installing searxng (isolated)")
        sh(
            f"git clone --depth 1 https://github.com/searxng/searxng {root}",
            check=False,
            quiet=True,
        )
    if not os.path.isdir(os.path.join(SEARX_LIB, "flask")):
        req = os.path.join(root, "requirements.txt")
        if not os.path.isfile(req):
            print("searxng checkout looks wrong — no requirements.txt")
            return False
        print("installing searxng requirements (isolated)")
        if not iso_install(SEARX_LIB, f'-r "{req}"'):
            boot_issue("searxng", "pip install -r requirements.txt failed")
            return False
    # Write our own minimal settings instead of using upstream's default,
    # which enables ~200 engines. Any one of them failing to register
    # kills the whole service — 'ahmia' is a Tor onion engine that needs
    # a Tor proxy we do not have, and it took searx down entirely.
    cfg = os.path.join(STATE_DIR, "searx-settings.yml")
    os.makedirs(STATE_DIR, exist_ok=True)
    open(cfg, "w").write(f"""use_default_settings:
  engines:
    keep_only:
      - google
      - duckduckgo
      - bing
      - brave
      - wikipedia
      - wikidata
      - github
      - stackexchange
      - hackernews
      - arxiv
      - openstreetmap
      - reddit
general:
  debug: false
  instance_name: "botsearx"
search:
  safe_search: 0
  formats:
    - html
    - json
server:
  secret_key: "{os.urandom(16).hex()}"
  limiter: false
  public_instance: false
  image_proxy: false
outgoing:
  request_timeout: 8.0
  max_request_timeout: 15.0
  pool_connections: 50
""")
    env = iso_env(
        SEARX_LIB,
        {
            "SEARXNG_SETTINGS_PATH": cfg,
            "SEARXNG_BIND_ADDRESS": "127.0.0.1",
            "SEARXNG_PORT": str(SEARX_PORT),
            "SEARXNG_SECRET": os.urandom(16).hex(),
        },
    )
    env["PYTHONPATH"] = root + os.pathsep + env["PYTHONPATH"]
    os.makedirs(JOB_DIR, exist_ok=True)
    sh("pkill -f searx.webapp || true", check=False, quiet=True)
    time.sleep(1)
    log = open(os.path.join(JOB_DIR, "searx.log"), "wb")
    subprocess.Popen(
        [sys.executable, "-m", "searx.webapp"],
        env=env,
        cwd=root,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    import requests as _rq

    for _ in range(45):
        try:
            if _rq.get(f"{SEARX_URL}/healthz", timeout=2).ok:
                print(f"searxng up on {SEARX_URL}")
                return True
        except _rq.RequestException:  # not listening yet
            pass
        time.sleep(1)
    try:
        tail = open(os.path.join(JOB_DIR, "searx.log"), errors="ignore").read()[-400:]
    except OSError:
        tail = ""
    boot_issue("searxng", f"service did not bind to {SEARX_URL}. log tail: {tail[-400:]}")
    return False


def ensure_crawl4ai() -> bool:
    """Crawl4AI isolated via --target: it pins aiofiles/cryptography
    versions that conflict with Kaggle's, so a plain install fails."""
    env = iso_env(CRAWL_LIB)
    probe = subprocess.run([sys.executable, "-c", "import crawl4ai"], capture_output=True, env=env)
    if probe.returncode != 0:
        print("installing crawl4ai (isolated)")
        if not iso_install(CRAWL_LIB, "crawl4ai"):
            boot_issue("crawl4ai", "pip install --target crawl4ai failed")
            return False
        env = iso_env(CRAWL_LIB)
        sh(
            f"{sys.executable} -m playwright install chromium || true",
            check=False,
            quiet=True,
        )
        probe = subprocess.run(
            [sys.executable, "-c", "import crawl4ai"], capture_output=True, env=env
        )
    if probe.returncode == 0:
        print("crawl4ai ready (isolated)")
        return True
    err = (probe.stderr or b"").decode(errors="ignore")[-400:]
    boot_issue("crawl4ai", f"import failed after install: {err}")
    return False


def ensure_browser() -> None:
    """Install Playwright and Chromium for the browser tools."""
    try:
        import playwright  # noqa
    except ImportError:
        sh(f"{sys.executable} -m pip install -q playwright", check=False)
    print("installing chromium (one-off)")
    if (
        sh(
            f"{sys.executable} -m playwright install --with-deps chromium",
            check=False,
            quiet=True,
        ).returncode
        != 0
    ):
        sh(f"{sys.executable} -m playwright install chromium", check=False, quiet=True)


def server_up(rq: ModuleType) -> bool:
    """True if the Ollama server answers."""
    try:
        return rq.get(f"{OLLAMA_URL}/api/tags", timeout=2).ok
    except rq.RequestException:
        return False
