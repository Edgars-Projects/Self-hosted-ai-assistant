"""Application wiring: model bootstrap, tools and Telegram handlers.

``bootstrap`` resolves which models are installed and records the result in
this module's namespace; ``main`` then builds the tool registry and starts
polling. The tools are closures over shared state (DB connection, locks,
scheduler) created inside ``main``.
"""

import ast
import asyncio
import glob
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback

# Imported into this module's namespace on purpose: bootstrap() rebinds
# the model names here once it knows what actually installed.
from .config import (
    AGENT_STEPS,
    ALLOW_FILE,
    AUTO_REFEREE,
    BROWSER_TIMEOUT,
    CRAWL_LIB,
    CRAWL_OK,
    CRITIQUE_MIN,
    DB_PATH,
    DL_DIR,
    EMBED_MODEL,
    ENABLE_BROWSER,
    ENABLE_CRAWL4AI,
    ENABLE_SEARX,
    ENABLE_TUNNEL,
    ENABLE_VOICE,
    EXAMPLES_IN_CTX,
    EXPERT_TIMEOUT,
    EXPERTS,
    FAST_CANDIDATES,
    FAST_MODEL,
    GOAL_MIN_SCORE,
    IDLE_ENABLED,
    IDLE_MAX_PER_DAY,
    IDLE_MINUTES,
    IDLE_QUIET_H,
    IMAGE_GPU,
    IMAGE_MODEL,
    IMAGE_TURBO,
    JOB_DIR,
    LEARNING,
    LLM_PARALLEL,
    LLM_TIMEOUT,
    MAX_RULE_LEN,
    MAX_RULES,
    MAX_STEPS,
    MAX_TEAM,
    MAX_TOOL_OUT,
    MAX_TURNS,
    MODEL,  # noqa: F401  (rebound by bootstrap)
    NON_CHAT,
    NUM_CTX,
    OLLAMA_URL,
    PROFILE_DIR,
    REFEREE,
    REFLECT_CRON,
    ROUTER,
    SEARX_LIB,
    SEARX_OK,
    SEARX_URL,
    SELF_CRITIQUE,
    SERVE_PORT,
    SMART_CANDIDATES,
    SMART_MODEL,
    STATE_DIR,
    TASK_HINTS,
    TG_MAX,
    TOOL_TIMEOUT,
    TOOLS_DIR,
    USE_UTILITY,
    UTILITY_CANDIDATES,
    UTILITY_MODEL,
    VISION_MODEL,
    WHISPER_SIZE,
    WORKSPACE,
)
from .db import Store, open_db
from .installers import (
    BOOT_ISSUES,
    CLEAN_ENV_GLOBAL,
    ensure_browser,
    ensure_crawl4ai,
    ensure_pip_deps,
    ensure_searx,
    find_ollama,
    install_ollama,
    iso_env,
    server_up,
    sh,
)
from .persona import PERSONA
from .secrets import get_secret
from .tools.clock import t_now
from .tools.crypto import t_crypto
from .tools.documents import t_make_pdf, t_make_xlsx
from .tools.mail import InboxTools, t_mail_send
from .tools.memory import MemoryTools
from .util import clip


def bootstrap():
    os.makedirs(STATE_DIR, exist_ok=True)
    ensure_pip_deps()
    import requests

    for d in ("/kaggle/temp", "/content", "/tmp"):
        if os.path.isdir(d) or os.path.isdir(os.path.dirname(d)):
            os.environ.setdefault("OLLAMA_MODELS", os.path.join(d, "ollama"))
            break
    os.environ.setdefault("OLLAMA_KEEP_ALIVE", "-1")
    os.environ.setdefault("OLLAMA_NUM_PARALLEL", str(LLM_PARALLEL))
    os.environ.setdefault("OLLAMA_MAX_QUEUE", "64")
    # the big model spans both cards, so only one can be resident
    os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
    os.environ.setdefault("OLLAMA_SCHED_SPREAD", "1")  # split across GPUs

    binary = find_ollama()
    if not server_up(requests):
        binary = binary or install_ollama()
        libdir = os.path.join(os.path.dirname(os.path.dirname(binary)), "lib", "ollama")
        if os.path.isdir(libdir):
            os.environ["LD_LIBRARY_PATH"] = libdir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        subprocess.Popen(
            [binary, "serve"],
            env=os.environ,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(90):
            if server_up(requests):
                break
            time.sleep(1)
        else:
            raise RuntimeError("ollama server never came up")
    else:
        binary = binary or "ollama"
        print("ollama already running")

    def _installed():
        """Ollama exits 0 on a failed pull, so ask what is REALLY there."""
        try:
            return {
                m["name"]
                for m in requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json().get("models", [])
            }
        except Exception:
            return set()

    def _pull_first(cands, label):
        for cand in cands:
            print(f"pulling {label}: {cand}")
            sh(f"{binary} pull {cand}", check=False)
            have = _installed()
            if cand in have or f"{cand}:latest" in have:
                print(f"  {label} confirmed: {cand}")
                return cand
            print(f"  {cand} did not install, trying next")
        return None

    got = _pull_first(FAST_CANDIDATES, "fast model")
    if not got:
        installed = sorted(_installed())
        raise RuntimeError(
            "No usable chat model could be installed.\n"
            f"Currently installed: {installed or 'nothing'}\n"
            "Edit FAST_CANDIDATES with a tag from https://ollama.com/search"
        )
    globals()["FAST_MODEL"] = got
    globals()["MODEL"] = got

    if USE_UTILITY:
        util = _pull_first(
            [c for c in UTILITY_CANDIDATES if c != FAST_MODEL] + [FAST_MODEL],
            "utility model",
        )
        globals()["UTILITY_MODEL"] = util or FAST_MODEL
        print(f"  utility model: {globals()['UTILITY_MODEL']}")

    if ROUTER:
        smart = _pull_first(SMART_CANDIDATES, "smart model")
        if smart:
            globals()["SMART_MODEL"] = smart
        else:
            print("no smart model available — running single-model")
            globals()["ROUTER"] = False

    print(f"pulling {EMBED_MODEL} (for semantic memory)")
    sh(f"{binary} pull {EMBED_MODEL}", check=False)

    if ENABLE_BROWSER:
        ensure_browser()
    if ENABLE_SEARX:
        globals()["SEARX_OK"] = ensure_searx()
    if ENABLE_CRAWL4AI:
        globals()["CRAWL_OK"] = ensure_crawl4ai()

    if shutil.which("nvidia-smi"):
        sh(
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
            check=False,
        )
    else:
        print("no GPU — CPU inference will be slow")


# ================================================================== main
def main():
    bootstrap()

    import numpy as np
    import requests
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    from bs4 import BeautifulSoup
    from telegram import Update
    from telegram.constants import ChatAction
    from telegram.ext import (
        ApplicationBuilder,
        ApplicationHandlerStop,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        TypeHandler,
        filters,
    )

    START = time.time()
    token = get_secret("TELEGRAM_TOKEN")  # the only secret required
    con = open_db()
    dblock = threading.Lock()
    scheduler = AsyncIOScheduler()

    # Tools inherit this env. Kaggle/Colab inject their own credentials,
    # so strip those too or the shell can just re-read your secrets.
    CLEAN_ENV = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "TELEGRAM_TOKEN",
            "KAGGLE_USER_SECRETS_TOKEN",
            "KAGGLE_KEY",
            "KAGGLE_USERNAME",
            "KAGGLE_DATA_PROXY_TOKEN",
            "COLAB_RELEASE_TAG",
        )
    }

    # Database access goes through Store; see db.py.
    store = Store(con, dblock)
    q = store.q
    setting = store.setting
    set_setting = store.set_setting

    # ------------------------------------------------------- embeddings
    def embed(text):
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text[:8000]},
            timeout=120,
        )
        r.raise_for_status()
        v = r.json()["embeddings"][0]
        a = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(a)
        return a / n if n else a

    def vec_add(chat_id, kind, source, text):
        try:
            v = embed(text)
        except Exception as e:
            print("embed failed:", e)
            return
        q(
            "INSERT INTO vectors (chat_id,kind,source,text,vec) VALUES (?,?,?,?,?)",
            (chat_id, kind, source, text, v.tobytes()),
        )

    def vec_search(chat_id, query, kind=None, k=6):
        rows = q(
            "SELECT text, source, vec, ts FROM vectors WHERE chat_id=?"
            + (" AND kind=?" if kind else ""),
            (chat_id, kind) if kind else (chat_id,),
            "all",
        )
        if not rows:
            return []
        qv = embed(query)
        scored = []
        for text, source, blob, ts in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            if v.shape != qv.shape:
                continue
            scored.append((float(np.dot(qv, v)), text, source, ts))
        scored.sort(reverse=True)
        return scored[:k]

    # ---------------------------------------------------------- browser
    class Browser:
        """Playwright objects are thread-bound, so everything runs on one
        long-lived thread and calls arrive over a queue."""

        def __init__(self):
            self.q = queue.Queue()
            self.ready = threading.Event()
            self.err = None
            self.started = False

        def start(self):
            if self.started:
                return
            self.started = True
            threading.Thread(target=self._loop, daemon=True).start()
            self.ready.wait(180)
            if self.err:
                raise RuntimeError(self.err)

        def _loop(self):
            try:
                from playwright.sync_api import sync_playwright

                os.makedirs(PROFILE_DIR, exist_ok=True)
                self.pw = sync_playwright().start()
                self.ctx = self.pw.chromium.launch_persistent_context(
                    PROFILE_DIR,
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
                    viewport={"width": 1280, "height": 900},
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"
                    ),
                )
                self.page = self.ctx.pages[0] if self.ctx.pages else self.ctx.new_page()
                self.page.set_default_timeout(30000)
            except Exception as e:
                self.err = f"browser failed to start: {type(e).__name__}: {e}"
                self.ready.set()
                return
            self.ready.set()
            while True:
                fn, kw, box = self.q.get()
                try:
                    box["r"] = fn(self.page, **kw)
                except Exception as e:
                    box["r"] = f"error: {type(e).__name__}: {e}"
                box["ev"].set()

        def call(self, fn, **kw):
            self.start()
            box = {"ev": threading.Event()}
            self.q.put((fn, kw, box))
            if not box["ev"].wait(BROWSER_TIMEOUT):
                return f"browser call timed out after {BROWSER_TIMEOUT}s"
            self._frame()
            return box["r"]

        def _frame(self):
            """Keep a current screenshot for the /watch live view."""
            try:
                box = {"ev": threading.Event()}
                path = os.path.join(DL_DIR, "_live.png")
                os.makedirs(DL_DIR, exist_ok=True)
                self.q.put((lambda pg: pg.screenshot(path=path), {}, box))
                if box["ev"].wait(15):
                    PUBLIC["shot"] = path
            except Exception:
                pass

    browser = Browser()

    def _text(page):
        try:
            return page.inner_text("body")
        except Exception:
            return page.content()

    def _b_open(page, url, wait=None):
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        page.goto(url, wait_until="domcontentloaded")
        if wait:
            try:
                page.wait_for_selector(wait, timeout=20000)
            except Exception:
                pass
        page.wait_for_timeout(1200)
        return clip(f"{page.title()}\n{page.url}\n\n{_text(page)}")

    def _b_read(page):
        return clip(f"{page.title()}\n{page.url}\n\n{_text(page)}")

    def _b_click(page, selector):
        try:
            page.click(selector)
        except Exception:
            page.get_by_text(selector, exact=False).first.click()
        page.wait_for_timeout(1500)
        return clip(f"clicked. now at {page.url}\n\n{_text(page)}")

    def _b_type(page, selector, text, enter=True):
        page.fill(selector, text)
        if enter:
            page.press(selector, "Enter")
        page.wait_for_timeout(2000)
        return clip(f"typed. now at {page.url}\n\n{_text(page)}")

    def _b_links(page):
        out = page.eval_on_selector_all(
            "a[href]",
            "els => els.slice(0,120).map(e => e.innerText.trim() + ' -> ' + e.href)",
        )
        return clip("\n".join(x for x in out if x.strip(" ->")))

    def _b_shot(page, full_page=False):
        os.makedirs(DL_DIR, exist_ok=True)
        path = os.path.join(DL_DIR, f"shot_{int(time.time())}.png")
        page.screenshot(path=path, full_page=bool(full_page))
        return f"saved {path} — use send_file to show the user"

    def _b_vision(page, question="Describe this page in detail."):
        """Screenshot the page and read it with the VL model. Use when the
        text extraction is empty or useless — canvas apps, image-heavy
        dashboards, anything rendered to pixels."""
        os.makedirs(DL_DIR, exist_ok=True)
        shot = os.path.join(DL_DIR, f"page_{int(time.time())}.png")
        page.screenshot(path=shot, full_page=False)
        return shot  # described outside the browser thread

    def _b_js(page, code):
        return clip(str(page.evaluate(code)))

    def _b_back(page):
        page.go_back()
        page.wait_for_timeout(1000)
        return clip(f"back at {page.url}\n\n{_text(page)}")

    # ------------------------------------------------------------ tools
    def t_search(query, max_results=8, engines=None, category=None):
        """SearxNG: unlimited, free, no key. Falls back to DuckDuckGo's
        HTML endpoint if the service is not running."""
        if SEARX_OK:
            params = {"q": query, "format": "json", "language": "en", "safesearch": 0}
            if engines:
                params["engines"] = engines
            if category:
                params["categories"] = category
            try:
                r = requests.get(f"{SEARX_URL}/search", params=params, timeout=45)
                r.raise_for_status()
                res = r.json().get("results", [])[: int(max_results)]
                if res:
                    return clip(
                        "\n\n".join(
                            f"{h.get('title', '')}\n{h.get('url', '')}\n{h.get('content', '')}"
                            for h in res
                        )
                    )
            except Exception as e:
                print("searx failed:", e)
        # fallback, no key required
        try:
            r = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=30,
            )
            soup = BeautifulSoup(r.text, "html.parser")
            out = []
            for a in soup.select("a.result__a")[: int(max_results)]:
                out.append(f"{a.get_text(' ', strip=True)}\n{a.get('href')}")
            return clip("\n\n".join(out) or "no results")
        except Exception as e:
            return f"search failed: {e}"

    def t_fetch(url, mode="markdown", contains=None, start_at=0):
        """Crawl4AI when available: runs JS, strips boilerplate, returns
        clean markdown. Falls back to plain extraction."""
        if CRAWL_OK:
            try:
                code = (
                    "import asyncio, json\n"
                    "from crawl4ai import AsyncWebCrawler\n"
                    "async def go():\n"
                    "    async with AsyncWebCrawler(verbose=False) as c:\n"
                    f"        r = await c.arun(url={url!r},\n"
                    "                          bypass_cache=True)\n"
                    "        print(json.dumps({'ok': r.success,\n"
                    "            'md': (r.markdown or '')[:20000],\n"
                    "            'links': [l.get('href') for l in\n"
                    "                      (r.links or {}).get('internal', [])[:25]]}))\n"
                    "asyncio.run(go())"
                )
                p = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    env=iso_env(CRAWL_LIB),
                )
                line = [l for l in (p.stdout or "").splitlines() if l.startswith("{")]
                if line:
                    d = json.loads(line[-1])
                    if d.get("ok") and d.get("md"):
                        body = d["md"]
                        if contains:
                            i = body.lower().find(str(contains).lower())
                            if i >= 0:
                                body = "…\n" + body[max(0, i - 500) : i + MAX_TOOL_OUT]
                            else:
                                body = f"[{contains!r} not found on the page]\n\n" + body
                        elif start_at:
                            body = body[int(start_at) :]
                        if mode == "links":
                            body += "\n\nLINKS:\n" + "\n".join(x for x in d.get("links", []) if x)
                        return clip(body)
            except Exception as e:
                print("crawl4ai failed, falling back:", e)
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return clip(re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True)))

    def t_python(code, timeout=None):
        p = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=int(timeout or TOOL_TIMEOUT),
            env=CLEAN_ENV,
            stdin=subprocess.DEVNULL,
        )
        out = (p.stdout or "") + (("\nSTDERR:\n" + p.stderr) if p.stderr else "")
        return clip(out.strip() or f"(no output, exit {p.returncode})")

    def t_shell(command, timeout=None):
        p = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=int(timeout or TOOL_TIMEOUT),
            env=CLEAN_ENV,
            cwd="/tmp",
            stdin=subprocess.DEVNULL,
        )
        auto_record(command, ok=(p.returncode == 0))
        out = (p.stdout or "") + (("\nSTDERR:\n" + p.stderr) if p.stderr else "")
        return clip(out.strip() or f"(no output, exit {p.returncode})")

    JOBS = {}

    def t_bg_start(command, name=None):
        os.makedirs(JOB_DIR, exist_ok=True)
        jid = name or f"job{len(JOBS) + 1}"
        log = os.path.join(JOB_DIR, f"{jid}.log")
        fh = open(log, "wb")
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=CLEAN_ENV,
            cwd="/tmp",
            start_new_session=True,
        )
        auto_record(command)
        JOBS[jid] = {"proc": proc, "log": log, "cmd": command, "started": time.time()}
        return f"started {jid} (pid {proc.pid}), no time limit"

    def t_bg_status(job_id, lines=40):
        j = JOBS.get(job_id)
        if not j:
            return f"no job {job_id!r}. known: {list(JOBS) or 'none'}"
        rc = j["proc"].poll()
        state = "running" if rc is None else f"finished (exit {rc})"
        tail = subprocess.run(
            ["tail", "-n", str(int(lines)), j["log"]], capture_output=True, text=True
        ).stdout
        return clip(
            f"{job_id}: {state} after "
            f"{int(time.time() - j['started'])}s\ncmd: {j['cmd']}\n"
            f"--- tail ---\n{tail}"
        )

    def t_bg_list():
        if not JOBS:
            return "no background jobs"
        return "\n".join(
            f"{jid}: {'running' if j['proc'].poll() is None else 'done'} "
            f"({int(time.time() - j['started'])}s) — {j['cmd'][:60]}"
            for jid, j in JOBS.items()
        )

    def t_bg_kill(job_id):
        j = JOBS.get(job_id)
        if not j:
            return f"no job {job_id!r}"
        j["proc"].terminate()
        return f"killed {job_id}"

    def t_download(url, filename=None):
        os.makedirs(DL_DIR, exist_ok=True)
        name = filename or url.split("?")[0].rstrip("/").split("/")[-1] or "download.bin"
        path = os.path.join(DL_DIR, re.sub(r"[^\w.\-]", "_", name))
        with requests.get(
            url, stream=True, timeout=300, headers={"User-Agent": "Mozilla/5.0"}
        ) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        return (
            f"downloaded to {path} "
            f"({os.path.getsize(path) / 1e6:.1f} MB). Use send_file to "
            f"deliver it."
        )

    def t_send_file(outbox, path, caption=None):
        if not os.path.isfile(path):
            bare = re.sub(r"^/?workspace/+", "", str(path))
            cands = []
            for c in (
                path,
                os.path.join(DL_DIR, os.path.basename(path)),
                os.path.join(WORKSPACE, bare),
                os.path.join(WORKSPACE, os.path.basename(path)),
            ):
                cands += glob.glob(c)
            if not cands:
                return (
                    f"no such file: {path}. Use list_files or "
                    f"workspace_list to see what actually exists."
                )
            path = cands[0]
        size = os.path.getsize(path)
        if size > TG_MAX:
            return (
                f"{path} is {size / 1e6:.0f} MB, over Telegram's 50 MB "
                f"cap. Split it with `split -b 45M` and send the parts."
            )
        if any(os.path.abspath(p) == os.path.abspath(path) for p, _ in outbox):
            return (
                f"{os.path.basename(path)} is already queued for this reply — do not send it again."
            )
        outbox.append((path, caption))
        return f"queued {os.path.basename(path)} ({size / 1e6:.1f} MB)"

    def t_list_files(directory=None):
        d = directory or DL_DIR
        if not os.path.isdir(d):
            return f"{d} does not exist"
        rows = []
        for n in sorted(os.listdir(d))[:100]:
            fp = os.path.join(d, n)
            rows.append(
                f"{n}  {os.path.getsize(fp) / 1e6:.1f} MB" if os.path.isfile(fp) else f"{n}/"
            )
        return clip("\n".join(rows) or "(empty)")

    # ------ memory
    memory = MemoryTools(store, vec_add)
    t_remember = memory.remember
    t_recall = memory.recall
    t_forget = memory.forget

    def t_recall_semantic(chat_id, query, k=6):
        hits = vec_search(chat_id, query, k=int(k))
        if not hits:
            return "nothing relevant in memory"
        return clip("\n\n".join(f"[{s:.2f}] ({src}, {ts}) {txt}" for s, txt, src, ts in hits))

    def t_search_docs(chat_id, query, k=6):
        hits = vec_search(chat_id, query, kind="doc", k=int(k))
        if not hits:
            return "no matching document passages"
        return clip("\n\n".join(f"[{s:.2f}] from {src}:\n{txt}" for s, txt, src, ts in hits))

    def chunk(text, size=1200, overlap=150):
        text = re.sub(r"\s+", " ", text).strip()
        out, i = [], 0
        while i < len(text):
            out.append(text[i : i + size])
            i += size - overlap
        return out

    def t_index_doc(chat_id, path):
        if not os.path.isfile(path):
            return f"no such file: {path}"
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            from pypdf import PdfReader

            text = "\n".join((pg.extract_text() or "") for pg in PdfReader(path).pages)
        else:
            text = open(path, "r", errors="ignore").read()
        parts = chunk(text)
        for c in parts:
            vec_add(chat_id, "doc", os.path.basename(path), c)
        return f"indexed {os.path.basename(path)} as {len(parts)} passages"

    # ================================================================
    # SYSTEMS TOOLKIT
    # Grouped by action rather than one tool per verb — 58 tools was
    # already straining selection accuracy on a 14B.
    # ================================================================

    # ---- 01 persistent terminal session ----------------------------
    SHELL = {"proc": None, "cwd": "/tmp"}
    SH_MARK = "___END_OF_CMD___"

    def _shell_start():
        SHELL["proc"] = subprocess.Popen(
            # NOT -i: interactive bash echoes the prompt and the command,
            # which desynchronises the reader by one call.
            ["/bin/bash", "--norc"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=CLEAN_ENV,
            cwd=SHELL["cwd"],
            start_new_session=True,
        )

    def t_terminal(command=None, action="run", timeout=120):
        """A shell that REMEMBERS: cd, exports and venvs persist between
        calls, unlike run_shell which starts fresh each time."""
        if action == "restart" or (SHELL["proc"] and SHELL["proc"].poll() is not None):
            if SHELL["proc"]:
                try:
                    SHELL["proc"].kill()
                except Exception:
                    pass
            SHELL["proc"] = None
            if action == "restart":
                _shell_start()
                return "terminal restarted"
        if SHELL["proc"] is None:
            _shell_start()
        if action == "status":
            return f"terminal alive (pid {SHELL['proc'].pid}), cwd tracked by the shell itself"
        if not command:
            return "give a command to run"
        p = SHELL["proc"]
        out, done = [], threading.Event()

        def reader():
            for line in p.stdout:
                if SH_MARK in line:
                    done.set()
                    return
                out.append(line)

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        auto_record(command)
        p.stdin.write(command + f"\necho {SH_MARK}\n")
        p.stdin.flush()
        done.wait(int(timeout))
        if not done.is_set():
            return clip(
                "".join(out) + f"\n[still running after {timeout}s "
                f"— the session stays alive, check again]"
            )
        return clip("".join(out).strip() or "(no output)")

    # ---- 07 persistent python session ------------------------------
    PY = {"ns": None}

    def t_python_session(code, action="run"):
        """Variables, imports and loaded data persist between calls —
        like a notebook kernel, unlike run_python."""
        import contextlib
        import io
        import traceback as tb

        if PY["ns"] is None or action == "reset":
            PY["ns"] = {"__name__": "__session__"}
            if action == "reset":
                return "python session reset"
        if action == "vars":
            return clip(
                ", ".join(
                    f"{k}={type(v).__name__}" for k, v in PY["ns"].items() if not k.startswith("__")
                )
                or "(empty)"
            )
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    val = eval(compile(code, "<session>", "eval"), PY["ns"])
                    if val is not None:
                        print(repr(val), file=buf)
                except SyntaxError:
                    exec(compile(code, "<session>", "exec"), PY["ns"])
        except Exception:
            return clip(buf.getvalue() + "\n" + tb.format_exc(limit=3))
        return clip(buf.getvalue().strip() or "(ok, no output)")

    # ---- 02/03/04/17/18/19/21/23/24 system + network ---------------
    def _run(cmd, t=60):
        p = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=t, env=CLEAN_ENV
        )
        return (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")

    def t_system(action="info", target=None):
        a = (action or "info").lower()
        if a == "info":
            return clip(
                _run(
                    "uname -a; echo; lsb_release -d 2>/dev/null; "
                    "echo; uptime; echo; nproc; free -h; "
                    "df -h / /tmp /kaggle 2>/dev/null"
                )
            )
        if a == "resources":
            return clip(_run("top -bn1 | head -20; echo; free -m; echo; df -h | head -10"))
        if a == "hardware":
            return clip(
                _run(
                    "lscpu | head -25; echo; lsblk 2>/dev/null "
                    "| head -15; echo; cat /proc/meminfo | head -6"
                )
            )
        if a == "gpu":
            return clip(
                _run(
                    "nvidia-smi --query-gpu=index,name,memory.used,memory.total,"
                    "utilization.gpu,temperature.gpu --format=csv; echo; "
                    "nvidia-smi --query-compute-apps=pid,used_memory "
                    "--format=csv"
                )
            )
        if a == "network":
            return clip(
                _run(
                    "ip -br addr 2>/dev/null || ifconfig; echo; "
                    "ip route 2>/dev/null; echo; cat /etc/resolv.conf"
                )
            )
        if a == "dns":
            host = target or "example.com"
            return clip(
                _run(
                    f"getent hosts {host}; echo; "
                    f"(dig +short {host} 2>/dev/null || "
                    f'python3 -c "import socket,sys;'
                    f"print(socket.gethostbyname_ex('{host}'))\")"
                )
            )
        if a == "ports":
            return clip(
                _run("ss -tulpn 2>/dev/null || netstat -tulpn 2>/dev/null || echo 'no ss/netstat'")
            )
        if a == "connectivity":
            host = target or "1.1.1.1"
            return clip(
                _run(
                    f"ping -c2 -W2 {host}; echo; "
                    f"curl -s -o /dev/null -w 'http %{{http_code}} "
                    f"in %{{time_total}}s\\n' https://{host} "
                    f"2>/dev/null || true",
                    30,
                )
            )
        if a == "logs":
            logs = sorted(glob.glob(os.path.join(JOB_DIR, "*.log")))
            if not logs:
                return "no job logs"
            if target:
                match = [l for l in logs if target in os.path.basename(l)]
                if match:
                    return clip(_run(f'tail -n 80 "{match[0]}"'))
            return "available logs:\n" + "\n".join(os.path.basename(l) for l in logs)
        return (
            f"unknown action {action!r}. try: info, resources, hardware, "
            f"gpu, network, dns, ports, connectivity, logs"
        )

    def t_process(action="list", pid=None, pattern=None, signal_name="TERM"):
        a = (action or "list").lower()
        if a == "list":
            if pattern:
                return clip(_run(f"ps aux | grep -i {pattern!r} | grep -v grep"))
            return clip(_run("ps aux --sort=-%mem | head -25"))
        if a == "tree":
            return clip(_run("ps -ejH | head -40"))
        if a == "info":
            if not pid:
                return "give a pid"
            return clip(
                _run(
                    f"ps -p {int(pid)} -o pid,ppid,user,%cpu,%mem,"
                    f"etime,cmd; echo; ls -l /proc/{int(pid)}/fd "
                    f"2>/dev/null | head -10"
                )
            )
        if a == "kill":
            if not pid:
                return "give a pid"
            return clip(
                _run(f"kill -{signal_name} {int(pid)} && echo 'sent {signal_name} to {pid}'")
            )
        if a == "monitor":
            samples = []
            for _ in range(3):
                samples.append(_run("head -1 /proc/loadavg; free -m | sed -n 2p").strip())
                time.sleep(2)
            return clip("\n---\n".join(samples))
        return "actions: list, tree, info, kill, monitor"

    # ---- 05/06 filesystem ------------------------------------------
    def t_fs(action, path=".", pattern=None, dest=None, depth=3):
        a = action.lower()
        if a == "tree":
            return clip(
                _run(f'find "{path}" -maxdepth {int(depth)} -not -path "*/.git/*" | head -200')
            )
        if a == "find":
            return clip(
                _run(
                    f'find "{path}" -name {pattern!r} -not -path "*/.git/*" 2>/dev/null | head -100'
                )
            )
        if a == "grep":
            return clip(
                _run(
                    f"grep -rn --binary-files=without-match "
                    f'{pattern!r} "{path}" 2>/dev/null | head -80'
                )
            )
        if a == "stat":
            return clip(
                _run(
                    f'ls -la "{path}"; echo; du -sh "{path}" 2>/dev/null; file "{path}" 2>/dev/null'
                )
            )
        if a in ("copy", "move"):
            if not dest:
                return "give a dest"
            cmd = "cp -r" if a == "copy" else "mv"
            return clip(_run(f'{cmd} "{path}" "{dest}" && echo done'))
        if a == "delete":
            if os.path.abspath(path) in ("/", "/kaggle", "/usr", "/etc"):
                return "refusing to delete a system root directory"
            return clip(_run(f'rm -rf "{path}" && echo deleted'))
        if a == "watch":
            before = {}
            for root, _, files in os.walk(path):
                for f in files[:500]:
                    fp = os.path.join(root, f)
                    try:
                        before[fp] = os.path.getmtime(fp)
                    except OSError:
                        pass
            time.sleep(8)
            changed = []
            for fp, mt in list(before.items()):
                try:
                    if os.path.getmtime(fp) != mt:
                        changed.append(f"modified {fp}")
                except OSError:
                    changed.append(f"deleted {fp}")
            for root, _, files in os.walk(path):
                for f in files[:500]:
                    fp = os.path.join(root, f)
                    if fp not in before:
                        changed.append(f"created {fp}")
            return clip("\n".join(changed) or "no changes in 8s")
        return "actions: tree, find, grep, stat, copy, move, delete, watch"

    # ---- 08 packages -----------------------------------------------
    def t_packages(action, names="", manager="auto"):
        if action.lower() == "install" and names:
            mgr = "pip" if manager in ("auto", "pip") else manager
            auto_record(f"{mgr} install {names}")
        a = action.lower()
        m = manager.lower()
        if m == "auto":
            m = "pip"
        if a == "list":
            cmds = {
                "pip": "pip list 2>/dev/null | head -60",
                "apt": "dpkg -l | tail -n +6 | awk '{print $2, $3}' | head -60",
                "npm": "npm ls -g --depth=0 2>/dev/null",
            }
            return clip(_run(cmds.get(m, cmds["pip"]), 120))
        if a == "install":
            if not names:
                return "give package names"
            cmds = {
                "pip": f"{sys.executable} -m pip install -q {names}",
                "apt": f"apt-get install -y -qq {names}",
                "npm": f"npm install -g {names}",
            }
            return clip(_run(cmds.get(m, cmds["pip"]), 900))
        if a == "info":
            return clip(
                _run(
                    f"pip show {names}" if m == "pip" else f"apt-cache show {names} | head -20",
                    120,
                )
            )
        return "actions: list, install, info | managers: pip, apt, npm"

    # ---- 09 compile / build ----------------------------------------
    def t_build(action, source=None, output=None, args=""):
        a = action.lower()
        if a == "toolchains":
            return clip(
                _run(
                    "for t in gcc g++ rustc go javac node tsc make "
                    "cmake; do printf '%-8s ' $t; "
                    "command -v $t >/dev/null && $t --version "
                    "2>&1 | head -1 || echo 'not installed'; done"
                )
            )
        if not source:
            return "give a source file"
        out = output or os.path.splitext(source)[0]
        cmds = {
            "c": f'gcc "{source}" -o "{out}" {args}',
            "cpp": f'g++ -std=c++17 "{source}" -o "{out}" {args}',
            "rust": f'rustc "{source}" -o "{out}" {args}',
            "go": f'go build -o "{out}" "{source}"',
            "make": f'make -C "{os.path.dirname(source) or "."}" {args}',
        }
        if a not in cmds:
            return f"actions: toolchains, {', '.join(cmds)}"
        r = _run(cmds[a], 600)
        if os.path.exists(out):
            return clip(f"built {out}\n{r}")
        return clip(f"build failed:\n{r}")

    # ---- 10/25/26 code analysis ------------------------------------
    def t_code(action, path=None, code=None):
        a = action.lower()
        src = code
        if src is None and path:
            try:
                src = open(path, errors="ignore").read()
            except Exception as e:
                return f"cannot read {path}: {e}"
        if a == "lint":
            target = path or "-"
            return clip(
                _run(
                    f"({sys.executable} -m ruff check {target} "
                    f"2>/dev/null || {sys.executable} -m pyflakes "
                    f"{target} 2>/dev/null || "
                    f"echo 'install ruff for linting')",
                    120,
                )
            )
        if a == "check":
            if not src:
                return "give code or a path"
            try:
                compile(src, path or "<code>", "exec")
                return "syntax OK"
            except SyntaxError as e:
                return f"SyntaxError line {e.lineno}: {e.msg}\n  {e.text or ''}"
        if a == "symbols":
            if not src:
                return "give code or a path"
            try:
                tree = ast.parse(src)
            except SyntaxError as e:
                return f"cannot parse: {e}"
            out = []
            for n in ast.walk(tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    a_ = [x.arg for x in n.args.args]
                    out.append(f"def {n.name}({', '.join(a_)})  L{n.lineno}")
                elif isinstance(n, ast.ClassDef):
                    out.append(f"class {n.name}  L{n.lineno}")
                elif isinstance(n, (ast.Import, ast.ImportFrom)):
                    mod = getattr(n, "module", "") or ""
                    out.append(f"import {mod or ','.join(x.name for x in n.names)}")
            return clip("\n".join(out) or "(no symbols)")
        if a == "complexity":
            if not src:
                return "give code or a path"
            tree = ast.parse(src)
            rows = []
            for n in ast.walk(tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    branches = sum(
                        1
                        for x in ast.walk(n)
                        if isinstance(
                            x,
                            (
                                ast.If,
                                ast.For,
                                ast.While,
                                ast.Try,
                                ast.BoolOp,
                                ast.ExceptHandler,
                            ),
                        )
                    )
                    length = (n.end_lineno or n.lineno) - n.lineno
                    rows.append((branches + 1, length, n.name, n.lineno))
            rows.sort(reverse=True)
            return clip(
                "\n".join(
                    f"{c:3d} complexity, {l:4d} lines — {nm} (L{ln})" for c, l, nm, ln in rows[:25]
                )
                or "(no functions)"
            )
        return "actions: lint, check, symbols, complexity"

    # ---- 27 git ----------------------------------------------------
    def t_git(action, repo=".", args="", message=None, url=None):
        a = action.lower()
        r = repo
        if a == "clone":
            if not url:
                return "give a url"
            dest = (
                repo
                if repo != "."
                else os.path.join(WORKSPACE, os.path.basename(url).replace(".git", ""))
            )
            return clip(_run(f'git clone --depth 20 "{url}" "{dest}"', 900))
        cmds = {
            "status": "git status -sb",
            "log": "git log --oneline -20",
            "diff": "git diff --stat; echo; git diff | head -200",
            "branches": "git branch -a",
            "remotes": "git remote -v",
            "blame": f"git blame {args} | head -60",
            "show": f"git show {args or 'HEAD'} | head -200",
        }
        if a == "commit":
            return clip(
                _run(
                    f'cd "{r}" && git add -A && git commit -q -m '
                    f'"{message or "update"}" && git log --oneline -3'
                )
            )
        if a in cmds:
            return clip(_run(f'cd "{r}" && {cmds[a]}', 120))
        return f"actions: clone, commit, {', '.join(cmds)}"

    # ---- 16 http client --------------------------------------------
    def t_http(url, method="GET", headers=None, body=None, json_body=None):
        try:
            h = json.loads(headers) if isinstance(headers, str) else (headers or {})
        except Exception:
            h = {}
        h.setdefault("User-Agent", "Mozilla/5.0")
        kw = {"headers": h, "timeout": 60, "allow_redirects": True}
        if json_body is not None:
            kw["json"] = json.loads(json_body) if isinstance(json_body, str) else json_body
        elif body is not None:
            kw["data"] = body
        r = requests.request(method.upper(), url, **kw)
        head = "\n".join(f"{k}: {v}" for k, v in list(r.headers.items())[:12])
        return clip(
            f"{r.status_code} {r.reason}  ({len(r.content)} bytes)\n{head}\n\n{r.text[:3000]}"
        )

    # ---- 14/15 ssh / remote ----------------------------------------
    def t_ssh(host, command, user=None, key_path=None, port=22):
        if not shutil.which("ssh"):
            _run("apt-get install -y -qq openssh-client", 300)
        target = f"{user}@{host}" if user else host
        key = f'-i "{key_path}"' if key_path else ""
        cmd = (
            f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "
            f"-p {int(port)} {key} {target} {command!r}"
        )
        return clip(_run(cmd, 300))

    # ---- 13 sandboxed execution ------------------------------------
    def t_sandbox(command, seconds=30, memory_mb=512):
        """Run something untrusted with hard resource limits."""
        pre = f"ulimit -v {int(memory_mb) * 1024}; ulimit -t {int(seconds)}; ulimit -f 102400; "
        if shutil.which("bwrap"):
            cmd = (
                f"bwrap --ro-bind / / --dev /dev --proc /proc "
                f"--tmpfs /tmp --unshare-pid --die-with-parent "
                f"/bin/bash -c {pre + command!r}"
            )
        else:
            cmd = f"/bin/bash -c {pre + command!r}"
        p = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=int(seconds) + 10,
            env=CLEAN_ENV,
            cwd="/tmp",
        )
        return clip(
            ((p.stdout or "") + (p.stderr or "")).strip() or f"(no output, exit {p.returncode})"
        )

    # ---- 12 containers (honest about the limits) -------------------
    def t_container(action="check", image=None, command=""):
        if not shutil.which("podman"):
            if action == "install":
                return clip(_run("apt-get update -qq && apt-get install -y -qq podman", 900))
            return (
                "podman is not installed. Run this tool with "
                "action='install' first. Note: Docker cannot work in "
                "this container at all — no privileges for the daemon."
            )
        if action == "check":
            return clip(
                _run(
                    "podman info --format '{{.Host.OCIRuntime.Name}} "
                    "{{.Store.GraphDriverName}}' 2>&1 | head -5",
                    120,
                )
            )
        if action == "run":
            if not image:
                return "give an image"
            return clip(_run(f"podman run --rm --storage-driver=vfs {image} {command}", 600))
        if action == "list":
            return clip(_run("podman ps -a; echo; podman images", 120))
        return "actions: check, install, run, list"

    # ================================================================
    # ENVIRONMENT SNAPSHOTS
    # Kaggle wipes the container on every restart. Record what was
    # installed, then replay it in one command next session.
    # ================================================================
    # Every successful install is logged automatically. Nobody has to
    # remember to declare anything.
    INSTALL_RE = [
        (re.compile(r"pip\s+(?:3\s+)?install\s+(?:-[^\s]+\s+)*([^\-][^&|;]*)"), "pip"),
        (re.compile(r"apt(?:-get)?\s+install\s+(?:-[^\s]+\s+)*([^\-][^&|;]*)"), "apt"),
        (re.compile(r"npm\s+i(?:nstall)?\s+(?:-[^\s]+\s+)*([^\-][^&|;]*)"), "npm"),
    ]

    def auto_record(command, ok=True):
        """Called after any shell/package tool. Extracts package names from
        install commands and appends them to the 'auto' environment."""
        if not ok or not command:
            return
        found = []
        for rx, mgr in INSTALL_RE:
            for m in rx.finditer(str(command)):
                for pkg in re.split(r"[\s,]+", m.group(1).strip()):
                    pkg = pkg.strip("\"'`")
                    if (
                        pkg
                        and not pkg.startswith("-")
                        and pkg not in ("install", "-y", "-q", "&&")
                        and not pkg.startswith("/")
                    ):
                        found.append((mgr, pkg))
        if not found:
            return
        row = q("SELECT payload FROM jobs WHERE id='env_auto'", (), "one")
        snap = (
            json.loads(row[0])
            if row
            else {
                "created": time.strftime("%Y-%m-%d %H:%M"),
                "declared": [],
                "apt_pkgs": [],
                "npm_pkgs": [],
                "commands": [],
                "pip": [],
                "apt": [],
            }
        )
        changed = False
        for mgr, pkg in found:
            bucket = {"pip": "declared", "apt": "apt_pkgs", "npm": "npm_pkgs"}[mgr]
            snap.setdefault(bucket, [])
            if pkg not in snap[bucket]:
                snap[bucket].append(pkg)
                changed = True
                print(f"[env] recorded {mgr} package: {pkg}")
        if changed:
            q(
                "INSERT OR REPLACE INTO jobs (id,chat_id,kind,spec,payload,"
                "state) VALUES (?,?,?,?,?,?)",
                ("env_auto", OWNER[0] or 0, "env", "auto", json.dumps(snap), "saved"),
            )

    def t_env(action, name="default", packages=None, commands=None):
        a = action.lower()
        key = f"env_{name}"

        if a == "snapshot":
            """Record what is installed NOW, so it can be restored later."""
            pip = _run(f"{sys.executable} -m pip freeze", 180)
            apt = _run(
                "comm -13 <(sort /var/lib/apt/extra_pkgs 2>/dev/null "
                "|| echo) <(dpkg --get-selections | awk '{print $1}' "
                "| sort) 2>/dev/null | head -200",
                120,
            )
            snap = {
                "created": time.strftime("%Y-%m-%d %H:%M"),
                "pip": [l for l in pip.splitlines() if "==" in l][:300],
                "apt_note": "apt list recorded for reference only",
                "apt": [l for l in apt.splitlines() if l.strip()][:200],
                "commands": [],
            }
            q(
                "INSERT OR REPLACE INTO jobs (id,chat_id,kind,spec,payload,"
                "state) VALUES (?,?,?,?,?,?)",
                (key, OWNER[0] or 0, "env", name, json.dumps(snap), "saved"),
            )
            return (
                f"snapshot {name!r}: {len(snap['pip'])} pip packages, "
                f"{len(snap['apt'])} apt packages recorded"
            )

        if a == "define":
            """Explicitly declare what this environment needs — usually
            better than a full snapshot, which records noise too."""
            pk = packages
            if isinstance(pk, str):
                pk = [x.strip() for x in re.split(r"[,\s]+", pk) if x.strip()]
            cm = commands
            if isinstance(cm, str):
                try:
                    cm = json.loads(cm)
                except Exception:
                    cm = [cm]
            snap = {
                "created": time.strftime("%Y-%m-%d %H:%M"),
                "declared": pk or [],
                "commands": cm or [],
                "pip": [],
                "apt": [],
            }
            q(
                "INSERT OR REPLACE INTO jobs (id,chat_id,kind,spec,payload,"
                "state) VALUES (?,?,?,?,?,?)",
                (key, OWNER[0] or 0, "env", name, json.dumps(snap), "saved"),
            )
            return (
                f"environment {name!r} defined: "
                f"{len(pk or [])} packages, {len(cm or [])} commands. "
                f"Restore it with env action='restore'."
            )

        if a == "restore":
            row = q("SELECT payload FROM jobs WHERE id=?", (key,), "one")
            if not row:
                return f"no environment {name!r}. Known: " + str(t_env("list"))
            snap = json.loads(row[0])
            steps = []
            declared = snap.get("declared") or []
            if declared:
                steps.append(f"{sys.executable} -m pip install -q " + " ".join(declared))
            if snap.get("apt_pkgs"):
                steps.append("apt-get install -y -qq " + " ".join(snap["apt_pkgs"]))
            if snap.get("npm_pkgs"):
                steps.append("npm install -g " + " ".join(snap["npm_pkgs"]))
            if snap.get("pip"):
                req = os.path.join(STATE_DIR, f"{name}-requirements.txt")
                open(req, "w").write("\n".join(snap["pip"]))
                steps.append(f"{sys.executable} -m pip install -q -r {req}")
            for c in snap.get("commands") or []:
                steps.append(c)
            if not steps:
                return f"environment {name!r} is empty"
            script = " && ".join(f"({st})" for st in steps)
            jid = t_bg_start(script, name=f"restore_{name}")
            return (
                f"restoring {name!r} in the background "
                f"({len(steps)} steps).\n{jid}\n"
                f"Check progress with job_status."
            )

        if a == "list":
            rows = q("SELECT spec, payload FROM jobs WHERE kind='env'", (), "all") or []
            if not rows:
                return "no environments saved"
            out = []
            for nm, pl in rows:
                d = json.loads(pl)
                n_auto = len(d.get("apt_pkgs") or []) + len(d.get("npm_pkgs") or [])
                out.append(
                    f"{nm}: {len(d.get('declared') or [])} pip, "
                    f"{n_auto} apt/npm, "
                    f"{len(d.get('commands') or [])} commands "
                    f"(saved {d.get('created', '?')})"
                )
            return "\n".join(out)

        if a == "show":
            row = q("SELECT payload FROM jobs WHERE id=?", (key,), "one")
            if not row:
                return f"no environment {name!r}"
            d = json.loads(row[0])
            return clip(
                f"{name} (saved {d.get('created')})\n"
                f"pip: {', '.join(d.get('declared') or []) or '-'}\n"
                f"apt: {', '.join(d.get('apt_pkgs') or []) or '-'}\n"
                f"npm: {', '.join(d.get('npm_pkgs') or []) or '-'}\n"
                f"commands:\n"
                + ("\n".join(f"  {c}" for c in d.get("commands") or []) or "  -")
                + f"\nfrozen pip: {len(d.get('pip') or [])} entries"
            )

        if a == "delete":
            q("DELETE FROM jobs WHERE id=?", (key,))
            return f"deleted environment {name!r}"

        return "actions: define, snapshot, restore, list, show, delete"

    # ================================================================
    # WORKFLOWS — the n8n-shaped layer.
    # The agent figures a sequence out once; you freeze it; it then runs
    # deterministically with no model in the loop.
    # ================================================================
    WF_DEPTH = {"n": 0}

    def t_workflow(action, name=None, steps=None, inputs=None):
        a = action.lower()
        if a == "run" and WF_DEPTH["n"] >= 3:
            return (
                "workflow nesting limit reached (3) — a workflow is "
                "calling itself or a chain of workflows. Stopped."
            )
        if a == "save":
            if not (name and steps):
                return "give a name and steps"
            if isinstance(steps, str):
                try:
                    steps = json.loads(steps)
                except Exception as e:
                    return f"steps must be JSON array: {e}"
            for i, st in enumerate(steps, 1):
                if "tool" not in st:
                    return f"step {i} has no 'tool' key"
                if st["tool"] not in {t[0] for t in TOOLS}:
                    return f"step {i}: unknown tool {st['tool']!r}"
            q(
                "INSERT OR REPLACE INTO jobs (id,chat_id,kind,spec,payload,"
                "state) VALUES (?,?,?,?,?,?)",
                (
                    f"wf_{name}",
                    OWNER[0] or 0,
                    "workflow",
                    name,
                    json.dumps(steps),
                    "saved",
                ),
            )
            return (
                f"saved workflow {name!r} with {len(steps)} steps. "
                f"Run it with workflow action='run'."
            )
        if a == "list":
            rows = q("SELECT spec, payload FROM jobs WHERE kind='workflow'", (), "all") or []
            if not rows:
                return "no workflows saved"
            return clip(
                "\n".join(
                    f"{nm}: " + " → ".join(st.get("tool", "?") for st in json.loads(pl))
                    for nm, pl in rows
                )
            )
        if a == "show":
            r = q("SELECT payload FROM jobs WHERE id=?", (f"wf_{name}",), "one")
            return clip(json.dumps(json.loads(r[0]), indent=1)) if r else f"no workflow {name!r}"
        if a == "delete":
            q("DELETE FROM jobs WHERE id=?", (f"wf_{name}",))
            return f"deleted {name!r}"
        if a == "run":
            r = q("SELECT payload FROM jobs WHERE id=?", (f"wf_{name}",), "one")
            if not r:
                return f"no workflow {name!r}"
            steps = json.loads(r[0])
            ctx = dict(inputs or {})
            if isinstance(inputs, str):
                try:
                    ctx = json.loads(inputs)
                except Exception:
                    ctx = {"input": inputs}
            log, outbox = [], []
            WF_DEPTH["n"] += 1
            try:
                return _run_workflow_steps(steps, ctx, log, outbox)
            finally:
                WF_DEPTH["n"] -= 1

        return "actions: save, run, list, show, delete"

    def _run_workflow_steps(steps, ctx, log, outbox):
        for i, st in enumerate(steps, 1):
            args = dict(st.get("args") or {})
            # {{var}} substitution from inputs and earlier step results
            for k, v in list(args.items()):
                if isinstance(v, str):
                    for var, val in ctx.items():
                        v = v.replace("{{" + str(var) + "}}", str(val))
                    args[k] = v
            if st.get("if"):
                cond = st["if"]
                for var, val in ctx.items():
                    cond = cond.replace("{{" + str(var) + "}}", str(val))
                if cond.strip().lower() in ("false", "0", "", "none"):
                    log.append(f"{i}. {st['tool']} SKIPPED (condition)")
                    continue
            out = dispatch(st["tool"], args, OWNER[0] or 0, outbox)
            ctx[st.get("save_as") or f"step{i}"] = str(out)[:4000]
            ok = not str(out).lower().startswith(("error", "no such"))
            log.append(f"{i}. {st['tool']} {'ok' if ok else 'FAILED'}: {str(out)[:120]}")
            if not ok and st.get("stop_on_error", True):
                log.append("stopped — step failed")
                break
        for pth, cap in outbox:
            log.append(f"produced file: {pth}")
        return clip("\n".join(log))

    # ================================================================
    # MEDIA — ffmpeg is already installed for the voice pipeline
    # ================================================================
    def t_media(action, source=None, output=None, start=None, duration=None, args=""):
        a = action.lower()
        if not source and a != "formats":
            return "give a source file"
        out = output or os.path.join(
            DL_DIR,
            f"media_{int(time.time())}."
            + {"audio": "mp3", "gif": "gif", "frames": "png"}.get(a, "mp4"),
        )
        cmds = {
            "info": f'ffprobe -v error -show_format -show_streams "{source}"',
            "audio": f'ffmpeg -y -i "{source}" -vn -acodec libmp3lame "{out}"',
            "clip": f'ffmpeg -y -ss {start or 0} -i "{source}" -t {duration or 10} -c copy "{out}"',
            "compress": f'ffmpeg -y -i "{source}" -vcodec libx264 -crf 28 "{out}"',
            "gif": f"ffmpeg -y -ss {start or 0} -t {duration or 5} "
            f'-i "{source}" -vf "fps=12,scale=480:-1" "{out}"',
            "thumbnail": f'ffmpeg -y -ss {start or 1} -i "{source}" -vframes 1 "{out}"',
            "frames": f'ffmpeg -y -i "{source}" -vf fps=1 "{os.path.dirname(out)}/frame_%03d.png"',
            "convert": f'ffmpeg -y -i "{source}" {args} "{out}"',
            "formats": "ffmpeg -formats 2>/dev/null | head -40",
        }
        if a not in cmds:
            return f"actions: {', '.join(cmds)}"
        r = _run(cmds[a], 1800)
        if a in ("info", "formats"):
            return clip(r)
        if os.path.exists(out):
            return f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB) — send with send_file"
        return clip(f"failed:\n{r[-800:]}")

    # ================================================================
    # DATA — DuckDB handles files bigger than RAM, no server needed
    # ================================================================
    DUCK = {"con": None}

    def t_data(action, query=None, path=None, table=None):
        try:
            import duckdb
        except ImportError:
            sh(f"{sys.executable} -m pip install -q duckdb", check=False)
            try:
                import duckdb
            except ImportError:
                return "could not install duckdb"
        if DUCK["con"] is None:
            DUCK["con"] = duckdb.connect(os.path.join(STATE_DIR, "analysis.duckdb"))
        con_ = DUCK["con"]
        a = action.lower()
        try:
            if a == "load":
                if not (path and table):
                    return "give path and table"
                fn = (
                    "read_csv_auto"
                    if path.lower().endswith((".csv", ".tsv"))
                    else "read_parquet"
                    if path.lower().endswith(".parquet")
                    else "read_json_auto"
                )
                con_.execute(f"CREATE OR REPLACE TABLE \"{table}\" AS SELECT * FROM {fn}('{path}')")
                n = con_.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                cols = con_.execute(f'DESCRIBE "{table}"').fetchall()
                return f"loaded {n:,} rows into {table}\n" + "\n".join(
                    f"  {c[0]} {c[1]}" for c in cols[:30]
                )
            if a == "query":
                if not query:
                    return "give a SQL query"
                rows = con_.execute(query).fetchall()
                cols = [d[0] for d in con_.description] if con_.description else []
                head = " | ".join(cols)
                body = "\n".join(" | ".join(str(v)[:40] for v in r) for r in rows[:50])
                return clip(f"{head}\n{'-' * len(head)}\n{body}\n\n({len(rows)} rows)")
            if a == "tables":
                rows = con_.execute("SHOW TABLES").fetchall()
                return "\n".join(r[0] for r in rows) or "(no tables)"
            if a == "describe":
                rows = con_.execute(f'DESCRIBE "{table}"').fetchall()
                return "\n".join(f"{r[0]}  {r[1]}" for r in rows)
            if a == "export":
                if not (query and path):
                    return "give query and path"
                con_.execute(f"COPY ({query}) TO '{path}' (HEADER)")
                return f"wrote {path} — send with send_file"
        except Exception as e:
            return f"duckdb error: {type(e).__name__}: {e}"
        return "actions: load, query, tables, describe, export"

    # ================================================================
    # CRYPTO — read-only. No keys, no signing, nothing that can move money.
    # ================================================================
    # ================================================================
    # SELF-IMPROVEMENT
    # Four layers, all additive and all reversible:
    #   rules      - behavioural lessons, injected into the system prompt
    #   examples   - past successes, retrieved by similarity
    #   tool_stats - reliability telemetry that feeds reflection
    #   tools      - self-written code, tested in staging before activation
    # The model's weights never change; everything above them can.
    # ================================================================

    # ---- layer 1: learned rules ------------------------------------
    def rules_active(chat_id):
        rows = q(
            "SELECT id, text FROM rules WHERE chat_id=? AND active=1 "
            "ORDER BY score DESC, id DESC LIMIT ?",
            (chat_id, MAX_RULES),
            "all",
        )
        return rows or []

    def rule_add(chat_id, text, source="reflection"):
        text = " ".join(str(text).split())[:MAX_RULE_LEN]
        if len(text) < 12:
            return "rule too short to be useful"
        existing = [r[1].lower() for r in rules_active(chat_id)]
        if text.lower() in existing:
            return "already learned that"
        # semantic near-duplicate check
        try:
            v = embed(text)
            for _, t in rules_active(chat_id):
                if float(np.dot(v, embed(t))) > 0.93:
                    return f"too similar to an existing rule: {t[:80]}"
        except Exception:
            pass
        n = q("SELECT COUNT(*) FROM rules WHERE chat_id=? AND active=1", (chat_id,), "one")[0]
        if n >= MAX_RULES:  # retire the weakest rather than grow forever
            q(
                "UPDATE rules SET active=0 WHERE id = (SELECT id FROM rules "
                "WHERE chat_id=? AND active=1 ORDER BY score ASC, id ASC "
                "LIMIT 1)",
                (chat_id,),
            )
        q(
            "INSERT INTO rules (chat_id,text,source) VALUES (?,?,?)",
            (chat_id, text, source),
        )
        return f"learned: {text}"

    def rule_list(chat_id):
        rows = q(
            "SELECT id, text, score, source FROM rules WHERE chat_id=? "
            "AND active=1 ORDER BY score DESC, id DESC",
            (chat_id,),
            "all",
        )
        if not rows:
            return "no learned rules yet"
        return clip("\n".join(f"[{i}] (score {sc:+.0f}, {src}) {t}" for i, t, sc, src in rows))

    def rule_revoke(chat_id, rule_id):
        q("UPDATE rules SET active=0 WHERE chat_id=? AND id=?", (chat_id, int(rule_id)))
        return f"revoked rule {rule_id}"

    # ---- layer 2: example bank -------------------------------------
    def example_add(chat_id, task, approach, score=1.0):
        try:
            v = embed(task)
        except Exception:
            return
        q(
            "INSERT INTO examples (chat_id,task,approach,vec,score) VALUES (?,?,?,?,?)",
            (chat_id, str(task)[:500], str(approach)[:900], v.tobytes(), float(score)),
        )

    def examples_for(chat_id, task, k=EXAMPLES_IN_CTX):
        rows = q(
            "SELECT task, approach, vec, score FROM examples WHERE chat_id=? AND score > 0",
            (chat_id,),
            "all",
        )
        if not rows:
            return []
        try:
            qv = embed(task)
        except Exception:
            return []
        scored = []
        for t, a, blob, sc in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            if v.shape != qv.shape:
                continue
            scored.append((float(np.dot(qv, v)) + 0.05 * float(sc), t, a))
        scored.sort(reverse=True)
        return [(t, a) for s, t, a in scored[:k] if s > 0.6]

    # ---- layer 3: tool telemetry -----------------------------------
    def stat_record(name, ok, ms, err=None):
        q(
            "INSERT INTO tool_stats (name,calls,fails,total_ms,last_error) "
            "VALUES (?,1,?,?,?) ON CONFLICT(name) DO UPDATE SET "
            "calls=calls+1, fails=fails+?, total_ms=total_ms+?, "
            "last_error=COALESCE(?,last_error)",
            (name, 0 if ok else 1, ms, err, 0 if ok else 1, ms, err),
        )

    REPAIR_PLANS = {
        "crawl4ai": [
            (
                "install with deps resolved separately",
                f'{sys.executable} -m pip install -q --target "{CRAWL_LIB}" '
                f"--break-system-packages --no-deps crawl4ai && "
                f'{sys.executable} -m pip install -q --target "{CRAWL_LIB}" '
                f"--break-system-packages playwright lxml beautifulsoup4 "
                f"aiosqlite aiofiles rank-bm25 snowballstemmer",
            ),
            (
                "install an older, lighter release",
                f'{sys.executable} -m pip install -q --target "{CRAWL_LIB}" '
                f"--break-system-packages --ignore-installed "
                f'"crawl4ai<0.4"',
            ),
            (
                "install ignoring all host packages",
                f'{sys.executable} -m pip install -q --target "{CRAWL_LIB}" '
                f"--break-system-packages --ignore-installed crawl4ai",
            ),
        ],
        "searxng": [
            (
                "rewrite the config to only reliable engines (the usual "
                "cause: one broken engine kills the whole service)",
                "true",
            ),  # handled specially below — config is rewritten
            (
                "install searx core dependencies directly",
                f'{sys.executable} -m pip install -q --target "{SEARX_LIB}" '
                f"--break-system-packages --ignore-installed flask flask-babel "
                f"lxml httpx[http2] babel pyyaml jinja2 brotli msgspec "
                f"python-dateutil pygments typing-extensions certifi",
            ),
            (
                "reinstall from the repo requirements, ignoring host versions",
                f'{sys.executable} -m pip install -q --target "{SEARX_LIB}" '
                f"--break-system-packages --ignore-installed "
                f"-r /opt/searxng/requirements.txt",
            ),
            (
                "re-clone the repository",
                "rm -rf /opt/searxng && git clone --depth 1 "
                "https://github.com/searxng/searxng /opt/searxng",
            ),
        ],
    }

    def t_repair(component=None, action="diagnose"):
        """Read the logs, then try progressively different install
        strategies. This is what to use when a service failed at boot."""
        if action == "issues":
            if not BOOT_ISSUES:
                return "nothing failed at boot"
            return clip("\n\n".join(f"{i['component']}: {i['detail']}" for i in BOOT_ISSUES))

        if action == "diagnose":
            out = [t_services("status"), ""]
            if BOOT_ISSUES:
                out.append("BOOT FAILURES:")
                for i in BOOT_ISSUES:
                    out.append(f"  {i['component']}: {i['detail'][:400]}")
            for f in ("searx.log",):
                fp = os.path.join(JOB_DIR, f)
                if os.path.isfile(fp):
                    out.append(f"\n--- {f} (tail) ---\n" + open(fp, errors="ignore").read()[-1000:])
            out.append("\nRepairable components: " + ", ".join(REPAIR_PLANS))
            out.append("Call repair with component=<name> action='fix' to try the next strategy.")
            return clip("\n".join(out))

        if action == "fix":
            if component not in REPAIR_PLANS:
                return f"nothing to try for {component!r}. Options: {', '.join(REPAIR_PLANS)}"
            log = []
            for label, cmd in REPAIR_PLANS[component]:
                log.append(f"trying: {label}")
                if component == "searxng" and cmd == "true":
                    # remove the stale config so ensure_searx rewrites it
                    try:
                        os.remove(os.path.join(STATE_DIR, "searx-settings.yml"))
                    except OSError:
                        pass
                    if ensure_searx():
                        globals()["SEARX_OK"] = True
                        BOOT_ISSUES[:] = [i for i in BOOT_ISSUES if i["component"] != "searxng"]
                        log.append("  searxng now responding — fixed")
                        return clip("\n".join(log))
                    log.append("  still not binding")
                    continue
                r = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    env=CLEAN_ENV_GLOBAL,
                )
                tail = ((r.stdout or "") + (r.stderr or ""))[-300:].strip()
                if r.returncode != 0:
                    log.append(f"  failed: {tail}")
                    continue
                log.append("  install ok, verifying…")
                if component == "crawl4ai":
                    ok = (
                        subprocess.run(
                            [sys.executable, "-c", "import crawl4ai"],
                            capture_output=True,
                            env=iso_env(CRAWL_LIB),
                        ).returncode
                        == 0
                    )
                    if ok:
                        globals()["CRAWL_OK"] = True
                        log.append("  crawl4ai now imports — fixed")
                        return clip("\n".join(log))
                else:
                    ok = ensure_searx()
                    if ok:
                        globals()["SEARX_OK"] = True
                        log.append("  searxng now responding — fixed")
                        return clip("\n".join(log))
                log.append("  installed but still not working")
            log.append(
                "\nAll strategies failed. The fallbacks "
                "(duckduckgo html search, basic page extraction) "
                "are still working, so nothing is broken for the "
                "user — this is an optimisation, not an outage."
            )
            return clip("\n".join(log))

        return "actions: diagnose, fix, issues"

    def t_services(action="status"):
        rows = []
        try:
            ok = requests.get(f"{SEARX_URL}/healthz", timeout=3).ok
        except Exception:
            ok = False
        rows.append(f"searxng: {'up' if ok else 'down'} ({SEARX_URL})")
        rows.append(f"crawl4ai: {'ready' if CRAWL_OK else 'unavailable'}")
        rows.append(f"ollama: up ({OLLAMA_URL})")
        rows.append(f"utility model: {UTILITY_MODEL or 'none'} (runs the internal checks)")
        rows.append(f"referee: {'on' if REFEREE else 'off'}")
        rows.append(f"public url: {PUBLIC['url'] or 'not started'}")
        if action == "logs":
            for f in ("searx.log", "tunnel.log"):
                fp = os.path.join(JOB_DIR, f)
                if os.path.isfile(fp):
                    rows.append(f"\n--- {f} ---\n" + open(fp, errors="ignore").read()[-1200:])
        return clip("\n".join(rows))

    def t_models(action="list", name=None):
        """Real installed models, and a way to add one by exact tag."""
        b = find_ollama() or "ollama"
        if action == "list":
            try:
                ms = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json().get("models", [])
                return clip(
                    "\n".join(f"{m['name']}  {m.get('size', 0) / 1e9:.1f}GB" for m in ms)
                    or "none installed"
                )
            except Exception as e:
                return f"could not list: {e}"
        if action == "pull":
            if not name:
                return "give an exact model tag"
            r = subprocess.run(
                f"{b} pull {name}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            ok = name in str(t_models("list"))
            return clip((r.stdout or "") + (r.stderr or "") + f"\n\ninstalled: {ok}")
        if action == "use":
            if not name:
                return "give a model name"
            globals()["SMART_MODEL"] = name
            ACTIVE["model"] = name
            ACTIVE["pin"] = "smart"
            return f"now using {name}"
        return "actions: list, pull, use"

    def t_tool_health():
        rows = q(
            "SELECT name, calls, fails, total_ms, last_error "
            "FROM tool_stats ORDER BY fails DESC, calls DESC LIMIT 30",
            (),
            "all",
        )
        if not rows:
            return "no tool usage recorded yet"
        out = []
        for n, c, f, ms, err in rows:
            rate = (f / c * 100) if c else 0
            line = f"{n}: {c} calls, {f} failed ({rate:.0f}%), {ms / max(c, 1):.0f}ms avg"
            if f and err:
                line += f"\n    last error: {str(err)[:120]}"
            out.append(line)
        return clip("\n".join(out))

    # ---- layer 4: reflection ---------------------------------------
    def reflect(chat_id):
        """Look at what went wrong lately and write durable lessons.
        This is the actual learning step."""
        fb = (
            q(
                "SELECT verdict, note, context FROM feedback WHERE chat_id=? "
                "AND ts > datetime('now','-7 days') ORDER BY id DESC LIMIT 30",
                (chat_id,),
                "all",
            )
            or []
        )
        health = t_tool_health()
        turns = (
            q(
                "SELECT role, content FROM turns WHERE chat_id=? AND "
                "ts > datetime('now','-24 hours') ORDER BY id DESC "
                "LIMIT 40",
                (chat_id,),
                "all",
            )
            or []
        )
        if not fb and not turns:
            return "nothing to reflect on"
        convo = "\n".join(f"{r}: {c[:400]}" for r, c in reversed(turns))[:9000]
        fbtxt = "\n".join(f"{v}: {n or ''} (about: {(c or '')[:120]})" for v, n, c in fb)[:3000]
        current = rule_list(chat_id)
        try:
            out = clean(
                call_model(
                    [
                        {
                            "role": "system",
                            "content": "You are reviewing an AI assistant's recent behaviour to "
                            "extract durable lessons.\n\n"
                            "Write at most 4 NEW rules. A good rule is specific, "
                            "actionable, and generalises beyond the single incident — "
                            "e.g. 'Before quoting a price, open the vendor's own "
                            "pricing page; search snippets are often stale.' A bad "
                            "rule is vague ('be more helpful') or restates something "
                            "already learned.\n\n"
                            "Rules must be about HOW to work: verification habits, "
                            "tool choice, formatting the user prefers, mistakes to "
                            "avoid. Never write rules about safety limits, "
                            "credentials, or who is permitted to use the system — "
                            "those are fixed and not yours to change.\n\n"
                            "Output one rule per line, no numbering, no commentary. "
                            "If there is genuinely nothing new to learn, output "
                            "exactly: NOTHING",
                        },
                        {
                            "role": "user",
                            "content": f"ALREADY LEARNED:\n{current}\n\n"
                            f"TOOL RELIABILITY:\n{health}\n\n"
                            f"USER FEEDBACK:\n{fbtxt or '(none)'}\n\n"
                            f"RECENT CONVERSATION:\n{convo}",
                        },
                    ],
                    use_tools=False,
                ).get("content", "")
            )
        except Exception as e:
            return f"reflection failed: {e}"
        if not out or out.strip().upper().startswith("NOTHING"):
            return "reflected: nothing new worth learning"
        added = []
        for line in out.splitlines():
            line = line.strip(" -*•\t")
            if len(line) < 12:
                continue
            res = rule_add(chat_id, line, source="reflection")
            if res.startswith("learned:"):
                added.append(line)
            if len(added) >= 4:
                break
        return clip(
            "learned " + str(len(added)) + " new rule(s):\n" + "\n".join(f"- {a}" for a in added)
            if added
            else "reflected: no new rules passed the filter"
        )

    # ---- layer 5: self-written tools, tested before activation ------
    STAGING = os.path.join(TOOLS_DIR, "_staging")

    def tool_selftest(path):
        """Import the file in a subprocess. If it hangs, crashes, or
        doesn't declare what it promises, it never reaches the live set."""
        probe = f"""
import json, sys
ns = {{}}
exec(open({path!r}).read(), ns)
tools = ns.get("TOOLS")
assert isinstance(tools, list) and tools, "no TOOLS list"
for spec in tools:
    assert len(spec) == 4, "each TOOLS entry needs 4 fields"
    name, desc, props, req = spec
    assert callable(ns.get(name)), "no function named " + str(name)
    assert isinstance(props, dict) and isinstance(req, list)
print(json.dumps([t[0] for t in tools]))
"""
        p = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=30,
            env=CLEAN_ENV,
        )
        if p.returncode != 0:
            return None, (p.stderr or "")[-400:]
        try:
            return json.loads(p.stdout.strip().splitlines()[-1]), None
        except Exception as e:
            return None, f"could not read tool names: {e}"

    def t_write_tool_safe(name, code):
        os.makedirs(STAGING, exist_ok=True)
        safe = re.sub(r"[^\w]", "_", str(name)) + ".py"
        staged = os.path.join(STAGING, safe)
        open(staged, "w").write(code)
        names, err = tool_selftest(staged)
        if err:
            os.remove(staged)
            return (
                f"REJECTED — the tool failed its self-test, so it was "
                f"not installed:\n{err}\nFix the code and try again."
            )
        live = os.path.join(TOOLS_DIR, safe)
        shutil.move(staged, live)
        subprocess.run(
            f'cd "{TOOLS_DIR}" && git init -q 2>/dev/null; '
            f"git config user.email bot@local; git config user.name bot; "
            f'git add -A && git commit -q -m "tool: {safe}" '
            f"|| true",
            shell=True,
            capture_output=True,
            timeout=60,
        )
        load_custom()
        return (
            f"installed and activated {safe} providing: "
            f"{', '.join(names)}. It is live now — no reload needed."
        )

    def t_tool_rollback(steps=1):
        p = subprocess.run(
            f'cd "{TOOLS_DIR}" && git reset -q --hard HEAD~{int(steps)} && git log --oneline -3',
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if p.returncode != 0:
            return f"rollback failed: {(p.stderr or '')[-300:]}"
        load_custom()
        return clip("rolled back. now at:\n" + (p.stdout or ""))

    LAST_EXPERT = {"model": None, "kind": "general"}

    def t_feedback(chat_id, verdict, note=""):
        if LAST_EXPERT["model"]:
            score_model(
                LAST_EXPERT["model"],
                LAST_EXPERT["kind"],
                1.0 if verdict == "good" else -1.0,
                note=f"user said {verdict}: {note[:60]}",
            )
        last = q(
            "SELECT content FROM turns WHERE chat_id=? AND "
            "role='assistant' ORDER BY id DESC LIMIT 1",
            (chat_id,),
            "one",
        )
        q(
            "INSERT INTO feedback (chat_id,verdict,note,context) VALUES (?,?,?,?)",
            (chat_id, verdict, note, (last[0] if last else "")[:600]),
        )
        return "noted"

    # ------ memory consolidation
    def consolidate(chat_id, hours=24):
        """Read recent turns, distil them into durable notes, and drop the
        raw fragments. Stops the vector store turning into confetti."""
        rows = q(
            "SELECT role, content FROM turns WHERE chat_id=? "
            "AND ts > datetime('now', ?) ORDER BY id",
            (chat_id, f"-{int(hours)} hours"),
            "all",
        )
        if not rows or len(rows) < 4:
            return "not enough recent conversation to consolidate"
        convo = "\n".join(f"{r}: {c[:800]}" for r, c in rows)[:14000]
        try:
            notes = clean(
                call_model(
                    [
                        {
                            "role": "system",
                            "content": "Read this conversation log and write durable notes worth "
                            "remembering long-term: decisions made, facts about the "
                            "user, preferences, open threads, things to follow up. "
                            "One short bullet per item. Skip small talk and anything "
                            "already obvious. If nothing is worth keeping, say NOTHING.",
                        },
                        {"role": "user", "content": convo},
                    ],
                    use_tools=False,
                ).get("content", "")
            )
        except Exception as e:
            return f"consolidation failed: {e}"
        if not notes or notes.strip().upper().startswith("NOTHING"):
            return "nothing worth keeping"
        vec_add(chat_id, "summary", f"digest {time.strftime('%Y-%m-%d')}", notes)
        q(
            "INSERT OR REPLACE INTO facts (chat_id,key,value) VALUES (?,?,?)",
            (chat_id, f"digest_{time.strftime('%Y%m%d_%H%M')}", notes[:2000]),
        )
        # raw per-turn vectors older than the window are now redundant
        q(
            "DELETE FROM vectors WHERE chat_id=? AND kind='turn' AND ts < datetime('now', ?)",
            (chat_id, f"-{int(hours)} hours"),
        )
        return clip(f"consolidated {len(rows)} turns into notes:\n\n{notes}")

    # ================================================================
    # ASK_EXPERT — borrow a frontier model for hard reasoning
    # ================================================================
    EXPERT_CACHE = {
        "available": None,
        "catalogue": None,
        "fetched": 0,
        "dead": set(),
        "bootstrapping": False,
    }

    def _expert_key(name, env_key):
        v = os.environ.get(env_key)
        if v:
            return v
        for getter in (
            lambda: __import__("kaggle_secrets").UserSecretsClient().get_secret(env_key),
            lambda: __import__("google.colab", fromlist=["userdata"]).userdata.get(env_key),
        ):
            try:
                v = getter()
                if v:
                    return v
            except Exception:
                pass
        return None

    def experts_available():
        if EXPERT_CACHE["available"] is None:
            EXPERT_CACHE["available"] = [e for e in EXPERTS if _expert_key(e[0], e[1])]
        return EXPERT_CACHE["available"]

    def _catalogue_openrouter(key, url):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        out = []
        for m in r.json().get("data", []):
            pr = m.get("pricing") or {}
            try:
                if not (
                    float(pr.get("prompt", 1) or 0) == 0
                    and float(pr.get("completion", 1) or 0) == 0
                ):
                    continue
            except (TypeError, ValueError):
                continue
            out.append(
                {
                    "id": m.get("id", ""),
                    "ctx": int(m.get("context_length") or 0),
                    "name": m.get("name", ""),
                }
            )
        return out

    def _catalogue_openai_style(key, url):
        r = requests.get(url, timeout=30, headers={"Authorization": f"Bearer {key}"})
        r.raise_for_status()
        out = []
        for m in r.json().get("data", []):
            out.append(
                {
                    "id": m.get("id", ""),
                    "ctx": int(m.get("context_window") or m.get("context_length") or 0),
                    "name": m.get("id", ""),
                }
            )
        return out

    def _catalogue_gemini(key, url):
        r = requests.get(f"{url}?key={key}", timeout=30)
        r.raise_for_status()
        out = []
        for m in r.json().get("models", []):
            if "generateContent" not in (m.get("supportedGenerationMethods") or []):
                continue
            out.append(
                {
                    "id": (m.get("name") or "").replace("models/", ""),
                    "ctx": int(m.get("inputTokenLimit") or 0),
                    "name": m.get("displayName", ""),
                }
            )
        return out

    CATALOGUE_FN = {
        "openrouter": _catalogue_openrouter,
        "groq": _catalogue_openai_style,
        "cerebras": _catalogue_openai_style,
        "gemini": _catalogue_gemini,
    }

    def free_models(force=False):
        """Every configured provider's models, merged into one pool with
        the provider attached. Hardcoding model tags goes stale in weeks;
        four hardcoded tags goes stale four times as fast."""
        if not force and EXPERT_CACHE["catalogue"] and time.time() - EXPERT_CACHE["fetched"] < 3600:
            return EXPERT_CACHE["catalogue"]
        pool = []
        for name, env_key, chat_url, cat_url, style in EXPERTS:
            key = _expert_key(name, env_key)
            if not key:
                continue
            try:
                got = CATALOGUE_FN[name](key, cat_url)
                for m in got:
                    mid = m.get("id") or ""
                    if not mid or any(b in mid.lower() for b in NON_CHAT):
                        continue
                    pool.append(
                        {
                            "id": mid,
                            "ctx": m.get("ctx", 0),
                            "name": m.get("name", ""),
                            "provider": name,
                        }
                    )
                print(f"[catalogue] {name}: {len(got)} models")
            except Exception as e:
                print(f"[catalogue] {name} failed: {str(e)[:120]}")
        pool.sort(key=lambda x: -x["ctx"])
        if pool:
            EXPERT_CACHE["catalogue"] = pool
            EXPERT_CACHE["fetched"] = time.time()
        return pool or EXPERT_CACHE["catalogue"] or []

    def provider_of(model_id):
        for m in free_models():
            if m["id"] == model_id:
                return m["provider"]
        return None

    def pick_expert_model(task="general", need_ctx=0):
        """Choose the best currently-free model for this kind of work."""
        cat = [m for m in free_models() if m["id"] not in EXPERT_CACHE["dead"]]
        if not cat:
            return None
        if need_ctx:
            big = [m for m in cat if m["ctx"] >= need_ctx]
            cat = big or cat
        for hint in TASK_HINTS.get(task, TASK_HINTS["general"]):
            for m in cat:
                if hint in m["id"].lower():
                    return m["id"]
        return cat[0]["id"]

    def t_expert_models(task=None):
        cat = free_models(force=True)
        if not cat:
            return (
                "could not reach the OpenRouter catalogue — check "
                "connectivity, or the free tier list may be empty"
            )
        picks = {t: pick_expert_model(t) for t in TASK_HINTS}
        head = "chosen for each task:\n" + "\n".join(f"  {t:<10} {p}" for t, p in picks.items())
        if task:
            return (
                f"best free chat model for {task!r}: "
                f"{pick_expert_model(task)}\n\n{head}\n\n"
                + "\n".join(f"  {m['id']}  ({m['ctx']:,} ctx)" for m in cat[:25])
            )
        return clip(
            f"{len(cat)} free CHAT models right now "
            f"(music/vision/safety models excluded):\n\n{head}"
            f"\n\nfull list:\n" + "\n".join(f"  {m['id']}  ({m['ctx']:,} ctx)" for m in cat[:40])
        )

    def t_ask_expert(question, context=None, provider=None, task="general", model=None):
        """Delegate hard thinking to a frontier model. Free tiers only.
        The model is chosen first; the provider that owns it is then used,
        so all four accounts act as one pool."""
        avail = experts_available()
        if not avail:
            return (
                "No expert model configured. Add any of these as "
                "secrets to unlock frontier reasoning for free:\n"
                + "\n".join(f"  {e[1]}  ({e[0]})" for e in EXPERTS)
            )

        prompt = str(question)
        if context:
            prompt = f"Context gathered from tools:\n{str(context)[:8000]}\n\nQuestion: {question}"
        sysmsg = (
            "You are a reasoning expert being consulted by another AI "
            "agent. Be precise and concrete. If the context is "
            "insufficient, say exactly what is missing rather than "
            "speculating. Do not pad."
        )
        need_ctx = (len(prompt) // 3) + 2000
        t_start = time.time()

        # build the ordered list of (model, provider) to try
        plan = []
        if model:
            plan.append((model, provider_of(model) or "openrouter", "explicitly requested"))
        elif EXPERT_CACHE["bootstrapping"]:
            # no routing table yet, and asking for one is what got us here
            for m in free_models():
                if m["id"] not in EXPERT_CACHE["dead"] and (
                    not need_ctx or not m["ctx"] or m["ctx"] >= need_ctx
                ):
                    plan.append((m["id"], m["provider"], "bootstrap"))
                    break
        else:
            picked, why = smart_pick(question, task, need_ctx)
            if picked:
                plan.append((picked, provider_of(picked) or "openrouter", why))
        if provider:
            for m in free_models():
                if m["provider"] == provider.lower() and (
                    m["id"],
                    m["provider"],
                ) not in [(a, b) for a, b, _ in plan]:
                    plan.append((m["id"], m["provider"], "provider requested"))
                    break
        # fall back across every other provider, largest context first
        seen = {a for a, _, _ in plan}
        for m in free_models():
            if m["id"] in seen or m["id"] in EXPERT_CACHE["dead"]:
                continue
            if need_ctx and m["ctx"] and m["ctx"] < need_ctx:
                continue
            plan.append((m["id"], m["provider"], "fallback"))
            seen.add(m["id"])
            if len(plan) >= 6:
                break

        errors = []
        for chosen, prov, pick_why in plan:
            spec = next((e for e in EXPERTS if e[0] == prov), None)
            if not spec:
                continue
            name, env_key, chat_url, _cat, style = spec
            key = _expert_key(name, env_key)
            if not key:
                continue
            try:
                if style == "query":
                    url = chat_url.replace("{model}", chosen)
                    r = requests.post(
                        f"{url}?key={key}",
                        timeout=EXPERT_TIMEOUT,
                        json={"contents": [{"parts": [{"text": sysmsg + "\n\n" + prompt}]}]},
                    )
                    if r.status_code in (400, 404, 429):
                        EXPERT_CACHE["dead"].add(chosen)
                        score_model(chosen, task, -0.5, note=f"http {r.status_code}")
                        errors.append(f"{prov}/{chosen}: {r.status_code}")
                        continue
                    r.raise_for_status()
                    d = r.json()
                    txt = d["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    r = requests.post(
                        chat_url,
                        timeout=EXPERT_TIMEOUT,
                        headers={
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": chosen,
                            "messages": [
                                {"role": "system", "content": sysmsg},
                                {"role": "user", "content": prompt},
                            ],
                        },
                    )
                    if r.status_code in (400, 404, 429):
                        EXPERT_CACHE["dead"].add(chosen)
                        score_model(chosen, task, -0.5, note=f"http {r.status_code}")
                        errors.append(
                            f"{prov}/{chosen}: {r.status_code}"
                            + (" rate limited" if r.status_code == 429 else "")
                        )
                        continue
                    r.raise_for_status()
                    txt = r.json()["choices"][0]["message"]["content"]

                if txt and txt.strip():
                    score_model(
                        chosen,
                        task,
                        0,
                        ms=(time.time() - t_start) * 1000,
                        note="answered",
                    )
                    LAST_EXPERT["model"] = chosen
                    LAST_EXPERT["kind"] = task
                    return clip(
                        f"[expert: {prov}/{chosen} · {pick_why}]\n\n" + txt.strip(),
                        8000,
                    )
                errors.append(f"{prov}/{chosen}: empty reply")
            except Exception as e:
                msg = str(e)[:150]
                errors.append(f"{prov}/{chosen}: {msg}")
                if "refus" in msg.lower():
                    return (
                        f"The {prov} model declined that request. Tell "
                        f"the user plainly rather than working around it."
                    )
        return clip("every provider failed:\n" + "\n".join(errors))

    TOOL_SPEC = (
        "Write a Python tool module. Rules, all mandatory:"
        "\n"
        ""
        "\n"
        '1. Define TOOLS = [(name, description, {param: "type"}, [required])]'
        "\n"
        "2. Define one function per name, taking exactly those parameters."
        "\n"
        "3. Return a STRING from every function. Never print."
        "\n"
        "4. Only these are pre-injected: requests, os, re, json, subprocess,"
        "\n"
        "   time. Import anything else INSIDE the function, and pip install it"
        "\n"
        "   there if it may be missing:"
        "\n"
        "       try:"
        "\n"
        "           import foo"
        "\n"
        "       except ImportError:"
        "\n"
        "           import sys, subprocess"
        "\n"
        '           subprocess.run([sys.executable, "-m", "pip", "install",'
        "\n"
        '                           "-q", "foo", "--break-system-packages"],'
        "\n"
        "                          timeout=600)"
        "\n"
        "           import foo"
        "\n"
        "5. Handle errors: return a readable string, never raise."
        "\n"
        "6. No API keys. If one is required, return a message naming it."
        "\n"
        "7. No top-level code that runs on import: no network calls, no sleeps."
        "\n"
        ""
        "\n"
        "Output ONLY the Python. No markdown fences, no commentary."
        "\n"
    )

    def t_build_tool(need, name=None):
        """Describe a capability you lack and get a working tool. A
        frontier model writes it, the self-test gate proves it imports and
        declares what it promises, then it is live immediately."""
        tool_name = re.sub(r"[^\w]", "_", (name or need)[:40].strip().lower()) or "tool"
        existing = ", ".join(sorted({t[0] for t in TOOLS} | set(CUSTOM["fns"])))
        attempts = []
        for attempt in range(3):
            fix = ""
            if attempts:
                fix = (
                    "\n\nYour previous attempt FAILED its self-test "
                    "with this error. Fix it:\n" + attempts[-1]
                )
            code = t_ask_expert(
                "Write a tool that does this: "
                + str(need)
                + "\n\nTool module name: "
                + tool_name
                + "\nTools that already exist (do not duplicate): "
                + existing
                + fix,
                context=TOOL_SPEC,
                task="coding",
            )
            code = re.sub(r"^\[expert:[^\]]*\]\s*", "", str(code))
            code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code.strip(), flags=re.M)
            if not code or "TOOLS" not in code:
                attempts.append("no TOOLS list in the output")
                continue
            os.makedirs(STAGING, exist_ok=True)
            staged = os.path.join(STAGING, tool_name + ".py")
            open(staged, "w").write(code)
            names, err = tool_selftest(staged)
            if err:
                attempts.append(str(err)[-300:])
                try:
                    os.remove(staged)
                except OSError:
                    pass
                continue
            live = os.path.join(TOOLS_DIR, tool_name + ".py")
            shutil.move(staged, live)
            subprocess.run(
                'cd "' + TOOLS_DIR + '" && git init -q 2>/dev/null; '
                "git config user.email bot@local; "
                "git config user.name bot; git add -A && "
                'git commit -q -m "built: ' + tool_name + '" || true',
                shell=True,
                capture_output=True,
                timeout=60,
            )
            load_custom()
            return (
                "built and activated "
                + tool_name
                + " on attempt "
                + str(attempt + 1)
                + ", providing: "
                + ", ".join(names)
                + ". It is live now — call it immediately."
            )
        return clip("could not build a working tool after 3 attempts:\n" + "\n---\n".join(attempts))

    # ================================================================
    # EMPIRICAL MODEL ROUTING
    # Instead of guessing which model suits which job, keep a scorecard
    # built from real outcomes: did the two referees agree, did the
    # fact-checks pass, did the user complain. Route by what has actually
    # worked on THIS user's questions.
    # ================================================================
    def score_model(model, kind, delta, ms=0, note=None):
        """delta: +1 good outcome, -1 bad, 0 neutral-but-used."""
        q(
            "INSERT INTO model_scores (model,kind,uses,wins,losses,ms,"
            "last_used,last_note) VALUES (?,?,1,?,?,?,datetime('now'),?) "
            "ON CONFLICT(model,kind) DO UPDATE SET uses=uses+1, "
            "wins=wins+?, losses=losses+?, ms=ms+?, "
            "last_used=datetime('now'), last_note=COALESCE(?,last_note)",
            (
                model,
                kind,
                max(delta, 0),
                max(-delta, 0),
                ms,
                note,
                max(delta, 0),
                max(-delta, 0),
                ms,
                note,
            ),
        )

    def model_rank(kind):
        """Wilson-ish score: reward wins, punish losses, and keep trying
        models we have barely used so the table does not calcify."""
        rows = (
            q(
                "SELECT model, uses, wins, losses, ms FROM model_scores WHERE kind=?",
                (kind,),
                "all",
            )
            or []
        )
        out = {}
        for m, uses, wins, losses, ms in rows:
            n = max(uses, 1)
            rate = (wins + 1.0) / (wins + losses + 2.0)  # smoothed
            explore = 0.35 / (n**0.5)  # try new ones
            speed = 1.0 / (1.0 + (ms / n) / 60000.0)  # mild speed bias
            out[m] = rate * 0.75 + explore + speed * 0.1
        return out

    def describe_catalogue(limit=30):
        cat = [m for m in free_models() if m["id"] not in EXPERT_CACHE["dead"]][:limit]
        return "\n".join(
            f"{m['id']}  |  {m.get('provider', '?')}  |  {m['ctx']:,} ctx" for m in cat
        )

    def reason_about_models(question, kind, cat_text):
        """Let the model READ the live catalogue and choose. It knows
        far more about model families than any list I could hardcode —
        and the catalogue changes weekly."""
        try:
            out = clean(
                call_utility(
                    [
                        {
                            "role": "system",
                            "content": "You are choosing which model to delegate a question to. "
                            "You will see a catalogue of currently-available models "
                            "with their context sizes. Use what you know about these "
                            "model families — size, specialisation, reasoning "
                            "ability, coding ability — to pick the single best one "
                            "for this question.\n"
                            "Reply with exactly two lines:\n"
                            "MODEL: <exact id from the catalogue>\n"
                            "WHY: <one short line>",
                        },
                        {
                            "role": "user",
                            "content": f"QUESTION KIND: {kind}\n"
                            f"QUESTION: {question[:800]}\n\n"
                            f"AVAILABLE MODELS:\n{cat_text}",
                        },
                    ],
                    max_tokens=120,
                ).get("content", "")
            )
            m = re.search(r"MODEL:\s*([^\s\n]+)", out or "")
            w = re.search(r"WHY:\s*(.+)", out or "")
            if m:
                mid = m.group(1).strip().strip("`\"'")
                if any(c["id"] == mid for c in free_models()):
                    return mid, (w.group(1).strip()[:120] if w else "")
        except Exception as e:
            print("model reasoning failed:", e)
        return None, ""

    def bootstrap_routing():
        """Ask a frontier model to map the live catalogue to task types.
        It knows these model families; waiting a week to learn what it
        could tell us in one call is silly. Cached in the database, so it
        survives restarts via /backup."""
        if setting(OWNER[0] or 0, "routing_bootstrapped"):
            return "already bootstrapped"
        if setting(OWNER[0] or 0, "routing_bootstrap_failed"):
            return "bootstrap previously failed; using catalogue reasoning"
        cat = describe_catalogue(40)
        if not cat:
            return "no catalogue to map"
        try:
            out = t_ask_expert(
                "Below is a live catalogue of models I can delegate work "
                "to. For each of these task types — reasoning, coding, "
                "long, general — name the best and second-best model for "
                "that job, using what you know about these families "
                "(size, specialisation, instruction-following, speed).\n"
                "Reply ONLY as lines of the form:\n"
                "reasoning: <best id> | <second id>\n"
                "coding: <best id> | <second id>\n"
                "long: <best id> | <second id>\n"
                "general: <best id> | <second id>",
                context=cat,
                task="reasoning",
            )
        except Exception as e:
            return f"bootstrap failed: {e}"
        ids = {m["id"] for m in free_models()}
        seeded = 0
        for line in str(out).splitlines():
            if ":" not in line:
                continue
            kind, _, rest = line.partition(":")
            kind = kind.strip().lower()
            if kind not in TASK_HINTS:
                continue
            picks = [x.strip().strip("`\"'*") for x in rest.split("|")]
            for i, mid in enumerate(picks[:2]):
                if mid in ids:
                    # seed as prior evidence, not as fact
                    score_model(
                        mid,
                        kind,
                        0.8 if i == 0 else 0.4,
                        note="seeded by expert at first use",
                    )
                    seeded += 1
        if seeded:
            set_setting(OWNER[0] or 0, "routing_bootstrapped", time.strftime("%Y-%m-%d %H:%M"))
        if seeded:
            return f"routing bootstrapped from expert knowledge: {seeded} model/task pairs seeded"
        # do not retry every single call
        set_setting(OWNER[0] or 0, "routing_bootstrap_failed", time.strftime("%Y-%m-%d %H:%M"))
        return f"expert reply could not be parsed:\n{str(out)[:400]}"

    def smart_pick(question, kind="general", need_ctx=0):
        """Three layers, best first:
        1. measured performance on this user's questions
        2. the model reading the live catalogue and reasoning about it
        3. the static hints, as a floor
        """
        cat = [m for m in free_models() if m["id"] not in EXPERT_CACHE["dead"]]
        if not cat:
            return None, "no catalogue"
        if need_ctx:
            cat = [m for m in cat if m["ctx"] >= need_ctx] or cat
        ids = {m["id"] for m in cat}

        # bootstrap_routing calls ask_expert, which calls smart_pick.
        # Without this guard that is an infinite loop.
        if not model_rank(kind) and experts_available() and not EXPERT_CACHE["bootstrapping"]:
            try:
                EXPERT_CACHE["bootstrapping"] = True
                print("[routing]", bootstrap_routing())
            except Exception as e:
                print("routing bootstrap failed:", e)
            finally:
                EXPERT_CACHE["bootstrapping"] = False

        ranks = {k: v for k, v in model_rank(kind).items() if k in ids}
        tried = sum(1 for k in ranks if ranks[k])
        # once there is real evidence, trust it
        if ranks and tried >= 1:
            best = max(ranks, key=ranks.get)
            return best, f"measured best for {kind} ({ranks[best]:.2f})"

        chosen, why = reason_about_models(question, kind, describe_catalogue())
        if chosen and chosen in ids:
            return chosen, f"chosen by reasoning: {why}"

        return pick_expert_model(kind, need_ctx), "static fallback"

    # ================================================================
    # AUTOMATIC ESCALATION
    # The user should not have to remember which tool to reach for. High
    # stakes get cross-checked whether or not anyone asks.
    # ================================================================
    HIGH_STAKES_RE = re.compile(
        r"\b(?:should i|shall i|is it safe|is it legal|am i liable|"
        r"how much (?:should|do) i|worth (?:buying|paying|investing)|"
        r"invest|mortgage|contract|lawsuit|sue|tribunal|dismissal|"
        r"redundan|tax|hmrc|visa|immigration|deport|"
        r"dosage|dose|mg\b|symptom|diagnos|medication|overdose|"
        r"side effect|allerg|surgery|treatment|"
        r"before i (?:buy|sign|send|pay|commit)|"
        r"is this (?:right|correct|accurate|true|safe)|"
        r"double.?check|are you sure|verify this|fact.?check|"
        r"my (?:money|savings|salary|deposit|health|landlord|employer))\b",
        re.I,
    )
    CASUAL_RE = re.compile(
        r"^\s*(?:hi|hey|hello|thanks|ok|cool|lol|yes|no|sure|nice|"
        r"good morning|what'?s up|you there)\b",
        re.I,
    )

    def assess_stakes(question):
        """Cheap checks first, the utility model only for genuinely
        ambiguous questions. Returns 'high', 'normal' or 'low'."""
        q = (question or "").strip()
        if not q or CASUAL_RE.match(q) or len(q) < 25:
            return "low", "casual"
        m = HIGH_STAKES_RE.search(q)
        if m:
            return "high", f"matched {m.group(0)!r}"
        if len(q) < 80:
            return "normal", "short question"
        try:
            out = clean(
                call_utility(
                    [
                        {
                            "role": "system",
                            "content": "Classify how costly it would be if the answer to this "
                            "question were wrong. Reply with exactly one word:\n"
                            "HIGH - money, health, legal, safety, an irreversible "
                            "decision, or something the user will act on\n"
                            "NORMAL - useful to get right, low cost if wrong\n"
                            "LOW - chat, trivia, creative, exploratory",
                        },
                        {"role": "user", "content": q[:1200]},
                    ],
                    max_tokens=12,
                ).get("content", "")
            )
            w = (out or "").strip().upper()
            if w.startswith("HIGH"):
                return "high", "classifier"
            if w.startswith("LOW"):
                return "low", "classifier"
        except Exception as e:
            print("stakes check failed:", e)
        return "normal", "default"

    def t_referee(question, context=None, task="reasoning"):
        """Ask two DIFFERENT frontier models the same question. Free, so
        the only cost is time — and disagreement between two independent
        models is the strongest unreliability signal available."""
        cat = [m for m in free_models() if m["id"] not in EXPERT_CACHE["dead"]]
        if not cat:
            return "no free models available — try ask_expert, or expert_models to see what is live"

        # two genuinely different models, not two sizes of the same one
        first, _why = smart_pick(question, task)
        first = first or pick_expert_model(task)
        family = (first or "").split("/")[0].lower()
        second = next(
            (m["id"] for m in cat if m["id"] != first and m["id"].split("/")[0].lower() != family),
            None,
        )
        if not second:
            second = next((m["id"] for m in cat if m["id"] != first), None)
        if not second:
            return "only one free model available — cannot cross-check"

        answers = {}
        for mid in (first, second):
            res = t_ask_expert(question, context=context, model=mid)
            if res and not res.lower().startswith(("every expert", "no expert")):
                answers[mid] = re.sub(r"^\[expert:[^\]]*\]\s*", "", res)
        if len(answers) < 2:
            got = list(answers.values())
            return "only one model answered:\n\n" + (got[0] if got else "neither model answered")

        ids = list(answers)
        try:
            verdict = clean(
                call_utility(
                    [
                        {
                            "role": "system",
                            "content": "Two independent AI models answered the same question. "
                            "Compare them. Output exactly:\n"
                            "AGREE: <what they both say>\n"
                            "DISAGREE: <where they differ, or 'nothing substantive'>\n"
                            "CONFIDENCE: high | medium | low\n"
                            "Judge only the substance. Wording differences are not "
                            "disagreement. If they contradict each other on a fact, "
                            "confidence is low.",
                        },
                        {
                            "role": "user",
                            "content": f"QUESTION: {question}\n\n"
                            f"MODEL A ({ids[0]}):\n{answers[ids[0]][:5000]}\n\n"
                            f"MODEL B ({ids[1]}):\n{answers[ids[1]][:5000]}",
                        },
                    ],
                    max_tokens=900,
                ).get("content", "")
            )
        except Exception as e:
            verdict = f"(could not compare: {e})"

        # agreement is evidence both were sane; disagreement means one of
        # them is wrong and we do not yet know which
        vlow = (verdict or "").lower()
        conf = (
            "high"
            if "confidence: high" in vlow
            else "low"
            if "confidence: low" in vlow
            else "medium"
        )
        delta = {"high": 1.0, "medium": 0.2, "low": -0.4}[conf]
        for mid in ids:
            score_model(mid, task, delta, note=f"referee confidence {conf}")

        return clip(
            f"=== CROSS-CHECK: {ids[0]} vs {ids[1]} ===\n\n{verdict}\n\n"
            f"--- {ids[0]} ---\n{answers[ids[0]][:3000]}\n\n"
            f"--- {ids[1]} ---\n{answers[ids[1]][:3000]}",
            12000,
        )

    def t_model_report(kind=None):
        rows = (
            q(
                "SELECT model, kind, uses, wins, losses, ms, last_note "
                "FROM model_scores"
                + (" WHERE kind=?" if kind else "")
                + " ORDER BY uses DESC LIMIT 40",
                (kind,) if kind else (),
                "all",
            )
            or []
        )
        if not rows:
            return (
                "no measurements yet — it routes by reasoning over "
                "the live catalogue until real outcomes accumulate"
            )
        ranks = {}
        for k in {r[1] for r in rows}:
            ranks.update({(m, k): v for m, v in model_rank(k).items()})
        out = []
        for m, k, uses, wins, losses, ms, note in rows:
            sc = ranks.get((m, k), 0)
            out.append(
                f"{sc:.2f}  {k:<9} {m[:44]:<46} "
                f"{int(wins)}W/{int(losses)}L of {uses}  "
                f"{ms / max(uses, 1) / 1000:.0f}s avg"
            )
            if note:
                out.append(f"        last: {note[:70]}")
        best = {
            k: max((v for (mm, kk), v in ranks.items() if kk == k), default=0)
            for k in {r[1] for r in rows}
        }
        head = "currently preferred:\n" + "\n".join(
            f"  {k:<9} "
            + max(
                (mm for (mm, kk), v in ranks.items() if kk == k and v == best[k]),
                default="-",
            )
            for k in best
        )
        return clip(head + "\n\n" + "\n".join(out))

    def t_expert_status():
        avail = experts_available()
        rows = []
        cat = free_models()
        for name, env_key, _chat, _cat_url, _ in EXPERTS:
            have = any(a[0] == name for a in avail)
            n = sum(1 for m in cat if m.get("provider") == name)
            rows.append(
                f"{'✓' if have else '·'} {name:<11} "
                + (f"{n} models available" if have else f"add secret {env_key}")
            )
        if not avail:
            rows.append("\nNone configured — the bot is running on its local model alone.")
        return "\n".join(rows)

    # ================================================================
    # STANDING GOALS / IDLE LOOP / SELF-EVALUATION
    # ================================================================
    def t_goal(action, text=None, goal_id=None, cadence_min=1440, success=None, kind="inform"):
        a = action.lower()
        if a == "add":
            if not text:
                return "give the goal text"
            if not success:
                return (
                    "a goal needs a success condition — how will you "
                    "know a run was worth telling the user about? "
                    "e.g. success='only report if something changed "
                    "since last time'"
                )
            q(
                "INSERT INTO goals (chat_id,text,kind,cadence_min,success) VALUES (?,?,?,?,?)",
                (OWNER[0] or 0, text, kind, int(cadence_min), success),
            )
            row = q(
                "SELECT id FROM goals WHERE chat_id=? ORDER BY id DESC LIMIT 1",
                (OWNER[0] or 0,),
                "one",
            )
            return f"goal #{row[0]} added, runs every {int(cadence_min)} min when idle."
        if a == "list":
            rows = (
                q(
                    "SELECT id,text,cadence_min,active,runs,wins,last_run "
                    "FROM goals WHERE chat_id=? ORDER BY id",
                    (OWNER[0] or 0,),
                    "all",
                )
                or []
            )
            if not rows:
                return "no standing goals"
            return clip(
                "\n".join(
                    f"#{i} [{'on' if act else 'off'}] every {c}min — "
                    f"{t[:70]}\n     {w}/{r} runs useful, last {lr or 'never'}"
                    for i, t, c, act, r, w, lr in rows
                )
            )
        if a in ("pause", "resume"):
            q(
                "UPDATE goals SET active=? WHERE chat_id=? AND id=?",
                (0 if a == "pause" else 1, OWNER[0] or 0, int(goal_id)),
            )
            return f"goal #{goal_id} {a}d"
        if a == "delete":
            q(
                "DELETE FROM goals WHERE chat_id=? AND id=?",
                (OWNER[0] or 0, int(goal_id)),
            )
            return f"goal #{goal_id} deleted"
        if a == "history":
            rows = (
                q(
                    "SELECT g.id, l.score, l.verdict, l.summary, l.ts "
                    "FROM goal_log l JOIN goals g ON g.id=l.goal_id "
                    "WHERE g.chat_id=? ORDER BY l.id DESC LIMIT 15",
                    (OWNER[0] or 0,),
                    "all",
                )
                or []
            )
            if not rows:
                return "no goal runs yet"
            return clip(
                "\n".join(
                    f"#{i} {ts[:16]} score {sc}/10 — {(v or '')[:60]}" for i, sc, v, sm, ts in rows
                )
            )
        if a == "run":
            row = q(
                "SELECT id,text,success FROM goals WHERE chat_id=? AND id=?",
                (OWNER[0] or 0, int(goal_id)),
                "one",
            )
            if not row:
                return f"no goal #{goal_id}"
            return run_goal(row[0], row[1], row[2], forced=True)
        return "actions: add, list, run, pause, resume, delete, history"

    def grade_output(goal_text, success, result, evidence=None):
        """Decide whether a self-initiated run produced anything worth
        interrupting the user for. Without this the idle loop is a spam
        machine."""
        try:
            out = clean(
                call_utility(
                    [
                        {
                            "role": "system",
                            "content": "You grade whether an autonomous agent's work is worth "
                            "sending to the user. Be harsh: the default is NOT "
                            "worth sending. Restating known information, vague "
                            "summaries, 'no change detected', and anything the user "
                            "did not need to be interrupted for all score low.\n"
                            "Reply in exactly this format:\n"
                            "SCORE: <0-10>\n"
                            "VERDICT: <one short line on why>",
                        },
                        {
                            "role": "user",
                            "content": f"GOAL: {goal_text}\n"
                            f"SUCCESS CONDITION: {success}\n\n"
                            f"WHAT THE AGENT PRODUCED:\n{str(result)[:4000]}",
                        },
                    ]
                ).get("content", "")
            )
        except Exception as e:
            print("grading failed:", e)
            return 5, "could not grade"
        m = re.search(r"SCORE:\s*(\d+)", out or "")
        score = int(m.group(1)) if m else 5
        vm = re.search(r"VERDICT:\s*(.+)", out or "")
        return min(max(score, 0), 10), (vm.group(1).strip()[:200] if vm else "")

    def run_goal(gid, text, success, forced=False):
        """Work one standing goal, grade the result, and only deliver it
        if it clears the bar."""
        cid = OWNER[0] or 0
        prompt = (
            f"Standing goal: {text}\n"
            f"It is worth reporting only if: {success}\n\n"
            f"Work on this now using your tools. Check what you "
            f"already know first with recall_semantic so you do not "
            f"repeat yourself. If there is nothing new or nothing "
            f"worth the user's attention, reply with exactly: "
            f"NOTHING TO REPORT."
        )
        outbox = []
        try:
            result = agent_turn(
                build_messages(cid, prompt),
                cid,
                outbox,
                max_steps=AGENT_STEPS,
                force_model="smart",
            )
        except Exception as e:
            q(
                "UPDATE goals SET runs=runs+1, last_run=datetime('now'), last_result=? WHERE id=?",
                (f"error: {e}", gid),
            )
            return f"goal #{gid} failed: {e}"

        if not result or result.strip().upper().startswith("NOTHING"):
            q(
                "UPDATE goals SET runs=runs+1, last_run=datetime('now'), "
                "last_result='nothing to report' WHERE id=?",
                (gid,),
            )
            q(
                "INSERT INTO goal_log (goal_id,score,verdict,summary) VALUES (?,?,?,?)",
                (gid, 0, "nothing to report", ""),
            )
            return f"goal #{gid}: nothing to report"

        score, verdict = grade_output(text, success, result)
        deliver = forced or score >= GOAL_MIN_SCORE
        q(
            "INSERT INTO goal_log (goal_id,score,verdict,summary,delivered) VALUES (?,?,?,?,?)",
            (gid, score, verdict, str(result)[:1500], 1 if deliver else 0),
        )
        q(
            "UPDATE goals SET runs=runs+1, wins=wins+?, "
            "last_run=datetime('now'), last_result=? WHERE id=?",
            (1 if deliver else 0, str(result)[:500], gid),
        )

        if not deliver:
            print(f"[goal {gid}] suppressed, score {score}: {verdict}")
            return (
                f"goal #{gid}: produced something but it scored "
                f"{score}/10 ({verdict}) — not worth sending"
            )
        GOAL_OUTBOX.append({"gid": gid, "text": result, "score": score, "files": outbox})
        return f"goal #{gid}: delivered (score {score}/10)"

    GOAL_OUTBOX = []

    def idle_pick():
        """What to work on, if anything. Respects cadence, quiet hours
        and the daily cap."""
        if not IDLE_ENABLED or not OWNER[0]:
            return None
        hour = int(time.strftime("%H"))
        lo, hi = IDLE_QUIET_H
        if lo <= hour or hour < hi:
            return None
        today = q(
            "SELECT COUNT(*) FROM goal_log WHERE date(ts)=date('now') AND delivered=1",
            (),
            "one",
        )[0]
        if today >= IDLE_MAX_PER_DAY:
            return None
        rows = (
            q(
                "SELECT id,text,success,cadence_min,last_run FROM goals "
                "WHERE chat_id=? AND active=1",
                (OWNER[0],),
                "all",
            )
            or []
        )
        due = []
        for gid, text, success, cad, last in rows:
            if not last:
                due.append((99999, gid, text, success))
                continue
            age = q("SELECT (julianday('now') - julianday(?)) * 1440", (last,), "one")[0] or 0
            if age >= (cad or 1440):
                due.append((age, gid, text, success))
        if not due:
            return None
        due.sort(reverse=True)  # most overdue first
        return due[0][1:]

    # ------ scheduling
    def _register(job_id, chat_id, kind, spec, payload):
        if kind == "cron":
            trig = CronTrigger.from_crontab(spec)
        else:
            trig = IntervalTrigger(minutes=int(spec))
        scheduler.add_job(
            lambda: asyncio.create_task(run_scheduled(job_id, chat_id, kind, payload)),
            trig,
            id=job_id,
            replace_existing=True,
        )

    def t_cron_add(chat_id, crontab, prompt, name=None):
        jid = name or f"cron{int(time.time())}"
        q(
            "INSERT OR REPLACE INTO jobs (id,chat_id,kind,spec,payload,state) VALUES (?,?,?,?,?,?)",
            (jid, chat_id, "cron", crontab, prompt, ""),
        )
        _register(jid, chat_id, "cron", crontab, prompt)
        return f"scheduled {jid}: '{crontab}' -> {prompt[:60]}"

    def t_watch_add(chat_id, url, minutes=60, note="", name=None):
        jid = name or f"watch{int(time.time())}"
        payload = json.dumps({"url": url, "note": note})
        q(
            "INSERT OR REPLACE INTO jobs (id,chat_id,kind,spec,payload,state) VALUES (?,?,?,?,?,?)",
            (jid, chat_id, "watch", str(int(minutes)), payload, ""),
        )
        _register(jid, chat_id, "watch", str(int(minutes)), payload)
        return f"watching {url} every {minutes} min as {jid}"

    def t_job_list(chat_id):
        rows = q("SELECT id,kind,spec,payload FROM jobs WHERE chat_id=?", (chat_id,), "all")
        if not rows:
            return "no scheduled jobs"
        return "\n".join(f"{i} [{k}] {s} -> {str(p)[:60]}" for i, k, s, p in rows)

    def t_job_remove(chat_id, job_id):
        q("DELETE FROM jobs WHERE chat_id=? AND id=?", (chat_id, job_id))
        try:
            scheduler.remove_job(job_id)
        except Exception:
            pass
        return f"removed {job_id}"

    VISION = {"ready": False}

    def describe_image(path, prompt):
        import base64

        if not VISION["ready"]:
            b = find_ollama() or "ollama"
            sh(f"{b} pull {VISION_MODEL}", check=False)
            VISION["ready"] = True
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        with LLM_GATE:  # vision shares the same model server
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                timeout=LLM_TIMEOUT,
                json={
                    "model": VISION_MODEL,
                    "stream": False,
                    "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                },
            )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    # ================= youtube / media transcription ===================
    def t_transcribe_url(url, keep_audio=False):
        """Pull audio from YouTube (or any yt-dlp supported site) and
        transcribe it with whisper."""
        os.makedirs(DL_DIR, exist_ok=True)
        stem = os.path.join(DL_DIR, f"yt_{int(time.time())}")
        cmd = f'yt-dlp -x --audio-format mp3 --no-playlist -o "{stem}.%(ext)s" --print-json "{url}"'
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
        if p.returncode != 0:
            return f"download failed: {(p.stderr or '')[-400:]}"
        title = ""
        try:
            title = json.loads(p.stdout.strip().splitlines()[0]).get("title", "")
        except Exception:
            pass
        audio = next(iter(glob.glob(stem + ".*")), None)
        if not audio:
            return "downloaded but no audio file found"
        text = transcribe(audio)
        if not keep_audio:
            try:
                os.remove(audio)
            except Exception:
                pass
        return clip(f"TITLE: {title}\n\nTRANSCRIPT:\n{text}", 12000)

    # ================= real documents ==================================
    # ================= music ===========================================
    MUSIC = {"pipe": None}

    def t_music(prompt, seconds=10):
        if MUSIC["pipe"] is None:
            sh(f"{sys.executable} -m pip install -q transformers scipy", check=False)
            import torch
            from transformers import pipeline as hf_pipeline

            MUSIC["pipe"] = hf_pipeline(
                "text-to-audio",
                model="facebook/musicgen-small",
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16,
            )
        secs = max(3, min(int(seconds), 30))
        out = MUSIC["pipe"](prompt, forward_params={"max_new_tokens": secs * 50})
        import scipy.io.wavfile

        os.makedirs(DL_DIR, exist_ok=True)
        path = os.path.join(DL_DIR, f"music_{int(time.time())}.wav")
        scipy.io.wavfile.write(path, rate=out["sampling_rate"], data=out["audio"][0].T)
        return f"wrote {path} ({secs}s) — send it with send_file"

    # ================= public URL + dashboard ==========================
    import secrets as _secrets

    PUBLIC = {
        "url": None,
        "httpd": None,
        "shot": None,
        "token": _secrets.token_urlsafe(12),
    }
    LOOP = [None]

    def _serve():
        """Tiny HTTP server: status dashboard plus file download for things
        too big for Telegram's 50 MB cap."""
        import html as _html
        import http.server
        import socketserver
        import urllib.parse

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body, ctype="text/html; charset=utf-8", code=200):
                b = body.encode() if isinstance(body, str) else body
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                u = urllib.parse.urlparse(self.path)
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n).decode(errors="ignore") if n else ""
                try:
                    payload = json.loads(raw) if raw.strip().startswith("{") else {"prompt": raw}
                except Exception:
                    payload = {"prompt": raw}

                if u.path == "/ask":
                    if payload.get("token") != PUBLIC["token"]:
                        return self._send(
                            json.dumps({"error": "bad token — get it with /url in Telegram"}),
                            "application/json",
                            401,
                        )
                    try:
                        msgs = build_messages(OWNER[0] or 0, payload.get("prompt", ""))
                        reply = agent_turn(msgs, OWNER[0] or 0, [])
                    except Exception as e:
                        reply = f"error: {e}"
                    return self._send(json.dumps({"reply": reply}), "application/json")

                if u.path.startswith("/hook/"):
                    hook = u.path.split("/hook/", 1)[1]
                    row = q("SELECT payload FROM jobs WHERE id=?", (f"hook_{hook}",), "one")
                    if not row:
                        return self._send(
                            json.dumps({"error": "no such webhook"}),
                            "application/json",
                            404,
                        )
                    spec = json.loads(row[0])
                    body = raw[:4000]
                    threading.Thread(
                        target=_fire_hook, args=(hook, spec, body), daemon=True
                    ).start()
                    return self._send(json.dumps({"ok": True, "hook": hook}), "application/json")
                return self._send("not found", "text/plain", 404)

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                if u.path == "/health":
                    return self._send("ok", "text/plain")
                if u.path == "/browser.png":
                    shot = PUBLIC.get("shot")
                    if shot and os.path.isfile(shot):
                        with open(shot, "rb") as fh:
                            data = fh.read()
                        self.send_response(200)
                        self.send_header("Content-Type", "image/png")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        return
                    return self._send("no frame yet", "text/plain", 404)
                if u.path == "/watch":
                    return self._send(
                        "<html><body style='margin:0;background:#111'>"
                        "<img id=f src='/browser.png' "
                        "style='width:100%'>"
                        "<script>setInterval(()=>{document.getElementById"
                        "('f').src='/browser.png?'+Date.now()},1200)"
                        "</script></body></html>"
                    )
                if u.path == "/chat":
                    return self._send("""<html><head><meta name=viewport
content="width=device-width,initial-scale=1"><style>
body{font-family:system-ui;margin:0;background:#f6f6f7;display:flex;
flex-direction:column;height:100vh}
#log{flex:1;overflow:auto;padding:1rem}
.m{margin:.5rem 0;padding:.6rem .8rem;border-radius:.8rem;max-width:80%}
.u{background:#2563eb;color:#fff;margin-left:auto}
.b{background:#fff;border:1px solid #ddd}
form{display:flex;gap:.5rem;padding:.8rem;background:#fff}
input{flex:1;padding:.7rem;border:1px solid #ccc;border-radius:.6rem}
button{padding:.7rem 1rem;border:0;border-radius:.6rem;background:#2563eb;
color:#fff}</style></head><body>
<div id=log></div>
<form onsubmit="send(event)">
<input id=i placeholder="token:message  (see /url in Telegram)" autofocus>
<button>Send</button></form>
<script>
const log=document.getElementById('log');
function add(t,c){const d=document.createElement('div');d.className='m '+c;
d.textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight;}
async function send(e){e.preventDefault();const el=document.getElementById('i');
const raw=el.value.trim();if(!raw)return;const ix=raw.indexOf(':');
const tok=raw.slice(0,ix),msg=raw.slice(ix+1);add(msg,'u');el.value=tok+':';
add('thinking...','b');
const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':
'application/json'},body:JSON.stringify({token:tok,prompt:msg})});
const j=await r.json();log.lastChild.textContent=j.reply||j.error;}
</script></body></html>""")
                if u.path == "/dl":
                    qs = urllib.parse.parse_qs(u.query)
                    fp = (qs.get("f") or [""])[0]
                    safe = os.path.abspath(fp)
                    if not (
                        safe.startswith(os.path.abspath(DL_DIR))
                        or safe.startswith(os.path.abspath(WORKSPACE))
                    ) or not os.path.isfile(safe):
                        return self._send("not found", "text/plain", 404)
                    with open(safe, "rb") as fh:
                        data = fh.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header(
                        "Content-Disposition",
                        f'attachment; filename="{os.path.basename(safe)}"',
                    )
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                rows = []
                for aid, a in AGENTS.items():
                    rows.append(
                        f"<tr><td>{_html.escape(aid)}</td>"
                        f"<td>{a['state']}</td>"
                        f"<td>{int(time.time() - a['started'])}s</td>"
                        f"<td>{_html.escape(a['task'][:90])}</td></tr>"
                    )
                jobs = "".join(
                    f"<tr><td>{k}</td><td>"
                    f"{'running' if v['proc'].poll() is None else 'done'}"
                    f"</td><td>{_html.escape(v['cmd'][:90])}</td></tr>"
                    for k, v in JOBS.items()
                )
                files = (
                    "".join(
                        f'<li><a href="/dl?f={urllib.parse.quote(os.path.join(DL_DIR, n))}">'
                        f"{_html.escape(n)}</a> "
                        f"({os.path.getsize(os.path.join(DL_DIR, n)) / 1e6:.1f} MB)</li>"
                        for n in sorted(os.listdir(DL_DIR))[:60]
                        if os.path.isfile(os.path.join(DL_DIR, n))
                    )
                    if os.path.isdir(DL_DIR)
                    else ""
                )
                self._send(f"""<html><head><title>bot</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:60rem}}
table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:.4rem;text-align:left;
font-size:.9rem}}</style></head><body>
<h2>agent dashboard</h2>
<p>model {_html.escape(ACTIVE["model"])} · {len(TOOLS)} tools ·
uptime {int(time.time() - START)}s</p>
<h3>sub-agents</h3><table><tr><th>id</th><th>state</th><th>age</th>
<th>task</th></tr>{"".join(rows) or "<tr><td>none</td></tr>"}</table>
<h3>background jobs</h3><table><tr><th>id</th><th>state</th><th>cmd</th></tr>
{jobs or "<tr><td>none</td></tr>"}</table>
<h3>files</h3><ul>{files or "<li>none</li>"}</ul>
</body></html>""")

        socketserver.TCPServer.allow_reuse_address = True
        httpd = socketserver.TCPServer(("0.0.0.0", SERVE_PORT), H)
        PUBLIC["httpd"] = httpd
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def _fire_hook(hook, spec, body):
        """A webhook either replays a saved workflow or asks the agent."""
        cid = spec.get("chat_id") or OWNER[0]
        try:
            if spec.get("workflow"):
                out = t_workflow("run", name=spec["workflow"], inputs=json.dumps({"body": body}))
            else:
                prompt = (
                    spec.get("prompt") or "Handle this webhook."
                ) + f"\n\nIncoming payload:\n{body}"
                out = agent_turn(build_messages(cid, prompt), cid, [])
            if cid and spec.get("notify", True):
                asyncio.run_coroutine_threadsafe(
                    app.bot.send_message(cid, f"🪝 webhook {hook}:\n\n{str(out)[:3500]}"),
                    LOOP[0],
                )
        except Exception as e:
            print("webhook failed:", e)

    def t_webhook(action, name=None, prompt=None, workflow=None):
        a = action.lower()
        if a == "create":
            if not name:
                return "give a name"
            q(
                "INSERT OR REPLACE INTO jobs (id,chat_id,kind,spec,payload,"
                "state) VALUES (?,?,?,?,?,?)",
                (
                    f"hook_{name}",
                    OWNER[0] or 0,
                    "webhook",
                    name,
                    json.dumps({"chat_id": OWNER[0], "prompt": prompt, "workflow": workflow}),
                    "active",
                ),
            )
            if not PUBLIC["url"]:
                start_tunnel()
            return (
                f"webhook live: {PUBLIC['url']}/hook/{name}\n"
                f"POST anything to that URL and it will "
                + (f"run workflow {workflow!r}" if workflow else "be handled by the agent")
            )
        if a == "list":
            rows = q("SELECT spec, payload FROM jobs WHERE kind='webhook'", (), "all") or []
            if not rows:
                return "no webhooks"
            base = PUBLIC["url"] or "(no url yet)"
            return "\n".join(
                f"{base}/hook/{nm} -> {json.loads(pl).get('workflow') or 'agent'}"
                for nm, pl in rows
            )
        if a == "delete":
            q("DELETE FROM jobs WHERE id=?", (f"hook_{name}",))
            return f"deleted webhook {name!r}"
        return "actions: create, list, delete"

    def start_tunnel():
        """cloudflared quick tunnel — a real public https URL, no account."""
        if not shutil.which("cloudflared"):
            arch = "amd64" if platform.machine().lower() in ("x86_64", "amd64") else "arm64"
            sh(
                f"curl -fsSL https://github.com/cloudflare/cloudflared/"
                f"releases/latest/download/cloudflared-linux-{arch} "
                f"-o /usr/local/bin/cloudflared && "
                f"chmod +x /usr/local/bin/cloudflared",
                check=False,
            )
        if not shutil.which("cloudflared"):
            return "cloudflared unavailable"
        os.makedirs(JOB_DIR, exist_ok=True)
        log = os.path.join(JOB_DIR, "tunnel.log")
        subprocess.Popen(
            f"cloudflared tunnel --url http://localhost:{SERVE_PORT} "
            f'--no-autoupdate > "{log}" 2>&1',
            shell=True,
            start_new_session=True,
        )
        for _ in range(40):
            time.sleep(1)
            try:
                m = re.search(
                    r"https://[a-z0-9-]+\.trycloudflare\.com",
                    open(log, errors="ignore").read(),
                )
                if m:
                    PUBLIC["url"] = m.group(0)
                    return PUBLIC["url"]
            except Exception:
                pass
        return "tunnel did not come up"

    def t_public_url():
        if not PUBLIC["url"]:
            start_tunnel()
        u = PUBLIC["url"]
        if not u:
            return "no public URL available"
        return (
            f"{u}            dashboard\n"
            f"{u}/chat       web chat\n"
            f"{u}/watch      live browser view\n"
            f"{u}/hook/NAME  webhook endpoint\n"
            f"POST {u}/ask   API\n"
            f"api token: {PUBLIC['token']}"
        )

    def t_share_file(path):
        """For files over Telegram's 50 MB limit — hand back a link."""
        if not os.path.isfile(path):
            return f"no such file: {path}"
        if not PUBLIC["url"]:
            start_tunnel()
        if not PUBLIC["url"]:
            return "no public URL available"
        import urllib.parse

        return f"{PUBLIC['url']}/dl?f={urllib.parse.quote(os.path.abspath(path))}"

    # ================= email: throwaway inbox + optional SMTP ==========
    inbox_tools = InboxTools(store)
    t_mail_create = inbox_tools.create
    t_mail_inbox = inbox_tools.inbox
    t_mail_read = inbox_tools.read

    # ================= persistent project workspace ====================
    def _ws(path=""):
        os.makedirs(WORKSPACE, exist_ok=True)
        # the model often writes "workspace/x.md"; don't nest it twice
        path = re.sub(r"^/?workspace/+", "", str(path or ""))
        full = os.path.abspath(os.path.join(WORKSPACE, path))
        if not full.startswith(os.path.abspath(WORKSPACE)):
            raise ValueError("path escapes the workspace")
        return full

    def t_ws_write(path, content):
        full = _ws(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "w").write(content)
        return f"wrote {path} ({len(content)} chars)"

    def t_ws_read(path):
        return clip(open(_ws(path), errors="ignore").read())

    def t_ws_list(path=""):
        base = _ws(path)
        if os.path.isfile(base):
            return f"{path} is a file ({os.path.getsize(base)} bytes)"
        out = []
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d != ".git"]
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), WORKSPACE)
                out.append(f"{rel}  {os.path.getsize(os.path.join(root, f))}b")
        return clip("\n".join(sorted(out)[:200]) or "(workspace empty)")

    def t_git_snapshot(message="snapshot"):
        os.makedirs(WORKSPACE, exist_ok=True)
        cmds = (
            f'cd "{WORKSPACE}" && git init -q 2>/dev/null; '
            f"git config user.email bot@local; "
            f"git config user.name bot; "
            f'git add -A && git commit -q -m "{message}" '
            f'|| echo "nothing to commit"'
        )
        p = subprocess.run(cmds, shell=True, capture_output=True, text=True, timeout=120)
        log = subprocess.run(
            f'cd "{WORKSPACE}" && git log --oneline -5',
            shell=True,
            capture_output=True,
            text=True,
        )
        return clip((p.stdout or "") + "\n" + (log.stdout or ""))

    # ================= image generation (second GPU) ===================
    IMG = {"pipe": None}

    NEG_PROMPT = (
        "text, watermark, signature, letters, words, logo, "
        "blurry, low quality, jpeg artifacts, deformed, "
        "oversaturated, extra limbs"
    )

    def t_image(prompt, steps=None):
        # Turbo breaks past ~4 steps; full SDXL needs ~30. Don't let the
        # model pick a number that ruins the image.
        if IMAGE_TURBO:
            steps = max(1, min(int(steps or 4), 4))
            guidance = 0.0
        else:
            steps = max(15, min(int(steps or 30), 50))
            guidance = 7.0
        if IMG["pipe"] is None:
            sh(
                f"{sys.executable} -m pip install -q diffusers transformers accelerate safetensors",
                check=False,
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = IMAGE_GPU
            import torch
            from diffusers import AutoPipelineForText2Image

            IMG["pipe"] = AutoPipelineForText2Image.from_pretrained(
                IMAGE_MODEL, torch_dtype=torch.float16, variant="fp16"
            ).to("cuda")
        os.makedirs(DL_DIR, exist_ok=True)
        path = os.path.join(DL_DIR, f"img_{int(time.time())}.png")
        img = IMG["pipe"](
            prompt=prompt,
            negative_prompt=NEG_PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance,
        ).images[0]
        img.save(path)
        return f"generated {path} ({steps} steps) — use send_file to show the user"

    # ================= sub-agents ======================================
    AGENTS = {}

    def t_agent_spawn(chat_id, task, name=None):
        aid = name or f"agent{len(AGENTS) + 1}"
        log = []

        def worker():
            AGENTS[aid]["state"] = "running"
            try:
                msgs = [
                    {
                        "role": "system",
                        "content": PERSONA + "\n\nYou are running as an "
                        "autonomous background agent. Work the task to "
                        "completion using your tools. Be decisive.\n\n"
                        "RESEARCH STANDARDS — these matter more than "
                        "finishing fast:\n"
                        "- Search snippets are leads, not facts. Before "
                        "stating a price, spec or claim, open the actual "
                        "source page with browser_open or web_fetch and "
                        "read it.\n"
                        "- Check what a site actually is. Blogs, deal "
                        "aggregators and listicles are not vendors; do not "
                        "present them as one.\n"
                        "- Run several different searches, not one. The "
                        "first three results are rarely the best three "
                        "answers.\n"
                        "- If you could not verify something, say so "
                        "explicitly rather than stating it as fact.\n"
                        "When finished, reply with a clear summary.",
                    },
                    {"role": "user", "content": task},
                ]
                ob = []
                res = agent_turn(
                    msgs,
                    chat_id,
                    ob,
                    on_progress=lambda s: log.append(s),
                    max_steps=AGENT_STEPS,
                    force_model="smart",
                )
                AGENTS[aid]["result"] = res
                AGENTS[aid]["outbox"] = ob
                AGENTS[aid]["state"] = "done"
            except Exception as e:
                msg = f"failed: {type(e).__name__}: {e}"
                if "11434" in str(e) or "ReadTimeout" in type(e).__name__:
                    msg += (
                        "\n\n(This is the local model server being "
                        "overloaded, not the websites. Too many agents "
                        "generating at once — try fewer parallel tasks.)"
                    )
                AGENTS[aid]["result"] = msg
                AGENTS[aid]["state"] = "failed"

        AGENTS[aid] = {
            "task": task,
            "state": "starting",
            "log": log,
            "result": None,
            "outbox": [],
            "chat_id": chat_id,
            "started": time.time(),
        }
        threading.Thread(target=worker, daemon=True).start()
        return f"spawned {aid} with a {AGENT_STEPS}-step budget. Check with agent_status('{aid}')."

    def t_team_spawn(chat_id, tasks, name=None):
        """Run several sub-agents in parallel, then synthesise their
        findings into one answer."""
        if isinstance(tasks, str):
            try:
                tasks = json.loads(tasks)
            except Exception:
                tasks = [t.strip() for t in tasks.split("|") if t.strip()]
        tasks = [str(t) for t in tasks][:MAX_TEAM]
        if len(tasks) < 2:
            return "give at least 2 tasks; for one task use agent_spawn"
        tid = name or f"team{len(AGENTS) + 1}"
        ids = []
        for i, t in enumerate(tasks, 1):
            sub = f"{tid}_{i}"
            t_agent_spawn(chat_id, t, name=sub)
            ids.append(sub)
            time.sleep(2)  # stagger so they don't all fire at once

        def joiner():
            while any(AGENTS[i]["state"] in ("starting", "running") for i in ids):
                time.sleep(3)
            parts = [f"### {i}: {AGENTS[i]['task'][:80]}\n{AGENTS[i]['result']}" for i in ids]
            merged = "\n\n".join(parts)
            try:
                summary = clean(
                    call_model(
                        [
                            {
                                "role": "system",
                                "content": "Synthesise these parallel research results into one "
                                "clear answer. Note where they disagree. Flag anything "
                                "that looks unverified.",
                            },
                            {"role": "user", "content": merged[:12000]},
                        ],
                        use_tools=False,
                    ).get("content", "")
                )
            except Exception as e:
                summary = f"(synthesis failed: {e})\n\n{merged}"
            AGENTS[tid]["result"] = summary
            AGENTS[tid]["outbox"] = [o for i in ids for o in AGENTS[i].get("outbox", [])]
            AGENTS[tid]["state"] = "done"

        AGENTS[tid] = {
            "task": f"team of {len(ids)}: " + "; ".join(t[:40] for t in tasks),
            "state": "running",
            "log": [],
            "result": None,
            "outbox": [],
            "chat_id": chat_id,
            "started": time.time(),
        }
        threading.Thread(target=joiner, daemon=True).start()
        return (
            f"spawned team {tid} with {len(ids)} agents working in "
            f"parallel: {', '.join(ids)}. Check agent_status('{tid}')."
        )

    def t_agent_status(agent_id):
        a = AGENTS.get(agent_id)
        if not a:
            return f"no agent {agent_id!r}. known: {list(AGENTS) or 'none'}"
        age = int(time.time() - a["started"])
        if a["state"] == "running" and not a["log"]:
            return (
                f"{agent_id}: running for {age}s, no tool calls yet — "
                f"it is still planning or the model is loading"
            )
        head = f"{agent_id}: {a['state']} after {age}s\ntask: {a['task'][:120]}"
        if a["state"] in ("done", "failed"):
            return clip(f"{head}\n\nRESULT:\n{a['result']}")
        return clip(f"{head}\n\nrecent steps:\n" + "\n".join(a["log"][-8:]))

    def t_agent_list():
        if not AGENTS:
            return "no sub-agents"
        return "\n".join(
            f"{k}: {v['state']} ({int(time.time() - v['started'])}s) — {v['task'][:60]}"
            for k, v in AGENTS.items()
        )

    # ------ self-written tools
    CUSTOM = {"tools": [], "fns": {}}

    def load_custom():
        os.makedirs(TOOLS_DIR, exist_ok=True)
        CUSTOM["tools"], CUSTOM["fns"] = [], {}
        for f in sorted(glob.glob(os.path.join(TOOLS_DIR, "*.py"))):
            try:
                ns = {
                    "requests": requests,
                    "os": os,
                    "re": re,
                    "json": json,
                    "subprocess": subprocess,
                    "time": time,
                }
                exec(open(f).read(), ns)
                for spec in ns.get("TOOLS", []):
                    name = spec[0]
                    if name in ns:
                        CUSTOM["tools"].append(tuple(spec))
                        CUSTOM["fns"][name] = ns[name]
            except Exception as e:
                print(f"custom tool {f} failed to load: {e}")
        return f"loaded {len(CUSTOM['tools'])} custom tools"

    def t_write_tool(name, code):
        os.makedirs(TOOLS_DIR, exist_ok=True)
        path = os.path.join(TOOLS_DIR, re.sub(r"[^\w]", "_", name) + ".py")
        open(path, "w").write(code)
        try:
            compile(code, path, "exec")
        except SyntaxError as e:
            return f"written to {path} but it has a syntax error: {e}"
        return f"wrote {path}. Tell the user to run /reload to activate it."

    # ------------------------------------------------------ tool schema
    TOOLS = [
        (
            "web_search",
            "Search the web via the local SearxNG service — "
            "unlimited and free. Optionally restrict engines "
            "(e.g. 'google,bing') or category (news, images, "
            "science, it, videos).",
            {
                "query": "string",
                "max_results": "integer",
                "engines": "string",
                "category": "string",
            },
            ["query"],
        ),
        (
            "web_fetch",
            "Fetch and clean a page with Crawl4AI — runs "
            "JavaScript, returns readable markdown. If the "
            "output gets truncated, call it again with "
            "contains='some phrase' to jump to that part of the "
            "page, or start_at=<character offset> to continue "
            "where it cut off. mode='links' lists the page's "
            "links. Use browser_open only to click or log in.",
            {
                "url": "string",
                "mode": "string",
                "contains": "string",
                "start_at": "integer",
            },
            ["url"],
        ),
        (
            "browser_open",
            "Open a URL in real headless Chrome; returns visible "
            "text. Cookies persist across restarts.",
            {"url": "string", "wait": "string"},
            ["url"],
        ),
        ("browser_read", "Re-read the current page.", {}, []),
        (
            "browser_click",
            "Click a CSS selector or visible text.",
            {"selector": "string"},
            ["selector"],
        ),
        (
            "browser_type",
            "Type into a field, Enter by default.",
            {"selector": "string", "text": "string", "enter": "boolean"},
            ["selector", "text"],
        ),
        ("browser_links", "List links on the current page.", {}, []),
        (
            "browser_screenshot",
            "Screenshot the page to PNG; send with send_file.",
            {"full_page": "boolean"},
            [],
        ),
        (
            "browser_look",
            "LOOK at the current page with vision instead of "
            "reading its text. Use when browser_read returns "
            "nothing useful, or when layout/visual detail "
            "matters.",
            {"question": "string"},
            [],
        ),
        ("browser_eval", "Run JavaScript in the page.", {"code": "string"}, ["code"]),
        ("browser_back", "Go back one page.", {}, []),
        (
            "run_python",
            "Run Python, return stdout.",
            {"code": "string", "timeout": "integer"},
            ["code"],
        ),
        (
            "run_shell",
            "Run bash and wait (15 min cap, raise with timeout).",
            {"command": "string", "timeout": "integer"},
            ["command"],
        ),
        (
            "run_background",
            "Start a shell command with NO time limit; returns a job id immediately.",
            {"command": "string", "name": "string"},
            ["command"],
        ),
        (
            "job_status",
            "Check a background job's state and output.",
            {"job_id": "string", "lines": "integer"},
            ["job_id"],
        ),
        ("job_list", "List background jobs.", {}, []),
        ("job_kill", "Stop a background job.", {"job_id": "string"}, ["job_id"]),
        (
            "download_file",
            "Download a URL into the container.",
            {"url": "string", "filename": "string"},
            ["url"],
        ),
        (
            "send_file",
            "Send a container file to the user as a Telegram attachment (max 50 MB).",
            {"path": "string", "caption": "string"},
            ["path"],
        ),
        ("list_files", "List files in a directory.", {"directory": "string"}, []),
        (
            "index_document",
            "Index a local file (pdf/txt/md) into the searchable document library.",
            {"path": "string"},
            ["path"],
        ),
        (
            "search_documents",
            "Search indexed documents by meaning.",
            {"query": "string", "k": "integer"},
            ["query"],
        ),
        (
            "remember",
            "Save a durable fact about the user.",
            {"key": "string", "value": "string"},
            ["key", "value"],
        ),
        ("recall", "Look up saved facts; omit key to list all.", {"key": "string"}, []),
        (
            "recall_semantic",
            "Search all past conversations and facts by "
            "meaning. Use before claiming you don't know "
            "something.",
            {"query": "string", "k": "integer"},
            ["query"],
        ),
        ("forget", "Delete a saved fact.", {"key": "string"}, ["key"]),
        (
            "cron_add",
            "Run a prompt on a schedule and message the user the "
            "result. Standard 5-field crontab, e.g. '0 8 * * *'.",
            {"crontab": "string", "prompt": "string", "name": "string"},
            ["crontab", "prompt"],
        ),
        (
            "watch_add",
            "Watch a page and message the user when it changes.",
            {"url": "string", "minutes": "integer", "note": "string", "name": "string"},
            ["url"],
        ),
        ("job_schedule_list", "List scheduled prompts and watchers.", {}, []),
        (
            "job_schedule_remove",
            "Delete a scheduled prompt or watcher.",
            {"job_id": "string"},
            ["job_id"],
        ),
        (
            "agent_spawn",
            "Hand a big multi-step job to an autonomous "
            "background agent with its own large tool budget. "
            "Returns immediately. Use for research, building "
            "things, anything long.",
            {"task": "string", "name": "string"},
            ["task"],
        ),
        (
            "team_spawn",
            "Run 2-3 sub-agents IN PARALLEL on different parts "
            "of a job, then merge their findings automatically. "
            "Far faster than one agent doing everything. Pass "
            "tasks as a list of independent instructions.",
            {"tasks": "array", "name": "string"},
            ["tasks"],
        ),
        (
            "agent_status",
            "Check a sub-agent's progress or final result.",
            {"agent_id": "string"},
            ["agent_id"],
        ),
        ("agent_list", "List sub-agents.", {}, []),
        (
            "email_create_inbox",
            "Create a throwaway email address for this chat. Needed before reading mail.",
            {},
            [],
        ),
        (
            "email_inbox",
            "List recent messages in the throwaway inbox.",
            {"limit": "integer"},
            [],
        ),
        (
            "email_read",
            "Read one message in full by its id.",
            {"message_id": "string"},
            ["message_id"],
        ),
        (
            "email_send",
            "Send an email (needs SMTP secrets configured).",
            {"to": "string", "subject": "string", "body": "string"},
            ["to", "subject", "body"],
        ),
        (
            "workspace_write",
            "Write a file into the persistent project folder.",
            {"path": "string", "content": "string"},
            ["path", "content"],
        ),
        (
            "workspace_read",
            "Read a file from the project folder.",
            {"path": "string"},
            ["path"],
        ),
        ("workspace_list", "List the project folder.", {"path": "string"}, []),
        (
            "git_snapshot",
            "Commit the project folder so the work survives.",
            {"message": "string"},
            [],
        ),
        (
            "generate_image",
            "Generate an image on the second GPU. Write a "
            "rich, detailed prompt — style, lighting, lens, "
            "mood — not just the subject. Leave steps unset; "
            "the correct value is chosen for you. Returns a "
            "PNG path; send it with send_file.",
            {"prompt": "string"},
            ["prompt"],
        ),
        (
            "terminal",
            "A PERSISTENT shell: cd, exports, activated venvs "
            "and background state survive between calls, unlike "
            "run_shell. Use this for multi-step work in a "
            "directory. action: run|restart|status.",
            {"command": "string", "action": "string", "timeout": "integer"},
            [],
        ),
        (
            "python_session",
            "A PERSISTENT Python kernel: variables, imports "
            "and loaded dataframes survive between calls, "
            "like a notebook. action: run|reset|vars.",
            {"code": "string", "action": "string"},
            ["code"],
        ),
        (
            "system",
            "Inspect the machine. action: info, resources, hardware, "
            "gpu, network, dns, ports, connectivity, logs. Pass "
            "target for dns/connectivity/logs.",
            {"action": "string", "target": "string"},
            ["action"],
        ),
        (
            "process",
            "Manage processes. action: list, tree, info, kill, monitor. Pass pid or pattern.",
            {
                "action": "string",
                "pid": "integer",
                "pattern": "string",
                "signal_name": "string",
            },
            ["action"],
        ),
        (
            "fs",
            "Filesystem operations. action: tree, find, grep, stat, "
            "copy, move, delete, watch. pattern for find/grep, dest for "
            "copy/move.",
            {
                "action": "string",
                "path": "string",
                "pattern": "string",
                "dest": "string",
                "depth": "integer",
            },
            ["action"],
        ),
        (
            "packages",
            "Install or inspect packages. action: list, install, info. manager: pip, apt, npm.",
            {"action": "string", "names": "string", "manager": "string"},
            ["action"],
        ),
        (
            "build",
            "Compile code. action: toolchains (see what is available), c, cpp, rust, go, make.",
            {
                "action": "string",
                "source": "string",
                "output": "string",
                "args": "string",
            },
            ["action"],
        ),
        (
            "code",
            "Analyse source. action: lint, check (syntax), symbols "
            "(functions/classes/imports), complexity. Give path or "
            "code.",
            {"action": "string", "path": "string", "code": "string"},
            ["action"],
        ),
        (
            "git",
            "Git operations on any repo. action: clone, status, log, "
            "diff, branches, remotes, blame, show, commit.",
            {
                "action": "string",
                "repo": "string",
                "url": "string",
                "args": "string",
                "message": "string",
            },
            ["action"],
        ),
        (
            "http",
            "Full-control HTTP request — any method, headers, body. "
            "Returns status, headers and body. Use for APIs; use "
            "web_fetch for reading articles.",
            {
                "url": "string",
                "method": "string",
                "headers": "string",
                "body": "string",
                "json_body": "string",
            },
            ["url"],
        ),
        (
            "ssh",
            "Run a command on a remote host over SSH.",
            {
                "host": "string",
                "command": "string",
                "user": "string",
                "key_path": "string",
                "port": "integer",
            },
            ["host", "command"],
        ),
        (
            "sandbox",
            "Run an untrusted command with hard CPU, memory and file-size limits.",
            {"command": "string", "seconds": "integer", "memory_mb": "integer"},
            ["command"],
        ),
        (
            "container",
            "Podman containers. action: check, install, run, "
            "list. Docker is impossible in this environment.",
            {"action": "string", "image": "string", "command": "string"},
            [],
        ),
        (
            "tool_rollback",
            "Undo the last self-written tool change if something broke.",
            {"steps": "integer"},
            [],
        ),
        (
            "env",
            "Save and restore the container's software environment. "
            "The container is wiped on every restart, so record what "
            "you install. action: define (declare packages and setup "
            "commands — preferred), snapshot (record everything "
            "currently installed), restore, list, show, delete.",
            {
                "action": "string",
                "name": "string",
                "packages": "string",
                "commands": "string",
            },
            ["action"],
        ),
        (
            "webhook",
            "Create a public URL that anything on the internet can "
            "POST to — GitHub, Stripe, IFTTT, forms, alerts. It can "
            "either run a saved workflow or be handled by you. "
            "action: create, list, delete.",
            {
                "action": "string",
                "name": "string",
                "prompt": "string",
                "workflow": "string",
            },
            ["action"],
        ),
        (
            "workflow",
            "Save and replay a fixed sequence of tool calls — "
            "deterministic, no model in the loop. steps is a JSON "
            'array like [{"tool":"web_search","args":'
            '{"query":"{{topic}}"},"save_as":"hits"}]. '
            "Use {{var}} for inputs and earlier results. action: "
            "save, run, list, show, delete.",
            {
                "action": "string",
                "name": "string",
                "steps": "string",
                "inputs": "string",
            },
            ["action"],
        ),
        (
            "media",
            "Video and audio processing with ffmpeg. action: info, "
            "audio (extract), clip, compress, gif, thumbnail, frames, "
            "convert, formats.",
            {
                "action": "string",
                "source": "string",
                "output": "string",
                "start": "string",
                "duration": "string",
                "args": "string",
            },
            ["action"],
        ),
        (
            "data",
            "Analyse large files with SQL via DuckDB — handles files "
            "bigger than RAM. action: load (csv/parquet/json into a "
            "table), query, tables, describe, export.",
            {
                "action": "string",
                "query": "string",
                "path": "string",
                "table": "string",
            },
            ["action"],
        ),
        (
            "crypto",
            "Read-only crypto data: price, market, balance (any "
            "public address), gas. Holds no keys and cannot move "
            "funds.",
            {
                "action": "string",
                "symbol": "string",
                "address": "string",
                "chain": "string",
            },
            ["action"],
        ),
        (
            "ask_expert",
            "Consult a frontier reasoning model when a problem "
            "is beyond your own judgement: hard analysis, "
            "subtle code review, maths, weighing conflicting "
            "evidence, planning something complex. Pass the "
            "tool output you gathered as 'context' so it "
            "reasons over real data. Set task to 'reasoning', "
            "'coding', 'long' or 'general' and the best "
            "currently-free model is chosen automatically. YOU "
            "still answer the user — integrate its reasoning, "
            "do not just relay it.",
            {
                "question": "string",
                "context": "string",
                "task": "string",
                "provider": "string",
                "model": "string",
            },
            ["question"],
        ),
        (
            "build_tool",
            "Lacking a capability? Describe it and a "
            "frontier model writes the code, it is tested "
            "automatically, and it goes live immediately — no "
            "restart, no reload. Use this whenever a task needs "
            "something you cannot currently do, instead of "
            "telling the user you cannot do it.",
            {"need": "string", "name": "string"},
            ["need"],
        ),
        (
            "model_report",
            "Show the measured scorecard of every frontier "
            "model tried: win/loss from referee agreement, "
            "fact-check results and user feedback, plus which "
            "is currently preferred per task type.",
            {"kind": "string"},
            [],
        ),
        (
            "stakes",
            "Judge how costly it would be to get a question wrong: "
            "high, normal or low. Called automatically, but useful "
            "if you want to check before committing to an answer.",
            {"question": "string"},
            ["question"],
        ),
        (
            "referee",
            "Ask TWO different frontier models the same question "
            "and compare their answers. Use when the answer really "
            "matters and being wrong would be costly — "
            "disagreement between independent models is the "
            "strongest warning sign that an answer is unreliable. "
            "Slower than ask_expert, so save it for things worth "
            "double-checking.",
            {"question": "string", "context": "string", "task": "string"},
            ["question"],
        ),
        (
            "expert_status",
            "Show which frontier models are configured and which keys would unlock more.",
            {},
            [],
        ),
        (
            "expert_models",
            "List the models that are actually free on "
            "OpenRouter right now, and which one would be "
            "chosen for a given task (reasoning, coding, "
            "long, general). The free line-up changes, so "
            "check rather than assume.",
            {"task": "string"},
            [],
        ),
        (
            "goal",
            "Standing goals the bot works on by itself when idle. "
            "action='add' needs text and a success condition (how to "
            "tell a run was worth reporting) plus cadence_min. Also: "
            "list, run, pause, resume, delete, history.",
            {
                "action": "string",
                "text": "string",
                "goal_id": "integer",
                "cadence_min": "integer",
                "success": "string",
            },
            ["action"],
        ),
        (
            "check_sources",
            "Rate the quality of the sources behind a set "
            "of URLs: 1=primary/authoritative, 2=established "
            "press, 3=secondary, 4=unknown. Also flags "
            "blogs, deal aggregators and forums. Use before "
            "presenting research as fact.",
            {"urls": "string"},
            ["urls"],
        ),
        (
            "repair",
            "Diagnose and fix a service that failed to start. "
            "action='diagnose' reads the logs and boot failures, "
            "action='fix' with component='crawl4ai' or 'searxng' "
            "tries progressively different install strategies, "
            "action='issues' lists what failed at boot.",
            {"component": "string", "action": "string"},
            [],
        ),
        (
            "services",
            "Check the local services — searxng, crawl4ai, "
            "ollama, public url. action='logs' also shows their "
            "logs. Use when search or fetching misbehaves.",
            {"action": "string"},
            [],
        ),
        (
            "models",
            "Inspect or change the language models. action: list "
            "(what is actually installed), pull (add one by exact "
            "tag), use (switch to one).",
            {"action": "string", "name": "string"},
            [],
        ),
        (
            "tool_health",
            "See how reliable each tool has been — call counts, "
            "failure rates, last errors. Use this before "
            "blaming yourself for a failure.",
            {},
            [],
        ),
        (
            "learn_rule",
            "Record a durable lesson about how to work better. "
            "Use when the user corrects you or you discover a "
            "better approach. Be specific and actionable.",
            {"text": "string"},
            ["text"],
        ),
        ("list_rules", "List the lessons you have learned.", {}, []),
        (
            "revoke_rule",
            "Drop a learned rule that turned out to be wrong.",
            {"rule_id": "integer"},
            ["rule_id"],
        ),
        (
            "reflect_now",
            "Review recent work and feedback, and extract new lessons. Runs nightly on its own.",
            {},
            [],
        ),
        (
            "write_tool",
            "Write a new permanent tool for yourself. Define a "
            "TOOLS list of (name, description, {param: type}, "
            "[required]) tuples plus a function per name. It is "
            "tested automatically and only goes live if it "
            "passes — if it fails you get the error back to fix.",
            {"name": "string", "code": "string"},
            ["name", "code"],
        ),
        (
            "public_url",
            "Get (or start) the bot's public https dashboard URL, showing agents, jobs and files.",
            {},
            [],
        ),
        (
            "share_file",
            "Get a public download link for a file. Use this "
            "instead of send_file when the file is over 50 MB.",
            {"path": "string"},
            ["path"],
        ),
        (
            "transcribe_url",
            "Download and transcribe a YouTube (or other) "
            "video or podcast. Returns the full transcript "
            "so you can summarise or quote it.",
            {"url": "string"},
            ["url"],
        ),
        (
            "make_pdf",
            "Produce a real formatted PDF. Content accepts markdown-style headings and bullets.",
            {"filename": "string", "title": "string", "content": "string"},
            ["filename", "title", "content"],
        ),
        (
            "make_spreadsheet",
            "Produce a real .xlsx. sheets is an object of "
            "sheet name -> array of row arrays. Cells "
            "starting with = stay live formulas.",
            {"filename": "string", "sheets": "object"},
            ["filename", "sheets"],
        ),
        (
            "generate_music",
            "Generate a short instrumental clip from a text description.",
            {"prompt": "string", "seconds": "integer"},
            ["prompt"],
        ),
        (
            "consolidate_memory",
            "Distil recent conversation into durable "
            "long-term notes and clear the redundant "
            "fragments. Runs nightly on its own; call it "
            "manually if a lot just happened.",
            {"hours": "integer"},
            [],
        ),
        ("current_time", "Current date and time.", {}, []),
    ]

    def tool_schema():
        return [
            {
                "type": "function",
                "function": {
                    "name": n,
                    "description": d,
                    "parameters": {
                        "type": "object",
                        "required": r,
                        "properties": {k: {"type": v} for k, v in p.items()},
                    },
                },
            }
            for n, d, p, r in (TOOLS + CUSTOM["tools"])
        ]

    VALID = {n: (set(p.keys()), set(r)) for n, d, p, r in TOOLS}

    def check_args(name, args):
        """Drop hallucinated kwargs and report missing required ones, so a
        bad call returns a useful message instead of a TypeError."""
        spec = VALID.get(name)
        if not spec:
            return args, None
        allowed, required = spec
        extra = set(args) - allowed
        clean_args = {k: v for k, v in args.items() if k in allowed}
        missing = required - set(clean_args)
        if missing:
            return clean_args, (
                f"missing required argument(s): "
                f"{', '.join(sorted(missing))}. "
                f"Accepted: {', '.join(sorted(allowed))}"
            )
        if extra:
            print(f"[warn] dropped unknown args for {name}: {extra}")
        return clean_args, None

    def dispatch(name, args, chat_id, outbox):
        t0 = time.time()
        try:
            res = _dispatch(name, args, chat_id, outbox)
            low = str(res).lower()
            ok = not (
                low.startswith("error:")
                or low.startswith("no such")
                or low.startswith("rejected")
                or "failed" in low[:40]
            )
            stat_record(name, ok, (time.time() - t0) * 1000, None if ok else str(res)[:200])
            return res
        except Exception as e:
            stat_record(name, False, (time.time() - t0) * 1000, f"{type(e).__name__}: {e}")
            raise

    def _dispatch(name, args, chat_id, outbox):
        args, problem = check_args(name, args)
        if problem:
            return problem
        try:
            if name == "web_search":
                return t_search(**args)
            if name == "web_fetch":
                return t_fetch(**args)
            if name == "browser_open":
                return browser.call(_b_open, **args)
            if name == "browser_read":
                return browser.call(_b_read)
            if name == "browser_click":
                return browser.call(_b_click, **args)
            if name == "browser_type":
                return browser.call(_b_type, **args)
            if name == "browser_links":
                return browser.call(_b_links)
            if name == "browser_screenshot":
                return browser.call(_b_shot, **args)
            if name == "browser_look":
                shot = browser.call(_b_vision)
                if not str(shot).endswith(".png"):
                    return str(shot)
                q = args.get("question") or "Describe this page in detail."
                return clip(describe_image(shot, q) + f"\n\n(screenshot at {shot})")
            if name == "browser_eval":
                return browser.call(_b_js, **args)
            if name == "browser_back":
                return browser.call(_b_back)
            if name == "run_python":
                return t_python(**args)
            if name == "run_shell":
                return t_shell(**args)
            if name == "run_background":
                return t_bg_start(**args)
            if name == "job_status":
                return t_bg_status(**args)
            if name == "job_list":
                return t_bg_list()
            if name == "job_kill":
                return t_bg_kill(**args)
            if name == "download_file":
                return t_download(**args)
            if name == "send_file":
                return t_send_file(outbox, **args)
            if name == "list_files":
                return t_list_files(**args)
            if name == "index_document":
                return t_index_doc(chat_id, **args)
            if name == "search_documents":
                return t_search_docs(chat_id, **args)
            if name == "remember":
                return t_remember(chat_id, **args)
            if name == "recall":
                return t_recall(chat_id, **args)
            if name == "recall_semantic":
                return t_recall_semantic(chat_id, **args)
            if name == "forget":
                return t_forget(chat_id, **args)
            if name == "cron_add":
                return t_cron_add(chat_id, **args)
            if name == "watch_add":
                return t_watch_add(chat_id, **args)
            if name == "job_schedule_list":
                return t_job_list(chat_id)
            if name == "job_schedule_remove":
                return t_job_remove(chat_id, **args)
            if name == "agent_spawn":
                return t_agent_spawn(chat_id, **args)
            if name == "team_spawn":
                return t_team_spawn(chat_id, **args)
            if name == "agent_status":
                return t_agent_status(**args)
            if name == "agent_list":
                return t_agent_list()
            if name == "email_create_inbox":
                return t_mail_create(chat_id)
            if name == "email_inbox":
                return t_mail_inbox(chat_id, **args)
            if name == "email_read":
                return t_mail_read(chat_id, **args)
            if name == "email_send":
                return t_mail_send(**args)
            if name == "workspace_write":
                return t_ws_write(**args)
            if name == "workspace_read":
                return t_ws_read(**args)
            if name == "workspace_list":
                return t_ws_list(**args)
            if name == "git_snapshot":
                return t_git_snapshot(**args)
            if name == "generate_image":
                return t_image(**args)
            if name == "write_tool":
                return t_write_tool_safe(**args)
            if name == "terminal":
                return t_terminal(**args)
            if name == "python_session":
                return t_python_session(**args)
            if name == "system":
                return t_system(**args)
            if name == "process":
                return t_process(**args)
            if name == "fs":
                return t_fs(**args)
            if name == "packages":
                return t_packages(**args)
            if name == "build":
                return t_build(**args)
            if name == "code":
                return t_code(**args)
            if name == "git":
                return t_git(**args)
            if name == "http":
                return t_http(**args)
            if name == "ssh":
                return t_ssh(**args)
            if name == "sandbox":
                return t_sandbox(**args)
            if name == "container":
                return t_container(**args)
            if name == "tool_rollback":
                return t_tool_rollback(**args)
            if name == "env":
                return t_env(**args)
            if name == "webhook":
                return t_webhook(**args)
            if name == "workflow":
                return t_workflow(**args)
            if name == "media":
                return t_media(**args)
            if name == "data":
                return t_data(**args)
            if name == "crypto":
                return t_crypto(**args)
            if name == "ask_expert":
                if AUTO_REFEREE:
                    lvl, why = assess_stakes(args.get("question", ""))
                    if lvl == "high":
                        print(f"[escalate] ask_expert -> referee ({why})")
                        out = t_referee(
                            args.get("question", ""),
                            context=args.get("context"),
                            task=args.get("task", "reasoning"),
                        )
                        return (
                            "[Escalated automatically: this looked "
                            "high-stakes, so TWO independent models "
                            "were asked and compared. If they "
                            "disagree, tell the user plainly.]\n\n" + str(out)
                        )
                return t_ask_expert(**args)
            if name == "build_tool":
                return t_build_tool(**args)
            if name == "model_report":
                return t_model_report(**args)
            if name == "stakes":
                lvl, why = assess_stakes(args.get("question", ""))
                return f"{lvl} ({why})"
            if name == "referee":
                return t_referee(**args)
            if name == "expert_status":
                return t_expert_status()
            if name == "expert_models":
                return t_expert_models(**args)
            if name == "goal":
                return t_goal(**args)
            if name == "check_sources":
                rep = source_report([args.get("urls", "")])
                if not rep:
                    return "no urls found"
                lines = [
                    f"tier {r['tier']}  {r['domain']}"
                    + ("   ⚠️ aggregator/blog" if r["aggregator"] else "")
                    for r in sorted(rep["sources"], key=lambda x: x["tier"])
                ]
                if rep["weak_only"]:
                    lines.append(
                        "\nAll sources are low quality — find a "
                        "primary source before relying on this."
                    )
                return clip("\n".join(lines))
            if name == "repair":
                return t_repair(**args)
            if name == "services":
                return t_services(**args)
            if name == "models":
                return t_models(**args)
            if name == "tool_health":
                return t_tool_health()
            if name == "learn_rule":
                return rule_add(chat_id, **args)
            if name == "list_rules":
                return rule_list(chat_id)
            if name == "revoke_rule":
                return rule_revoke(chat_id, **args)
            if name == "reflect_now":
                return reflect(chat_id)
            if name == "public_url":
                return t_public_url()
            if name == "share_file":
                return t_share_file(**args)
            if name == "transcribe_url":
                return t_transcribe_url(**args)
            if name == "make_pdf":
                return t_make_pdf(**args)
            if name == "make_spreadsheet":
                return t_make_xlsx(**args)
            if name == "generate_music":
                return t_music(**args)
            if name == "consolidate_memory":
                return consolidate(chat_id, **args)
            if name == "current_time":
                return t_now()
            if name in CUSTOM["fns"]:
                return clip(CUSTOM["fns"][name](**args))
            return f"unknown tool {name}"
        except subprocess.TimeoutExpired:
            return f"timed out after {TOOL_TIMEOUT}s"
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"

    # ------------------------------------------------------- model loop
    ACTIVE = {
        "model": FAST_MODEL,
        "pin": "auto",
        "swaps": 0,
        "last": None,
        "swapping": False,
    }

    def _recover_model(missing):
        """A model vanished mid-session. Switch to any chat model that is
        actually installed rather than dying."""
        try:
            have = [
                m["name"]
                for m in requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json().get("models", [])
            ]
        except Exception:
            return None
        usable = [
            h for h in have if EMBED_MODEL.split(":")[0] not in h and "embed" not in h.lower()
        ]
        if not usable:
            return None
        pick = FAST_MODEL if FAST_MODEL in usable else usable[0]
        print(f"[router] {missing} is missing — recovering with {pick}")
        if missing == SMART_MODEL:
            globals()["SMART_MODEL"] = None
        globals()["FAST_MODEL"] = pick
        ACTIVE["model"] = pick
        ACTIVE["pin"] = "fast"
        return pick

    LLM_GATE = threading.BoundedSemaphore(LLM_PARALLEL)

    # Cheap signals — no extra LLM call, because a classifier round-trip
    # would cost more than it saves.
    HEAVY_WORDS = (
        "research",
        "compare",
        "analyse",
        "analyze",
        "build",
        "write a",
        "debug",
        "fix",
        "plan",
        "investigate",
        "verify",
        "spawn",
        "agent",
        "team",
        "why",
        "how does",
        "explain",
        "design",
        "review",
        "summarise",
        "summarize",
    )
    LIGHT_WORDS = (
        "hi",
        "hey",
        "hello",
        "thanks",
        "ok",
        "cool",
        "yes",
        "no",
        "lol",
        "what's up",
        "you there",
        "good morning",
    )

    def pick_model(user_msg, force=None):
        if not (ROUTER and SMART_MODEL):
            return FAST_MODEL
        if ACTIVE["pin"] == "fast":
            return FAST_MODEL
        if ACTIVE["pin"] == "smart":
            return SMART_MODEL
        if force:
            return SMART_MODEL if force == "smart" else FAST_MODEL
        m = (user_msg or "").strip().lower()
        if len(m) <= 25 and any(m.startswith(w) for w in LIGHT_WORDS):
            return FAST_MODEL
        if len(m) > 120 or any(w in m for w in HEAVY_WORDS):
            return SMART_MODEL
        if "?" in m and len(m) > 60:
            return SMART_MODEL
        return FAST_MODEL

    def set_model(name, note=None):
        if name == ACTIVE["model"]:
            return False
        ACTIVE["swapping"] = True
        ACTIVE["model"] = name
        ACTIVE["swaps"] += 1
        short = name.split("/")[-1]
        if note:
            note(f"🧠 switching to {short}")
        print(f"[router] -> {name} (swap #{ACTIVE['swaps']})")
        return True

    def call_model(messages, use_tools=True, on_delta=None, _retry=0):
        """Serialised through a semaphore: if every slot is busy we wait
        here rather than leaving an HTTP request open until it times out."""
        with LLM_GATE:
            return _call_model(messages, use_tools, on_delta, _retry)

    def _call_model(messages, use_tools=True, on_delta=None, _retry=0):
        payload = {
            "model": ACTIVE["model"],
            "messages": messages,
            "think": False,
            "stream": bool(on_delta),
            "options": {"num_ctx": NUM_CTX, "temperature": 0.78, "top_p": 0.9},
        }
        if use_tools:
            payload["tools"] = tool_schema()

        if not on_delta:
            try:
                r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=LLM_TIMEOUT)
            except requests.exceptions.ReadTimeout:
                if _retry < 1:
                    print("llm timeout, retrying once")
                    time.sleep(5)
                    return _call_model(messages, use_tools, None, _retry + 1)
                raise
            if r.status_code == 400:
                payload.pop("think", None)
                r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=LLM_TIMEOUT)
            if r.status_code == 404:
                alt = _recover_model(payload["model"])
                if alt:
                    payload["model"] = alt
                    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=LLM_TIMEOUT)
            r.raise_for_status()
            ACTIVE["swapping"] = False
            return r.json()["message"]

        # streamed: accumulate content and any tool calls
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat", json=payload, timeout=LLM_TIMEOUT, stream=True
            )
            if r.status_code == 404:
                alt = _recover_model(payload["model"])
                if alt:
                    payload["model"] = alt
                    r = requests.post(
                        f"{OLLAMA_URL}/api/chat",
                        json=payload,
                        timeout=LLM_TIMEOUT,
                        stream=True,
                    )
            if r.status_code == 400:
                payload.pop("think", None)
                r = requests.post(
                    f"{OLLAMA_URL}/api/chat",
                    json=payload,
                    timeout=LLM_TIMEOUT,
                    stream=True,
                )
            r.raise_for_status()
            content, calls = "", []
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                m = d.get("message") or {}
                if m.get("content"):
                    content += m["content"]
                    on_delta(m["content"])
                if m.get("tool_calls"):
                    calls += m["tool_calls"]
                if d.get("done"):
                    break
            ACTIVE["swapping"] = False
            return {"content": content, "tool_calls": calls}
        except Exception as e:
            print("stream failed, falling back:", e)
            return _call_model(messages, use_tools, None, _retry)

    def call_utility(messages, max_tokens=600):
        """Small local model for internal checks. Runs without swapping
        the main model out, so it costs about a second instead of a
        minute."""
        model = UTILITY_MODEL or FAST_MODEL
        try:
            with LLM_GATE:
                r = requests.post(
                    f"{OLLAMA_URL}/api/chat",
                    timeout=180,
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "think": False,
                        "options": {
                            "num_ctx": 8192,
                            "temperature": 0.2,
                            "num_predict": max_tokens,
                        },
                    },
                )
            if r.status_code == 404:
                return call_model(messages, use_tools=False)
            r.raise_for_status()
            return r.json()["message"]
        except RecursionError:
            print("utility model hit recursion — refusing to escalate")
            return {"content": ""}
        except Exception as e:
            print("utility model failed, using main:", e)
            try:
                return call_model(messages, use_tools=False)
            except Exception as e2:
                print("main model also failed:", e2)
                return {"content": ""}

    def clean(text):
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
        if "</think>" in text:
            text = text.split("</think>")[-1]
        return text.strip()

    QUERY_STOP = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "how",
        "what",
        "best",
        "free",
        "get",
        "use",
        "using",
        "can",
        "are",
        "does",
        "available",
        "online",
        "you",
        "your",
        "about",
        "that",
        "this",
        "any",
        "all",
        "some",
    }

    def query_shape(fn, args):
        """Keyword set for near-duplicate detection. Rewording a search
        does not make it a different search."""
        if fn not in ("web_search", "web_fetch", "ask_expert", "referee"):
            return None
        text = " ".join(
            str(v) for k, v in sorted(args.items()) if k in ("query", "question", "url")
        )
        return frozenset(
            w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in QUERY_STOP
        )

    def too_similar(shape, seen, threshold=0.6):
        """Jaccard overlap. 60% shared keywords means the same search."""
        if not shape:
            return False
        for prev in seen:
            if not prev:
                continue
            union = len(shape | prev)
            if union and len(shape & prev) / union >= threshold:
                return True
        return False

    POLLABLE = {
        "agent_status",
        "agent_list",
        "job_status",
        "job_list",
        "browser_read",
        "workspace_list",
        "list_files",
        "recall",
        "email_inbox",
        "current_time",
        "job_schedule_list",
    }

    TOOL_ICON = {
        "web_search": "🔍",
        "web_fetch": "🌐",
        "browser_open": "🧭",
        "browser_click": "👆",
        "browser_type": "⌨️",
        "browser_screenshot": "📸",
        "run_python": "🐍",
        "run_shell": "💻",
        "run_background": "⚙️",
        "download_file": "⬇️",
        "send_file": "📎",
        "recall_semantic": "🧠",
        "search_documents": "📚",
        "cron_add": "⏰",
        "watch_add": "👀",
        "write_tool": "🔧",
    }

    def agent_turn(
        messages,
        chat_id,
        outbox,
        on_progress=None,
        on_delta=None,
        max_steps=None,
        force_model=None,
        no_critique=False,
    ):
        budget = max_steps or MAX_STEPS
        if ROUTER and SMART_MODEL:
            last_user = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            set_model(pick_model(str(last_user), force_model), on_progress)
        failed = set()  # repeats of FAILED calls get blocked
        attempted = set()  # repeats of ANY call get blocked too: three
        # identical successful searches is a loop
        shapes = set()  # and near-duplicates, reworded slightly
        evidence = []  # everything the tools actually returned,
        # used to fact-check the final answer

        def note(s):
            if on_progress:
                on_progress(s)

        # For anything non-trivial, plan once before acting. Cheap, and it
        # stops the model charging at the first idea that occurs to it.
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if AUTO_REFEREE and not no_critique:
            lvl, why = assess_stakes(str(last_user))
            if lvl == "high":
                note("⚖️ high stakes — will cross-check")
                messages.append(
                    {
                        "role": "system",
                        "content": "This question is high-stakes: being wrong would cost "
                        "the user money, health, legal standing, or an "
                        "irreversible decision. Verify with tools rather than "
                        "answering from memory, and use referee (not "
                        "ask_expert) so two independent models are compared. "
                        "If they disagree, say so plainly instead of picking "
                        "one.",
                    }
                )

        if len(str(last_user)) > 120 and budget > 4 and not no_critique:
            try:
                note("🗺 planning my approach")
                plan = clean(
                    call_model(
                        messages
                        + [
                            {
                                "role": "user",
                                "content": "Before acting: in under 60 words, state "
                                "your plan and the single most likely "
                                "reason it might fail. No tools yet.",
                            }
                        ],
                        use_tools=False,
                    ).get("content", "")
                )
                if plan:
                    messages.append({"role": "assistant", "content": f"My plan: {plan}"})
            except Exception as e:
                print("plan step skipped:", e)

        for step_i in range(budget):
            if step_i:
                note(f"↻ step {step_i + 1} of up to {budget}")
            msg = call_model(messages, on_delta=on_delta)
            calls = msg.get("tool_calls") or []
            if not calls:
                ans = clean(msg.get("content", ""))
                if no_critique:
                    return ans
                return critique(messages, ans, chat_id, outbox, note, evidence)
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": calls,
                }
            )
            for c in calls:
                fn = c["function"]["name"]
                args = c["function"].get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                shown = ", ".join(str(v) for v in args.values())[:50]
                note(f"{TOOL_ICON.get(fn, '🔧')} {fn}({shown})")

                sig = f"{fn}:{json.dumps(args, sort_keys=True, default=str)}"
                if fn not in POLLABLE and sig in failed:
                    res = (
                        "You already tried this exact call and it failed. "
                        "Do not repeat it. Either try a genuinely "
                        "different approach or stop and explain to the "
                        "user what is blocking you."
                    )
                shape = query_shape(fn, args)
                if shape and sig not in attempted and too_similar(shape, shapes):
                    res = (
                        "You have already searched for essentially this "
                        "— same keywords, different wording. It "
                        "returned the same kind of result. The problem "
                        "is your search STRATEGY, not the phrasing.\n\n"
                        "Think about what page would actually contain "
                        "the answer, and search for THAT. Name a "
                        "specific site, a specific product, or the "
                        "exact phrase that page would use. If no such "
                        "page exists, say so and move on."
                    )
                elif fn not in POLLABLE and sig in attempted:
                    # succeeded before, but the model is going round in
                    # circles: same call, same result, no progress
                    res = (
                        "You already made this EXACT call in this turn "
                        "and you have its result above. Repeating it "
                        "will return the same thing.\n\n"
                        + (
                            "The results did not answer the question, so "
                            "the QUERY is wrong, not the tool. Change "
                            "the search terms substantially — use "
                            "different words, name a specific site or "
                            "source, or search for the thing itself "
                            "rather than for how to use it.\n"
                            if fn in ("web_search", "web_fetch")
                            else "Try a different approach entirely, or "
                            "tell the user what is blocking you.\n"
                        )
                        + "Do NOT call it again with the same arguments."
                    )
                else:
                    attempted.add(sig)
                    if shape:
                        shapes.add(shape)
                    res = dispatch(fn, args, chat_id, outbox)
                    low = str(res).lower()
                    if (
                        low.startswith("error:")
                        or low.startswith("no such")
                        or "not found" in low
                        or "timed out" in low
                        or "can't operate" in low
                    ):
                        failed.add(sig)
                    if "not been booted with systemd" in low or "failed to connect to bus" in low:
                        res = (
                            str(res) + "\n\nNOTE: this container has no "
                            "systemd and never will. Stop trying to start "
                            "services and tell the user."
                        )
                ok_mark = (
                    "✓" if not str(res).lower().startswith(("error", "no such", "failed")) else "✗"
                )
                note(f"{ok_mark} {fn} done ({len(str(res)):,} chars)")
                print(f"[tool] {fn} {str(args)[:90]} -> {str(res)[:120]}")
                messages.append({"role": "tool", "name": fn, "content": str(res)})
                evidence.append(f"[{fn}] {str(res)[:2500]}")
        messages.append({"role": "user", "content": "Stop using tools and answer now."})
        return clean(call_model(messages, use_tools=False).get("content", ""))

    CRITIQUE_JUNK = re.compile(
        r"^\s*(?:the\s+)?(?:answer|response|previous answer)\s+(?:does|is|"
        r"doesn'?t|was)[^\n]*\n+|^\s*corrected\s+answer\s*:?\s*\n+|"
        r"^\s*(?:revised|improved|updated)\s+answer\s*:?\s*\n+|"
        r"^\s*critique\s*:?[^\n]*\n+",
        re.I,
    )

    def strip_critique_meta(text):
        """The model likes to narrate its own review. That is internal —
        the user should only ever see the answer itself."""
        prev = None
        while prev != text:
            prev = text
            text = CRITIQUE_JUNK.sub("", text).lstrip()
        # if it still explains itself before answering, cut to the answer
        m = re.search(
            r"(?:corrected|revised|here is the corrected)[^\n:]*:"
            r"\s*\n+",
            text,
            re.I,
        )
        if m:
            text = text[m.end() :].lstrip()
        return text.strip()

    SPECIFIC_RE = re.compile(
        r"(?:[£$€]\s?[\d,.]+\s?(?:million|billion|bn|m|k)?"  # money
        r"|\b\d[\d,.]*\s?(?:%|percent|million|billion|thousand)"  # counts
        r"|\b(?:19|20)\d\d\b"  # years
        r"|\b(?:CNN|CNBC|BBC|Reuters|Nature|Axios|NYT|New York Times"
        r"|New Scientist|Guardian|Bloomberg|TechCrunch|Wired|Forbes|WSJ)\b)",
        re.I,
    )

    CURRENCY_RE = re.compile(
        r"\b(?:as of (?:now|today|20\d\d)|currently|at present|as it "
        r"stands|remains (?:unsolved|open|unresolved)|still (?:unsolved|"
        r"open|the case|no)|to date|so far|there is no (?:single|specific|"
        r"known|widely)|has not been (?:solved|announced|released)"
        r"|no one has|nobody has|is the (?:latest|current|newest))\b",
        re.I,
    )

    def unverified_currency(answer, evidence):
        """Present-tense claims about the state of the world, made with
        no tool evidence at all. This is the 'answered from stale
        training data' failure."""
        if evidence:
            return []
        return sorted({m.group(0).strip() for m in CURRENCY_RE.finditer(answer or "")})[:6]

    def unsupported_specifics(answer, evidence):
        """Numbers, dates and outlet names that never appeared in any tool
        result. This is where the fabrication actually happens."""
        if not evidence:
            return []
        hay = " ".join(evidence).lower()
        bad = []
        for m in set(SPECIFIC_RE.findall(answer)):
            token = m.strip().lower()
            core = token.lstrip("£$€ ").rstrip("%")
            if token and token not in hay and core not in hay:
                bad.append(m.strip())
        return bad[:12]

    # ================================================================
    # SOURCE QUALITY + CLAIM ENTAILMENT
    # The specifics check catches invented numbers. These catch the two
    # failures it cannot see: trusting rubbish sources, and stating
    # things the sources never said.
    # ================================================================
    SOURCE_TIERS = [
        # tier 1: primary / authoritative
        (
            1,
            re.compile(
                r"\.(?:gov|gov\.uk|mil)(?:/|$)|\.edu(?:/|$)"
                r"|\.ac\.uk|europa\.eu|who\.int|nature\.com"
                r"|science\.org|sciencedirect|springer|wiley"
                r"|arxiv\.org|pubmed|nih\.gov|ieee\.org|acm\.org"
                r"|jstor|doi\.org|claymath\.org",
                re.I,
            ),
        ),
        # tier 2: established press and official company sources
        (
            2,
            re.compile(
                r"reuters\.com|apnews\.com|bbc\.(?:co\.uk|com)"
                r"|ft\.com|economist\.com|wsj\.com|nytimes\.com"
                r"|washingtonpost|theguardian|cnn\.com|axios\.com"
                r"|bloomberg\.com|npr\.org|aljazeera"
                r"|openai\.com|anthropic\.com|google\.com"
                r"|microsoft\.com|github\.com",
                re.I,
            ),
        ),
        # tier 3: usable but secondary
        (
            3,
            re.compile(
                r"wikipedia\.org|techcrunch|theverge|arstechnica"
                r"|wired\.com|zdnet|stackoverflow|stackexchange"
                r"|news\.ycombinator",
                re.I,
            ),
        ),
    ]
    AGGREGATOR_RE = re.compile(
        r"lowendbox|lowendtalk|slickdeals|dealnews|coupon|top10|bestof"
        r"|listicle|\breview[sz]?site|affiliate|blogspot|medium\.com"
        r"|quora\.com|reddit\.com/r/\w+/comments",
        re.I,
    )

    URL_RE = re.compile(r"https?://[^\s)\]\"'>]+")

    def source_report(evidence):
        """Tier every URL the tools returned. Low-tier-only sourcing is
        the LowEndBox failure: presenting a deals blog as a vendor."""
        urls = []
        for e in evidence or []:
            urls += URL_RE.findall(str(e))
        if not urls:
            return None
        seen, rows = set(), []
        for u in urls:
            dom = re.sub(r"^https?://(?:www\.)?", "", u).split("/")[0]
            if dom in seen:
                continue
            seen.add(dom)
            tier = 4
            for t, rx in SOURCE_TIERS:
                if rx.search(u):
                    tier = t
                    break
            agg = bool(AGGREGATOR_RE.search(u))
            rows.append({"domain": dom, "tier": tier, "aggregator": agg})
        best = min((r["tier"] for r in rows), default=4)
        return {
            "sources": rows,
            "best_tier": best,
            "weak_only": best >= 4,
            "aggregators": [r["domain"] for r in rows if r["aggregator"]],
        }

    def entailment_check(answer, evidence, note=None):
        """Split the answer into claims and test each against the
        evidence. A narrow supported/absent/contradicted judgement is far
        more reliable than open-ended self-review."""
        if not evidence or len(answer) < 200:
            return None
        ev = "\n\n".join(str(e)[:2000] for e in evidence)[:12000]
        try:
            if note:
                note("🧾 checking claims against sources")
            out = clean(
                call_utility(
                    [
                        {
                            "role": "system",
                            "content": "You are a fact-checking function. You will be given "
                            "EVIDENCE (raw tool output) and an ANSWER. Split the "
                            "answer into its individual factual claims. For each "
                            "claim output exactly one line:\n"
                            "SUPPORTED | <claim>\n"
                            "ABSENT | <claim>     (evidence does not mention it)\n"
                            "CONTRADICTED | <claim>   (evidence says otherwise)\n"
                            "Judge only against the evidence given — not your own "
                            "knowledge. Ignore hedged statements and opinions. "
                            "Output nothing else.",
                        },
                        {
                            "role": "user",
                            "content": f"EVIDENCE:\n{ev}\n\nANSWER:\n{answer[:4000]}",
                        },
                    ]
                ).get("content", "")
            )
        except Exception as e:
            print("entailment check failed:", e)
            return None
        bad = {"absent": [], "contradicted": []}
        for line in (out or "").splitlines():
            if "|" not in line:
                continue
            verdict, _, claim = line.partition("|")
            v = verdict.strip().upper()
            c = claim.strip()
            if not c:
                continue
            if v.startswith("CONTRADICT"):
                bad["contradicted"].append(c[:160])
            elif v.startswith("ABSENT"):
                bad["absent"].append(c[:160])
        return bad if (bad["absent"] or bad["contradicted"]) else None

    def critique(messages, answer, chat_id=None, outbox=None, note=None, evidence=None):
        """One review pass. If the review finds unverified claims it gets
        TOOLS, so it can go and check rather than rewriting the same
        guess more carefully."""
        if not SELF_CRITIQUE or len(answer) < CRITIQUE_MIN:
            return answer
        try:
            if note:
                note("🔎 checking my answer")
            flagged = unsupported_specifics(answer, evidence or [])
            stale = unverified_currency(answer, evidence or [])
            srcs = source_report(evidence or [])
            ent = entailment_check(answer, evidence or [], note)
            check = (
                "Review your answer above, critically and briefly.\n"
                "- Does it actually answer what was asked?\n"
                "- Every number, name, date and news outlet you "
                "mentioned: did it appear in a tool result? Remove "
                "any that did not.\n"
                "- Did any tool output get truncated, leaving you "
                "guessing?\n\n"
            )
            if flagged:
                check += (
                    "AUTOMATED CHECK — these appear in your answer "
                    "but NOT in any tool output you received: "
                    + ", ".join(flagged)
                    + ". Either they came from a tool you did not "
                    "call, or you made them up. Remove or verify "
                    "each one.\n\n"
                )
            if stale:
                check += (
                    "STALENESS CHECK — you made present-tense "
                    "claims about the state of the world ("
                    + ", ".join(f"'{x}'" for x in stale)
                    + ") without calling a single tool. Your "
                    "training data is out of date. Search for the "
                    "current position before answering.\n\n"
                )
            if ent and ent.get("contradicted"):
                check += (
                    "CLAIM CHECK — your sources CONTRADICT these "
                    "statements:\n"
                    + "\n".join(f"  - {c}" for c in ent["contradicted"][:6])
                    + "\nFix or remove them.\n\n"
                )
            if ent and ent.get("absent"):
                check += (
                    "CLAIM CHECK — these statements do not appear "
                    "in any source you read:\n"
                    + "\n".join(f"  - {c}" for c in ent["absent"][:6])
                    + "\nEither verify them with a tool or drop "
                    "them.\n\n"
                )
            if srcs and srcs.get("weak_only"):
                check += (
                    "SOURCE CHECK — everything you read came from "
                    "low-quality sources ("
                    + ", ".join(r["domain"] for r in srcs["sources"][:6])
                    + "). Find a primary source — the official "
                    "site, the company's own announcement, the "
                    "paper itself — before presenting this as "
                    "fact.\n\n"
                )
            if srcs and srcs.get("aggregators"):
                check += (
                    "SOURCE CHECK — these are blogs, deal "
                    "aggregators or forums, not primary sources: "
                    + ", ".join(srcs["aggregators"][:5])
                    + ". Do not present them as vendors, "
                    "manufacturers or authorities.\n\n"
                )
            check += (
                "Reply with exactly OK if it is fine.\n"
                "Reply with exactly FETCH if you need tools to "
                "verify or complete it.\n"
                "Otherwise reply with the corrected answer only — "
                "no commentary about the previous answer."
            )
            if flagged and note:
                note(f"⚠️ unverified: {', '.join(flagged[:4])}")
            if stale and note:
                note("⚠️ answered from memory — verifying")
            review = clean(
                call_utility(
                    messages[-6:]
                    + [
                        {"role": "assistant", "content": answer},
                        {"role": "user", "content": check},
                    ]
                ).get("content", "")
            )
        except Exception as e:
            print("critique skipped:", e)
            return answer

        serious = bool(
            flagged
            or stale
            or (ent and ent.get("contradicted"))
            or (srcs and srcs.get("weak_only"))
        )
        if serious and LAST_EXPERT["model"] and evidence:
            if any("[expert:" in str(e) for e in evidence):
                score_model(
                    LAST_EXPERT["model"],
                    LAST_EXPERT["kind"],
                    -0.6,
                    note="answer failed the fact checks",
                )
        if serious and AUTO_REFEREE and evidence:
            used_expert = any("ask_expert" in str(e)[:40] or "[expert:" in str(e) for e in evidence)
            if used_expert and note:
                note("⚖️ expert answer looked shaky — cross-checking")
        verdict = (review or "").strip()
        if not verdict or verdict.upper().startswith("OK"):
            if serious and chat_id is not None:
                # it said OK but the automated check disagrees — verify
                verdict = "FETCH"
            else:
                return answer

        # The important case: it knows it needs to go and look.
        if verdict.upper().startswith("FETCH") and chat_id is not None:
            if note:
                note("🔁 verifying properly")
            try:
                follow = messages + [
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": "Your answer contained unverified or incomplete "
                        "information"
                        + (f" — invented specifics: {', '.join(flagged)}" if flagged else "")
                        + (
                            f" — contradicted by your own sources: "
                            f"{'; '.join((ent or {}).get('contradicted', [])[:3])}"
                            if (ent and ent.get("contradicted"))
                            else ""
                        )
                        + (
                            " — and every source you used was low quality, so find a primary one"
                            if (srcs and srcs.get("weak_only"))
                            else ""
                        )
                        + (
                            " — you answered from memory about something that "
                            "may have changed since your training cutoff; "
                            "SEARCH for the current state now"
                            if stale
                            else ""
                        )
                        + ". Use your tools NOW to check it. If a page was "
                        "truncated, fetch it again with contains='keyword' "
                        "to reach the part you need. If you cannot verify a "
                        "figure or a source, DROP IT from your answer rather "
                        "than keeping it. Then answer the user completely "
                        "and without speculation.",
                    },
                ]
                fixed = agent_turn(
                    follow,
                    chat_id,
                    outbox if outbox is not None else [],
                    on_progress=note,
                    max_steps=6,
                    no_critique=True,
                )
                if fixed and len(fixed) > 40:
                    return strip_critique_meta(fixed)
            except Exception as e:
                print("verification pass failed:", e)
            return answer

        cleaned = strip_critique_meta(verdict)
        if len(cleaned) < len(answer) * 0.4:  # lost content, keep original
            return answer
        return cleaned

    def build_messages(chat_id, user_msg):
        facts = t_recall(chat_id)
        system = PERSONA
        if BOOT_ISSUES:
            system += (
                "\n\nSOMETHING FAILED WHEN YOU STARTED UP:\n"
                + "\n".join(f"- {i['component']}: {i['detail'][:200]}" for i in BOOT_ISSUES)
                + "\nFallbacks are active so nothing is broken for "
                "the user, but you can try to fix it: call repair "
                "action='fix' with the component name. Do this if "
                "the user asks you to, or if a failure is getting "
                "in the way of what they want."
            )
        if LEARNING:
            rules = rules_active(chat_id)
            if rules:
                system += (
                    "\n\nLESSONS YOU HAVE LEARNED (follow these; "
                    "they came from real mistakes):\n" + "\n".join(f"- {t}" for _, t in rules)
                )
            try:
                ex = examples_for(chat_id, user_msg)
                if ex:
                    system += "\n\nHOW YOU HANDLED SIMILAR REQUESTS BEFORE:\n" + "\n".join(
                        f'- when asked "{t[:90]}" you: {a[:220]}' for t, a in ex
                    )
            except Exception as e:
                print("example lookup failed:", e)
        if facts and facts != "(empty)":
            system += f"\n\nWhat you know about this user:\n{facts}"
        try:
            hits = vec_search(chat_id, user_msg, k=4)
            good = [h for h in hits if h[0] > 0.55]
            if good:
                system += "\n\nPossibly relevant from earlier:\n" + "\n".join(
                    f"- {t[:300]}" for _, t, _, _ in good
                )
        except Exception as e:
            print("semantic recall failed:", e)
        rows = q(
            "SELECT role,content FROM turns WHERE chat_id=? ORDER BY id DESC LIMIT ?",
            (chat_id, MAX_TURNS * 2),
            "all",
        )
        convo = [{"role": r, "content": c} for r, c in reversed(rows or [])]
        return (
            [{"role": "system", "content": system}]
            + convo
            + [{"role": "user", "content": user_msg}]
        )

    def save_turn(chat_id, role, content):
        q(
            "INSERT INTO turns (chat_id,role,content) VALUES (?,?,?)",
            (chat_id, role, content),
        )

    # ------------------------------------------------------------ voice
    _whisper = {}

    def transcribe(path):
        if "m" not in _whisper:
            from faster_whisper import WhisperModel

            dev = "cuda" if shutil.which("nvidia-smi") else "cpu"
            _whisper["m"] = WhisperModel(
                WHISPER_SIZE,
                device=dev,
                compute_type="float16" if dev == "cuda" else "int8",
            )
        segs, _ = _whisper["m"].transcribe(path, beam_size=5)
        return " ".join(s.text.strip() for s in segs).strip()

    async def speak(text, out_ogg):
        import edge_tts

        mp3 = out_ogg.replace(".ogg", ".mp3")
        await edge_tts.Communicate(text[:1500], "en-GB-RyanNeural").save(mp3)
        r = subprocess.run(
            f'ffmpeg -y -i "{mp3}" -c:a libopus -b:a 48k "{out_ogg}"',
            shell=True,
            capture_output=True,
        )
        return out_ogg if r.returncode == 0 else mp3

    # ------------------------------------------------- scheduled runner
    async def run_scheduled(job_id, chat_id, kind, payload):
        try:
            if kind == "cron":
                msgs = build_messages(chat_id, payload)
                outbox = []
                reply = await asyncio.to_thread(agent_turn, msgs, chat_id, outbox)
                if reply:
                    await app.bot.send_message(chat_id, f"⏰ {reply[:3800]}")
                for p, cap in outbox:
                    with open(p, "rb") as fh:
                        await app.bot.send_document(
                            chat_id, fh, filename=os.path.basename(p), caption=cap
                        )
            else:
                info = json.loads(payload)
                text = await asyncio.to_thread(t_fetch, info["url"])
                digest = str(hash(text))
                prev = q("SELECT state FROM jobs WHERE id=?", (job_id,), "one")
                if prev and prev[0] and prev[0] != digest:
                    await app.bot.send_message(
                        chat_id, f"👀 {info['url']} changed. {info.get('note', '')}"
                    )
                q("UPDATE jobs SET state=? WHERE id=?", (digest, job_id))
        except Exception:
            traceback.print_exc()

    # ---------------------------------------------------------- handlers
    async def send_long(bot, cid, text):
        for i in range(0, len(text), 4000):
            await bot.send_message(cid, text[i : i + 4000])

    async def respond(update, context, user_msg, spoken=False):
        cid = update.effective_chat.id
        await context.bot.send_chat_action(cid, ChatAction.TYPING)
        outbox, progress = [], []
        buf = {"text": "", "seen_think": False, "stale": False}

        def on_progress(s):
            progress.append(s)
            # next streamed token starts a fresh block
            buf["stale"] = True
            print("[tool]", s)

        def on_delta(chunk):
            # don't show reasoning tokens to the user
            if buf.get("stale"):
                buf["text"] = ""
                buf["stale"] = False
            buf["text"] += chunk
            if "</think>" in buf["text"]:
                buf["text"] = buf["text"].split("</think>")[-1]
                buf["seen_think"] = True

        status = None
        try:
            status = await context.bot.send_message(cid, "…")
        except Exception:
            pass

        msgs = build_messages(cid, user_msg)
        task = asyncio.ensure_future(
            asyncio.to_thread(agent_turn, msgs, cid, outbox, on_progress, on_delta)
        )

        last, t0 = "", time.time()
        stage_t, stage_n = time.time(), 0
        while not task.done():
            await asyncio.sleep(1.3)
            live = buf["text"].strip()
            secs = int(time.time() - t0)
            model_short = ACTIVE["model"].split("/")[-1].split(":")[0]

            # reset the per-stage clock whenever something new happens
            if len(progress) != stage_n:
                stage_n = len(progress)
                stage_t = time.time()
            stage_secs = int(time.time() - stage_t)

            spin = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[(secs * 3 // 4) % 10]

            # Progress ALWAYS shows. Previously any streamed token hid it,
            # so the display froze while tools ran in the background.
            if progress:
                done_lines = progress[:-1][-3:]
                current = progress[-1]
                status_block = (
                    "\n".join(done_lines)
                    + ("\n" if done_lines else "")
                    + f"{spin} {current}  ({stage_secs}s)"
                )
            elif ACTIVE.get("swapping"):
                status_block = (
                    f"{spin} loading {model_short} into VRAM — 30-60s the first time  ({secs}s)"
                )
            elif secs < 8:
                status_block = f"{spin} thinking…"
            else:
                status_block = f"{spin} {model_short} is thinking  ({secs}s)"

            if live:
                # streamed text on top, what it is doing underneath
                body = live[-3000:] + "\n\n─────\n" + status_block + f"  · {secs}s total"
            else:
                bar = "▓" * min(secs // 5, 12) + "░" * max(0, 12 - secs // 5)
                body = f"{status_block}\n\n{bar} {secs}s total"

            if body and body != last and status:
                try:
                    await context.bot.edit_message_text(body, cid, status.message_id)
                    last = body
                except Exception:
                    pass
            # Telegram clears the typing indicator after ~5s
            if secs % 4 == 0:
                try:
                    await context.bot.send_chat_action(cid, ChatAction.TYPING)
                except Exception:
                    pass

        try:
            reply = await task
        except Exception:
            traceback.print_exc()
            reply = ""

        if status:
            try:
                await context.bot.delete_message(cid, status.message_id)
            except Exception:
                pass

        for path, cap in outbox:
            try:
                await context.bot.send_chat_action(cid, ChatAction.UPLOAD_DOCUMENT)
                with open(path, "rb") as fh:
                    await context.bot.send_document(
                        cid, fh, filename=os.path.basename(path), caption=cap
                    )
            except Exception as e:
                await context.bot.send_message(cid, f"couldn't send: {e}")

        if not reply:
            if not outbox:
                await context.bot.send_message(cid, "(brain froze, try again)")
            return

        save_turn(cid, "user", user_msg)
        save_turn(cid, "assistant", reply)
        vec_add(cid, "turn", "chat", f"User: {user_msg}\nYou: {reply}")
        if LEARNING and progress:
            # remember the tool path that satisfied this kind of request
            example_add(cid, user_msg, " → ".join(progress[:8]))

        want_voice = setting(cid, "voice", "0") == "1"
        if ENABLE_VOICE and (want_voice or spoken):
            try:
                os.makedirs(DL_DIR, exist_ok=True)
                ogg = os.path.join(DL_DIR, f"say_{int(time.time())}.ogg")
                path = await speak(reply, ogg)
                with open(path, "rb") as fh:
                    await context.bot.send_voice(cid, fh)
                if len(reply) > 1500:
                    await send_long(context.bot, cid, reply)
                return
            except Exception as e:
                print("tts failed:", e)

        await send_long(context.bot, cid, reply)

    def with_reply_context(update, text):
        """If the user replied to an earlier message, show the model what
        they were pointing at — otherwise the reference is invisible."""
        r = getattr(update.message, "reply_to_message", None)
        if not r:
            return text
        who = "you" if (r.from_user and r.from_user.is_bot) else "the user"
        quoted = (r.text or r.caption or "")[:1500]
        if not quoted:
            return text
        return f'[The user is replying to this earlier message from {who}:]\n"{quoted}"\n\n{text}'

    async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await respond(update, context, with_reply_context(update, update.message.text))

    async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
        cid = update.effective_chat.id
        await context.bot.send_chat_action(cid, ChatAction.TYPING)
        os.makedirs(DL_DIR, exist_ok=True)
        src = os.path.join(DL_DIR, f"vn_{int(time.time())}.ogg")
        tgf = await (update.message.voice or update.message.audio).get_file()
        await tgf.download_to_drive(src)
        try:
            text = await asyncio.to_thread(transcribe, src)
        except Exception as e:
            await context.bot.send_message(cid, f"couldn't transcribe: {e}")
            return
        if not text:
            await context.bot.send_message(cid, "didn't catch that")
            return
        await context.bot.send_message(cid, f"🎤 {text}")
        await respond(update, context, with_reply_context(update, text), spoken=True)

    async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
        cid = update.effective_chat.id
        await context.bot.send_chat_action(cid, ChatAction.TYPING)
        os.makedirs(DL_DIR, exist_ok=True)
        path = os.path.join(DL_DIR, f"img_{int(time.time())}.jpg")
        tgf = await update.message.photo[-1].get_file()
        await tgf.download_to_drive(path)
        caption = update.message.caption or "Describe this image in detail."
        note = await context.bot.send_message(cid, "👁 looking…")
        try:
            desc = await asyncio.to_thread(describe_image, path, caption)
        except Exception as e:
            await context.bot.edit_message_text(f"couldn't read image: {e}", cid, note.message_id)
            return
        try:
            await context.bot.delete_message(cid, note.message_id)
        except Exception:
            pass
        # feed what it saw back into the main model so tools stay available
        await respond(
            update,
            context,
            with_reply_context(update, f"[The user sent an image. It shows: {desc}]\n\n{caption}"),
        )

    async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
        cid = update.effective_chat.id
        doc = update.message.document
        os.makedirs(DL_DIR, exist_ok=True)
        path = os.path.join(DL_DIR, doc.file_name)
        tgf = await doc.get_file()
        await tgf.download_to_drive(path)

        if doc.file_name.endswith(".db"):
            shutil.copy(path, DB_PATH)
            await context.bot.send_message(cid, "memory restored from backup — restart to load it")
            return
        if os.path.splitext(path)[1].lower() in (".pdf", ".txt", ".md", ".csv"):
            res = await asyncio.to_thread(t_index_doc, cid, path)
            await context.bot.send_message(cid, f"📚 {res}")
            return
        await context.bot.send_message(cid, f"saved to {path}")

    async def cmd_reset(update, context):
        q("DELETE FROM turns WHERE chat_id=?", (update.effective_chat.id,))
        await update.message.reply_text("conversation cleared (memory kept)")

    async def cmd_wipe(update, context):
        cid = update.effective_chat.id
        for t in ("turns", "facts", "vectors"):
            q(f"DELETE FROM {t} WHERE chat_id=?", (cid,))
        await update.message.reply_text("everything wiped for this chat")

    async def cmd_voice(update, context):
        cid = update.effective_chat.id
        new = "0" if setting(cid, "voice", "0") == "1" else "1"
        set_setting(cid, "voice", new)
        await update.message.reply_text(f"voice replies {'on' if new == '1' else 'off'}")

    async def cmd_backup(update, context):
        cid = update.effective_chat.id
        with dblock:
            con.commit()
        snap = os.path.join(DL_DIR, f"bot_state_{time.strftime('%Y%m%d')}.db")
        os.makedirs(DL_DIR, exist_ok=True)
        shutil.copy(DB_PATH, snap)
        with open(snap, "rb") as fh:
            await context.bot.send_document(
                cid,
                fh,
                filename=os.path.basename(snap),
                caption="Your memory. Send this file back any time to restore.",
            )

    async def cmd_reload(update, context):
        res = await asyncio.to_thread(load_custom)
        names = ", ".join(t[0] for t in CUSTOM["tools"]) or "none"
        await update.message.reply_text(f"{res}\n{names}")

    async def cmd_model(update, context):
        arg = context.args[0].lower() if context.args else ""
        if arg in ("fast", "smart", "auto"):
            ACTIVE["pin"] = arg
            if arg != "auto":
                set_model(FAST_MODEL if arg == "fast" else (SMART_MODEL or FAST_MODEL))
        await update.message.reply_text(
            f"mode: {ACTIVE['pin']}\n"
            f"loaded: {ACTIVE['model'].split('/')[-1]}\n"
            f"fast:  {FAST_MODEL.split('/')[-1]}\n"
            f"smart: {(SMART_MODEL or 'unavailable').split('/')[-1]}\n"
            f"swaps this session: {ACTIVE['swaps']}\n"
            f"(/model fast | smart | auto)"
        )

    async def cmd_start(update, context):
        await cmd_help(update, context)

    async def cmd_id(update, context):
        await update.message.reply_text(f"chat id: {update.effective_chat.id}")

    async def cmd_digest(update, context):
        cid = update.effective_chat.id
        await update.message.reply_text("consolidating…")
        res = await asyncio.to_thread(consolidate, cid, 24)
        await send_long(context.bot, cid, str(res))

    async def cmd_url(update, context):
        url = await asyncio.to_thread(t_public_url)
        await update.message.reply_text(str(url))

    async def cmd_good(update, context):
        cid = update.effective_chat.id
        t_feedback(cid, "good", " ".join(context.args or []))
        await update.message.reply_text("👍 noted")

    async def cmd_bad(update, context):
        cid = update.effective_chat.id
        note = " ".join(context.args or [])
        t_feedback(cid, "bad", note)
        await update.message.reply_text(
            "👎 noted"
            + (
                " — I'll factor that in"
                if note
                else " (add a few words after /bad to say what was wrong)"
            )
        )

    async def cmd_rules(update, context):
        cid = update.effective_chat.id
        await send_long(context.bot, cid, str(rule_list(cid)))

    async def cmd_reflect(update, context):
        cid = update.effective_chat.id
        await update.message.reply_text("reflecting…")
        res = await asyncio.to_thread(reflect, cid)
        await send_long(context.bot, cid, str(res))

    async def cmd_health(update, context):
        await send_long(context.bot, update.effective_chat.id, str(t_tool_health()))

    COMMAND_HELP = [
        ("model", "fast / smart / auto — switch brain, see swap count"),
        ("reset", "clear this conversation (keeps saved memory)"),
        ("stats", "how much it remembers, db size, active jobs"),
        ("backup", "send me the memory file — do this before you finish"),
        ("digest", "condense today's chat into durable notes"),
        ("bad", "/bad <what was wrong> — it learns from this"),
        ("good", "that answer was right, do more of it"),
        ("reflect", "turn recent feedback into permanent rules"),
        ("rules", "show the lessons it has learned"),
        ("health", "failure rate per tool — use when something keeps breaking"),
        ("url", "dashboard, web chat, live browser view, API token"),
        ("voice", "toggle spoken replies"),
        ("reload", "activate tools it wrote for itself"),
        ("wipe", "delete everything for this chat (careful)"),
        ("id", "show this chat's id"),
        ("goals", "standing goals it works on by itself"),
        ("idle", "turn self-directed work on or off"),
        ("help", "show this list"),
    ]

    async def cmd_help(update, context):
        body = "\n".join(f"/{c} — {d}" for c, d in COMMAND_HELP)
        await update.message.reply_text(
            "Commands:\n\n" + body + "\n\nThe loop worth using: when it gets something wrong send "
            "/bad <reason>. At the end of a session, /reflect then /backup."
        )

    async def cmd_goals(update, context):
        await send_long(context.bot, update.effective_chat.id, str(t_goal("list")))

    async def cmd_idle(update, context):
        globals()["IDLE_ENABLED"] = not IDLE_ENABLED
        await update.message.reply_text(f"self-directed work {'on' if IDLE_ENABLED else 'off'}")

    async def cmd_stats(update, context):
        cid = update.effective_chat.id
        t = q("SELECT COUNT(*) FROM turns WHERE chat_id=?", (cid,), "one")[0]
        f = q("SELECT COUNT(*) FROM facts WHERE chat_id=?", (cid,), "one")[0]
        v = q("SELECT COUNT(*) FROM vectors WHERE chat_id=?", (cid,), "one")[0]
        j = q("SELECT COUNT(*) FROM jobs WHERE chat_id=?", (cid,), "one")[0]
        size = os.path.getsize(DB_PATH) / 1e6 if os.path.exists(DB_PATH) else 0
        live = [
            f"{k}: {a['state']}" for k, a in AGENTS.items() if a["state"] in ("starting", "running")
        ]
        await update.message.reply_text(
            f"{t} turns, {f} facts, {v} memories, {j} scheduled jobs\n"
            f"db {size:.1f} MB\n"
            f"model: {ACTIVE['model'].split('/')[-1]} "
            f"({ACTIVE['pin']}, {ACTIVE['swaps']} swaps)\n"
            f"busy: {', '.join(live) if live else 'nothing running'}"
        )

    OWNER = [None]

    def allowed(cid):
        """First chat to message becomes the owner; after that it's a
        whitelist. Edit allowed_chats.json to add more."""
        try:
            ids = json.load(open(ALLOW_FILE))
        except Exception:
            ids = []
        if not ids:
            json.dump([cid], open(ALLOW_FILE, "w"))
            OWNER[0] = cid
            print(f"owner set to chat {cid}")
            return True
        OWNER[0] = OWNER[0] or ids[0]
        return cid in ids

    async def gate(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_chat and not allowed(update.effective_chat.id):
            raise ApplicationHandlerStop

    app = ApplicationBuilder().token(token).concurrent_updates(True).build()
    app.add_handler(TypeHandler(Update, gate), group=-1)
    for c, h in (
        ("reset", cmd_reset),
        ("wipe", cmd_wipe),
        ("voice", cmd_voice),
        ("backup", cmd_backup),
        ("stats", cmd_stats),
        ("reload", cmd_reload),
        ("model", cmd_model),
        ("id", cmd_id),
        ("digest", cmd_digest),
        ("url", cmd_url),
        ("help", cmd_help),
        ("start", cmd_start),
        ("good", cmd_good),
        ("bad", cmd_bad),
        ("rules", cmd_rules),
        ("goals", cmd_goals),
        ("idle", cmd_idle),
        ("reflect", cmd_reflect),
        ("health", cmd_health),
    ):
        app.add_handler(CommandHandler(c, h))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, on_voice))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    async def idle_loop():
        """Every IDLE_MINUTES, see if a standing goal is due. One at a
        time, never during quiet hours, capped per day."""
        await asyncio.sleep(120)  # let boot settle
        while True:
            try:
                pick = idle_pick()
                if pick:
                    gid, text, success = pick
                    print(f"[idle] working goal #{gid}: {text[:60]}")
                    res = await asyncio.to_thread(run_goal, gid, text, success)
                    print(f"[idle] {res}")
            except Exception as e:
                print("idle loop error:", e)
            await asyncio.sleep(IDLE_MINUTES * 60)

    async def goal_deliverer():
        """Send graded goal output. Separate from the worker so a slow
        send cannot stall the idle loop."""
        while True:
            await asyncio.sleep(10)
            while GOAL_OUTBOX:
                item = GOAL_OUTBOX.pop(0)
                cid = OWNER[0]
                if not cid:
                    continue
                try:
                    await send_long(
                        app.bot,
                        cid,
                        f"🎯 goal #{item['gid']} (self-scored "
                        f"{item['score']}/10)\n\n{item['text']}",
                    )
                    for path, cap in item.get("files") or []:
                        with open(path, "rb") as fh:
                            await app.bot.send_document(
                                cid, fh, filename=os.path.basename(path), caption=cap
                            )
                except Exception as e:
                    print("goal delivery failed:", e)

    async def agent_watcher():
        """Push a sub-agent's result to the chat the moment it finishes."""
        while True:
            await asyncio.sleep(5)
            for aid, a in list(AGENTS.items()):
                if a["state"] in ("done", "failed") and not a.get("sent"):
                    a["sent"] = True
                    try:
                        await send_long(
                            app.bot,
                            a.get("chat_id") or OWNER[0],
                            f"🤖 {aid} finished:\n\n{a['result']}",
                        )
                        for path, cap in a.get("outbox", []):
                            with open(path, "rb") as fh:
                                await app.bot.send_document(
                                    a.get("chat_id") or OWNER[0],
                                    fh,
                                    filename=os.path.basename(path),
                                    caption=cap,
                                )
                    except Exception as e:
                        print("agent delivery failed:", e)

    async def nightly_reflect():
        cid = OWNER[0]
        if not cid:
            return
        try:
            res = await asyncio.to_thread(reflect, cid)
            print("[reflect]", str(res)[:300])
            if str(res).startswith("learned"):
                await app.bot.send_message(cid, f"🧠 {res}")
        except Exception as e:
            print("nightly reflection failed:", e)

    async def nightly():
        cid = OWNER[0]
        if not cid:
            return
        try:
            res = await asyncio.to_thread(consolidate, cid, 24)
            print("[nightly]", str(res)[:200])
        except Exception as e:
            print("nightly consolidation failed:", e)

    async def on_start(_):
        LOOP[0] = asyncio.get_running_loop()
        try:
            if BOOT_ISSUES and OWNER[0]:
                names = ", ".join(sorted({i["component"] for i in BOOT_ISSUES}))
                await app.bot.send_message(
                    OWNER[0],
                    f"⚠️ {names} failed to start. Fallbacks are running so "
                    f"everything still works, just less well.\n"
                    f'Say "fix it" and I\'ll try other install '
                    f"strategies.",
                )
            row = q("SELECT payload FROM jobs WHERE id='env_auto'", (), "one")
            if row and OWNER[0]:
                d = json.loads(row[0])
                n = (
                    len(d.get("declared") or [])
                    + len(d.get("apt_pkgs") or [])
                    + len(d.get("npm_pkgs") or [])
                )
                if n:
                    await app.bot.send_message(
                        OWNER[0],
                        f"♻️ Fresh container. I previously installed {n} "
                        f"packages for you — I recorded them automatically."
                        f'\nSay "restore" and I\'ll put them back.',
                    )
        except Exception as e:
            print("env notice failed:", e)
        try:
            from telegram import BotCommand

            await app.bot.set_my_commands([BotCommand(c, d[:250]) for c, d in COMMAND_HELP])
            print(f"registered {len(COMMAND_HELP)} commands in the Telegram menu")
        except Exception as e:
            print("could not register commands:", e)
        load_custom()
        if ENABLE_TUNNEL:
            try:
                _serve()
                url = await asyncio.to_thread(start_tunnel)
                print("public dashboard:", url)
            except Exception as e:
                print("tunnel failed:", e)
        scheduler.start()
        scheduler.add_job(
            lambda: asyncio.create_task(nightly()),
            CronTrigger.from_crontab("0 4 * * *"),
            id="_nightly_consolidate",
            replace_existing=True,
        )
        if LEARNING:
            scheduler.add_job(
                lambda: asyncio.create_task(nightly_reflect()),
                CronTrigger.from_crontab(REFLECT_CRON),
                id="_nightly_reflect",
                replace_existing=True,
            )
        asyncio.ensure_future(agent_watcher())
        if IDLE_ENABLED:
            asyncio.ensure_future(idle_loop())
            asyncio.ensure_future(goal_deliverer())
            n = q("SELECT COUNT(*) FROM goals WHERE active=1", (), "one")[0]
            print(
                f"idle loop on ({n} standing goals, every "
                f"{IDLE_MINUTES}min, quiet {IDLE_QUIET_H[0]}:00-"
                f"{IDLE_QUIET_H[1]}:00)"
            )
        rows = q("SELECT id,chat_id,kind,spec,payload FROM jobs", (), "all")
        for jid, cid, kind, spec, payload in rows or []:
            try:
                _register(jid, cid, kind, spec, payload)
            except Exception as e:
                print("could not restore job", jid, e)
        print(f"restored {len(rows or [])} scheduled jobs")

    app.post_init = on_start

    try:
        asyncio.get_running_loop()
        import nest_asyncio

        nest_asyncio.apply()
    except RuntimeError:
        pass

    async def on_error(update, context):
        print("handler error:", repr(context.error))

    app.add_error_handler(on_error)

    print(f"live with {len(TOOLS)} tools — message it on Telegram")
    print(
        f"router: fast={FAST_MODEL.split('/')[-1]} "
        f"smart={(SMART_MODEL or 'none').split('/')[-1]} "
        f"utility={(UTILITY_MODEL or 'none').split('/')[-1]}"
    )

    # Free notebooks drop connections. Reconnect instead of dying.
    backoff = 5
    while True:
        try:
            app.run_polling(
                bootstrap_retries=-1,
                poll_interval=1.0,
                timeout=30,
                drop_pending_updates=True,
            )
            break  # clean shutdown, e.g. Ctrl-C
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"polling died ({type(e).__name__}: {e}); retrying in {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)
