"""Configuration defaults.

Everything tunable lives here. Values marked "resolved at boot" are
overwritten by :func:`assistant.app.bootstrap` once models are installed.
"""

import os

# Two brains. The small one answers instantly; the big one is split across
# both GPUs and handles anything that needs judgment. Only one can be
# resident at a time (the 27B needs both cards), so routing is deliberately
# sticky — swapping costs ~30-60s and thrashing would be worse than either
# model alone.
FAST_CANDIDATES = [  # first one that REALLY installs wins
    "qwen3:14b",
    "qwen2.5:14b",
]
FAST_MODEL = FAST_CANDIDATES[0]
SMART_CANDIDATES = [  # first one that pulls wins
    "qwen3:30b-a3b",
    "qwen3:32b",
    "qwq:32b",
]
SMART_MODEL = None  # resolved at boot
MODEL = FAST_MODEL  # default; router overrides per message
ROUTER = True
VISION_MODEL = "qwen2.5vl:7b"  # pulled on first photo; any Ollama vision
# model tag works here
EMBED_MODEL = "nomic-embed-text"
WHISPER_SIZE = "small"  # tiny/base/small/medium — small is the sweet spot
OLLAMA_URL = "http://127.0.0.1:11434"

MAX_TURNS = 12  # recent turns kept verbatim in context
MAX_STEPS = 10  # tool calls per message before forcing an answer
TOOL_TIMEOUT = 900  # blocking shell/python cap; background = no cap
MAX_TOOL_OUT = 4000  # chars of tool output returned to the model
TG_MAX = 49 * 1024 * 1024

ENABLE_BROWSER = True
ENABLE_VOICE = True
IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_TURBO = "turbo" in IMAGE_MODEL.lower()  # drives steps + guidance
IMAGE_GPU = "1"  # keep GPU 0 free for the LLM
MAIL_API = "https://api.mail.tm"  # throwaway inbox, no signup
# Self-hosted services. Both run as separate processes, so the heavy
# lifting lives outside this script and outside the model's context.
SEARX_PORT = 8888  # SearxNG: unlimited free meta-search, no key
SEARX_URL = f"http://127.0.0.1:{SEARX_PORT}"
ENABLE_SEARX = True
ENABLE_CRAWL4AI = True  # LLM-oriented scraper: JS, clean markdown
SEARX_OK = False
CRAWL_OK = False
# Services install into private directories via pip --target, then run
# with PYTHONPATH pointed at them. Two reasons not to use venvs here:
# Kaggle's broken sitecustomize kills ensurepip, and --target resolves
# dependencies fresh instead of fighting the 400 preinstalled packages.
LIB_ROOT = "/opt/botlibs"
CRAWL_LIB = f"{LIB_ROOT}/crawl4ai"
SEARX_LIB = f"{LIB_ROOT}/searx"
AGENT_STEPS = 40  # tool budget for a sub-agent
# Ollama serialises requests by default, so parallel agents queue behind
# each other and time out. Let it handle several at once, and hold a
# client-side semaphore so waiting happens locally instead of in an open
# HTTP request that eventually dies.
LLM_PARALLEL = 3  # concurrent generations Ollama will accept
MAX_TEAM = 3  # sub-agents per team
LLM_TIMEOUT = 900
NUM_CTX = 16384  # 71 tool schemas eat a lot before you even start
ENABLE_TUNNEL = True  # public URL via cloudflared
SELF_CRITIQUE = True  # one revision pass on long answers
CRITIQUE_MIN = 400  # only critique answers this long
SERVE_PORT = 8765

# --- self-improvement limits. These are the safety envelope: learning is
# additive only, capped, versioned, and always reversible.
LEARNING = True
MAX_RULES = 25  # hard cap on learned behavioural rules
MAX_RULE_LEN = 240  # a rule is a sentence, not a new persona
EXAMPLES_IN_CTX = 3  # past successes injected per message
REFLECT_CRON = "30 4 * * *"
# Self-direction. Deliberately conservative: an unsupervised model with
# open-ended goals produces expensive noise, so goals must be narrow,
# the idle loop is slow, and output is graded before it reaches you.
IDLE_ENABLED = True
IDLE_MINUTES = 45  # how often it looks for something to do
IDLE_QUIET_H = (23, 7)  # do not message during these hours (local)
IDLE_MAX_PER_DAY = 6  # hard cap on self-initiated work
GOAL_MIN_SCORE = 6  # out of 10; below this you are not told

# ---- borrowed brains -------------------------------------------------
# The local model stays the front-end: it talks to you and holds
# the tools. When it needs reasoning beyond its
# weight class it CONSULTS a frontier model as a tool and integrates the
# answer itself. All of these have free tiers.
# Add whichever keys you have as secrets; it uses the first that works.
EXPERTS = [
    # name, key-name, chat endpoint, catalogue endpoint, auth style
    (
        "openrouter",
        "OPENROUTER_KEY",
        "https://openrouter.ai/api/v1/chat/completions",
        "https://openrouter.ai/api/v1/models",
        "bearer",
    ),
    (
        "groq",
        "GROQ_KEY",
        "https://api.groq.com/openai/v1/chat/completions",
        "https://api.groq.com/openai/v1/models",
        "bearer",
    ),
    (
        "cerebras",
        "CEREBRAS_KEY",
        "https://api.cerebras.ai/v1/chat/completions",
        "https://api.cerebras.ai/v1/models",
        "bearer",
    ),
    (
        "gemini",
        "GEMINI_KEY",
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models",
        "query",
    ),
]
EXPERT_TIMEOUT = 120

# ---- utility brain ---------------------------------------------------
# The internal checks (critique, entailment, grading, planning) are simple
# classification jobs, but they were running on the main model at 30-60s
# each — the real reason replies felt slow. A tiny local model does them
# in about a second.
UTILITY_CANDIDATES = [
    "qwen3:4b",
    "qwen3:1.7b",
    "qwen3:14b",  # fall back to fast
]
UTILITY_MODEL = None
USE_UTILITY = True

# ---- referee ---------------------------------------------------------
# Two independent frontier models on the same question. Agreement is
# reassuring; DISAGREEMENT is the strongest unreliability signal we have,
# because it is two outside models rather than one grading itself.
REFEREE = True
AUTO_REFEREE = True  # escalate high-stakes questions without being asked
# OpenRouter's free line-up changes constantly, so the live catalogue is
# fetched rather than hardcoded. These are preference hints, matched as
# substrings against whatever is actually free at the time.
# Substring hints, biggest/most-capable first. The free line-up churns,
# so these are preferences rather than requirements — anything unmatched
# falls back to the largest-context text model available.
TASK_HINTS = {
    "reasoning": [
        "ultra",
        "-550b",
        "reasoning",
        "-120b",
        "nemotron-3-super",
        "inkling",
        "nex-n2.5-pro",
        "ling-3.0",
        "qwq",
        "deepseek-r1",
        "-70b",
        "gemma-4-31b",
    ],
    "coding": [
        "code",
        "laguna",
        "coder",
        "devstral",
        "codestral",
        "nex-n2.5-pro",
        "ultra",
        "-120b",
        "gemma-4",
    ],
    "long": ["ultra", "-120b", "gemma-4-31b", "inkling", "kimi", "gemini", "ling-3.0"],
    "general": [
        "nemotron-3-super",
        "gemma-4-31b",
        "inkling",
        "ling-3.0",
        "ultra",
        "-70b",
        "nex-n2.5",
    ],
}
# Not chat models: music/audio/video generators, embedding models, and
# safety classifiers. Sending a question to these returns nonsense.
NON_CHAT = (
    "lyria",
    "whisper",
    "tts",
    "audio",
    "music",
    "embed",
    "rerank",
    "content-safety",
    "guard",
    "moderation",
    "clip",
    "image",
    "video",
    "veo",
    "imagen",
    "dall-e",
    "flux",
    "sdxl",
    "stable-diffusion",
)

# Kaggle keeps /kaggle/working between sessions; elsewhere use the home dir.
STATE_DIR = os.environ.get(
    "BOT_STATE",
    "/kaggle/working"
    if os.path.isdir("/kaggle/working")
    else os.path.join(os.path.expanduser("~"), ".assistant"),
)
DB_PATH = os.path.join(STATE_DIR, "bot_state.db")
PROFILE_DIR = os.path.join(STATE_DIR, "browser_profile")
TOOLS_DIR = os.path.join(STATE_DIR, "custom_tools")
WORKSPACE = os.path.join(STATE_DIR, "workspace")
ALLOW_FILE = os.path.join(STATE_DIR, "allowed_chats.json")
JOB_DIR = "/tmp/botjobs"
DL_DIR = "/tmp/downloads"
BROWSER_TIMEOUT = 180
