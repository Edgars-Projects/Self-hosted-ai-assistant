# Changelog

## 1.1.0 — 2026-09-15

- Extracted memory, mail, document, crypto and clock tools into `assistant.tools`
- Added a thread-safe `Store` class for database access
- Added end-to-end tests that run the real bot against fake Telegram and Ollama servers
- Added unit tests for every extracted module (36 tests in total)
- Added type hints and mypy checking for all modules outside `app.py`
- Narrowed exception handling where failure types are known; documented intentional catches at tool boundaries

## 1.0.0 — 2026-09-15

- Restructured the single-file bot into the `assistant` package (config, persona, installers, db, secrets, app)
- Switched to standard Ollama Qwen models
- Added unit tests, Ruff linting and GitHub Actions CI
- Added `pyproject.toml`, `requirements.txt` and `run.py`
- Portable default state directory outside Kaggle
- Removed dead code
