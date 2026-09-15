from assistant import installers


def test_clean_env_never_contains_secrets():
    for key in ("TELEGRAM_TOKEN", "KAGGLE_KEY", "KAGGLE_USER_SECRETS_TOKEN"):
        assert key not in installers.CLEAN_ENV_GLOBAL


def test_iso_env_isolates_python_path():
    env = installers.iso_env("/opt/example", extra={"FOO": "bar"})
    assert env["PYTHONPATH"].split(":")[0] == "/opt/example"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["FOO"] == "bar"
    assert "TELEGRAM_TOKEN" not in env


def test_boot_issue_is_recorded_and_truncated(capsys):
    installers.BOOT_ISSUES.clear()
    installers.boot_issue("searx", "x" * 2000)
    assert installers.BOOT_ISSUES[0]["component"] == "searx"
    assert len(installers.BOOT_ISSUES[0]["detail"]) == 600
    installers.BOOT_ISSUES.clear()
