import pytest

from assistant.secrets import get_secret


def test_environment_variable_is_used(monkeypatch):
    monkeypatch.setenv("EXAMPLE_SECRET", "abc123")
    assert get_secret("EXAMPLE_SECRET") == "abc123"


def test_missing_secret_raises_helpful_error(monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    with pytest.raises(RuntimeError, match="DEFINITELY_NOT_SET"):
        get_secret("DEFINITELY_NOT_SET")
