from assistant.util import clip


def test_short_text_is_unchanged():
    assert clip("hello", 10) == "hello"


def test_long_text_is_truncated_with_instructions():
    out = clip("x" * 50, 10)
    assert out.startswith("x" * 10)
    assert "TRUNCATED: 50 characters total, 10 shown" in out
    assert "Do NOT guess" in out


def test_non_strings_are_converted():
    assert clip(12345, 10) == "12345"
