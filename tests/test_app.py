from assistant import app, persona


def test_importing_app_does_not_start_the_bot():
    assert callable(app.main)
    assert callable(app.bootstrap)


def test_persona_forbids_leaking_secrets():
    assert "Never print API keys" in persona.PERSONA


def test_persona_does_not_encourage_bypassing_refusals():
    assert "no such restriction" not in persona.PERSONA
