from assistant import config


def test_model_candidates_are_defined():
    assert config.FAST_CANDIDATES
    assert config.SMART_CANDIDATES
    assert config.UTILITY_CANDIDATES
    assert config.FAST_MODEL == config.FAST_CANDIDATES[0]


def test_only_standard_models_are_configured():
    tags = config.FAST_CANDIDATES + config.SMART_CANDIDATES + config.UTILITY_CANDIDATES
    for tag in tags:
        assert "abliterat" not in tag.lower(), tag


def test_limits_are_sane():
    assert config.MAX_STEPS > 0
    assert config.NUM_CTX >= 4096
    assert 0 < config.MAX_RULES <= 100
    assert config.IDLE_MAX_PER_DAY > 0


def test_state_paths_live_under_state_dir():
    for path in (config.DB_PATH, config.ALLOW_FILE, config.WORKSPACE):
        assert path.startswith(config.STATE_DIR)
