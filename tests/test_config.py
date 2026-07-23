from munbench.config import DEFAULT_JUDGES, Settings


def test_default_judges_are_openrouter_prefixed():
    # Fix: the default `judges` fallback used to require 3 separate provider API
    # keys (OPENAI/GEMINI/ANTHROPIC), contradicting the single-key OpenRouter
    # quickstart. All defaults must work with only OPENROUTER_API_KEY set.
    assert len(DEFAULT_JUDGES) == 3
    for judge in DEFAULT_JUDGES:
        assert judge.startswith("openrouter/"), judge


def test_settings_uses_default_judges_when_omitted():
    settings = Settings(models=["openrouter/openai/gpt-5"])
    assert settings.judges == DEFAULT_JUDGES


def test_settings_cli_concurrency_default():
    assert Settings().cli_concurrency == 2
