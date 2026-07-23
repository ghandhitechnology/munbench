import pytest

import munbench.cli as cli_mod


async def test_check_env_async_passes_when_all_ok(monkeypatch):
    monkeypatch.setattr(cli_mod.litellm, "validate_environment", lambda m: {"keys_in_environment": True})

    async def fake_cli_check(model):
        return None

    monkeypatch.setattr(cli_mod.providers, "check_cli_backend", fake_cli_check)
    await cli_mod._check_env_async(["gpt-5", "claude-cli/claude-opus-4-6"])  # should not raise


async def test_check_env_async_reports_missing_api_key(monkeypatch):
    monkeypatch.setattr(
        cli_mod.litellm,
        "validate_environment",
        lambda m: {"keys_in_environment": False, "missing_keys": ["OPENAI_API_KEY"]},
    )

    with pytest.raises(cli_mod.EnvCheckError) as exc_info:
        await cli_mod._check_env_async(["gpt-5"])
    assert any("OPENAI_API_KEY" in line for line in exc_info.value.lines)


async def test_check_env_async_reports_missing_cli(monkeypatch):
    async def fake_cli_check(model):
        return "not found on PATH. Install and run `claude login`."

    monkeypatch.setattr(cli_mod.providers, "check_cli_backend", fake_cli_check)
    monkeypatch.setattr(cli_mod.litellm, "validate_environment", lambda m: {"keys_in_environment": True})

    with pytest.raises(cli_mod.EnvCheckError) as exc_info:
        await cli_mod._check_env_async(["claude-cli/claude-opus-4-6"])
    assert any("claude login" in line for line in exc_info.value.lines)


async def test_check_env_async_skips_litellm_check_for_cli_models(monkeypatch):
    calls = {"n": 0}

    def fake_validate(m):
        calls["n"] += 1
        return {"keys_in_environment": True}

    async def fake_cli_check(model):
        return None

    monkeypatch.setattr(cli_mod.litellm, "validate_environment", fake_validate)
    monkeypatch.setattr(cli_mod.providers, "check_cli_backend", fake_cli_check)
    await cli_mod._check_env_async(["claude-cli/claude-opus-4-6", "codex-cli/gpt-5.6-sol"])
    assert calls["n"] == 0
