import warnings

import pytest
import typer

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


def test_run_stage_never_builds_coroutine_when_env_check_fails(monkeypatch):
    # Fix: the stage coroutine used to be created eagerly (as a function-call
    # argument) before the env check ran, leaving it unawaited - and a
    # RuntimeWarning - whenever the check failed. `_run_stage` must take a thunk and
    # only invoke it after the check passes.
    monkeypatch.setattr(
        cli_mod.litellm,
        "validate_environment",
        lambda m: {"keys_in_environment": False, "missing_keys": ["SOME_API_KEY"]},
    )

    factory_called = {"n": 0}

    def stage_factory():
        factory_called["n"] += 1

        async def _coro():
            return "should never run"

        return _coro()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # any "never awaited" warning fails the test
        with pytest.raises(typer.Exit):
            cli_mod._run_stage(["some-model"], stage_factory)

    assert factory_called["n"] == 0


def test_run_stage_calls_factory_and_returns_result_when_env_check_passes(monkeypatch):
    monkeypatch.setattr(cli_mod.litellm, "validate_environment", lambda m: {"keys_in_environment": True})

    async def fake_stage():
        return "ok"

    result = cli_mod._run_stage(["some-model"], fake_stage)
    assert result == "ok"
