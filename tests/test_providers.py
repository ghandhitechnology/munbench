import asyncio
from types import SimpleNamespace

import pytest

import munbench.providers as providers
from munbench.config import Settings

SETTINGS = Settings(temperature=0.7, max_tokens=64, cli_concurrency=2, max_retries=0)


# --------------------------------------------------------------------------
# Message rendering
# --------------------------------------------------------------------------


def test_split_system_message_present():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    system, rest = providers.split_system_message(messages)
    assert system == "sys"
    assert rest == [{"role": "user", "content": "hi"}]


def test_split_system_message_absent():
    messages = [{"role": "user", "content": "hi"}]
    system, rest = providers.split_system_message(messages)
    assert system is None
    assert rest == messages


def test_render_prompt_empty():
    assert providers.render_prompt([]) == ""


def test_render_prompt_single_user_message_passthrough():
    messages = [{"role": "user", "content": "이 프롬프트 그대로"}]
    assert providers.render_prompt(messages) == "이 프롬프트 그대로"


def test_render_prompt_multiturn_transcript_is_labeled():
    messages = [
        {"role": "user", "content": "첫 질문"},
        {"role": "assistant", "content": "첫 답변"},
        {"role": "user", "content": "둘째 질문"},
    ]
    rendered = providers.render_prompt(messages)
    assert "[대화 기록]" in rendered
    assert "user: 첫 질문" in rendered
    assert "assistant: 첫 답변" in rendered
    assert "user: 둘째 질문" in rendered
    # instructs the CLI to continue as only the assistant's next line
    assert "assistant" in rendered.splitlines()[-1]


def test_is_cli_model():
    assert providers.is_cli_model("claude-cli/claude-opus-4-6")
    assert providers.is_cli_model("codex-cli/gpt-5.6-sol")
    assert not providers.is_cli_model("openrouter/openai/gpt-5")
    assert not providers.is_cli_model("gpt-5")


# --------------------------------------------------------------------------
# argv construction
# --------------------------------------------------------------------------


def test_build_claude_cli_argv_no_system():
    argv = providers.build_claude_cli_argv("claude-opus-4-6", "hello", None)
    assert argv == ["claude", "-p", "hello", "--model", "claude-opus-4-6", "--output-format", "json", "--tools", ""]


def test_build_claude_cli_argv_with_system():
    argv = providers.build_claude_cli_argv("claude-opus-4-6", "hello", "be nice")
    assert "--append-system-prompt" in argv
    assert argv[argv.index("--append-system-prompt") + 1] == "be nice"


def test_build_codex_cli_argv_flags():
    argv = providers.build_codex_cli_argv("gpt-5.6-sol", "hello", "/tmp/out.txt")
    assert argv[:2] == ["codex", "exec"]
    assert "--sandbox" in argv and argv[argv.index("--sandbox") + 1] == "read-only"
    assert "--ask-for-approval" in argv and argv[argv.index("--ask-for-approval") + 1] == "never"
    assert "--skip-git-repo-check" in argv
    assert "--output-last-message" in argv and argv[argv.index("--output-last-message") + 1] == "/tmp/out.txt"
    assert argv[-1] == "hello"


# --------------------------------------------------------------------------
# Claude JSON envelope parsing
# --------------------------------------------------------------------------


def test_parse_claude_cli_output_extracts_result():
    stdout = '{"is_error": false, "result": "생성된 텍스트", "type": "result"}'
    assert providers.parse_claude_cli_output(stdout) == "생성된 텍스트"


def test_parse_claude_cli_output_raises_on_is_error():
    stdout = '{"is_error": true, "result": "model not found"}'
    with pytest.raises(providers.CliBackendError, match="model not found"):
        providers.parse_claude_cli_output(stdout)


def test_parse_claude_cli_output_raises_on_malformed_json():
    with pytest.raises(providers.CliBackendError):
        providers.parse_claude_cli_output("not json at all")


def test_parse_claude_cli_output_raises_on_missing_result_field():
    with pytest.raises(providers.CliBackendError):
        providers.parse_claude_cli_output('{"is_error": false}')


# --------------------------------------------------------------------------
# Subprocess dispatch (mocked asyncio.create_subprocess_exec)
# --------------------------------------------------------------------------


class FakeProcess:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0, hang: bool = False):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self.killed = False

    async def communicate(self):
        if self._hang:
            await asyncio.sleep(100)
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


async def test_run_subprocess_success(monkeypatch):
    async def fake_exec(*args, **kwargs):
        return FakeProcess(stdout=b"hello out", returncode=0)

    monkeypatch.setattr(providers.asyncio, "create_subprocess_exec", fake_exec)
    out = await providers._run_subprocess(["echo", "hi"], timeout=5)
    assert out == "hello out"


async def test_run_subprocess_nonzero_exit_raises(monkeypatch):
    async def fake_exec(*args, **kwargs):
        return FakeProcess(stderr=b"boom", returncode=1)

    monkeypatch.setattr(providers.asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(providers.CliBackendError, match="exited 1"):
        await providers._run_subprocess(["fail"], timeout=5)


async def test_run_subprocess_timeout_raises(monkeypatch):
    async def fake_exec(*args, **kwargs):
        return FakeProcess(hang=True)

    monkeypatch.setattr(providers.asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(providers.CliBackendError, match="timed out"):
        await providers._run_subprocess(["stuck"], timeout=0.05)


async def test_run_subprocess_missing_binary_raises(monkeypatch):
    async def fake_exec(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(providers.asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(providers.CliBackendError, match="not found"):
        await providers._run_subprocess(["nope"], timeout=5)


# --------------------------------------------------------------------------
# complete() dispatch, end to end per backend (mocked)
# --------------------------------------------------------------------------


async def test_complete_dispatches_to_claude_cli(monkeypatch):
    monkeypatch.setattr(providers.shutil, "which", lambda name: "/usr/local/bin/claude" if name == "claude" else None)

    captured_argv = {}

    async def fake_exec(*args, **kwargs):
        captured_argv["argv"] = list(args)
        envelope = '{"is_error": false, "result": "안녕하세요"}'
        return FakeProcess(stdout=envelope.encode("utf-8"), returncode=0)

    monkeypatch.setattr(providers.asyncio, "create_subprocess_exec", fake_exec)

    messages = [{"role": "system", "content": "친절하게 답하라"}, {"role": "user", "content": "인사해줘"}]
    result = await providers.complete("claude-cli/claude-opus-4-6", messages, SETTINGS)

    assert result == "안녕하세요"
    argv = captured_argv["argv"]
    assert argv[0] == "claude"
    assert "--model" in argv and argv[argv.index("--model") + 1] == "claude-opus-4-6"
    assert "--append-system-prompt" in argv


async def test_complete_dispatches_to_codex_cli(monkeypatch):
    monkeypatch.setattr(providers.shutil, "which", lambda name: "/usr/local/bin/codex" if name == "codex" else None)

    async def fake_exec(*args, **kwargs):
        argv = list(args)
        out_path = argv[argv.index("--output-last-message") + 1]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("codex의 응답")
        return FakeProcess(returncode=0)

    monkeypatch.setattr(providers.asyncio, "create_subprocess_exec", fake_exec)

    messages = [{"role": "user", "content": "코드리뷰 없이 그냥 텍스트만 생성해줘"}]
    result = await providers.complete("codex-cli/gpt-5.6-sol", messages, SETTINGS)
    assert result == "codex의 응답"


async def test_complete_cli_backend_nonzero_exit_raises(monkeypatch):
    monkeypatch.setattr(providers.shutil, "which", lambda name: "/usr/local/bin/claude")

    async def fake_exec(*args, **kwargs):
        return FakeProcess(stderr=b"auth expired", returncode=1)

    monkeypatch.setattr(providers.asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(providers.CliBackendError):
        await providers.complete("claude-cli/claude-opus-4-6", [{"role": "user", "content": "hi"}], SETTINGS)


async def test_complete_missing_binary_raises_helpful_error(monkeypatch):
    monkeypatch.setattr(providers.shutil, "which", lambda name: None)
    with pytest.raises(providers.CliBackendError, match="claude login"):
        await providers.complete("claude-cli/claude-opus-4-6", [{"role": "user", "content": "hi"}], SETTINGS)


async def test_complete_json_mode_appends_instruction_for_cli(monkeypatch):
    monkeypatch.setattr(providers.shutil, "which", lambda name: "/usr/local/bin/claude")
    captured = {}

    async def fake_exec(*args, **kwargs):
        captured["prompt"] = list(args)[2]  # ["claude", "-p", <prompt>, ...]
        return FakeProcess(stdout=b'{"is_error": false, "result": "{}"}', returncode=0)

    monkeypatch.setattr(providers.asyncio, "create_subprocess_exec", fake_exec)
    await providers.complete(
        "claude-cli/claude-opus-4-6", [{"role": "user", "content": "채점해줘"}], SETTINGS, json_mode=True
    )
    assert "JSON" in captured["prompt"]


# --------------------------------------------------------------------------
# Regression: plain (non-CLI) model ids still hit the litellm path
# --------------------------------------------------------------------------


async def test_complete_non_cli_model_uses_litellm(monkeypatch):
    calls = {"n": 0}

    async def fake_acompletion(**kwargs):
        calls["n"] += 1
        assert kwargs["model"] == "gpt-5"
        message = SimpleNamespace(content="litellm 경로 응답")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(providers.litellm, "acompletion", fake_acompletion)

    # subprocess must never be touched for a non-CLI model id
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess should not be invoked for a non-CLI model")

    monkeypatch.setattr(providers.asyncio, "create_subprocess_exec", fail_if_called)

    result = await providers.complete("gpt-5", [{"role": "user", "content": "hi"}], SETTINGS)
    assert result == "litellm 경로 응답"
    assert calls["n"] == 1


async def test_complete_litellm_json_mode_sets_response_format(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content="{}")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(providers.litellm, "acompletion", fake_acompletion)
    await providers.complete("gpt-5", [{"role": "user", "content": "hi"}], SETTINGS, json_mode=True, temperature=0.0)
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0.0


# --------------------------------------------------------------------------
# Pre-flight availability probes
# --------------------------------------------------------------------------


async def test_check_cli_backend_missing_binary(monkeypatch):
    monkeypatch.setattr(providers.shutil, "which", lambda name: None)
    error = await providers.check_cli_backend("claude-cli/claude-opus-4-6")
    assert error is not None
    assert "not found" in error


async def test_check_cli_backend_ok(monkeypatch):
    monkeypatch.setattr(providers.shutil, "which", lambda name: "/usr/local/bin/claude")

    async def fake_exec(*args, **kwargs):
        return FakeProcess(stdout=b"2.1.218", returncode=0)

    monkeypatch.setattr(providers.asyncio, "create_subprocess_exec", fake_exec)
    error = await providers.check_cli_backend("claude-cli/claude-opus-4-6")
    assert error is None


async def test_check_cli_backend_non_cli_model_returns_none():
    assert await providers.check_cli_backend("gpt-5") is None
