"""Unified model-calling entrypoint.

Every place in the harness that needs a model completion (generation and both
judge passes) goes through `complete()`. It dispatches on the model-id prefix:

- `claude-cli/<model-id>`  -> runs Claude Code headless (`claude -p ...`), billed
  against the user's Claude Pro/Max subscription instead of the metered API.
- `codex-cli/<model-id>`   -> runs Codex CLI headless (`codex exec ...`), billed
  against the user's ChatGPT subscription instead of the metered API.
- anything else            -> unchanged litellm `acompletion` path (API keys).

Both CLI backends only accept a single prompt string, not a structured chat
messages list, so multi-turn transcripts (track 1 roleplay, judge prompts) are
flattened by `render_prompt()`. Retries live in the callers (generate.py,
judge_rubric.py, judge_pairwise.py) exactly as they did for the litellm-only
path; `complete()` itself makes one attempt and raises on failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile

import litellm

from munbench.config import Settings

logger = logging.getLogger(__name__)

Message = dict[str, str]

CLAUDE_CLI_PREFIX = "claude-cli/"
CODEX_CLI_PREFIX = "codex-cli/"

# Generous but finite: subscription CLI sessions can be slow, but must never hang
# the harness forever (a stuck subprocess would otherwise stall a whole run).
CLI_TIMEOUT_SECONDS = 300.0
# Short timeout for the one-off auth/availability probe run by `munbench`'s
# pre-flight env check (cli.py) - should fail fast with a clear message rather
# than hang the CLI on a broken/unauthenticated install.
CLI_PROBE_TIMEOUT_SECONDS = 20.0

_JSON_ONLY_INSTRUCTION_KO = (
    "\n\n반드시 JSON 객체만 응답하세요. 코드 블록이나 설명 없이 JSON 텍스트만 출력하세요."
)


class CliBackendError(RuntimeError):
    """A claude-cli/ or codex-cli/ subprocess failed, timed out, or returned
    unparseable output. Callers' existing retry/error-record machinery handles this
    exactly like a litellm exception."""


def is_cli_model(model: str) -> bool:
    return model.startswith(CLAUDE_CLI_PREFIX) or model.startswith(CODEX_CLI_PREFIX)


# --------------------------------------------------------------------------
# Message -> single-prompt rendering (shared by both CLI backends)
# --------------------------------------------------------------------------


def split_system_message(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Pulls a leading system message out, if present. Returns (system, rest)."""
    if messages and messages[0].get("role") == "system":
        return messages[0]["content"], messages[1:]
    return None, messages


def render_prompt(messages: list[Message]) -> str:
    """Render a chat messages list (system already split out via split_system_message)
    into a single prompt string, since CLI backends take one prompt, not a messages list.

    - A single user message (track 2/3 generation, judge prompts) passes through verbatim.
    - A multi-turn transcript (track 1 roleplay) is rendered as a labeled dialogue log
      with an instruction to continue as only the assistant's next line.
    """
    if not messages:
        return ""
    if len(messages) == 1 and messages[0].get("role") == "user":
        return messages[0]["content"]

    lines = ["[대화 기록]"]
    for m in messages:
        speaker = "user" if m.get("role") == "user" else "assistant"
        lines.append(f"{speaker}: {m['content']}")
    lines.append("")
    lines.append(
        "위 대화 기록에 이어서 assistant의 다음 메시지만 작성하세요. "
        "설명, 서두, 따옴표 없이 응답 내용만 출력하세요."
    )
    return "\n".join(lines)


def _build_cli_prompt(messages: list[Message], json_mode: bool) -> tuple[str | None, str]:
    system, rest = split_system_message(messages)
    prompt = render_prompt(rest)
    if json_mode:
        prompt += _JSON_ONLY_INSTRUCTION_KO
    return system, prompt


# --------------------------------------------------------------------------
# Subprocess plumbing
# --------------------------------------------------------------------------

# id(settings) -> semaphore. Settings is loaded once per CLI invocation and threaded
# through everywhere, so keying on identity is enough to give each run its own cap
# without needing a mutable slot on the (otherwise plain) pydantic Settings model.
_cli_semaphores: dict[int, asyncio.Semaphore] = {}


def _cli_semaphore(settings: Settings) -> asyncio.Semaphore:
    key = id(settings)
    sem = _cli_semaphores.get(key)
    if sem is None:
        sem = asyncio.Semaphore(settings.cli_concurrency)
        _cli_semaphores[key] = sem
    return sem


async def _run_subprocess(argv: list[str], *, timeout: float) -> str:
    """Run argv with stdin closed, capturing stdout/stderr. Raises CliBackendError on
    missing binary, non-zero exit, or timeout."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise CliBackendError(f"command not found: {argv[0]}") from e

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as e:
        proc.kill()
        await proc.wait()
        raise CliBackendError(f"{argv[0]} timed out after {timeout:.0f}s") from e

    if proc.returncode != 0:
        stderr_text = stderr.decode("utf-8", "replace").strip()[:500]
        raise CliBackendError(f"{argv[0]} exited {proc.returncode}: {stderr_text}")

    return stdout.decode("utf-8", "replace")


# --------------------------------------------------------------------------
# claude-cli/ backend — Claude Code headless (`claude -p`)
# --------------------------------------------------------------------------


def build_claude_cli_argv(model_id: str, prompt: str, system: str | None) -> list[str]:
    # -p/--print: non-interactive, print-and-exit.
    # --output-format json: single JSON result envelope with a "result" field.
    # --tools "": disable all tool use (per `claude --help`) so this behaves as pure
    #   text generation, not an agentic coding session.
    argv = ["claude", "-p", prompt, "--model", model_id, "--output-format", "json", "--tools", ""]
    if system:
        argv += ["--append-system-prompt", system]
    return argv


def parse_claude_cli_output(stdout: str) -> str:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise CliBackendError(f"claude CLI returned unparseable JSON: {e}; output: {stdout[:300]!r}") from e
    if not isinstance(data, dict):
        raise CliBackendError(f"claude CLI JSON envelope was not an object: {type(data).__name__}")
    if data.get("is_error"):
        raise CliBackendError(f"claude CLI reported an error: {str(data.get('result', ''))[:300]}")
    result = data.get("result")
    if not isinstance(result, str):
        raise CliBackendError(f"claude CLI JSON envelope missing a 'result' string field: {stdout[:300]!r}")
    return result


async def _complete_claude_cli(model_id: str, messages: list[Message], settings: Settings, json_mode: bool) -> str:
    if shutil.which("claude") is None:
        raise CliBackendError(
            "claude CLI not found on PATH. Install with `npm i -g @anthropic-ai/claude-code`, then run `claude login`."
        )
    system, prompt = _build_cli_prompt(messages, json_mode)
    argv = build_claude_cli_argv(model_id, prompt, system)
    async with _cli_semaphore(settings):
        stdout = await _run_subprocess(argv, timeout=CLI_TIMEOUT_SECONDS)
    return parse_claude_cli_output(stdout)


async def _probe_claude_cli() -> str | None:
    if shutil.which("claude") is None:
        return "not found on PATH. Install with `npm i -g @anthropic-ai/claude-code`, then run `claude login`."
    try:
        await _run_subprocess(["claude", "--version"], timeout=CLI_PROBE_TIMEOUT_SECONDS)
    except CliBackendError as e:
        return f"found but not usable ({e}). Try `claude login`."
    return None


# --------------------------------------------------------------------------
# codex-cli/ backend — Codex CLI headless (`codex exec`)
# --------------------------------------------------------------------------


def build_codex_cli_argv(model_id: str, prompt: str, output_path: str) -> list[str]:
    # exec: non-interactive automation subcommand (as opposed to the interactive TUI).
    # --sandbox read-only: most restricted mode (pure text generation, not code
    #   execution). --ephemeral: no session file persisted per benchmark call.
    #   (--ask-for-approval was removed from `codex exec` as of CLI 0.145.0.)
    # --skip-git-repo-check: harness may run from a directory codex doesn't trust.
    # --output-last-message <file>: simplest way to get the final agent message as
    #   plain text, without parsing a --json event stream.
    return [
        "codex", "exec",
        "--model", model_id,
        "--sandbox", "read-only",
        "--skip-git-repo-check",
        "--ephemeral",
        "--output-last-message", output_path,
        prompt,
    ]


async def _complete_codex_cli(model_id: str, messages: list[Message], settings: Settings, json_mode: bool) -> str:
    if shutil.which("codex") is None:
        raise CliBackendError("codex CLI not found on PATH. Install the Codex CLI, then run `codex login`.")
    system, prompt = _build_cli_prompt(messages, json_mode)
    if system:
        prompt = f"[시스템 지침]\n{system}\n\n{prompt}"

    fd, output_path = tempfile.mkstemp(prefix="munbench-codex-", suffix=".txt")
    os.close(fd)
    try:
        argv = build_codex_cli_argv(model_id, prompt, output_path)
        async with _cli_semaphore(settings):
            await _run_subprocess(argv, timeout=CLI_TIMEOUT_SECONDS)
        with open(output_path, encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            raise CliBackendError("codex CLI produced empty output (--output-last-message file was empty)")
        return text
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


async def _probe_codex_cli() -> str | None:
    if shutil.which("codex") is None:
        return "not found on PATH. Install the Codex CLI, then run `codex login`."
    try:
        await _run_subprocess(["codex", "login", "status"], timeout=CLI_PROBE_TIMEOUT_SECONDS)
    except CliBackendError as e:
        return f"found but not usable ({e}). Try `codex login`."
    return None


# --------------------------------------------------------------------------
# litellm backend — unchanged behavior
# --------------------------------------------------------------------------


def split_effort(model: str) -> tuple[str, str | None]:
    """Split an optional reasoning-effort suffix off a model id:
    "openrouter/x/y@high" -> ("openrouter/x/y", "high"). "max" maps to "high",
    the top documented OpenRouter effort level."""
    if "@" not in model:
        return model, None
    base, effort = model.rsplit("@", 1)
    effort = effort.strip().lower()
    return base, "high" if effort == "max" else effort


async def _complete_litellm(
    model: str,
    messages: list[Message],
    settings: Settings,
    json_mode: bool,
    temperature: float | None,
    max_tokens: int | None,
) -> str:
    model, effort = split_effort(model)
    kwargs: dict = dict(
        model=model,
        messages=messages,
        temperature=settings.temperature if temperature is None else temperature,
        max_tokens=settings.max_tokens if max_tokens is None else max_tokens,
    )
    if effort:
        # OpenRouter's native reasoning param via extra_body: works uniformly
        # across providers, unlike litellm's reasoning_effort mapping which
        # rejects models it doesn't recognize as reasoning-capable.
        kwargs["extra_body"] = {"reasoning": {"effort": effort}}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = await litellm.acompletion(**kwargs)
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------------


async def complete(
    model: str,
    messages: list[Message],
    settings: Settings,
    json_mode: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Single-attempt completion, dispatched by model-id prefix. Raises on failure
    (CliBackendError for CLI backends, litellm's own exceptions otherwise) - callers
    are responsible for retries, exactly as before this existed.

    `temperature`/`max_tokens` are litellm-path-only overrides (default to
    `settings.temperature`/`settings.max_tokens`); the CLI backends ignore them since
    neither `claude -p` nor `codex exec` exposes sampling controls - product defaults
    apply for subscription-routed calls.
    """
    if model.startswith(CLAUDE_CLI_PREFIX):
        return await _complete_claude_cli(model[len(CLAUDE_CLI_PREFIX):], messages, settings, json_mode)
    if model.startswith(CODEX_CLI_PREFIX):
        return await _complete_codex_cli(model[len(CODEX_CLI_PREFIX):], messages, settings, json_mode)
    return await _complete_litellm(model, messages, settings, json_mode, temperature, max_tokens)


async def check_cli_backend(model: str) -> str | None:
    """Pre-flight availability check for a claude-cli/ or codex-cli/ model id.
    Returns None if OK, else a human-readable problem description."""
    if model.startswith(CLAUDE_CLI_PREFIX):
        return await _probe_claude_cli()
    if model.startswith(CODEX_CLI_PREFIX):
        return await _probe_codex_cli()
    return None
