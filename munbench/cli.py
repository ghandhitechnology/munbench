"""Typer CLI for the MunBench harness.

Stages are resumable/independent: each reads the previous stage's output files
from disk, so `generate` -> `judge --mode rubric` / `judge --mode pairwise` ->
`elo` -> `report` can be re-run individually.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import litellm
import typer
from rich.console import Console

from munbench import elo as elo_mod
from munbench import generate as generate_mod
from munbench import judge_pairwise
from munbench import judge_rubric
from munbench import providers
from munbench import report as report_mod
from munbench.config import Settings
from munbench.items import DataFileError, load_rubric, load_slop_list, load_track1, load_track2, load_track3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = typer.Typer(help="MunBench (문벤치) — Korean EQ / creative-writing / nuance benchmark harness")
console = Console()

CONFIG_OPTION = typer.Option(Path("munbench.yaml"), "--config", help="Path to munbench.yaml")
MODELS_OPTION = typer.Option(None, "--models", help="Model ids to use; defaults to `models` in config")


def _load_settings(config: Path) -> Settings:
    try:
        return Settings.load(config)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


def _parse_tracks(track: str) -> list[int]:
    if track == "all":
        return [1, 2, 3]
    if track in ("1", "2", "3"):
        return [int(track)]
    raise typer.BadParameter("track must be 'all', '1', '2', or '3'")


class EnvCheckError(Exception):
    """Raised by _check_env_async; carries one human-readable problem line per model."""

    def __init__(self, lines: list[str]) -> None:
        super().__init__("; ".join(lines))
        self.lines = lines


async def _check_env_async(models: list[str]) -> None:
    """Fail with a clear message before spending any calls: missing API keys for
    litellm-routed models (per litellm's env-var convention), or a missing/unusable
    `claude`/`codex` CLI install for claude-cli/ and codex-cli/ models."""
    problems: list[str] = []

    api_models = sorted({m for m in models if not providers.is_cli_model(m)})
    for model in api_models:
        try:
            result = litellm.validate_environment(model)
        except Exception:  # noqa: BLE001 - best-effort; unknown providers just skip the check
            continue
        if not result.get("keys_in_environment", True):
            keys = result.get("missing_keys") or []
            if keys:
                problems.append(f"{model}: missing API key(s) {', '.join(keys)}")

    cli_models = sorted({m for m in models if providers.is_cli_model(m)})
    for model in cli_models:
        error = await providers.check_cli_backend(model)
        if error:
            problems.append(f"{model}: {error}")

    if problems:
        raise EnvCheckError(problems)


def _run_stage(models: list[str], stage_factory):
    """asyncio.run() wrapper shared by generate/judge: env-checks `models` first and
    only creates the stage's coroutine (via `stage_factory`, a zero-arg callable) once
    the check passes. `stage_factory` must not be called before the check succeeds -
    building the coroutine eagerly and only conditionally awaiting it leaves an
    unawaited coroutine object (and a RuntimeWarning) on the failure path."""

    async def _inner():
        await _check_env_async(models)
        return await stage_factory()

    try:
        return asyncio.run(_inner())
    except EnvCheckError as e:
        console.print("[red]Environment check failed:[/red]")
        for line in e.lines:
            console.print(f"  {line}")
        raise typer.Exit(1)


@app.command()
def generate(
    track: str = typer.Option("all", "--track", help="'all', '1', '2', or '3'"),
    models: list[str] | None = MODELS_OPTION,
    config: Path = CONFIG_OPTION,
) -> None:
    """Generate model outputs for the given track(s), writing results/generations/*.jsonl."""
    settings = _load_settings(config)
    use_models = models or settings.models
    if not use_models:
        console.print("[red]No models specified. Use --models or set `models` in munbench.yaml.[/red]")
        raise typer.Exit(1)
    tracks = _parse_tracks(track)
    out_paths = _run_stage(use_models, lambda: generate_mod.run_generation(use_models, tracks, settings))
    for model, path in out_paths.items():
        console.print(f"[green]{model}[/green] -> {path}")


@app.command()
def judge(
    mode: str = typer.Option(..., "--mode", help="'rubric' or 'pairwise'"),
    models: list[str] | None = MODELS_OPTION,
    config: Path = CONFIG_OPTION,
) -> None:
    """Score generations: absolute rubric pass, or pairwise comparison pass for Elo."""
    settings = _load_settings(config)
    use_models = models or settings.models
    if not use_models:
        console.print("[red]No models specified. Use --models or set `models` in munbench.yaml.[/red]")
        raise typer.Exit(1)

    if mode == "rubric":
        out_paths = _run_stage(
            list(use_models) + settings.judges,
            lambda: judge_rubric.run_rubric_for_models(use_models, settings),
        )
        for model, path in out_paths.items():
            console.print(f"[green]{model}[/green] -> {path}")
    elif mode == "pairwise":
        comparisons = _run_stage(
            list(use_models) + settings.judges + settings.pairwise.anchors,
            lambda: judge_pairwise.run_pairwise_judging(use_models, settings),
        )
        judge_pairwise.write_comparisons(comparisons, settings.paths.pairwise_comparisons)
        console.print(f"[green]wrote {len(comparisons)} comparisons[/green] -> {settings.paths.pairwise_comparisons}")
    else:
        raise typer.BadParameter("mode must be 'rubric' or 'pairwise'")


@app.command()
def elo(config: Path = CONFIG_OPTION) -> None:
    """Fit Elo ratings from results/pairwise/comparisons.jsonl."""
    settings = _load_settings(config)
    try:
        ratings = elo_mod.run_elo(settings)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    for model, rating in sorted(ratings.items(), key=lambda kv: -kv[1]):
        console.print(f"{model}: {rating:.1f}")
    console.print(f"[green]wrote[/green] -> {settings.paths.elo_file}")


@app.command()
def report(
    models: list[str] | None = MODELS_OPTION,
    config: Path = CONFIG_OPTION,
) -> None:
    """Assemble the leaderboard from rubric results, Elo ratings, and judge-free metrics."""
    settings = _load_settings(config)
    use_models = models or settings.models
    if not use_models:
        console.print("[red]No models specified. Use --models or set `models` in munbench.yaml.[/red]")
        raise typer.Exit(1)
    leaderboard = report_mod.build_leaderboard(use_models, settings)
    report_mod.write_leaderboard(leaderboard, settings, console)
    console.print(f"[green]wrote[/green] -> {settings.paths.leaderboard_md}, {settings.paths.leaderboard_json}")


@app.command(name="validate-data")
def validate_data(config: Path = CONFIG_OPTION) -> None:
    """Schema-check data/items/*.json, data/rubrics/*.json, and data/slop_list.json."""
    settings = _load_settings(config)
    checks: list[tuple[str, callable]] = [
        ("track1 items", lambda: load_track1(settings.paths.track1_items)),
        ("track2 items", lambda: load_track2(settings.paths.track2_items)),
        ("track3 items", lambda: load_track3(settings.paths.track3_items)),
        ("track1 rubric", lambda: load_rubric(settings.paths.rubric(1))),
        ("track2 rubric", lambda: load_rubric(settings.paths.rubric(2))),
        ("track3 rubric", lambda: load_rubric(settings.paths.rubric(3))),
        ("slop list", lambda: load_slop_list(settings.paths.slop_list)),
    ]
    errors: list[str] = []
    for name, check in checks:
        try:
            result = check()
            n = len(result) if isinstance(result, list) else ""
            console.print(f"[green]OK[/green] {name}{f' ({n} entries)' if n != '' else ''}")
        except DataFileError as e:
            errors.append(f"{name}: {e}")
            console.print(f"[red]FAIL[/red] {name}: {e}")

    if errors:
        console.print(f"\n[red]{len(errors)} data file(s) failed validation.[/red]")
        raise typer.Exit(1)
    console.print("\n[green]All data files valid.[/green]")


@app.command()
def run(
    track: str = typer.Option("all", "--track", help="'all', '1', '2', or '3'"),
    models: list[str] | None = MODELS_OPTION,
    config: Path = CONFIG_OPTION,
    skip_generate: bool = typer.Option(False, "--skip-generate", help="Resume: skip generation"),
    skip_rubric: bool = typer.Option(False, "--skip-rubric", help="Resume: skip rubric judging"),
    skip_pairwise: bool = typer.Option(False, "--skip-pairwise", help="Resume: skip pairwise judging"),
    skip_elo: bool = typer.Option(False, "--skip-elo", help="Resume: skip Elo fitting"),
    skip_report: bool = typer.Option(False, "--skip-report", help="Resume: skip the leaderboard"),
) -> None:
    """Run the full pipeline in order: validate-data -> generate -> judge rubric ->
    judge pairwise -> elo -> report. Stops with a clear message on the first stage
    that fails; --skip-* flags let you resume a partial run without redoing earlier,
    already-completed stages."""
    settings = _load_settings(config)
    use_models = models or settings.models
    if not use_models:
        console.print("[red]No models specified. Use --models or set `models` in munbench.yaml.[/red]")
        raise typer.Exit(1)

    stages = [
        ("validate-data", False, lambda: validate_data(config=config)),
        ("generate", skip_generate, lambda: generate(track=track, models=use_models, config=config)),
        ("judge --mode rubric", skip_rubric, lambda: judge(mode="rubric", models=use_models, config=config)),
        ("judge --mode pairwise", skip_pairwise, lambda: judge(mode="pairwise", models=use_models, config=config)),
        ("elo", skip_elo, lambda: elo(config=config)),
        ("report", skip_report, lambda: report(models=use_models, config=config)),
    ]
    for name, skip, action in stages:
        if skip:
            console.print(f"[yellow]skip[/yellow] {name}")
            continue
        console.print(f"\n[bold]== {name} ==[/bold]")
        try:
            action()
        except typer.Exit as e:
            if e.exit_code != 0:
                console.print(f"[red]Stage '{name}' failed (exit {e.exit_code}); stopping `munbench run`.[/red]")
                raise

    console.print("\n[green]munbench run complete.[/green]")


if __name__ == "__main__":
    app()
