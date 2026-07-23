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


def _check_env(models: list[str]) -> None:
    """Fail with a clear message listing missing API keys before spending any calls."""
    missing: dict[str, list[str]] = {}
    for model in sorted(set(models)):
        try:
            result = litellm.validate_environment(model)
        except Exception:  # noqa: BLE001 - best-effort; unknown providers just skip the check
            continue
        if not result.get("keys_in_environment", True):
            keys = result.get("missing_keys") or []
            if keys:
                missing[model] = keys
    if missing:
        console.print("[red]Missing API keys (set as environment variables, per litellm's provider convention):[/red]")
        for model, keys in missing.items():
            console.print(f"  {model}: {', '.join(keys)}")
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
    _check_env(use_models)
    out_paths = asyncio.run(generate_mod.run_generation(use_models, tracks, settings))
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
        _check_env(list(use_models) + settings.judges)
        out_paths = asyncio.run(judge_rubric.run_rubric_for_models(use_models, settings))
        for model, path in out_paths.items():
            console.print(f"[green]{model}[/green] -> {path}")
    elif mode == "pairwise":
        _check_env(list(use_models) + settings.judges + settings.pairwise.anchors)
        comparisons = asyncio.run(judge_pairwise.run_pairwise_judging(use_models, settings))
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


if __name__ == "__main__":
    app()
