"""Assemble the leaderboard from rubric results, pairwise Elo, and judge-free metrics."""

from __future__ import annotations

import logging
import statistics

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from munbench.config import Settings
from munbench.elo import load_elo
from munbench.generate import GenerationRecord, load_generations, record_text
from munbench.items import SlopList, load_slop_list, load_track2
from munbench.judge_rubric import RubricResult, load_rubric_results
from munbench.metrics import (
    distinct_trigram_ratio,
    language_consistency_flag,
    length_compliance,
    slop_hits_per_1000_chars,
)

logger = logging.getLogger(__name__)

TRACKS = (1, 2, 3)


class TrackMetrics(BaseModel):
    n_samples: int
    rubric_mean: float | None
    rubric_std: float | None
    slop_per_1k: float | None
    code_switch_rate: float | None
    distinct_trigram_ratio: float | None
    length_compliance_rate: float | None  # only meaningful for track 2


class LeaderboardRow(BaseModel):
    model: str
    overall_rubric_mean: float | None  # mean of the 3 track rubric means
    elo: float | None
    tracks: dict[int, TrackMetrics]


class Leaderboard(BaseModel):
    rows: list[LeaderboardRow]


# --------------------------------------------------------------------------
# Per-model aggregation
# --------------------------------------------------------------------------


def _track_metrics(
    track: int,
    generations: list[GenerationRecord],
    rubric_results: list[RubricResult],
    slop_list: SlopList,
    length_specs: dict[str, str],
) -> TrackMetrics:
    gens = [g for g in generations if g.track == track and not g.error]
    rubrics = [r for r in rubric_results if r.track == track and r.final_mean is not None]

    rubric_means = [r.final_mean for r in rubrics if r.final_mean is not None]
    rubric_mean = statistics.fmean(rubric_means) if rubric_means else None
    rubric_std = statistics.pstdev(rubric_means) if len(rubric_means) > 1 else (0.0 if rubric_means else None)

    slop_vals, trigram_vals, switch_flags, length_flags = [], [], [], []
    for g in gens:
        text = record_text(g)
        if not text:
            continue
        slop_vals.append(slop_hits_per_1000_chars(text, slop_list))
        trigram_vals.append(distinct_trigram_ratio(text))
        _, flagged = language_consistency_flag(text)
        switch_flags.append(flagged)
        if track == 2 and g.item_id in length_specs:
            compliant = length_compliance(text, length_specs[g.item_id])
            if compliant is not None:
                length_flags.append(compliant)

    return TrackMetrics(
        n_samples=len(gens),
        rubric_mean=rubric_mean,
        rubric_std=rubric_std,
        slop_per_1k=statistics.fmean(slop_vals) if slop_vals else None,
        code_switch_rate=(sum(switch_flags) / len(switch_flags)) if switch_flags else None,
        distinct_trigram_ratio=statistics.fmean(trigram_vals) if trigram_vals else None,
        length_compliance_rate=(sum(length_flags) / len(length_flags)) if length_flags else None,
    )


def build_leaderboard(models: list[str], settings: Settings) -> Leaderboard:
    slop_list = load_slop_list(settings.paths.slop_list)
    elo_ratings = load_elo(settings.paths.elo_file)
    length_specs: dict[str, str] = {}
    if settings.paths.track2_items.exists():
        length_specs = {i.id: i.length_spec for i in load_track2(settings.paths.track2_items)}

    rows: list[LeaderboardRow] = []
    for model in models:
        gen_path = settings.paths.generations_dir / f"{settings.model_slug(model)}.jsonl"
        rubric_path = settings.paths.rubric_results_dir / f"{settings.model_slug(model)}.jsonl"
        if not gen_path.exists():
            logger.warning("no generations for %s; skipping in leaderboard", model)
            continue
        generations = load_generations(gen_path)
        rubric_results = load_rubric_results(rubric_path) if rubric_path.exists() else []

        track_metrics = {
            t: _track_metrics(t, generations, rubric_results, slop_list, length_specs) for t in TRACKS
        }
        track_means = [tm.rubric_mean for tm in track_metrics.values() if tm.rubric_mean is not None]
        overall = statistics.fmean(track_means) if track_means else None

        rows.append(
            LeaderboardRow(
                model=model,
                overall_rubric_mean=overall,
                elo=elo_ratings.get(model),
                tracks=track_metrics,
            )
        )

    rows.sort(key=lambda r: (r.overall_rubric_mean is None, -(r.overall_rubric_mean or 0)))
    return Leaderboard(rows=rows)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _fmt(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def render_console_table(leaderboard: Leaderboard) -> Table:
    table = Table(title="MunBench Leaderboard")
    table.add_column("Model")
    table.add_column("Overall")
    table.add_column("Elo")
    for t in TRACKS:
        table.add_column(f"T{t} rubric")
        table.add_column(f"T{t} slop/1k")
        table.add_column(f"T{t} switch%")
    for row in leaderboard.rows:
        cells = [row.model, _fmt(row.overall_rubric_mean), _fmt(row.elo, 0)]
        for t in TRACKS:
            tm = row.tracks[t]
            cells += [
                f"{_fmt(tm.rubric_mean)} ± {_fmt(tm.rubric_std)}",
                _fmt(tm.slop_per_1k),
                _fmt((tm.code_switch_rate or 0) * 100, 1) if tm.code_switch_rate is not None else "—",
            ]
        table.add_row(*cells)
    return table


def render_markdown(leaderboard: Leaderboard) -> str:
    header = "| Model | Overall | Elo | " + " | ".join(f"T{t} rubric | T{t} slop/1k | T{t} switch%" for t in TRACKS) + " |"
    sep = "|" + "---|" * (3 + 3 * len(TRACKS))
    lines = [header, sep]
    for row in leaderboard.rows:
        cells = [row.model, _fmt(row.overall_rubric_mean), _fmt(row.elo, 0)]
        for t in TRACKS:
            tm = row.tracks[t]
            cells += [
                f"{_fmt(tm.rubric_mean)} ± {_fmt(tm.rubric_std)}",
                _fmt(tm.slop_per_1k),
                _fmt((tm.code_switch_rate or 0) * 100, 1) if tm.code_switch_rate is not None else "—",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_leaderboard(leaderboard: Leaderboard, settings: Settings, console: Console | None = None) -> None:
    console = console or Console()
    console.print(render_console_table(leaderboard))

    settings.paths.leaderboard_md.parent.mkdir(parents=True, exist_ok=True)
    md = (
        "# MunBench Leaderboard\n\n"
        "Scores are LLM-judged (ensemble + bias controls, not yet human-validated — see README).\n\n"
        + render_markdown(leaderboard)
    )
    settings.paths.leaderboard_md.write_text(md, encoding="utf-8")
    settings.paths.leaderboard_json.write_text(
        leaderboard.model_dump_json(indent=2), encoding="utf-8"
    )
