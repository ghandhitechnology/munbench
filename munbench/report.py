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


class CompletionStats(BaseModel):
    """Generation errors + judge failures (fix for judges silently dropped without a
    surfaced signal — see judge_rubric.RubricResult.judges_failed)."""

    generation_total: int
    generation_errors: int
    generation_error_rate: float | None
    judge_slots_total: int  # sum of judges attempted across all scored samples
    judge_slots_failed: int  # sum of judges that produced zero parseable iterations
    judge_failure_rate: float | None


class CulturePairStats(BaseModel):
    """Track 3 culture-pair (specified-vs-neutral) delta — the actual point of that
    item subtype (see DESIGN.md "문화 명시/비명시 쌍"), computed only here rather than
    pooled into the track3 headline mean."""

    n_pairs: int
    specified_mean: float | None
    neutral_mean: float | None
    delta: float | None  # specified_mean - neutral_mean; None if either side is missing


class LeaderboardRow(BaseModel):
    model: str
    overall_rubric_mean: float | None  # mean of the 3 track rubric means
    elo: float | None
    tracks: dict[int, TrackMetrics]
    completion: CompletionStats
    judge_means: dict[str, float | None]  # per-judge mean score for this model (self-preference check)
    culture_pair: CulturePairStats


class Leaderboard(BaseModel):
    judges: list[str]
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
    # Headline track metrics exclude "neutral"-variant records: track3's 33
    # culture-pair items each produce a "specified" AND a "neutral" generation, and
    # pooling both would double-weight those items relative to single-variant ones.
    # The "specified" response is every item's canonical one; "neutral" only exists
    # for the culture-pair delta (see _culture_pair_stats), not the headline mean.
    # No-op for tracks 1/2, where variant is always None.
    gens = [g for g in generations if g.track == track and not g.error and g.variant != "neutral"]
    rubrics = [
        r for r in rubric_results if r.track == track and r.final_mean is not None and r.variant != "neutral"
    ]

    rubric_means = [r.final_mean for r in rubrics if r.final_mean is not None]
    rubric_mean = statistics.fmean(rubric_means) if rubric_means else None
    rubric_std = statistics.pstdev(rubric_means) if len(rubric_means) > 1 else (0.0 if rubric_means else None)

    slop_vals, trigram_vals, switch_flags, length_flags = [], [], [], []
    for g in gens:
        text = record_text(g)
        if not text:
            continue
        slop_vals.append(slop_hits_per_1000_chars(text, slop_list))
        dtr = distinct_trigram_ratio(text)
        if dtr is not None:  # None = text too short to measure; exclude, don't treat as 1.0
            trigram_vals.append(dtr)
        _, flagged = language_consistency_flag(text)
        switch_flags.append(flagged)
        if track == 2 and g.item_id in length_specs:
            compliant = length_compliance(text, length_specs[g.item_id])
            if compliant is not None:  # None = unparsable spec; exclude, never penalize
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


def _completion_stats(generations: list[GenerationRecord], rubric_results: list[RubricResult]) -> CompletionStats:
    gen_total = len(generations)
    gen_errors = sum(1 for g in generations if g.error)
    judge_slots_total = sum(len(r.judges) for r in rubric_results)
    judge_slots_failed = sum(len(r.judges_failed) for r in rubric_results)
    return CompletionStats(
        generation_total=gen_total,
        generation_errors=gen_errors,
        generation_error_rate=(gen_errors / gen_total) if gen_total else None,
        judge_slots_total=judge_slots_total,
        judge_slots_failed=judge_slots_failed,
        judge_failure_rate=(judge_slots_failed / judge_slots_total) if judge_slots_total else None,
    )


def _per_judge_means(rubric_results: list[RubricResult], judges: list[str]) -> dict[str, float | None]:
    """Mean rubric score attributed to each individual judge, across every sample it
    scored for this model — lets a self-preferring judge (e.g. a judge that's also one
    of the tested models) show up as a visible outlier column rather than being
    smoothed away inside final_mean."""
    means: dict[str, float | None] = {}
    for judge in judges:
        vals = [ja.mean for r in rubric_results for ja in r.judges if ja.judge == judge and ja.mean is not None]
        means[judge] = statistics.fmean(vals) if vals else None
    return means


def _culture_pair_stats(rubric_results: list[RubricResult]) -> CulturePairStats:
    specified = {
        r.item_id: r.final_mean
        for r in rubric_results
        if r.track == 3 and r.variant == "specified" and r.final_mean is not None
    }
    neutral = {
        r.item_id: r.final_mean
        for r in rubric_results
        if r.track == 3 and r.variant == "neutral" and r.final_mean is not None
    }
    paired_ids = sorted(set(specified) & set(neutral))
    if not paired_ids:
        return CulturePairStats(n_pairs=0, specified_mean=None, neutral_mean=None, delta=None)
    spec_mean = statistics.fmean(specified[i] for i in paired_ids)
    neut_mean = statistics.fmean(neutral[i] for i in paired_ids)
    return CulturePairStats(n_pairs=len(paired_ids), specified_mean=spec_mean, neutral_mean=neut_mean, delta=spec_mean - neut_mean)


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
                completion=_completion_stats(generations, rubric_results),
                judge_means=_per_judge_means(rubric_results, settings.judges),
                culture_pair=_culture_pair_stats(rubric_results),
            )
        )

    rows.sort(key=lambda r: (r.overall_rubric_mean is None, -(r.overall_rubric_mean or 0)))
    return Leaderboard(judges=settings.judges, rows=rows)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _fmt(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _fmt_pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


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
                _fmt_pct(tm.code_switch_rate),
            ]
        table.add_row(*cells)
    return table


def render_judge_means_table(leaderboard: Leaderboard) -> Table:
    table = Table(title="Per-judge mean score (self-preference check)")
    table.add_column("Model")
    for judge in leaderboard.judges:
        table.add_column(judge)
    for row in leaderboard.rows:
        table.add_row(row.model, *(_fmt(row.judge_means.get(j)) for j in leaderboard.judges))
    return table


def render_completion_table(leaderboard: Leaderboard) -> Table:
    table = Table(title="Completion / error rates")
    table.add_column("Model")
    table.add_column("Generation errors")
    table.add_column("Judge-slot failures")
    for row in leaderboard.rows:
        c = row.completion
        table.add_row(
            row.model,
            f"{c.generation_errors}/{c.generation_total} ({_fmt_pct(c.generation_error_rate)})",
            f"{c.judge_slots_failed}/{c.judge_slots_total} ({_fmt_pct(c.judge_failure_rate)})",
        )
    return table


def render_culture_pair_table(leaderboard: Leaderboard) -> Table:
    table = Table(title="Track 3 culture-pair: specified vs. neutral (명시/비명시)")
    table.add_column("Model")
    table.add_column("N pairs")
    table.add_column("Specified mean")
    table.add_column("Neutral mean")
    table.add_column("Delta (specified − neutral)")
    for row in leaderboard.rows:
        cp = row.culture_pair
        table.add_row(row.model, str(cp.n_pairs), _fmt(cp.specified_mean), _fmt(cp.neutral_mean), _fmt(cp.delta))
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
                _fmt_pct(tm.code_switch_rate),
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_judge_means_markdown(leaderboard: Leaderboard) -> str:
    header = "| Model | " + " | ".join(leaderboard.judges) + " |"
    sep = "|" + "---|" * (1 + len(leaderboard.judges))
    lines = [header, sep]
    for row in leaderboard.rows:
        cells = [row.model] + [_fmt(row.judge_means.get(j)) for j in leaderboard.judges]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_completion_markdown(leaderboard: Leaderboard) -> str:
    lines = ["| Model | Generation errors | Judge-slot failures |", "|---|---|---|"]
    for row in leaderboard.rows:
        c = row.completion
        lines.append(
            f"| {row.model} | {c.generation_errors}/{c.generation_total} ({_fmt_pct(c.generation_error_rate)}) "
            f"| {c.judge_slots_failed}/{c.judge_slots_total} ({_fmt_pct(c.judge_failure_rate)}) |"
        )
    return "\n".join(lines) + "\n"


def render_culture_pair_markdown(leaderboard: Leaderboard) -> str:
    lines = [
        "| Model | N pairs | Specified mean | Neutral mean | Delta (specified − neutral) |",
        "|---|---|---|---|---|",
    ]
    for row in leaderboard.rows:
        cp = row.culture_pair
        lines.append(
            f"| {row.model} | {cp.n_pairs} | {_fmt(cp.specified_mean)} | {_fmt(cp.neutral_mean)} | {_fmt(cp.delta)} |"
        )
    return "\n".join(lines) + "\n"


def write_leaderboard(leaderboard: Leaderboard, settings: Settings, console: Console | None = None) -> None:
    console = console or Console()
    console.print(render_console_table(leaderboard))
    console.print(render_judge_means_table(leaderboard))
    console.print(render_completion_table(leaderboard))
    console.print(render_culture_pair_table(leaderboard))

    settings.paths.leaderboard_md.parent.mkdir(parents=True, exist_ok=True)
    md = (
        "# MunBench Leaderboard\n\n"
        "Scores are LLM-judged (ensemble + bias controls, not yet human-validated — see README).\n\n"
        + render_markdown(leaderboard)
        + "\n## Per-judge mean score (self-preference check)\n\n"
        + render_judge_means_markdown(leaderboard)
        + "\n## Completion / error rates\n\n"
        + render_completion_markdown(leaderboard)
        + "\n## Track 3 culture-pair: specified vs. neutral (명시/비명시)\n\n"
        + render_culture_pair_markdown(leaderboard)
    )
    settings.paths.leaderboard_md.write_text(md, encoding="utf-8")
    settings.paths.leaderboard_json.write_text(
        leaderboard.model_dump_json(indent=2), encoding="utf-8"
    )
