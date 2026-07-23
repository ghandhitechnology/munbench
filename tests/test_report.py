"""Tests for report.py: variant-aware track aggregation (fixes 3/10), completion
stats (fix 12 surfacing), per-judge means, and the culture-pair delta (PART B)."""

import json

from munbench.config import Settings
from munbench.generate import GenerationRecord, write_generations
from munbench.items import SlopList
from munbench.judge_rubric import JudgeAggregate, RubricResult, write_rubric_results
from munbench.report import (
    _completion_stats,
    _culture_pair_stats,
    _per_judge_means,
    _track_metrics,
    build_leaderboard,
)

SLOP = SlopList(phrases=["뻔한 표현"])


def _gen(item_id, track, variant, outputs, error=None):
    return GenerationRecord(item_id=item_id, track=track, variant=variant, transcript=[], outputs=outputs, model="m", error=error)


def _rubric(item_id, track, variant, final_mean, judges_failed=None):
    return RubricResult(
        item_id=item_id, track=track, variant=variant, model="m",
        judges=[JudgeAggregate(judge="j1", iterations=[], mean=final_mean, std=0.0)],
        final_mean=final_mean, judge_disagreement_std=0.0, judges_failed=judges_failed or [],
    )


# --------------------------------------------------------------------------
# Fix 3/10: track3 headline mean must not double-weight culture-pair items
# --------------------------------------------------------------------------


def test_track_metrics_excludes_neutral_variant_from_headline():
    generations = [
        _gen("t3-001", 3, "specified", ["명시 응답"]),
        _gen("t3-001", 3, "neutral", ["비명시 응답"]),
        _gen("t3-002", 3, "specified", ["단일 응답"]),  # no contrast pair
    ]
    rubrics = [
        _rubric("t3-001", 3, "specified", 8.0),
        _rubric("t3-001", 3, "neutral", 4.0),
        _rubric("t3-002", 3, "specified", 6.0),
    ]
    tm = _track_metrics(3, generations, rubrics, SLOP, {})
    # Only the 2 "specified" records count toward the headline: (8+6)/2 = 7, not
    # (8+4+6)/3 which would double-weight t3-001's contrast pair.
    assert tm.n_samples == 2
    assert tm.rubric_mean == 7.0


def test_track_metrics_track1_and_track2_unaffected_by_variant_filter():
    generations = [_gen("t2-001", 2, None, ["텍스트"])]
    rubrics = [_rubric("t2-001", 2, None, 5.0)]
    tm = _track_metrics(2, generations, rubrics, SLOP, {})
    assert tm.n_samples == 1
    assert tm.rubric_mean == 5.0


# --------------------------------------------------------------------------
# Culture-pair delta
# --------------------------------------------------------------------------


def test_culture_pair_stats_computes_delta():
    rubrics = [
        _rubric("t3-001", 3, "specified", 8.0),
        _rubric("t3-001", 3, "neutral", 4.0),
        _rubric("t3-002", 3, "specified", 6.0),
        _rubric("t3-002", 3, "neutral", 6.0),
    ]
    stats = _culture_pair_stats(rubrics)
    assert stats.n_pairs == 2
    assert stats.specified_mean == 7.0  # (8+6)/2
    assert stats.neutral_mean == 5.0  # (4+6)/2
    assert stats.delta == 2.0


def test_culture_pair_stats_ignores_unpaired_items():
    rubrics = [_rubric("t3-003", 3, "specified", 9.0)]  # no neutral counterpart
    stats = _culture_pair_stats(rubrics)
    assert stats.n_pairs == 0
    assert stats.specified_mean is None
    assert stats.delta is None


# --------------------------------------------------------------------------
# Completion / error-rate stats (fix 12 surfacing)
# --------------------------------------------------------------------------


def test_completion_stats_counts_generation_errors_and_judge_failures():
    generations = [
        _gen("t2-001", 2, None, ["ok"]),
        _gen("t2-002", 2, None, [], error="timeout"),
    ]
    rubrics = [
        _rubric("t2-001", 2, None, 7.0, judges_failed=[]),
        _rubric("t2-002", 2, None, None, judges_failed=["judge-b"]),
    ]
    stats = _completion_stats(generations, rubrics)
    assert stats.generation_total == 2
    assert stats.generation_errors == 1
    assert stats.generation_error_rate == 0.5
    assert stats.judge_slots_total == 2  # 1 judge aggregate per rubric result, 2 results
    assert stats.judge_slots_failed == 1
    assert stats.judge_failure_rate == 0.5


# --------------------------------------------------------------------------
# Per-judge means (self-preference visibility)
# --------------------------------------------------------------------------


def test_per_judge_means_averages_across_samples():
    rubrics = [
        RubricResult(
            item_id="t2-001", track=2, variant=None, model="m",
            judges=[
                JudgeAggregate(judge="judge-a", iterations=[], mean=8.0, std=0.0),
                JudgeAggregate(judge="judge-b", iterations=[], mean=6.0, std=0.0),
            ],
            final_mean=7.0, judge_disagreement_std=1.0,
        ),
        RubricResult(
            item_id="t2-002", track=2, variant=None, model="m",
            judges=[
                JudgeAggregate(judge="judge-a", iterations=[], mean=4.0, std=0.0),
            ],
            final_mean=4.0, judge_disagreement_std=None,
        ),
    ]
    means = _per_judge_means(rubrics, ["judge-a", "judge-b", "judge-c"])
    assert means["judge-a"] == 6.0  # (8+4)/2
    assert means["judge-b"] == 6.0
    assert means["judge-c"] is None  # never scored anything


# --------------------------------------------------------------------------
# End-to-end build_leaderboard smoke test
# --------------------------------------------------------------------------


def test_build_leaderboard_end_to_end(tmp_path):
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    (data_dir / "items").mkdir(parents=True)

    (data_dir / "items" / "track2_literary.json").write_text(
        json.dumps(
            [
                {
                    "id": "t2-001", "form": "수필", "prompt_ko": "p", "constraints_ko": [],
                    "targeted_weakness": "w", "length_spec": "10~20자", "judge_notes_ko": "n",
                }
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "slop_list.json").write_text(json.dumps({"phrases": []}), encoding="utf-8")

    settings = Settings(models=["fake-model"])
    settings.paths.data_dir = data_dir
    settings.paths.results_dir = results_dir

    gen_records = [_gen("t2-001", 2, None, ["가" * 15])]
    write_generations(gen_records, settings.paths.generations_dir / "fake-model.jsonl")
    rubric_records = [_rubric("t2-001", 2, None, 7.5)]
    write_rubric_results(rubric_records, settings.paths.rubric_results_dir / "fake-model.jsonl")

    leaderboard = build_leaderboard(["fake-model"], settings)
    assert len(leaderboard.rows) == 1
    row = leaderboard.rows[0]
    assert row.model == "fake-model"
    assert row.tracks[2].rubric_mean == 7.5
    assert row.tracks[2].length_compliance_rate == 1.0
    assert row.completion.generation_total == 1
    assert row.completion.generation_errors == 0
