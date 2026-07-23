"""Absolute rubric scoring.

For every generation record, each judge in the ensemble scores every rubric
criterion 0-10, `rubric_iterations` times. Output: results/rubric/{model_slug}.jsonl
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import statistics
from pathlib import Path

import litellm
from pydantic import BaseModel

from munbench.config import Settings
from munbench.generate import GenerationRecord, load_generations
from munbench.items import Rubric, Track1Item, Track2Item, Track3Item, load_rubric, load_track1, load_track2, load_track3

logger = logging.getLogger(__name__)

Item = Track1Item | Track2Item | Track3Item


# --------------------------------------------------------------------------
# Result schema
# --------------------------------------------------------------------------


class JudgeIterationResult(BaseModel):
    raw_scores: dict[str, float]
    weighted_score: float | None
    parse_error: str | None = None


class JudgeAggregate(BaseModel):
    judge: str
    iterations: list[JudgeIterationResult]
    mean: float | None
    std: float | None


class RubricResult(BaseModel):
    item_id: str
    track: int
    variant: str | None
    model: str
    judges: list[JudgeAggregate]
    final_mean: float | None
    judge_disagreement_std: float | None
    human_score: None = None


# --------------------------------------------------------------------------
# Item lookup
# --------------------------------------------------------------------------


def _load_items_index(settings: Settings) -> dict[int, dict[str, Item]]:
    return {
        1: {i.id: i for i in load_track1(settings.paths.track1_items)},
        2: {i.id: i for i in load_track2(settings.paths.track2_items)},
        3: {i.id: i for i in load_track3(settings.paths.track3_items)},
    }


# --------------------------------------------------------------------------
# Context rendering (what the judge sees)
# --------------------------------------------------------------------------


def render_context(record: GenerationRecord, item: Item) -> tuple[str, str]:
    """Returns (context, judge_notes_ko) for a generation record."""
    if record.track == 1:
        assert isinstance(item, Track1Item)
        lines = [
            f"상황: {item.setup_ko}",
            f"모델이 맡은 역할: {item.model_role_ko}",
            f"상대방: {item.interlocutor_ko}",
            "",
            "[대화]",
        ]
        for i, turn in enumerate(item.turns_ko):
            lines.append(f"상대방: {turn}")
            if i < len(record.outputs):
                lines.append(f"모델(연기): {record.outputs[i]}")
        lines.append("")
        lines.append(f"[분석 질문] {item.analysis_question_ko}")
        if len(record.outputs) > 3:
            lines.append(f"[모델의 분석] {record.outputs[3]}")
        return "\n".join(lines), item.judge_notes_ko

    if record.track == 2:
        assert isinstance(item, Track2Item)
        constraints = "\n".join(f"- {c}" for c in item.constraints_ko) or "(없음)"
        context = (
            f"프롬프트: {item.prompt_ko}\n제약 조건:\n{constraints}\n분량: {item.length_spec}\n\n"
            f"[모델의 응답]\n{record.outputs[0] if record.outputs else ''}"
        )
        return context, item.judge_notes_ko

    assert isinstance(item, Track3Item)
    prompt = item.prompt_ko if record.variant != "neutral" else (item.variant_neutral_ko or item.prompt_ko)
    context = f"프롬프트: {prompt}\n기대되는 처리 방식: {item.expected_behavior_ko}\n\n[모델의 응답]\n{record.outputs[0] if record.outputs else ''}"
    return context, item.judge_notes_ko


# --------------------------------------------------------------------------
# Prompt construction (Korean)
# --------------------------------------------------------------------------


def build_judge_prompt(context: str, judge_notes_ko: str, rubric: Rubric) -> list[dict[str, str]]:
    criteria_lines = []
    for c in rubric.criteria:
        signals = "; ".join(c.failure_signals_ko) if c.failure_signals_ko else "(없음)"
        criteria_lines.append(
            f"- {c.name_ko} (가중치 {c.weight}): {c.description_ko}\n  실패 신호: {signals}"
        )
    criteria_block = "\n".join(criteria_lines)
    criterion_names = ", ".join(f'"{c.name_ko}"' for c in rubric.criteria)

    system = (
        "당신은 한국어 창작/공감 능력 평가를 전문으로 하는 엄격한 평가자입니다. "
        "아래 항목별 기준에 따라 모델의 응답을 0에서 10 사이의 점수로 채점하세요. "
        "각 기준은 독립적으로 채점하며, 실패 신호가 하나라도 뚜렷하게 나타나면 해당 기준 점수를 크게 낮추세요. "
        "친절하거나 분량이 많다는 이유로 후하게 채점하지 마세요. "
        f"반드시 다음 키를 모두 포함하는 JSON 객체로만 답하세요: {{{criterion_names}}}. "
        "각 값은 0~10 사이의 숫자입니다. JSON 외의 텍스트는 절대 포함하지 마세요."
    )
    user = (
        f"[평가 대상 맥락]\n{context}\n\n"
        f"[이 항목의 채점 유의사항]\n{judge_notes_ko}\n\n"
        f"[채점 기준]\n{criteria_block}\n\n"
        "위 기준에 따라 JSON으로 채점하세요."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------
# Defensive JSON parsing
# --------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_response(content: str, rubric: Rubric) -> tuple[dict[str, float], str | None]:
    """Best-effort parse of the judge's JSON. Returns (scores, parse_error).

    scores only contains criteria that were found and coercible to a float in [0, 10];
    parse_error is set (but scores may still be partially populated) if anything went wrong.
    """
    candidate = content.strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(candidate)
        if not m:
            return {}, f"no JSON object found in judge response: {content[:200]!r}"
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return {}, f"malformed JSON in judge response: {e}"

    if not isinstance(data, dict):
        return {}, f"judge response JSON was not an object: {type(data).__name__}"

    normalized = {str(k).strip(): v for k, v in data.items()}
    scores: dict[str, float] = {}
    missing = []
    for criterion in rubric.criteria:
        name = criterion.name_ko
        if name not in normalized:
            missing.append(name)
            continue
        try:
            score = float(normalized[name])
        except (TypeError, ValueError):
            missing.append(name)
            continue
        scores[name] = max(0.0, min(10.0, score))

    error = f"missing/invalid criteria: {missing}" if missing else None
    return scores, error


def weighted_score(scores: dict[str, float], rubric: Rubric) -> float | None:
    if len(scores) != len(rubric.criteria):
        return None
    total_weight = sum(c.weight for c in rubric.criteria)
    if total_weight == 0:
        return None
    return sum(c.weight * scores[c.name_ko] for c in rubric.criteria) / total_weight


# --------------------------------------------------------------------------
# Judging orchestration
# --------------------------------------------------------------------------


async def _judge_once(
    context: str, judge_notes_ko: str, rubric: Rubric, judge_model: str, settings: Settings
) -> JudgeIterationResult:
    messages = build_judge_prompt(context, judge_notes_ko, rubric)
    last_err: Exception | None = None
    for attempt in range(settings.max_retries + 1):
        try:
            resp = await litellm.acompletion(
                model=judge_model,
                messages=messages,
                temperature=0.0,
                max_tokens=settings.max_tokens,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            scores, err = parse_judge_response(content, rubric)
            return JudgeIterationResult(raw_scores=scores, weighted_score=weighted_score(scores, rubric), parse_error=err)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < settings.max_retries:
                await asyncio.sleep(settings.retry_backoff_seconds * (2**attempt))
    return JudgeIterationResult(raw_scores={}, weighted_score=None, parse_error=f"call failed: {last_err}")


async def judge_record(record: GenerationRecord, item: Item, rubric: Rubric, settings: Settings) -> RubricResult:
    if record.error:
        return RubricResult(
            item_id=record.item_id, track=record.track, variant=record.variant, model=record.model,
            judges=[], final_mean=None, judge_disagreement_std=None,
        )

    context, judge_notes_ko = render_context(record, item)
    judge_aggs: list[JudgeAggregate] = []
    for judge_model in settings.judges:
        iterations = [
            await _judge_once(context, judge_notes_ko, rubric, judge_model, settings)
            for _ in range(settings.rubric_iterations)
        ]
        valid = [it.weighted_score for it in iterations if it.weighted_score is not None]
        mean = statistics.fmean(valid) if valid else None
        std = statistics.pstdev(valid) if len(valid) > 1 else (0.0 if valid else None)
        judge_aggs.append(JudgeAggregate(judge=judge_model, iterations=iterations, mean=mean, std=std))

    judge_means = [j.mean for j in judge_aggs if j.mean is not None]
    final_mean = statistics.fmean(judge_means) if judge_means else None
    disagreement = statistics.pstdev(judge_means) if len(judge_means) > 1 else (0.0 if judge_means else None)

    return RubricResult(
        item_id=record.item_id, track=record.track, variant=record.variant, model=record.model,
        judges=judge_aggs, final_mean=final_mean, judge_disagreement_std=disagreement,
    )


async def run_rubric_judging(generations_path: Path, settings: Settings) -> list[RubricResult]:
    records = load_generations(generations_path)
    items_index = _load_items_index(settings)
    rubrics = {t: load_rubric(settings.paths.rubric(t)) for t in (1, 2, 3)}

    sem = asyncio.Semaphore(settings.concurrency)

    async def bounded(record: GenerationRecord) -> RubricResult | None:
        item = items_index[record.track].get(record.item_id)
        if item is None:
            logger.warning("no matching item for record %s (track %d); skipping", record.item_id, record.track)
            return None
        async with sem:
            return await judge_record(record, item, rubrics[record.track], settings)

    results = await asyncio.gather(*(bounded(r) for r in records))
    return [r for r in results if r is not None]


def write_rubric_results(results: list[RubricResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")


def load_rubric_results(path: Path) -> list[RubricResult]:
    results = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(RubricResult.model_validate_json(line))
    return results


async def run_rubric_for_models(models: list[str], settings: Settings) -> dict[str, Path]:
    out_paths: dict[str, Path] = {}
    for model in models:
        gen_path = settings.paths.generations_dir / f"{settings.model_slug(model)}.jsonl"
        if not gen_path.exists():
            raise FileNotFoundError(f"No generations found for {model} at {gen_path}. Run `munbench generate` first.")
        results = await run_rubric_judging(gen_path, settings)
        out_path = settings.paths.rubric_results_dir / f"{settings.model_slug(model)}.jsonl"
        write_rubric_results(results, out_path)
        out_paths[model] = out_path
        logger.info("model=%s: wrote %d rubric results to %s", model, len(results), out_path)
    return out_paths
