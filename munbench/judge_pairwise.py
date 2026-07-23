"""Pairwise comparison judging for the Elo pass.

For a sampled set of (model_a, model_b, item) triples, each judge compares the two
models' anonymized outputs ("응답 A" / "응답 B", equal-length truncated) and votes a
winner. Every pair is judged in BOTH label orders and averaged to cancel position bias.
Comparisons: every tested model vs every anchor, plus a capped round-robin among tested
models. Output: results/pairwise/comparisons.jsonl
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import litellm
from pydantic import BaseModel

from munbench.config import Settings
from munbench.generate import GenerationRecord, load_generations, record_text
from munbench.items import Item, Rubric, load_rubric, load_track1, load_track2, load_track3

logger = logging.getLogger(__name__)

Winner = Literal["A", "B", "draw"]


class PairwiseComparison(BaseModel):
    track: int
    item_id: str
    variant: str | None
    model_a: str
    model_b: str
    judge: str
    order1_winner: Winner | None  # label order: model_a="응답 A", model_b="응답 B"
    order2_winner: Winner | None  # swapped: model_b="응답 A", model_a="응답 B"
    score_a: float | None  # model_a's win-share in [0, 1], averaged over both orders
    error: str | None = None


GenKey = tuple[int, str, str | None]  # (track, item_id, variant)


@dataclass
class PairSpec:
    track: int
    item_id: str
    variant: str | None
    model_a: str
    model_b: str


# --------------------------------------------------------------------------
# Loading generations indexed by (track, item_id, variant)
# --------------------------------------------------------------------------


def _load_generation_index(models: list[str], settings: Settings) -> dict[str, dict[GenKey, GenerationRecord]]:
    index: dict[str, dict[GenKey, GenerationRecord]] = {}
    for model in models:
        path = settings.paths.generations_dir / f"{settings.model_slug(model)}.jsonl"
        if not path.exists():
            logger.warning("no generations for %s at %s; skipping in pairwise judging", model, path)
            continue
        records = load_generations(path)
        index[model] = {(r.track, r.item_id, r.variant): r for r in records if not r.error}
    return index


def _load_items_index(settings: Settings) -> dict[int, dict[str, Item]]:
    return {
        1: {i.id: i for i in load_track1(settings.paths.track1_items)},
        2: {i.id: i for i in load_track2(settings.paths.track2_items)},
        3: {i.id: i for i in load_track3(settings.paths.track3_items)},
    }


# --------------------------------------------------------------------------
# Pair sampling
# --------------------------------------------------------------------------


def build_comparison_pairs(
    tested_models: list[str],
    anchors: list[str],
    gen_index: dict[str, dict[GenKey, GenerationRecord]],
    max_comparisons_per_model: int,
) -> list[PairSpec]:
    pairs: list[PairSpec] = []

    # 1. Every tested (non-anchor-self) model vs every anchor, on every common item.
    for m in tested_models:
        for a in anchors:
            if m == a or m not in gen_index or a not in gen_index:
                continue
            common = sorted(set(gen_index[m]) & set(gen_index[a]))
            for key in common:
                track, item_id, variant = key
                pairs.append(PairSpec(track=track, item_id=item_id, variant=variant, model_a=m, model_b=a))

    # 2. Capped round-robin among tested models, spread evenly across items/pairs.
    remaining = {m: max_comparisons_per_model for m in tested_models}
    combos = list(itertools.combinations(sorted(tested_models), 2))
    if combos:
        all_keys: set[GenKey] = set()
        for m in tested_models:
            all_keys |= set(gen_index.get(m, {}))
        for key in sorted(all_keys):
            for m1, m2 in combos:
                if m1 not in gen_index or m2 not in gen_index:
                    continue
                if key not in gen_index[m1] or key not in gen_index[m2]:
                    continue
                if remaining[m1] <= 0 or remaining[m2] <= 0:
                    continue
                track, item_id, variant = key
                pairs.append(PairSpec(track=track, item_id=item_id, variant=variant, model_a=m1, model_b=m2))
                remaining[m1] -= 1
                remaining[m2] -= 1

    return pairs


# --------------------------------------------------------------------------
# Prompt + parsing
# --------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + " …(생략)"


def build_pairwise_prompt(
    item_context: str, judge_notes_ko: str, rubric: Rubric, text_a: str, text_b: str
) -> list[dict[str, str]]:
    criteria_block = "\n".join(f"- {c.name_ko}: {c.description_ko}" for c in rubric.criteria)
    system = (
        "당신은 한국어 창작/공감 능력을 비교 평가하는 엄격한 평가자입니다. "
        "두 응답(A, B) 중 아래 기준을 더 잘 만족하는 쪽을 판정하세요. "
        "분량이 많다고 우월한 것이 아니며, 모호할 때만 draw로 판정하세요. "
        '반드시 다음 JSON 형식으로만 답하세요: {"winner": "A" 또는 "B" 또는 "draw", "reason": "한두 문장 이유"}'
    )
    user = (
        f"[과제 맥락]\n{item_context}\n\n"
        f"[채점 유의사항]\n{judge_notes_ko}\n\n"
        f"[평가 기준]\n{criteria_block}\n\n"
        f"[응답 A]\n{text_a}\n\n[응답 B]\n{text_b}\n\n"
        "위 두 응답 중 어느 쪽이 더 나은지 JSON으로 답하세요."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_verdict(content: str) -> tuple[Winner | None, str | None]:
    candidate = content.strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(candidate)
        if not m:
            return None, f"no JSON object found: {content[:200]!r}"
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return None, f"malformed JSON: {e}"
    winner = str(data.get("winner", "")).strip().upper() if isinstance(data, dict) else ""
    if winner in ("A", "B"):
        return winner, None  # type: ignore[return-value]
    if winner in ("DRAW", "TIE"):
        return "draw", None
    return None, f"unrecognized winner value: {winner!r}"


def item_context_and_notes(
    track: int, item_id: str, variant: str | None, items_index: dict[int, dict[str, Item]]
) -> tuple[str, str]:
    item = items_index[track].get(item_id)
    if item is None:
        return f"(item {item_id} not found)", ""
    if track == 1:
        context = f"상황: {item.setup_ko}\n모델 역할: {item.model_role_ko}\n분석 질문: {item.analysis_question_ko}"
    elif track == 2:
        context = f"프롬프트: {item.prompt_ko}\n분량: {item.length_spec}"
    else:
        prompt = item.prompt_ko if variant != "neutral" else (item.variant_neutral_ko or item.prompt_ko)
        context = f"프롬프트: {prompt}\n기대되는 처리 방식: {item.expected_behavior_ko}"
    return context, item.judge_notes_ko


# --------------------------------------------------------------------------
# Judging
# --------------------------------------------------------------------------


async def _get_verdict(messages: list[dict[str, str]], judge_model: str, settings: Settings) -> tuple[Winner | None, str | None]:
    last_err: Exception | None = None
    for attempt in range(settings.max_retries + 1):
        try:
            resp = await litellm.acompletion(
                model=judge_model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            return parse_verdict(resp.choices[0].message.content or "")
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < settings.max_retries:
                await asyncio.sleep(settings.retry_backoff_seconds * (2**attempt))
    return None, f"call failed: {last_err}"


def _score_from_winners(order1: Winner | None, order2: Winner | None) -> float | None:
    """order1: model_a labeled A. order2: model_a labeled B (model_b labeled A)."""
    scores = []
    if order1 == "A":
        scores.append(1.0)
    elif order1 == "B":
        scores.append(0.0)
    elif order1 == "draw":
        scores.append(0.5)
    if order2 == "B":  # model_a is B in this order, so B winning means model_a wins
        scores.append(1.0)
    elif order2 == "A":
        scores.append(0.0)
    elif order2 == "draw":
        scores.append(0.5)
    return sum(scores) / len(scores) if scores else None


async def judge_pair(
    pair: PairSpec,
    judge_model: str,
    text_a: str,
    text_b: str,
    context: str,
    judge_notes_ko: str,
    rubric: Rubric,
    settings: Settings,
) -> PairwiseComparison:
    ta = truncate(text_a, settings.pairwise.truncate_chars)
    tb = truncate(text_b, settings.pairwise.truncate_chars)

    order1_msgs = build_pairwise_prompt(context, judge_notes_ko, rubric, ta, tb)
    order2_msgs = build_pairwise_prompt(context, judge_notes_ko, rubric, tb, ta)

    (w1, e1), (w2, e2) = await asyncio.gather(
        _get_verdict(order1_msgs, judge_model, settings),
        _get_verdict(order2_msgs, judge_model, settings),
    )
    score_a = _score_from_winners(w1, w2)
    error = " | ".join(e for e in (e1, e2) if e) or None
    return PairwiseComparison(
        track=pair.track, item_id=pair.item_id, variant=pair.variant, model_a=pair.model_a, model_b=pair.model_b,
        judge=judge_model, order1_winner=w1, order2_winner=w2, score_a=score_a, error=error,
    )


async def run_pairwise_judging(
    tested_models: list[str], settings: Settings
) -> list[PairwiseComparison]:
    if not settings.pairwise.enabled:
        logger.info("pairwise judging disabled in config")
        return []

    anchors = settings.pairwise.anchors
    all_models = sorted(set(tested_models) | set(anchors))
    gen_index = _load_generation_index(all_models, settings)
    items_index = _load_items_index(settings)
    rubrics = {t: load_rubric(settings.paths.rubric(t)) for t in (1, 2, 3)}

    pairs = build_comparison_pairs(tested_models, anchors, gen_index, settings.pairwise.max_comparisons_per_model)
    if not pairs:
        logger.warning("no comparison pairs could be built (check generations exist for models/anchors)")
        return []

    sem = asyncio.Semaphore(settings.concurrency)

    async def bounded(pair: PairSpec, judge_model: str) -> PairwiseComparison:
        key: GenKey = (pair.track, pair.item_id, pair.variant)
        rec_a = gen_index[pair.model_a][key]
        rec_b = gen_index[pair.model_b][key]
        context, judge_notes_ko = item_context_and_notes(pair.track, pair.item_id, pair.variant, items_index)
        async with sem:
            return await judge_pair(
                pair, judge_model, record_text(rec_a), record_text(rec_b), context, judge_notes_ko,
                rubrics[pair.track], settings,
            )

    tasks = [bounded(pair, judge) for pair in pairs for judge in settings.judges]
    return await asyncio.gather(*tasks)


def write_comparisons(comparisons: list[PairwiseComparison], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in comparisons:
            f.write(c.model_dump_json() + "\n")


def load_comparisons(path: Path) -> list[PairwiseComparison]:
    comparisons = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                comparisons.append(PairwiseComparison.model_validate_json(line))
    return comparisons
