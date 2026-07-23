"""Async generation of model outputs for all three tracks.

Track 1 (감정): multi-turn roleplay — the model plays model_role_ko through
3 pre-scripted interlocutor turns, then answers an out-of-character analysis
question. 4 assistant outputs are stored per item.

Track 2 (문학): single completion of prompt_ko + constraints_ko + length_spec.

Track 3 (결): single completion of prompt_ko; culture-pair items additionally
generate a completion of variant_neutral_ko, stored as a separate record with
variant="neutral" (the prompt_ko record gets variant="specified").
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from pydantic import BaseModel
import litellm

from munbench.config import Settings
from munbench.items import Track1Item, Track2Item, Track3Item, load_track1, load_track2, load_track3

logger = logging.getLogger(__name__)

Message = dict[str, str]

ANALYSIS_PREFIX = "[역할극 종료 — 분석 질문] "


class GenerationRecord(BaseModel):
    item_id: str
    track: int
    variant: str | None = None
    transcript: list[Message]
    outputs: list[str]
    model: str
    error: str | None = None
    human_score: None = None  # hook for a future human-annotation pass


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def track1_system_prompt(item: Track1Item) -> str:
    return (
        f"당신은 아래 상황극에서 '{item.model_role_ko}' 역할을 맡아 연기합니다.\n\n"
        f"상황: {item.setup_ko}\n"
        f"상대방: {item.interlocutor_ko}\n\n"
        "지침:\n"
        "- 절대 역할에서 벗어나지 마세요 (AI라는 사실을 밝히지 마세요).\n"
        "- 자연스러운 한국어 구어체로, 실제 사람처럼 응답하세요.\n"
        "- 상대방의 대사에 감정적으로 진짜처럼 반응하세요.\n"
        "- 대화 상대가 '[역할극 종료 — 분석 질문]'으로 시작하는 메시지를 보내면 "
        "그때만 역할에서 벗어나 분석가로서 답하세요."
    )


def track2_user_prompt(item: Track2Item) -> str:
    parts = [item.prompt_ko]
    if item.constraints_ko:
        parts.append("제약 조건:\n" + "\n".join(f"- {c}" for c in item.constraints_ko))
    parts.append(f"분량: {item.length_spec}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Retry wrapper
# --------------------------------------------------------------------------


async def _complete_with_retry(
    messages: list[Message],
    model: str,
    settings: Settings,
) -> str:
    last_err: Exception | None = None
    for attempt in range(settings.max_retries + 1):
        try:
            resp = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 - litellm raises many provider-specific types
            last_err = e
            if attempt < settings.max_retries:
                wait = settings.retry_backoff_seconds * (2**attempt)
                logger.warning("generation attempt %d failed for %s: %s (retrying in %.1fs)", attempt + 1, model, e, wait)
                await asyncio.sleep(wait)
    assert last_err is not None
    raise last_err


# --------------------------------------------------------------------------
# Per-track item generation
# --------------------------------------------------------------------------


async def generate_track1_item(item: Track1Item, model: str, settings: Settings) -> GenerationRecord:
    messages: list[Message] = [{"role": "system", "content": track1_system_prompt(item)}]
    outputs: list[str] = []
    try:
        for turn in item.turns_ko:
            messages.append({"role": "user", "content": turn})
            reply = await _complete_with_retry(messages, model, settings)
            messages.append({"role": "assistant", "content": reply})
            outputs.append(reply)

        analysis_prompt = ANALYSIS_PREFIX + item.analysis_question_ko
        messages.append({"role": "user", "content": analysis_prompt})
        analysis_reply = await _complete_with_retry(messages, model, settings)
        messages.append({"role": "assistant", "content": analysis_reply})
        outputs.append(analysis_reply)

        return GenerationRecord(item_id=item.id, track=1, transcript=messages, outputs=outputs, model=model)
    except Exception as e:  # noqa: BLE001
        return GenerationRecord(
            item_id=item.id, track=1, transcript=messages, outputs=outputs, model=model, error=str(e)
        )


async def generate_track2_item(item: Track2Item, model: str, settings: Settings) -> GenerationRecord:
    messages: list[Message] = [{"role": "user", "content": track2_user_prompt(item)}]
    try:
        reply = await _complete_with_retry(messages, model, settings)
        messages.append({"role": "assistant", "content": reply})
        return GenerationRecord(item_id=item.id, track=2, transcript=messages, outputs=[reply], model=model)
    except Exception as e:  # noqa: BLE001
        return GenerationRecord(item_id=item.id, track=2, transcript=messages, outputs=[], model=model, error=str(e))


async def _generate_track3_variant(
    item: Track3Item, prompt: str, variant: str, model: str, settings: Settings
) -> GenerationRecord:
    messages: list[Message] = [{"role": "user", "content": prompt}]
    try:
        reply = await _complete_with_retry(messages, model, settings)
        messages.append({"role": "assistant", "content": reply})
        return GenerationRecord(
            item_id=item.id, track=3, variant=variant, transcript=messages, outputs=[reply], model=model
        )
    except Exception as e:  # noqa: BLE001
        return GenerationRecord(
            item_id=item.id, track=3, variant=variant, transcript=messages, outputs=[], model=model, error=str(e)
        )


async def generate_track3_item(item: Track3Item, model: str, settings: Settings) -> list[GenerationRecord]:
    if item.subtype == "culture-pair" and item.variant_neutral_ko:
        return [
            await _generate_track3_variant(item, item.prompt_ko, "specified", model, settings),
            await _generate_track3_variant(item, item.variant_neutral_ko, "neutral", model, settings),
        ]
    return [await _generate_track3_variant(item, item.prompt_ko, "specified", model, settings)]


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


async def generate_for_model(
    model: str,
    tracks: list[int],
    settings: Settings,
) -> list[GenerationRecord]:
    """Generate all records for one model across the requested tracks, bounded by
    settings.concurrency. Returns records in stable item order (not completion order).
    """
    sem = asyncio.Semaphore(settings.concurrency)
    records: list[GenerationRecord] = []

    async def bounded_t1(item: Track1Item) -> GenerationRecord:
        async with sem:
            return await generate_track1_item(item, model, settings)

    async def bounded_t2(item: Track2Item) -> GenerationRecord:
        async with sem:
            return await generate_track2_item(item, model, settings)

    async def bounded_t3(item: Track3Item) -> list[GenerationRecord]:
        async with sem:
            return await generate_track3_item(item, model, settings)

    if 1 in tracks:
        items1 = load_track1(settings.paths.track1_items)
        records.extend(await asyncio.gather(*(bounded_t1(i) for i in items1)))
    if 2 in tracks:
        items2 = load_track2(settings.paths.track2_items)
        records.extend(await asyncio.gather(*(bounded_t2(i) for i in items2)))
    if 3 in tracks:
        items3 = load_track3(settings.paths.track3_items)
        nested = await asyncio.gather(*(bounded_t3(i) for i in items3))
        for group in nested:
            records.extend(group)

    return records


def write_generations(records: list[GenerationRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


def record_text(record: GenerationRecord) -> str:
    """The generated text of a record, concatenated for metrics/pairwise comparison
    purposes (track 1 has multiple turns; tracks 2/3 have a single output)."""
    return "\n\n".join(record.outputs)


def load_generations(path: Path) -> list[GenerationRecord]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(GenerationRecord.model_validate_json(line))
    return records


async def run_generation(models: list[str], tracks: list[int], settings: Settings) -> dict[str, Path]:
    """Generate for each model sequentially (models run one at a time; requests within
    a model are concurrent), writing results/generations/{model_slug}.jsonl.
    """
    out_paths: dict[str, Path] = {}
    for model in models:
        records = await generate_for_model(model, tracks, settings)
        out_path = settings.paths.generations_dir / f"{settings.model_slug(model)}.jsonl"
        write_generations(records, out_path)
        out_paths[model] = out_path
        n_errors = sum(1 for r in records if r.error)
        logger.info("model=%s: wrote %d records (%d errors) to %s", model, len(records), n_errors, out_path)
    return out_paths
