from types import SimpleNamespace

import pytest

import munbench.generate as generate_mod
from munbench.config import Settings
from munbench.items import Track1Item, Track2Item, Track3Item

SETTINGS = Settings(
    models=["fake-model"],
    temperature=0.7,
    max_tokens=64,
    concurrency=4,
    max_retries=0,
    retry_backoff_seconds=0.0,
)

TRACK1_ITEM = Track1Item(
    id="t1-001",
    title="테스트",
    domain="직장",
    setup_ko="상황 설명",
    model_role_ko="후배",
    interlocutor_ko="상사",
    turns_ko=["첫 대사", "둘째 대사", "셋째 대사"],
    phenomena=["눈치"],
    analysis_question_ko="상대는 어떤 감정이었을까요?",
    judge_notes_ko="노트",
)

TRACK2_ITEM = Track2Item(
    id="t2-001",
    form="수필",
    prompt_ko="가을에 대해 써라",
    constraints_ko=["문어체 금지"],
    targeted_weakness="restraint",
    length_spec="100~200자",
    judge_notes_ko="노트",
)

TRACK3_CULTURE_PAIR_ITEM = Track3Item(
    id="t3-001",
    subtype="culture-pair",
    prompt_ko="한국에서의 상황",
    variant_neutral_ko="중립적 상황",
    expected_behavior_ko="한국 특유의 관습이 드러나야 함",
    judge_notes_ko="노트",
)

TRACK3_REGISTER_ITEM = Track3Item(
    id="t3-002",
    subtype="register",
    prompt_ko="반말을 존댓말로 바꿔라",
    expected_behavior_ko="자연스러운 존댓말",
    judge_notes_ko="노트",
)


def _fake_response(text: str):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def mock_acompletion(monkeypatch):
    calls = {"n": 0}

    async def fake(**kwargs):
        calls["n"] += 1
        return _fake_response(f"응답-{calls['n']}")

    monkeypatch.setattr(generate_mod.providers.litellm, "acompletion", fake)
    return calls


async def test_track1_generates_four_outputs(mock_acompletion):
    record = await generate_mod.generate_track1_item(TRACK1_ITEM, "fake-model", SETTINGS)
    assert record.error is None
    assert len(record.outputs) == 4
    assert record.outputs == ["응답-1", "응답-2", "응답-3", "응답-4"]
    # transcript = system + 4*(user, assistant)
    assert record.transcript[0]["role"] == "system"
    assert len([m for m in record.transcript if m["role"] == "user"]) == 4
    assert record.transcript[-2]["content"].startswith(generate_mod.ANALYSIS_PREFIX)


async def test_track2_generates_single_output(mock_acompletion):
    record = await generate_mod.generate_track2_item(TRACK2_ITEM, "fake-model", SETTINGS)
    assert record.error is None
    assert record.outputs == ["응답-1"]
    assert record.variant is None


async def test_track3_culture_pair_produces_both_variants(mock_acompletion):
    records = await generate_mod.generate_track3_item(TRACK3_CULTURE_PAIR_ITEM, "fake-model", SETTINGS)
    assert {r.variant for r in records} == {"specified", "neutral"}
    assert all(r.error is None for r in records)
    assert all(len(r.outputs) == 1 for r in records)


async def test_track3_non_culture_pair_produces_single_variant(mock_acompletion):
    records = await generate_mod.generate_track3_item(TRACK3_REGISTER_ITEM, "fake-model", SETTINGS)
    assert len(records) == 1
    assert records[0].variant == "specified"


async def test_generation_failure_is_recorded_not_raised(monkeypatch):
    async def failing(**kwargs):
        raise RuntimeError("provider is down")

    monkeypatch.setattr(generate_mod.providers.litellm, "acompletion", failing)
    record = await generate_mod.generate_track2_item(TRACK2_ITEM, "fake-model", SETTINGS)
    assert record.error is not None
    assert "provider is down" in record.error
    assert record.outputs == []


def test_write_and_load_generations_roundtrip(tmp_path):
    record = generate_mod.GenerationRecord(
        item_id="t2-001", track=2, transcript=[{"role": "user", "content": "hi"}], outputs=["hello"], model="fake-model"
    )
    path = tmp_path / "fake-model.jsonl"
    generate_mod.write_generations([record], path)
    loaded = generate_mod.load_generations(path)
    assert len(loaded) == 1
    assert loaded[0].item_id == "t2-001"
    assert loaded[0].human_score is None
