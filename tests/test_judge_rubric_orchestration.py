"""Async orchestration tests for judge_rubric.py: judges_failed surfacing, and the
track1 context reuse from rendering.py. Mocks providers.complete — no network calls.
"""

from munbench.config import Settings
from munbench.generate import GenerationRecord
from munbench.items import Rubric, RubricCriterion, Track1Item
from munbench.judge_rubric import judge_record, render_context

SETTINGS = Settings(
    judges=["judge-a", "judge-b"],
    rubric_iterations=2,
    max_retries=0,
    retry_backoff_seconds=0.0,
)

RUBRIC = Rubric(
    criteria=[RubricCriterion(name_ko="눈치", description_ko="d", weight=1.0, failure_signals_ko=[])]
)

TRACK1_ITEM = Track1Item(
    id="t1-001",
    title="테스트",
    domain="직장",
    setup_ko="회식 자리 상황",
    model_role_ko="후배",
    interlocutor_ko="상사",
    turns_ko=["첫 대사입니다", "둘째 대사입니다", "셋째 대사입니다"],
    phenomena=["눈치"],
    analysis_question_ko="상사의 속마음은?",
    judge_notes_ko="노트",
)

TRACK1_RECORD = GenerationRecord(
    item_id="t1-001",
    track=1,
    transcript=[],
    outputs=["응답1", "응답2", "응답3", "분석 응답"],
    model="fake-model",
)


def test_render_context_track1_includes_interlocutor_turns():
    context, notes = render_context(TRACK1_RECORD, TRACK1_ITEM)
    for turn in TRACK1_ITEM.turns_ko:
        assert turn in context
    assert "응답1" in context
    assert "분석 응답" in context
    assert notes == "노트"


async def test_judge_record_marks_failed_judge_in_judges_failed(monkeypatch):
    import munbench.judge_rubric as judge_rubric_mod

    async def fake_complete(judge_model, messages, settings, json_mode=False, temperature=None, max_tokens=None):
        if judge_model == "judge-a":
            return '{"눈치": 8}'
        return "이 응답은 채점할 수 없습니다"  # judge-b never produces parseable JSON

    monkeypatch.setattr(judge_rubric_mod.providers, "complete", fake_complete)

    result = await judge_record(TRACK1_RECORD, TRACK1_ITEM, RUBRIC, SETTINGS)

    assert "judge-b" in result.judges_failed
    assert "judge-a" not in result.judges_failed
    # final_mean still computed from the surviving judge, but disagreement is
    # unmeasurable with only one contributing judge (not to be read as "perfect
    # agreement" - judges_failed is what makes that distinction visible).
    assert result.final_mean == 8.0
    assert result.judge_disagreement_std == 0.0


async def test_judge_record_no_failures_when_all_judges_parse(monkeypatch):
    import munbench.judge_rubric as judge_rubric_mod

    async def fake_complete(judge_model, messages, settings, json_mode=False, temperature=None, max_tokens=None):
        return '{"눈치": 7}'

    monkeypatch.setattr(judge_rubric_mod.providers, "complete", fake_complete)

    result = await judge_record(TRACK1_RECORD, TRACK1_ITEM, RUBRIC, SETTINGS)
    assert result.judges_failed == []
