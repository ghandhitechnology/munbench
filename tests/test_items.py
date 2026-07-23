import json

import pytest

from munbench.items import (
    DataFileError,
    load_rubric,
    load_slop_list,
    load_track1,
    load_track2,
    load_track3,
)

TRACK1_ITEM = {
    "id": "t1-001",
    "title": "회식 자리 갈등",
    "domain": "직장",
    "setup_ko": "회식 자리에서 상사가 후배에게...",
    "model_role_ko": "후배",
    "interlocutor_ko": "상사",
    "turns_ko": ["첫 번째 대사", "두 번째 대사", "세 번째 대사"],
    "phenomena": ["눈치", "체면"],
    "analysis_question_ko": "상사의 속마음은 무엇이었을까요?",
    "judge_notes_ko": "눈치와 체면을 모두 살렸는지 확인.",
}

TRACK2_ITEM = {
    "id": "t2-001",
    "form": "단편소설 장면",
    "prompt_ko": "이별 통보를 받은 인물이 카페에 앉아있다.",
    "constraints_ko": ["대사 없이 묘사만 사용"],
    "targeted_weakness": "restraint over melodrama",
    "length_spec": "600~1200자",
    "judge_notes_ko": "감정 과잉 서술 여부 확인.",
}

TRACK3_ITEM = {
    "id": "t3-001",
    "subtype": "culture-pair",
    "prompt_ko": "한국에서 첫 소개팅에 나간 상황을 묘사하라.",
    "variant_neutral_ko": "첫 소개팅에 나간 상황을 묘사하라.",
    "expected_behavior_ko": "한국 특유의 소개팅 관습이 드러나야 함.",
    "judge_notes_ko": "서구식 데이트 관습으로 대체하지 않았는지 확인.",
}

RUBRIC = {
    "criteria": [
        {
            "name_ko": "감정 진실성",
            "description_ko": "감정 표현이 진솔한가",
            "weight": 1.0,
            "failure_signals_ko": ["과장된 문어체", "설명조 감정"],
        }
    ],
    "notes_ko": "테스트 루브릭",
}

SLOP_LIST = {"phrases": ["마음이 따뜻해지는", "잊지 못할 순간"]}


def test_load_track1_valid(tmp_path):
    path = tmp_path / "track1.json"
    path.write_text(json.dumps([TRACK1_ITEM], ensure_ascii=False), encoding="utf-8")
    items = load_track1(path)
    assert len(items) == 1
    assert items[0].id == "t1-001"
    assert len(items[0].turns_ko) == 3


def test_load_track1_wrong_turn_count_fails(tmp_path):
    bad = dict(TRACK1_ITEM, turns_ko=["only one turn"])
    path = tmp_path / "track1.json"
    path.write_text(json.dumps([bad], ensure_ascii=False), encoding="utf-8")
    with pytest.raises(DataFileError):
        load_track1(path)


def test_load_track2_valid(tmp_path):
    path = tmp_path / "track2.json"
    path.write_text(json.dumps([TRACK2_ITEM], ensure_ascii=False), encoding="utf-8")
    items = load_track2(path)
    assert items[0].length_spec == "600~1200자"


def test_load_track3_culture_pair_has_variant(tmp_path):
    path = tmp_path / "track3.json"
    path.write_text(json.dumps([TRACK3_ITEM], ensure_ascii=False), encoding="utf-8")
    items = load_track3(path)
    assert items[0].subtype == "culture-pair"
    assert items[0].variant_neutral_ko is not None


def test_load_track3_invalid_subtype_fails(tmp_path):
    bad = dict(TRACK3_ITEM, subtype="not-a-real-subtype")
    path = tmp_path / "track3.json"
    path.write_text(json.dumps([bad], ensure_ascii=False), encoding="utf-8")
    with pytest.raises(DataFileError):
        load_track3(path)


def test_load_rubric(tmp_path):
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps(RUBRIC, ensure_ascii=False), encoding="utf-8")
    rubric = load_rubric(path)
    assert rubric.criteria[0].weight == 1.0


def test_load_slop_list(tmp_path):
    path = tmp_path / "slop.json"
    path.write_text(json.dumps(SLOP_LIST, ensure_ascii=False), encoding="utf-8")
    slop = load_slop_list(path)
    assert "마음이 따뜻해지는" in slop.phrases


def test_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(DataFileError, match="not found"):
        load_track1(missing)


def test_invalid_json_raises_clear_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DataFileError, match="Invalid JSON"):
        load_track1(path)
