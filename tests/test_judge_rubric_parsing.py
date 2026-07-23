from munbench.items import Rubric, RubricCriterion
from munbench.judge_rubric import parse_judge_response, weighted_score

RUBRIC = Rubric(
    criteria=[
        RubricCriterion(name_ko="눈치", description_ko="d1", weight=2.0, failure_signals_ko=[]),
        RubricCriterion(name_ko="체면", description_ko="d2", weight=1.0, failure_signals_ko=[]),
    ]
)


def test_parse_judge_response_valid_json():
    content = '{"눈치": 8, "체면": 6.5}'
    scores, error = parse_judge_response(content, RUBRIC)
    assert error is None
    assert scores == {"눈치": 8.0, "체면": 6.5}


def test_parse_judge_response_extracts_json_from_markdown_fence():
    content = '```json\n{"눈치": 7, "체면": 9}\n```'
    scores, error = parse_judge_response(content, RUBRIC)
    assert error is None
    assert scores["눈치"] == 7.0
    assert scores["체면"] == 9.0


def test_parse_judge_response_clamps_out_of_range_scores():
    content = '{"눈치": 15, "체면": -3}'
    scores, error = parse_judge_response(content, RUBRIC)
    assert error is None
    assert scores["눈치"] == 10.0
    assert scores["체면"] == 0.0


def test_parse_judge_response_missing_criterion_reports_error():
    content = '{"눈치": 8}'
    scores, error = parse_judge_response(content, RUBRIC)
    assert error is not None
    assert "체면" in error
    assert scores == {"눈치": 8.0}


def test_parse_judge_response_garbage_returns_empty_and_error():
    scores, error = parse_judge_response("죄송하지만 채점할 수 없습니다.", RUBRIC)
    assert scores == {}
    assert error is not None


def test_parse_judge_response_non_dict_json():
    scores, error = parse_judge_response("[1, 2, 3]", RUBRIC)
    assert scores == {}
    assert error is not None


def test_weighted_score_full_scores():
    scores = {"눈치": 10.0, "체면": 0.0}
    # weighted mean: (2*10 + 1*0) / 3
    assert weighted_score(scores, RUBRIC) == 20 / 3


def test_weighted_score_none_when_incomplete():
    assert weighted_score({"눈치": 10.0}, RUBRIC) is None
