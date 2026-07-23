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


def test_weighted_score_partial_credit_when_incomplete():
    # Only "눈치" matched -> renormalized over just that criterion's weight, not None.
    assert weighted_score({"눈치": 10.0}, RUBRIC) == 10.0


def test_weighted_score_none_when_nothing_matched():
    assert weighted_score({}, RUBRIC) is None


# --------------------------------------------------------------------------
# Fix: greedy `\{.*\}` regex broke when trailing prose contained braces.
# --------------------------------------------------------------------------


def test_parse_judge_response_brace_in_trailing_prose():
    # A valid JSON object followed by unrelated prose that happens to contain
    # another, unrelated brace pair. A naive greedy regex spans to the LAST '}' and
    # produces invalid JSON; the fix extracts the first well-formed object instead.
    content = '이 응답을 보면 {"눈치": 8, "체면": 6} 정도로 평가할 수 있는데, 다른 예시로 {잘못된 형식}도 언급해봅니다.'
    scores, error = parse_judge_response(content, RUBRIC)
    assert error is None
    assert scores == {"눈치": 8.0, "체면": 6.0}


# --------------------------------------------------------------------------
# Fix: exact-match criterion names broke on whitespace/punctuation drift.
# --------------------------------------------------------------------------

PUNCT_RUBRIC = Rubric(
    criteria=[
        RubricCriterion(name_ko="절제·진정성 (아부 방지)", description_ko="d1", weight=1.0, failure_signals_ko=[]),
        RubricCriterion(name_ko="존댓말 정확성", description_ko="d2", weight=1.0, failure_signals_ko=[]),
    ]
)


def test_parse_judge_response_tolerates_punctuation_drift():
    # No space before the parenthesis, unlike the canonical name — this exact drift
    # is present between the shipped track1/track3 rubrics themselves.
    content = '{"절제·진정성(아부 방지)": 7, "존댓말 정확성": 9}'
    scores, error = parse_judge_response(content, PUNCT_RUBRIC)
    assert error is None
    assert scores["절제·진정성 (아부 방지)"] == 7.0
    assert scores["존댓말 정확성"] == 9.0


def test_parse_judge_response_tolerates_whitespace_and_dash_variants():
    content = '{"절제 진정성 아부 방지": 5, "존댓말정확성": 8}'
    scores, error = parse_judge_response(content, PUNCT_RUBRIC)
    assert error is None
    assert scores["절제·진정성 (아부 방지)"] == 5.0
    assert scores["존댓말 정확성"] == 8.0


def test_parse_judge_response_unambiguous_substring_fallback():
    # Judge drops the parenthetical entirely - still an unambiguous single candidate.
    content = '{"절제·진정성": 6, "존댓말 정확성": 9}'
    scores, error = parse_judge_response(content, PUNCT_RUBRIC)
    assert error is None
    assert scores["절제·진정성 (아부 방지)"] == 6.0


def test_parse_judge_response_partial_match_still_reports_missing():
    content = '{"절제·진정성(아부 방지)": 7}'
    scores, error = parse_judge_response(content, PUNCT_RUBRIC)
    assert scores == {"절제·진정성 (아부 방지)": 7.0}
    assert error is not None
    assert "존댓말 정확성" in error
    # And the partial result still yields a (renormalized) weighted_score, not None.
    assert weighted_score(scores, PUNCT_RUBRIC) == 7.0
