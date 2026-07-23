from munbench.judge_pairwise import _score_from_winners, parse_verdict


def test_parse_verdict_a_wins():
    winner, error = parse_verdict('{"winner": "A", "reason": "..."}')
    assert winner == "A"
    assert error is None


def test_parse_verdict_lowercase_and_whitespace():
    winner, error = parse_verdict('{"winner": " b ", "reason": "..."}')
    assert winner == "B"
    assert error is None


def test_parse_verdict_draw_and_tie_synonyms():
    for raw in ('{"winner": "draw"}', '{"winner": "tie"}', '{"winner": "TIE"}'):
        winner, error = parse_verdict(raw)
        assert winner == "draw"
        assert error is None


def test_parse_verdict_malformed_json():
    winner, error = parse_verdict("this is not json at all")
    assert winner is None
    assert error is not None


def test_parse_verdict_unrecognized_value():
    winner, error = parse_verdict('{"winner": "C"}')
    assert winner is None
    assert error is not None


def test_parse_verdict_extracts_from_surrounding_text():
    content = 'Sure, here is my answer:\n{"winner": "A", "reason": "more vivid"}\nThanks.'
    winner, error = parse_verdict(content)
    assert winner == "A"
    assert error is None


def test_score_from_winners_consistent_win():
    # model_a is "A" in order1 and wins; model_a is "B" in order2 and wins -> full win
    assert _score_from_winners("A", "B") == 1.0


def test_score_from_winners_consistent_loss():
    assert _score_from_winners("B", "A") == 0.0


def test_score_from_winners_draw():
    assert _score_from_winners("draw", "draw") == 0.5


def test_score_from_winners_position_bias_averaged():
    # judge always prefers whichever is labeled "A" (pure position bias) -> averages to 0.5
    assert _score_from_winners("A", "A") == 0.5


def test_score_from_winners_missing_data_returns_none():
    assert _score_from_winners(None, None) is None


def test_score_from_winners_single_surviving_order_returns_none():
    # Fix: a single surviving order can't cancel position bias by itself - must not
    # silently fall back to an unaveraged single-order score.
    assert _score_from_winners("A", None) is None
    assert _score_from_winners(None, "B") is None
    assert _score_from_winners("draw", None) is None


def test_parse_verdict_brace_in_trailing_prose():
    # A valid JSON object followed by unrelated prose containing another brace pair -
    # a naive greedy regex would span to the LAST '}' and fail to parse.
    content = '{"winner": "A", "reason": "더 자연스러움"} 참고로 다른 예시는 {이것} 같은 것입니다.'
    winner, error = parse_verdict(content)
    assert winner == "A"
    assert error is None
