from munbench.items import SlopList
from munbench.metrics import (
    distinct_trigram_ratio,
    language_consistency_flag,
    latin_char_fraction,
    length_compliance,
    longest_repeated_substring,
    parse_length_spec,
    repetition_metrics,
    slop_hits_per_1000_chars,
)


def test_slop_hits_counts_substring_occurrences():
    slop = SlopList(phrases=["마음이 따뜻해지는", "잊지 못할"])
    text = "마음이 따뜻해지는 하루였다. 잊지 못할 순간, 잊지 못할 기억."
    hits = slop_hits_per_1000_chars(text, slop)
    expected = (1 + 2) / len(text) * 1000
    assert hits == expected


def test_slop_hits_zero_when_no_match():
    slop = SlopList(phrases=["없는 표현"])
    assert slop_hits_per_1000_chars("전혀 관계없는 문장입니다.", slop) == 0.0


def test_slop_hits_empty_text():
    slop = SlopList(phrases=["아무거나"])
    assert slop_hits_per_1000_chars("", slop) == 0.0


def test_distinct_trigram_ratio_fully_unique():
    # No repeated 3-char windows -> ratio should be 1.0
    text = "abcdefghij"
    assert distinct_trigram_ratio(text) == 1.0


def test_distinct_trigram_ratio_repetitive_text_is_lower():
    repetitive = "가나다" * 20
    varied = "이것은 서로 다른 문장을 만들어 반복이 적은 텍스트입니다 여러 단어를 사용합니다"
    assert distinct_trigram_ratio(repetitive) < distinct_trigram_ratio(varied)


def test_longest_repeated_substring_detects_repeat():
    text = "가나다라마바사" + "abcdefghij" + "가나다라마바사"
    assert longest_repeated_substring(text) >= 7


def test_longest_repeated_substring_no_repeat():
    assert longest_repeated_substring("abcdefgh") == 0


def test_repetition_metrics_flags_long_repeat():
    long_chunk = "동일한 문장을 여러 번 반복해서 사용합니다 " * 5
    text = long_chunk + " 그리고 결말." + long_chunk
    m = repetition_metrics(text, long_repeat_threshold=30)
    assert m.has_long_repeat is True
    assert m.longest_repeated_substring_len >= 30


def test_latin_char_fraction_pure_korean():
    assert latin_char_fraction("완전히 한국어로만 쓰인 문장입니다") == 0.0


def test_latin_char_fraction_mixed():
    frac = latin_char_fraction("이것은 test 문장입니다")
    assert 0.0 < frac < 1.0


def test_language_consistency_flag_thresholds():
    mostly_korean = "한국어 문장입니다 " * 20 + "OK"
    frac, flagged = language_consistency_flag(mostly_korean, threshold=0.02)
    assert flagged is False

    heavy_english = "This is mostly English text with 조금 한국어."
    frac2, flagged2 = language_consistency_flag(heavy_english, threshold=0.02)
    assert flagged2 is True
    assert frac2 > frac


def test_parse_length_spec_range():
    assert parse_length_spec("600~1200자") == (600, 1200)
    assert parse_length_spec("600-1200자") == (600, 1200)


def test_parse_length_spec_approx():
    bounds = parse_length_spec("약 800자")
    assert bounds is not None
    lo, hi = bounds
    assert lo < 800 < hi


def test_parse_length_spec_unparsable_returns_none():
    assert parse_length_spec("적당한 분량") is None


def test_length_compliance_in_and_out_of_range():
    spec = "10~20자"
    assert length_compliance("가" * 15, spec) is True
    assert length_compliance("가" * 5, spec) is False


def test_length_compliance_unparsable_spec_returns_none():
    assert length_compliance("아무 텍스트", "정해지지 않음") is None
