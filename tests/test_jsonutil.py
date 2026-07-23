from munbench.jsonutil import extract_first_json_object


def test_extract_first_json_object_simple():
    assert extract_first_json_object('{"a": 1}') == {"a": 1}


def test_extract_first_json_object_with_leading_prose():
    text = 'Sure, here you go: {"a": 1, "b": 2}'
    assert extract_first_json_object(text) == {"a": 1, "b": 2}


def test_extract_first_json_object_ignores_trailing_brace_garbage():
    # The historical bug: a greedy `\{.*\}` regex spans to the LAST '}' in the
    # string, which breaks when trailing prose happens to contain brace characters.
    text = '{"a": 1} and then some unrelated {garbage} after it'
    assert extract_first_json_object(text) == {"a": 1}


def test_extract_first_json_object_skips_non_object_json_first():
    # First '{' that parses is an array-adjacent decoy... actually first char here
    # is '{' starting a real object, so this just confirms objects are preferred
    # over any bracketed non-object value found later.
    text = 'not json [1, 2, 3] but here is one: {"winner": "A"}'
    assert extract_first_json_object(text) == {"winner": "A"}


def test_extract_first_json_object_returns_none_when_absent():
    assert extract_first_json_object("no json here at all") is None


def test_extract_first_json_object_returns_none_for_malformed_braces():
    assert extract_first_json_object("{not valid json at all}") is None
