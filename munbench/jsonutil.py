"""Shared best-effort JSON-object extraction for judge responses.

Both judge_rubric.py and judge_pairwise.py need a fallback for judges that don't
return pure JSON (mainly the CLI backends, which only get a soft "JSON only"
prompt instruction rather than a real structured-output API parameter). A naive
greedy `\\{.*\\}` regex spans from the first '{' to the LAST '}' anywhere in the
string, which breaks as soon as trailing prose happens to contain brace
characters. This extracts the first well-formed JSON *object* instead.
"""

from __future__ import annotations

import json

_decoder = json.JSONDecoder()


def extract_first_json_object(text: str) -> dict | None:
    """Scan `text` for the first well-formed JSON object, trying each '{' in turn
    (via `json.JSONDecoder.raw_decode`) rather than regex-matching brace spans.
    Returns None if no valid JSON object is found anywhere in the text."""
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = _decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None
