"""Judge-free auxiliary metrics: slop-list hits, n-gram diversity,
language-consistency (code-switching), and length-spec compliance.

These are computed directly on generated text — no LLM calls involved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from munbench.items import SlopList

# --------------------------------------------------------------------------
# Slop-list hit rate
# --------------------------------------------------------------------------


def slop_hits_per_1000_chars(text: str, slop: SlopList) -> float:
    """Count substring occurrences of curated overused-phrase list, normalized per 1000 chars."""
    if not text:
        return 0.0
    hits = sum(text.count(phrase) for phrase in slop.phrases)
    return hits / len(text) * 1000


# --------------------------------------------------------------------------
# Repetition / n-gram diversity
# --------------------------------------------------------------------------


@dataclass
class RepetitionMetrics:
    distinct_trigram_ratio: float | None  # unique trigrams / total trigrams, in [0, 1]; None if too short to measure
    longest_repeated_substring_len: int
    has_long_repeat: bool  # flag: a suspiciously long verbatim repeat exists


def _char_trigrams(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 3:
        return []
    return [text[i : i + 3] for i in range(len(text) - 2)]


def distinct_trigram_ratio(text: str) -> float | None:
    """Unique/total trigram ratio, or None for text too short to have any trigram
    (< 3 non-whitespace-collapsed chars) — that's insufficient signal to call "diverse",
    not the maximum-diversity case, so callers should exclude it from aggregation
    rather than average in a sentinel 1.0."""
    trigrams = _char_trigrams(text)
    if not trigrams:
        return None
    return len(set(trigrams)) / len(trigrams)


def longest_repeated_substring(text: str) -> int:
    """Length of the longest substring that occurs at least twice (non-overlapping start
    positions). O(n log n) via binary search + rolling hash; text is assumed short
    (single LLM completions), so this is fast in practice.
    """
    n = len(text)
    if n < 2:
        return 0

    def has_repeat_of_length(length: int) -> bool:
        if length == 0:
            return True
        seen: set[str] = set()
        for i in range(n - length + 1):
            sub = text[i : i + length]
            if sub in seen:
                return True
            seen.add(sub)
        return False

    lo, hi = 0, n // 2
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if has_repeat_of_length(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def repetition_metrics(text: str, long_repeat_threshold: int = 30) -> RepetitionMetrics:
    lrs = longest_repeated_substring(text)
    return RepetitionMetrics(
        distinct_trigram_ratio=distinct_trigram_ratio(text),
        longest_repeated_substring_len=lrs,
        has_long_repeat=lrs >= long_repeat_threshold,
    )


# --------------------------------------------------------------------------
# Language consistency (code-switching penalty)
# --------------------------------------------------------------------------

_LATIN_RE = re.compile(r"[A-Za-z]")
_ANY_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)  # any alphabetic char, any script


def latin_char_fraction(text: str) -> float:
    """Fraction of alphabetic characters that are Latin-script, out of all alphabetic chars.

    Simple heuristic per DESIGN.md: code blocks are unlikely in these tracks so we don't
    special-case them.
    """
    letters = _ANY_LETTER_RE.findall(text)
    if not letters:
        return 0.0
    latin = _LATIN_RE.findall(text)
    return len(latin) / len(letters)


def language_consistency_flag(text: str, threshold: float = 0.02) -> tuple[float, bool]:
    """Returns (latin_fraction, flagged) — flagged if latin_fraction exceeds threshold."""
    frac = latin_char_fraction(text)
    return frac, frac > threshold


# --------------------------------------------------------------------------
# Length-spec compliance
# --------------------------------------------------------------------------
#
# Track 2's poem items (시/자유시) specify length in 행 (lines) or 연 (stanzas)
# instead of 자 (characters), e.g. "12~20행", "3~5연 (총 12~30행 내외)",
# "4~6연 (각 연 4~5행, 총 16~30행)". A "총 X~Y행" sub-clause, when present, is the
# most reliable bound (it's an explicit total regardless of stanza structure), so
# it's checked first; a bare "X~Y행" or "X~Y연" is checked next.

LengthUnit = str  # "char" | "line" | "stanza"

_TOTAL_LINES_RE = re.compile(r"총\s*(\d+)\s*[~\-–]\s*(\d+)\s*행")
_CHAR_RANGE_RE = re.compile(r"(\d+)\s*[~\-–]\s*(\d+)\s*자")
_LINE_RANGE_RE = re.compile(r"(\d+)\s*[~\-–]\s*(\d+)\s*행")
_STANZA_RANGE_RE = re.compile(r"(\d+)\s*[~\-–]\s*(\d+)\s*연")
_CHAR_APPROX_RE = re.compile(r"(?:약\s*)?(\d+)\s*자\s*(?:내외|안팎)?")


def _ordered(unit: LengthUnit, lo: int, hi: int) -> tuple[LengthUnit, int, int]:
    return (unit, lo, hi) if lo <= hi else (unit, hi, lo)


def parse_length_spec(spec: str) -> tuple[LengthUnit, int, int] | None:
    """Parse a Korean length spec string into (unit, min, max) inclusive bounds, unit
    being "char", "line", or "stanza" depending on what the spec actually counts.

    Returns None if the spec can't be parsed at all (caller should exclude it from
    compliance aggregation entirely — never treat "unparsable" as "non-compliant").
    """
    m = _TOTAL_LINES_RE.search(spec)
    if m:
        return _ordered("line", int(m.group(1)), int(m.group(2)))
    m = _CHAR_RANGE_RE.search(spec)
    if m:
        return _ordered("char", int(m.group(1)), int(m.group(2)))
    m = _LINE_RANGE_RE.search(spec)
    if m:
        return _ordered("line", int(m.group(1)), int(m.group(2)))
    m = _STANZA_RANGE_RE.search(spec)
    if m:
        return _ordered("stanza", int(m.group(1)), int(m.group(2)))
    m = _CHAR_APPROX_RE.search(spec)
    if m:
        n = int(m.group(1))
        tolerance = max(round(n * 0.2), 50)
        return ("char", max(n - tolerance, 0), n + tolerance)
    return None


def _count_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def _count_stanzas(text: str) -> int:
    stanzas = re.split(r"\n\s*\n+", text.strip())
    return sum(1 for s in stanzas if s.strip())


def measure_length(text: str, unit: LengthUnit) -> int:
    if unit == "line":
        return _count_lines(text)
    if unit == "stanza":
        return _count_stanzas(text)
    return len(text)


def length_compliance(text: str, spec: str) -> bool | None:
    """Whether text falls within the parsed length_spec range (measured in whatever
    unit the spec uses — chars, lines, or stanzas). None if spec unparsable."""
    parsed = parse_length_spec(spec)
    if parsed is None:
        return None
    unit, lo, hi = parsed
    return lo <= measure_length(text, unit) <= hi
