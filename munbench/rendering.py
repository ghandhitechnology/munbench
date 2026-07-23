"""Shared "what the judge sees" rendering for track 1 (roleplay) records.

Both judge_rubric.py and judge_pairwise.py need the labeled dialogue
reconstruction (interlocutor turns_ko interleaved with the model's replies) —
this used to be duplicated (rubric had it, pairwise didn't), so it lives here
once and both judging passes use it.
"""

from __future__ import annotations

from munbench.generate import GenerationRecord
from munbench.items import Track1Item

TURN_LABELS = ["[1턴 응답]", "[2턴 응답]", "[3턴 응답]", "[분석]"]


def render_track1_turns(item: Track1Item) -> str:
    """The interlocutor's scripted lines only, labeled by turn number — the shared,
    response-independent part of the script both compared models answered."""
    return "\n".join(f"{i + 1}턴 상대방: {t}" for i, t in enumerate(item.turns_ko))


def render_track1_response(record: GenerationRecord) -> str:
    """One model's track1 outputs (3 in-character turns + 1 out-of-character analysis
    answer), each labeled so a judge can tell roleplay from meta-analysis. Used in
    place of a bare newline-joined blob wherever a single model's track1 output needs
    to be shown on its own (e.g. pairwise comparison)."""
    return "\n\n".join(f"{label} {text}" for label, text in zip(TURN_LABELS, record.outputs))


def render_track1_dialogue(record: GenerationRecord, item: Track1Item) -> str:
    """Full labeled reconstruction of one record: interlocutor turns interleaved with
    this record's in-character replies, plus the analysis Q&A. Used by the rubric
    judge, which scores a single record in isolation and benefits from turns and
    replies being shown side by side rather than as two separate blocks."""
    lines = [
        f"상황: {item.setup_ko}",
        f"모델이 맡은 역할: {item.model_role_ko}",
        f"상대방: {item.interlocutor_ko}",
        "",
        "[대화]",
    ]
    for i, turn in enumerate(item.turns_ko):
        lines.append(f"상대방: {turn}")
        if i < len(record.outputs):
            lines.append(f"모델(연기): {record.outputs[i]}")
    lines.append("")
    lines.append(f"[분석 질문] {item.analysis_question_ko}")
    if len(record.outputs) > 3:
        lines.append(f"[모델의 분석] {record.outputs[3]}")
    return "\n".join(lines)
