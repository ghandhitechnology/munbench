"""Item, rubric, and slop-list schemas + loaders for the three MunBench tracks.

Data files under data/items/ and data/rubrics/ are authored separately; this
module only defines the schema they must conform to and loads/validates them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

# --------------------------------------------------------------------------
# Track 1 — 감정 (Korean EQ, multi-turn roleplay)
# --------------------------------------------------------------------------


class Track1Item(BaseModel):
    id: str
    title: str
    domain: str
    setup_ko: str
    model_role_ko: str
    interlocutor_ko: str
    turns_ko: list[str]
    phenomena: list[str]
    analysis_question_ko: str
    judge_notes_ko: str

    @field_validator("turns_ko")
    @classmethod
    def _exactly_three_turns(cls, v: list[str]) -> list[str]:
        if len(v) != 3:
            raise ValueError(f"turns_ko must have exactly 3 entries, got {len(v)}")
        return v


# --------------------------------------------------------------------------
# Track 2 — 문학 (Creative writing)
# --------------------------------------------------------------------------


class Track2Item(BaseModel):
    id: str
    form: str
    prompt_ko: str
    constraints_ko: list[str] = Field(default_factory=list)
    targeted_weakness: str
    length_spec: str
    judge_notes_ko: str


# --------------------------------------------------------------------------
# Track 3 — 결 (Korean nuance & culture)
# --------------------------------------------------------------------------

Track3Subtype = Literal["register", "idiom", "culture-pair", "subtext"]


class Track3Item(BaseModel):
    id: str
    subtype: Track3Subtype
    prompt_ko: str
    # Only present for subtype == "culture-pair" (see generate.py).
    variant_neutral_ko: str | None = None
    expected_behavior_ko: str
    judge_notes_ko: str


# --------------------------------------------------------------------------
# Rubrics
# --------------------------------------------------------------------------


class RubricCriterion(BaseModel):
    name_ko: str
    description_ko: str
    weight: float
    failure_signals_ko: list[str] = Field(default_factory=list)


class Rubric(BaseModel):
    criteria: list[RubricCriterion]
    notes_ko: str = ""


# --------------------------------------------------------------------------
# Slop list
# --------------------------------------------------------------------------


class SlopList(BaseModel):
    phrases: list[str]


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


class DataFileError(Exception):
    """Raised when a data file is missing or fails schema validation."""


def _load_json(path: Path) -> object:
    if not path.exists():
        raise DataFileError(
            f"Data file not found: {path}. Item/rubric data is authored separately "
            "and is not part of the harness — see DESIGN.md."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise DataFileError(f"Invalid JSON in {path}: {e}") from e


def _load_list(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise DataFileError(f"{path} must contain a JSON array, got {type(raw).__name__}")
    try:
        return [model.model_validate(entry) for entry in raw]
    except ValidationError as e:
        raise DataFileError(f"Schema validation failed for {path}:\n{e}") from e


def load_track1(path: Path) -> list[Track1Item]:
    return _load_list(path, Track1Item)  # type: ignore[return-value]


def load_track2(path: Path) -> list[Track2Item]:
    return _load_list(path, Track2Item)  # type: ignore[return-value]


def load_track3(path: Path) -> list[Track3Item]:
    return _load_list(path, Track3Item)  # type: ignore[return-value]


def load_rubric(path: Path) -> Rubric:
    raw = _load_json(path)
    try:
        return Rubric.model_validate(raw)
    except ValidationError as e:
        raise DataFileError(f"Schema validation failed for {path}:\n{e}") from e


def load_slop_list(path: Path) -> SlopList:
    raw = _load_json(path)
    try:
        return SlopList.model_validate(raw)
    except ValidationError as e:
        raise DataFileError(f"Schema validation failed for {path}:\n{e}") from e


TRACK_ITEM_MODEL: dict[int, type[BaseModel]] = {
    1: Track1Item,
    2: Track2Item,
    3: Track3Item,
}

Item = Track1Item | Track2Item | Track3Item
