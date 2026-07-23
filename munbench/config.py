"""Configuration loaded from munbench.yaml (see repo root for an example)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_JUDGES = [
    "gpt-5",
    "gemini/gemini-2.5-pro",
    "anthropic/claude-sonnet-5",
]


class PairwiseConfig(BaseModel):
    enabled: bool = True
    truncate_chars: int = 4000
    anchors: list[str] = Field(default_factory=list)
    max_comparisons_per_model: int = 40


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")

    @property
    def items_dir(self) -> Path:
        return self.data_dir / "items"

    @property
    def rubrics_dir(self) -> Path:
        return self.data_dir / "rubrics"

    @property
    def track1_items(self) -> Path:
        return self.items_dir / "track1_emotion.json"

    @property
    def track2_items(self) -> Path:
        return self.items_dir / "track2_literary.json"

    @property
    def track3_items(self) -> Path:
        return self.items_dir / "track3_nuance.json"

    def rubric(self, track: int) -> Path:
        return self.rubrics_dir / f"track{track}.json"

    @property
    def slop_list(self) -> Path:
        return self.data_dir / "slop_list.json"

    @property
    def generations_dir(self) -> Path:
        return self.results_dir / "generations"

    @property
    def rubric_results_dir(self) -> Path:
        return self.results_dir / "rubric"

    @property
    def pairwise_dir(self) -> Path:
        return self.results_dir / "pairwise"

    @property
    def pairwise_comparisons(self) -> Path:
        return self.pairwise_dir / "comparisons.jsonl"

    @property
    def elo_file(self) -> Path:
        return self.results_dir / "elo.json"

    @property
    def leaderboard_md(self) -> Path:
        return self.results_dir / "leaderboard.md"

    @property
    def leaderboard_json(self) -> Path:
        return self.results_dir / "leaderboard.json"


class Settings(BaseModel):
    models: list[str] = Field(default_factory=list)
    judges: list[str] = Field(default_factory=lambda: list(DEFAULT_JUDGES))
    rubric_iterations: int = 3
    pairwise: PairwiseConfig = Field(default_factory=PairwiseConfig)
    temperature: float = 1.0
    max_tokens: int = 2048
    concurrency: int = 8
    # Separate, lower concurrency cap for claude-cli/ and codex-cli/ models (see
    # providers.py): subscription CLI sessions are heavier than a plain API call and
    # subscription rate limits are tighter than API rate limits.
    cli_concurrency: int = 2
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @classmethod
    def load(cls, path: str | Path = "munbench.yaml") -> "Settings":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}. Copy the example munbench.yaml "
                "at the repo root and adjust it, or pass --config."
            )
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)

    def model_slug(self, model: str) -> str:
        """Filesystem-safe slug for a litellm model id (e.g. 'gemini/gemini-2.5-pro' -> 'gemini_gemini-2.5-pro')."""
        return model.replace("/", "_")
