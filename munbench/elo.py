"""Elo rating fit from pairwise comparisons.jsonl (simple iterative Bradley-Terry style
update, no external dependency). Anchor models are pinned at a fixed rating."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from munbench.config import Settings
from munbench.judge_pairwise import PairwiseComparison, load_comparisons

logger = logging.getLogger(__name__)

Game = tuple[str, str, float]  # (model_a, model_b, score_a in [0, 1])

ANCHOR_RATING = 1200.0


def games_from_comparisons(comparisons: list[PairwiseComparison]) -> list[Game]:
    """Only comparisons with a score AND no recorded error enter the Elo fit. In
    practice judge_pairwise's _score_from_winners already guarantees score_a is None
    whenever either label-order failed, so this is a belt-and-suspenders check on the
    consumption side against the position-bias-cancellation invariant, not just an
    internal one."""
    return [(c.model_a, c.model_b, c.score_a) for c in comparisons if c.score_a is not None and c.error is None]


def fit_elo(
    games: list[Game],
    anchors: list[str],
    initial_rating: float = ANCHOR_RATING,
    k: float = 16.0,
    epochs: int = 200,
    k_decay: float = 0.99,
) -> dict[str, float]:
    """Iterative Elo update: repeatedly sweep all games, nudging ratings toward the
    Bradley-Terry MLE. Anchor models never update and stay at `initial_rating`.
    """
    anchor_set = set(anchors)
    ratings: dict[str, float] = {}
    for a, b, _ in games:
        ratings.setdefault(a, initial_rating)
        ratings.setdefault(b, initial_rating)

    step = k
    for _ in range(epochs):
        for model_a, model_b, score_a in games:
            ra, rb = ratings[model_a], ratings[model_b]
            expected_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
            delta = step * (score_a - expected_a)
            if model_a not in anchor_set:
                ratings[model_a] = ra + delta
            if model_b not in anchor_set:
                ratings[model_b] = rb - delta
        step *= k_decay

    return ratings


def run_elo(settings: Settings) -> dict[str, float]:
    path = settings.paths.pairwise_comparisons
    if not path.exists():
        raise FileNotFoundError(f"No pairwise comparisons found at {path}. Run `munbench judge --mode pairwise` first.")
    comparisons = load_comparisons(path)
    games = games_from_comparisons(comparisons)
    if not games:
        logger.warning("no valid (parsed) comparisons to fit Elo from")
        return {}
    ratings = fit_elo(games, anchors=settings.pairwise.anchors)
    write_elo(ratings, settings.paths.elo_file)
    return ratings


def write_elo(ratings: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ratings, ensure_ascii=False, indent=2), encoding="utf-8")


def load_elo(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
