from munbench.elo import fit_elo, games_from_comparisons
from munbench.judge_pairwise import PairwiseComparison


def test_fit_elo_stronger_model_ends_up_higher():
    # model_a wins every game against model_b
    games = [("strong", "weak", 1.0) for _ in range(30)]
    ratings = fit_elo(games, anchors=[], epochs=100)
    assert ratings["strong"] > ratings["weak"]


def test_fit_elo_anchor_stays_pinned():
    games = [("anchor", "challenger", 0.0) for _ in range(30)]  # challenger always wins
    ratings = fit_elo(games, anchors=["anchor"], initial_rating=1200.0, epochs=100)
    assert ratings["anchor"] == 1200.0
    assert ratings["challenger"] > 1200.0


def test_fit_elo_evenly_matched_models_converge_close():
    games = []
    for _ in range(20):
        games.append(("a", "b", 1.0))
        games.append(("a", "b", 0.0))
    ratings = fit_elo(games, anchors=[], epochs=150)
    assert abs(ratings["a"] - ratings["b"]) < 25


def test_fit_elo_transitive_ordering():
    # a beats b, b beats c -> expect rating(a) > rating(b) > rating(c)
    games = [("a", "b", 1.0)] * 20 + [("b", "c", 1.0)] * 20
    ratings = fit_elo(games, anchors=[], epochs=150)
    assert ratings["a"] > ratings["b"] > ratings["c"]


def test_fit_elo_empty_games_returns_empty():
    assert fit_elo([], anchors=["anchor"]) == {}


def test_games_from_comparisons_excludes_error_even_with_score(): # fix 2
    # Reproduces the exact defect: score_a is non-null (from a single surviving
    # order) while error is also non-null - must be excluded from the Elo fit.
    comparisons = [
        PairwiseComparison(
            track=1, item_id="t1-001", variant=None, model_a="m1", model_b="m2", judge="j1",
            order1_winner="A", order2_winner=None, score_a=1.0, error="call failed: timeout",
        ),
        PairwiseComparison(
            track=1, item_id="t1-002", variant=None, model_a="m1", model_b="m2", judge="j1",
            order1_winner="A", order2_winner="B", score_a=1.0, error=None,
        ),
    ]
    games = games_from_comparisons(comparisons)
    assert games == [("m1", "m2", 1.0)]


def test_games_from_comparisons_excludes_none_score():
    comparisons = [
        PairwiseComparison(
            track=1, item_id="t1-001", variant=None, model_a="m1", model_b="m2", judge="j1",
            order1_winner=None, order2_winner=None, score_a=None, error="call failed",
        ),
    ]
    assert games_from_comparisons(comparisons) == []
