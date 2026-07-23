from munbench.elo import fit_elo


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
