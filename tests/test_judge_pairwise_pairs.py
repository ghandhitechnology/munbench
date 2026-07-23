"""Tests for judge_pairwise.py's pair-sampling (dedupe + anchor-awareness, fix 9) and
the track1 labeled-dialogue rendering used in pairwise context (fix 1)."""

from munbench.generate import GenerationRecord
from munbench.items import Track1Item
from munbench.judge_pairwise import build_comparison_pairs, item_context_and_notes, pairwise_response_text

# --------------------------------------------------------------------------
# Fix 9: anchor-aware, deduped round-robin pair sampling
# --------------------------------------------------------------------------


def _gen_index_for(models, keys):
    """Build a minimal gen_index: every model has a (fake) record for every key."""
    return {
        m: {k: GenerationRecord(item_id=k[1], track=k[0], variant=k[2], transcript=[], outputs=["x"], model=m) for k in keys}
        for m in models
    }


def test_build_comparison_pairs_no_anchor_vs_anchor():
    tested_models = ["m1", "m2", "anchorA", "anchorB"]
    anchors = ["anchorA", "anchorB"]
    keys = [(2, "t2-001", None), (2, "t2-002", None)]
    gen_index = _gen_index_for(tested_models, keys)

    pairs = build_comparison_pairs(tested_models, anchors, gen_index, max_comparisons_per_model=40)
    for p in pairs:
        assert not (p.model_a in anchors and p.model_b in anchors)


def test_build_comparison_pairs_no_exact_duplicates():
    tested_models = ["m1", "m2", "anchorA", "anchorB"]
    anchors = ["anchorA", "anchorB"]
    keys = [(2, "t2-001", None), (2, "t2-002", None), (2, "t2-003", None)]
    gen_index = _gen_index_for(tested_models, keys)

    pairs = build_comparison_pairs(tested_models, anchors, gen_index, max_comparisons_per_model=40)
    seen = set()
    for p in pairs:
        a, b = sorted((p.model_a, p.model_b))
        key = (p.track, p.item_id, p.variant, a, b)
        assert key not in seen, f"duplicate pair: {key}"
        seen.add(key)


def test_build_comparison_pairs_tested_vs_anchor_still_produced():
    tested_models = ["m1", "anchorA"]
    anchors = ["anchorA"]
    keys = [(2, "t2-001", None)]
    gen_index = _gen_index_for(tested_models, keys)

    pairs = build_comparison_pairs(tested_models, anchors, gen_index, max_comparisons_per_model=40)
    assert len(pairs) == 1
    assert {pairs[0].model_a, pairs[0].model_b} == {"m1", "anchorA"}


def test_build_comparison_pairs_round_robin_excludes_anchors_as_primary():
    # anchorA should never appear as model_a/model_b together with anchorB, and
    # round-robin among non-anchor tested models should still work.
    tested_models = ["m1", "m2", "m3", "anchorA"]
    anchors = ["anchorA"]
    keys = [(2, "t2-001", None)]
    gen_index = _gen_index_for(tested_models, keys)

    pairs = build_comparison_pairs(tested_models, anchors, gen_index, max_comparisons_per_model=40)
    non_anchor_pairs = [p for p in pairs if p.model_a not in anchors and p.model_b not in anchors]
    assert len(non_anchor_pairs) == 3  # m1-m2, m1-m3, m2-m3


# --------------------------------------------------------------------------
# Fix 1: track1 pairwise context includes turns_ko; responses are labeled
# --------------------------------------------------------------------------

TRACK1_ITEM = Track1Item(
    id="t1-001",
    title="t",
    domain="직장",
    setup_ko="설정",
    model_role_ko="후배",
    interlocutor_ko="상사",
    turns_ko=["첫 대사", "둘째 대사", "셋째 대사"],
    phenomena=["눈치"],
    analysis_question_ko="분석 질문",
    judge_notes_ko="노트",
)


def test_item_context_and_notes_track1_includes_turns():
    items_index = {1: {"t1-001": TRACK1_ITEM}, 2: {}, 3: {}}
    context, notes = item_context_and_notes(1, "t1-001", None, items_index)
    for turn in TRACK1_ITEM.turns_ko:
        assert turn in context
    assert notes == "노트"


def test_pairwise_response_text_labels_track1_outputs():
    record = GenerationRecord(
        item_id="t1-001", track=1, transcript=[], outputs=["응답1", "응답2", "응답3", "분석응답"], model="m"
    )
    text = pairwise_response_text(record)
    assert "[1턴 응답] 응답1" in text
    assert "[2턴 응답] 응답2" in text
    assert "[3턴 응답] 응답3" in text
    assert "[분석] 분석응답" in text


def test_pairwise_response_text_track2_uses_plain_join():
    record = GenerationRecord(item_id="t2-001", track=2, transcript=[], outputs=["텍스트"], model="m")
    assert pairwise_response_text(record) == "텍스트"
