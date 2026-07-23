# MunBench (문벤치) — Design Draft v0.1

A benchmark for **Korean emotional intelligence, creative writing, and Korean nuance/culture** in LLMs — evaluated generatively, grounded in Korean norms, not translated from English.

## Why this doesn't exist yet (the gap)

- Korean benchmarks are overwhelmingly multiple-choice knowledge tests (KMMLU, CLIcK, HAE-RAE…). Almost nothing tests **generation quality**; nothing tests creative writing or EQ as a first-class target. LogicKor (the closest) used a generic 0–10 judge rubric and was retired for score saturation.
- English EQ/creative benchmarks (EQ-Bench 3, Creative Writing v3) have strong methodology but zero Korean grounding.
- Ko-EQ-Bench (SNU) exists but is small, mechanically scored, and adapted-from-English rather than natively Korean.
- Nunchi-Bench's key finding: factual cultural knowledge barely predicts **applied** cultural competence in generation. So MCQ scores don't tell you if a model can actually *write* with 눈치.

**The open niche: natively-authored, generative, human-validated eval at the intersection of EQ × creative writing × Korean cultural nuance.**

## Three tracks

### Track 1 — 감정 (Korean EQ, multi-turn roleplay)
EQ-Bench 3 format, Korean-native scenarios. Model plays a character through 3–5 turn emotionally charged conversations, then writes an out-of-character analysis of the other party's mental state.

Scenario domains (authored natively in Korean, never translated): 직장 (상사/후배 갈등, 회식), 가족 (시댁/처가, 부모 기대), 친구/연애 (서운함, 거리두기), 서비스/이웃 (갑질, 층간소음). Each scenario embeds at least one of: 눈치 (reading the unstated), 정 (obligation vs boundary), 체면 (face-saving), 존댓말 register pressure, sarcasm/passive-aggression (KoCoSa-style, meaning resolves only from context).

### Track 2 — 문학 (Creative writing)
Short-form generation (600–1,200자), ~30 weakness-exposing prompts across: 단편소설 장면, 수필, 시/자유시, 대화문 (dialogue-driven scene), 장르물 (스릴러/로맨스/SF). Prompts follow EQ-Bench CW v3 philosophy — each targets a known LLM weakness (restraint over melodrama, showing-not-telling, distinct character voice, subtext) — plus lechmazur-style constraint prompts (mandatory disparate elements) to jointly test instruction-following.

### Track 3 — 결 (Korean nuance & culture, applied)
Generative (not MCQ) tasks where nuance is the graded object:
- **Register switching**: rewrite/continue a passage shifting 반말↔존댓말↔극존칭 correctly per relationship (Mi:dm K-Pragmatics framing: honorifics as instruction-following).
- **Idiom deployment**: write a scene where a given 속담/사자성어 is used aptly and naturally (not bolted on).
- **Specified-vs-neutral culture pairs** (Nunchi-Bench trick): same scenario with and without "in Korea" stated — does the model default to Western assumptions when Korean-ness is implicit?
- **Subtext/sarcasm resolution**: continue a dialogue where the last turn is sarcastic/indirect; grading hinges on whether the model read it correctly.

## Scoring pipeline

Two-layer scoring, per EQ-Bench 3:
1. **Absolute rubric pass** (cheap, per-sample): Korean-specific criteria per track — never a generic "quality 0–10" (that's what killed LogicKor). Honorific correctness is its own scored dimension (RULER/VERSE precedent). Rubric criteria drafted → reviewed by native speakers → revised (KorNAT's double-review practice).
2. **Pairwise Elo/TrueSkill pass** (discriminative at the frontier): anonymized model IDs, equal-length truncation before judging, every pair judged in both orders and averaged, Elo anchored to fixed reference models.

**Judge-independent auxiliary metrics** (computed alongside, not judged):
- Korean slop-list hit rate (curated overused-LLM-Korean phrase list — needs building; nothing exists)
- Repetition / n-gram diversity
- **Language-consistency penalty** — code-switching into English is auto-penalized (HRET practice; KUDGE shows judges miss this)

**Judge validation (non-negotiable, per KUDGE):** LLM-judge reliability on Korean tracks the judge's *English* skill and misses cultural misrepresentation. So: (a) validate judge–human agreement on a held-out native-annotated sample before trusting any score; (b) test judge with reasoning ON vs OFF (LitBench + PACLIC: CoT can *hurt* creative/pragmatic judging); (c) swap judge models and publish the delta (self/family bias); (d) sycophancy stress test — can a model game the EQ track by being maximally warm/validating?

**Ground truth:** where "correct" is contested (EQ responses, register appropriateness), score against a small native-Korean annotator consensus distribution (SECEU/KorNAT approach), not one author's opinion.

## Quality controls

- All items authored natively in Korean first (ToMBench order: native → translate for docs → verify).
- Iterative item pruning: drop items that don't separate a ladder of weak→strong models (EQ-Bench practice).
- Report repeatability (score mean/std across N judge iterations, minimum iteration count).
- Contamination: keep a private held-out split; n-gram overlap checks for idiom/culture items (KoBALT practice).
- Report human ceiling on a sample (KoBEST practice).

## Implementation sketch (later phase)

- Python harness (litellm or provider SDKs) → generate → score → leaderboard.
- Phase 1: prompt/scenario authoring + rubric design (the real work; ~30 items/track to start).
- Phase 2: harness + judge pipeline + bias controls.
- Phase 3: judge validation with native annotators; item pruning against a model ladder.
- Phase 4: leaderboard page + slop-list/aux metrics.

## Locked decisions (2026-07-23)

1. **Name:** MunBench (문벤치).
2. **Judge:** ensemble of 3 diverse-family frontier models (config-driven); scores aggregated across judges, cross-family to dilute self/family bias.
3. **Scale:** full — ~160 items: Track 1 감정 = 50 scenarios (5 domains × 10), Track 2 문학 = 48 prompts (6 forms × 8), Track 3 결 = 60 tasks (4 subtypes × 15).
4. **Human annotation:** none at v1, but every result record keeps `human_score` hook fields so a native-rater validation pass can be added without schema changes.
5. **Test set:** one public set, no private split.
