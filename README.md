# MunBench (문벤치)

**Can your model read the room in Korean?**

MunBench is a benchmark for what almost no benchmark measures: whether an LLM can actually *write* in Korean — with emotional intelligence, literary craft, and the cultural instincts that separate a fluent model from one that merely knows the grammar. Not "pick the right answer about Korean culture." Actually navigate a 회식-politics minefield, write a poem about loss without saying 슬프다, and know when "아니야 괜찮아" means anything but 괜찮아.

## Why another benchmark?

We surveyed 40+ existing benchmarks before building this one (receipts in [`existing_benches.md`](./existing_benches.md)). The pattern is stark:

- Korean benchmarks (KMMLU, CLIcK, HAE-RAE, KoBALT…) are almost entirely **multiple-choice knowledge tests**. Cheap to grade, contamination-resistant — and completely silent on whether a model can *produce* good Korean.
- The generative exceptions used a generic 0–10 "quality" judge rubric and died of score saturation at the frontier (RIP LogicKor).
- English EQ and creative-writing benchmarks (EQ-Bench 3, Creative Writing v3) have excellent methodology and zero Korean grounding.
- And Nunchi-Bench delivered the killer finding: a model's factual score on Korean culture questions **barely correlates** with whether it applies that culture correctly in open-ended scenarios. Knowing *about* 눈치 is not having it.

The intersection — EQ × creative writing × Korean nuance, evaluated generatively — was empty. MunBench fills it. All 158 items were authored natively in Korean (never translated), critic-reviewed, and each one is designed to *split the field*: an item that every model aces teaches us nothing.

## The three tracks

### Track 1 — 감정 (Korean EQ) · 50 scenarios

Multi-turn roleplay through emotionally loaded, very Korean situations. The model plays a character; the other party's three turns are **pre-scripted** (written to stay coherent after any plausible reply), so every model walks into the identical conversation. After the roleplay, an out-of-character analysis question: *what was that person actually feeling?*

A taste: you're 김대리. Your 팀장 just found out the team had a 회식 without him. He corners you at the 탕비실:

> "아니 뭐 서운하다는 건 아니고. 원래 젊은 사람들끼리 편하게 노는 게 낫지… 근데 그래도 '팀장님도 한잔 하실래요' 하고 한 번쯤 물어봐 주는 게 예의는 예의 아닌가 싶어서. 하하, 내가 꼰대인가?"

He denies being hurt twice, then insists on buying you coffee. A strong model reads the 서운함 under the denial, leaves his 체면 intact, and signals he'll be included next time — without groveling, without naming his feelings at him, and without lying that everyone talked about him all night. Scenarios span 직장, 가족, 친구·연애, 이웃·서비스, 학교·선후배, each tagged with the phenomena it stresses: 눈치, 정, 체면, 존댓말 pressure, 반어, 갑을관계.

### Track 2 — 문학 (Creative writing) · 48 prompts

Short-form commissions (mostly 600–1,200자) across six forms: 단편소설 장면, 수필, 시, 대화문, 장르물, and constraint stories with mandatory unrelated elements. Every prompt targets a documented LLM weakness:

- Write a school essay in an actual 12-year-old's voice — not an adult novelist doing precocious-child cosplay.
- A poem about someone now gone, where the person may only appear through what they left behind: 두 개였다가 하나가 된 것, 이제 아무도 끄지 않는 것.
- A granddaughter clearing out her late grandmother's room — with direct emotion statements ("보고 싶다", tears, wailing) **banned**. Objects and silence only.

The enemy here is 신파 — the melodrama reflex. Restraint scores; slop doesn't.

### Track 3 — 결 (Nuance & culture) · 60 tasks

The Korean-specific machinery, tested in generation:

- **경어 전환** (15): rewrite or continue a scene across register shifts — 반말→격식 존대, 존대→반말 between people growing close, 극존칭 for customer-facing scripts, and the genuinely hard cases (a same-age coworker you just met; first dinner with 시댁).
- **관용구** (15): weave a given 속담/사자성어 (언 발에 오줌 누기, 견문발검…) into a scene *naturally* — no "이런 말 있잖아" scaffolding, no dictionary definitions. Includes trap items with near-miss idioms that models love to confuse.
- **문화 명시/비명시 쌍** (15): the same task twice — once stating the setting is Korea, once not (though every name and circumstance is Korean). Does the model quietly default to hugs, first names, and Western funeral etiquette when Korean-ness is merely implied? This is where translated-soul models get caught.
- **반어·서브텍스트** (15): a dialogue ends with "아 그래? …아니야 괜찮아. 일이 우선이지 뭐." Respond to what she means, not what she said. Some items carry a **sincere-version control** — same words, context where she genuinely means it — so a model that hallucinates 서운함 everywhere fails too. Reading subtext that isn't there is also an EQ failure.

33 items generate contrast variants, so a full run produces ~191 outputs per model.

## How scoring works

Every output gets scored three independent ways, because each way has a known blind spot:

**1. Rubric (interpretable).** Each track has a weighted, Korean-specific rubric — 눈치·정·체면 reading, emotional granularity, honorific correctness (its own criterion, never buried in "fluency"), restraint/anti-sycophancy, cliché avoidance. A 3-judge ensemble scores every criterion 0–10, **three times each** (repeatability is reported as mean ± std, and per-judge scores are stored so you can see disagreement — and self-preference — directly).

**2. Elo (discriminative).** Rubric scores compress at the frontier; head-to-head choices don't. Judges compare two anonymized outputs on the same item — truncated to equal length (kills length bias), judged in both A/B orders and averaged (kills position bias) — and the results are fitted into Elo ratings. Three **anchor models are pinned at 1200** as a top/mid/floor ladder: Claude Opus 4.6 (ceiling), Solar Pro 3 (Korean-native mid reference — every rating implicitly answers "how does this compare to a dedicated Korean model?"), GPT-5.6 Luna (floor). Pinned anchors mean adding model #10 next month doesn't move models #1–9.

**3. Judge-free metrics (the tripwires).** No LLM opinions involved: slop hits per 1,000 chars against a curated 185-phrase Korean LLM-cliché list (마음 한켠, 심장이 쿵 내려앉…, translationese like ~에 다름 아니다), n-gram repetition, length compliance, and **code-switch detection** — Latin characters above 2% gets flagged, because per KUDGE, LLM judges reliably fail to notice English leaking into Korean text.

## Quickstart

```bash
uv sync                            # or: pip install -e ".[dev]"
export OPENROUTER_API_KEY=...      # single key, all models routed via OpenRouter

munbench validate-data             # schema-check items/rubrics/slop list
munbench generate --track all      # ~191 outputs per model
munbench judge --mode rubric       # 3 judges × 3 iterations per sample
munbench judge --mode pairwise     # anonymized A/B battles vs anchors + round-robin
munbench elo                       # fit ratings, anchors pinned at 1200
munbench report                    # leaderboard.md / leaderboard.json
```

Each stage reads the previous stage's files from `results/`, so everything is resumable and re-runnable independently. Prefer direct provider APIs? Use plain litellm ids in the config and set per-provider keys instead — the harness doesn't care.

## Config (`munbench.yaml`)

| Key | What it does |
|---|---|
| `models` | Models under test (litellm ids). Anchors must be listed here too — their generations are the reference points. |
| `judges` | The scoring ensemble. Cross-family on purpose. |
| `rubric_iterations` | Re-scores per judge per sample; the std is your noise floor. |
| `pairwise.anchors` | Elo reference models, pinned at 1200. |
| `pairwise.max_comparisons_per_model` | Caps round-robin cost as the roster grows. |
| `concurrency` / `max_retries` | Async throttling for API calls. |

## Honest caveats

- **Scores are LLM-judged and not yet human-validated.** The bias controls above are real, but per KUDGE, judge–human agreement on Korean text must be *measured*, not assumed. Every per-sample record carries a `human_score: null` field so a native-annotator validation pass can be added without schema changes. Until then, treat the leaderboard as provisional.
- **Self-preference bias.** If judges overlap with tested models, each model gets a home-field judge. Anonymization hides names, not style. Read the per-judge breakdown — if every model is ranked #1 by its own judge, that's the bias talking.
- **The test set is public**, so it can be trained on. Contamination checks matter for any model released after this repo.

## Repo map

```
data/items/       158 items across 3 tracks (native-Korean, critic-reviewed)
data/rubrics/     per-track weighted rubrics
data/slop_list.json   185 Korean LLM-slop phrases
munbench/         the harness: generate → judge → elo → report
DESIGN.md         full methodology + the research it's built on
existing_benches.md   the 40+ benchmarks surveyed, one line each
```

## Prior art, gratefully borrowed

EQ-Bench 3 (roleplay format, rubric+Elo dual scoring, bias-control playbook) · Nunchi-Bench (knowledge ≠ applied competence; specified-vs-neutral pairs) · KUDGE (judge-validation warning) · KorNAT (consensus over single gold labels) · RULER/VERSE (honorifics as a standalone criterion) · LitBench & LongJudgeBench (LLM-judge failure modes) · HRET (language-consistency checks) · KoCoSa (context-dependent sarcasm). Full list in [`existing_benches.md`](./existing_benches.md).
