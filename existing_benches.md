# Existing Benchmarks Survey

Name + one-line function of each benchmark relevant to a Korean creative-writing / EQ / nuance benchmark.
(Surveyed 2026-07-23 via web research workflow.)

## EQ / Emotional Intelligence (English)

- **EQ-Bench v1** — 60 questions where an LLM reads a tense dialogue and predicts the intensity of four named emotions felt by a character.
- **EQ-Bench v2 / v2.1** — Expanded to 171 questions with a re-worked scoring curve for better discrimination between models.
- **EQ-Bench 3** — Multi-turn roleplay + analysis benchmark judged by an LLM, using both an absolute rubric and pairwise Elo/TrueSkill to rank active emotional-intelligence skill in generated dialogue.
- **SECEU** — A validated human psychometric EQ test (40 situational items) repurposed to score LLMs against a real human-population norm distribution rather than a single reference answer.
- **EmoBench** — 400 hand-crafted bilingual (EN/ZH) multiple-choice questions on Emotional Understanding and Emotional Application, requiring implicit reasoning rather than emotion-word pattern-matching.
- **ToMBench** — 2,860-item bilingual from-scratch multiple-choice Theory-of-Mind benchmark spanning 8 classic ToM tasks and 31 social-cognitive abilities.

## Creative Writing (English)

- **EQ-Bench Creative Writing v3** — LLM-judged short-form creative writing using 32 deliberately weakness-exposing prompts, scored by both isolated rubric grading and pairwise Elo.
- **EQ-Bench Longform Creative Writing** — Novella-length generation (plan → reflect → 8 chapters), judged per-chapter and whole-piece with structural-degradation penalties.
- **WritingBench** — 1,000 queries across 6 domains using per-instance LLM-generated criteria scored by a fine-tuned critic model instead of a fixed rubric.
- **NoveltyBench** — Measures whether a model can produce multiple genuinely distinct high-quality answers to the same prompt (anti-mode-collapse), not just one best output.
- **LitBench** — 2,480 debiased human-labeled Reddit story comparisons showing off-the-shelf LLM judges underperform small trained reward models on creative-writing preference.
- **Chatbot Arena — Creative Writing category** — Crowdsourced blind pairwise-battle Elo leaderboard filtered to real user creative-writing conversations (revealed human preference).
- **LLM Creative Story-Writing Benchmark (lechmazur)** — Pairwise leaderboard requiring stories to incorporate 10 mandatory disparate elements, rated via Thurstone scaling of LLM-judge comparisons.
- **LongJudgeBench** — Meta-benchmark testing whether LLM-as-judge is reliable on long-form outputs (~9k tokens avg), exposing systematic judge failure modes.

## Korean — General / Generation

- **KMMLU (+ Redux / Pro)** — Massive Korean multiple-choice knowledge benchmark built from original Korean exams, not translated MMLU.
- **HAE-RAE Bench** — Six-task benchmark (loan words, rare words, nomenclature, knowledge, history, RC) built to punish models whose "Korean" is really transferred English knowledge.
- **KoBEST** — The original GLUE-style Korean benchmark: five human-annotated NLU tasks (BoolQ, COPA, WiC, HellaSwag, sentiment-negation).
- **LogicKor** — Multi-turn multi-domain Korean generation benchmark scored by an LLM judge 0–10; one of the few testing free-form Korean generation (retired due to frontier score saturation).
- **KUDGE** — Korean meta-evaluation benchmark: evaluates how trustworthy LLM-judges/reward-models are when judging Korean text.
- **KorNAT** — "National alignment" benchmark measuring whether an LLM's values and knowledge match the actual Korean population's, via a 6,174-person survey (scored against answer distributions).
- **Open Ko-LLM Leaderboard 2 (incl. Ko-Harmlessness / Ko-Helpfulness)** — Public Korean leaderboard aggregating 9 benchmarks spanning reasoning, instruction-following, EQ, safety, and Korean knowledge/values.
- **Horangi (W&B Korean LLM Leaderboard)** — Korean leaderboard combining a Q&A comprehension harness with a Korean-translated MT-Bench for generative ability.
- **KITE** — Dedicated Korean instruction-following benchmark, generation-based/open-ended, with a Korean-specific linguistic-phenomena subset.
- **KoBBQ / KoSBi / SQuARe / KOLD** — Cluster of Korean bias/toxicity/sensitive-question safety benchmarks, mostly classification rather than generation quality.
- **HRET (HAE-RAE Evaluation Toolkit)** — Open-source toolkit standardizing how Korean benchmarks are run (includes language-consistency / anti-code-switching checks).

## Korean — Culture / Nuance / Linguistics

- **CLIcK** — 1,995 multiple-choice QA pairs from real Korean exams/textbooks split into Language and Culture (society, tradition, history, pop-culture) categories.
- **K-Viscuit** — 657-item multiple-choice VQA testing whether vision-language models understand Korean cultural objects and practices.
- **KoCommonGEN v2** — Human-annotated multiple-choice Korean commonsense benchmark catching outputs that violate everyday Korean social/commonsense norms.
- **Nunchi-Bench** — 247 questions over 31 Korean superstition topics mixing factual MCQs, "trap" scenarios, and cultural-interpretation items — tests applying cultural knowledge in context, not reciting it.
- **KULTURE Bench** — Cultural comprehension via Korean news, idioms (사자성어/속담), and poetry, evaluated at word, sentence, and paragraph granularity.
- **KoBALT** — 700 expert-written multiple-choice questions covering 24 linguistic phenomena across syntax, semantics, pragmatics, phonology, and morphology.
- **Pragmatic Competence Evaluation for Korean** — PACLIC 2024 eval of context-dependent (pragmatic) Korean expressions using paired auto-graded MCQs and human-graded open-ended questions.
- **Mi:dm 2.0 K-Pragmatics / K-Referential** — Benchmark suite testing honorifics, Sino-Korean vs native vocabulary, proverbs, and tone/register constraints as instruction-following tasks.

## Gap-check additions

- **Ko-EQ-Bench (SNU)** — Korean-localized adaptation of EQ-Bench testing emotional-intensity prediction in Korean interpersonal-conflict dialogues (small dataset, mechanically scored).
- **BLEnD** — 16-country, 13-language everyday-cultural-knowledge benchmark treating South and North Korea as distinct cultures under one language.
- **Ko-Sovereign** — Korean journal-published benchmark evaluating depth of Korea-specific expert/civic knowledge across 9 domains.
- **KoCoSa** — 12.8K-dialogue Korean context-aware sarcasm detection dataset where sarcasm only resolves given preceding dialogue context.
- **KorT** — Korean translation benchmark scoring deliberately hard-to-translate sentences (ambiguity, idioms, cultural references) with an LLM judge.
- **RULER & VERSE** — Two complementary rubrics for judging EN→KO literary translation, scoring honorific correctness as its own category.
- **RPEval** — 9,018-scenario role-play benchmark scoring emotional understanding, moral alignment, and in-character knowledge consistency separately.
- **CharacterBench** — Largest bilingual generative role-play benchmark (22,859 samples, 3,956 characters), scoring 11 dimensions via a purpose-trained judge model.
