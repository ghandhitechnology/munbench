# MunBench (문벤치)

A benchmark for **Korean emotional intelligence, creative writing, and Korean
nuance/culture** in LLMs — evaluated generatively, grounded in Korean norms,
not translated from English. See [`DESIGN.md`](./DESIGN.md) for the full
rationale and methodology.

## Three tracks

- **Track 1 — 감정** (Korean EQ): multi-turn roleplay through a scripted
  emotionally-charged conversation, followed by an out-of-character analysis
  of the other party's mental state.
- **Track 2 — 문학** (Creative writing): short-form generation (600–1,200자)
  across weakness-exposing prompts (단편소설, 수필, 시, 대화문, 장르물).
- **Track 3 — 결** (Korean nuance & culture): register switching, idiom
  deployment, specified-vs-neutral culture pairs, and subtext/sarcasm
  resolution.

## Quickstart

```bash
# install
uv sync                      # or: pip install -e ".[dev]"

# Single-key setup (default config): all models routed through OpenRouter —
# use "openrouter/<provider>/<model>" ids (see openrouter.ai/models) and set:
export OPENROUTER_API_KEY=...

# Alternative: call providers directly — use plain litellm ids in the config
# (gpt-5, gemini/gemini-2.5-pro, anthropic/claude-sonnet-5) and set the
# provider-specific keys instead:
# export OPENAI_API_KEY=... GEMINI_API_KEY=... ANTHROPIC_API_KEY=...

# copy and edit the example config
cp munbench.yaml my-munbench.yaml   # or just edit munbench.yaml in place

# run the pipeline stage by stage — each stage reads the previous stage's
# output files, so they're independently resumable
munbench validate-data --config munbench.yaml
munbench generate --track all --config munbench.yaml
munbench judge --mode rubric --config munbench.yaml
munbench judge --mode pairwise --config munbench.yaml
munbench elo --config munbench.yaml
munbench report --config munbench.yaml
```

Output lands in `results/`: `generations/*.jsonl`, `rubric/*.jsonl`,
`pairwise/comparisons.jsonl`, `elo.json`, and `leaderboard.{md,json}`.

## Config (`munbench.yaml`)

- `models` — litellm model ids under test.
- `judges` — cross-family judge ensemble (default: GPT-5, Gemini 2.5 Pro,
  Claude Sonnet 5) used for both rubric and pairwise scoring, to dilute
  self/family bias.
- `rubric_iterations` — how many times each judge re-scores each sample;
  reported as mean/std for repeatability.
- `pairwise.anchors` — reference models pinned to a fixed Elo rating (1200);
  every tested model is compared against every anchor, plus a capped
  round-robin among tested models (`pairwise.max_comparisons_per_model`).
- `temperature` / `max_tokens` — generation and judging sampling params.
- `concurrency` / `max_retries` / `retry_backoff_seconds` — async request
  throttling and retry behavior.

## Judge-validation caveat (KUDGE)

Scores in this harness are **LLM-judged**. Ensemble judging (3 diverse-family
models), both-orders pairwise averaging, and judge-free auxiliary metrics
(slop-list, repetition, language-consistency) are applied as bias controls —
but per DESIGN.md, judge–human agreement has **not yet been validated against
native-Korean annotators**. Treat leaderboard scores as provisional until a
human-validation pass (tracked via the `human_score` field kept on every
per-sample record) is run.

## Data

Item and rubric content under `data/items/` and `data/rubrics/` is authored
separately (see DESIGN.md) — this harness only defines and validates their
schema (`munbench validate-data`).
