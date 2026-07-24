# MunBench — Initial Run (2026-07-24)

First real leaderboard: **13 models, 780 samples, 3-judge ensemble, zero errors.** This is a
budget-constrained pilot — a stratified **47-item subset** (~60 records/model incl. contrast
variants) with **1 rubric iteration** per judge — so treat scores as a first signal, not a final
verdict. Full config: [`configs/initial-run.yaml`](../configs/initial-run.yaml). Total cost: ~$41
in API credits + one Claude membership doing 780 free judge calls.

## Ranked leaderboard (overall rubric, 0–10)

| # | Model | Effort | Overall | Elo† | 감정 (EQ) | 문학 (Writing) | 결 (Nuance) | Code-switch |
|--:|---|---|--:|--:|--:|--:|--:|--:|
| 1 | **Claude Fable 5** | high | **8.59** | 1469 | 8.69 | 8.19 | 8.87 | 4.7% |
| 2 | **Claude Opus 4.8** ⚓ | high | **8.48** | 1200 | 8.46 | 8.15 | 8.81 | 1.4% |
| 3 | **GPT-5.6 Sol** | high | **8.27** | **1547** | 8.57 | 7.51 | 8.73 | 1.4% |
| 4 | **GPT-5.6 Luna** ⚓ | — | **8.12** | 1200 | 8.26 | 7.87 | 8.24 | 0.0% |
| 5 | **GPT-5.6 Terra** | high | **8.07** | 1441 | 8.73 | 7.01 | 8.46 | 0.0% |
| 6 | **Kimi K3** | max | **7.92** | 1226 | 8.16 | 7.27 | 8.34 | 2.8% |
| 7 | **Claude Sonnet 5** | high | **7.72** | 1221 | 8.27 | 6.30 | 8.58 | 0.0% |
| 8 | **DeepSeek V4 Pro** | high | **7.39** | 1188 | 7.87 | 6.23 | 8.08 | 4.7% |
| 9 | **Gemini 3.6 Flash** | high | **6.84** | 1040 | 7.04 | 5.00 | 8.48 | 5.1% |
| 10 | **Grok 4.5** | high | **6.78** | 1195 | 6.43 | 6.91 | 6.99 | 19.6% |
| 11 | **GLM 5.2** | max | **6.59** | 1157 | 7.71 | 5.37 | 6.68 | 3.2% |
| 12 | **MiniMax M3** | — | **5.43** | 1115 | 7.84 | 1.67 | 6.79 | 8.3% |
| 13 | **Solar Pro 3** ⚓ | — | **3.41** | 1200 | 3.21 | 2.39 | 4.62 | 13.9% |

† Elo from a capped pairwise pass; ⚓ = anchor, **pinned at 1200 by definition** (their Elo is the
reference scale, not a measurement — read their strength from the rubric column).

## Overall rubric score

```
Claude Fable 5    high  8.59 ████████████████████████████████████████████▏
Claude Opus 4.8   high  8.48 ███████████████████████████████████████████▋
GPT-5.6 Sol       high  8.27 ██████████████████████████████████████████▌
GPT-5.6 Luna      —     8.12 █████████████████████████████████████████▊
GPT-5.6 Terra     high  8.07 █████████████████████████████████████████▌
Kimi K3           max   7.92 ████████████████████████████████████████▊
Claude Sonnet 5   high  7.72 ███████████████████████████████████████▋
DeepSeek V4 Pro   high  7.39 ██████████████████████████████████████
Gemini 3.6 Flash  high  6.84 ███████████████████████████████████▏
Grok 4.5          high  6.78 ██████████████████████████████████▊
GLM 5.2           max   6.59 █████████████████████████████████▉
MiniMax M3        —     5.43 ███████████████████████████▉
Solar Pro 3       —     3.41 █████████████████▌
                             0        2        4        6        8       10
```

## 문학 (creative writing) — where the field actually splits

The EQ and nuance tracks compress near the top; creative writing is what separates models:

```
Claude Fable 5    8.19 ██████████████████████████████████████████
Claude Opus 4.8   8.15 █████████████████████████████████████████▊
GPT-5.6 Luna      7.87 ████████████████████████████████████████▍
GPT-5.6 Sol       7.51 ██████████████████████████████████████▌
Kimi K3           7.27 █████████████████████████████████████▎
GPT-5.6 Terra     7.01 ████████████████████████████████████
Grok 4.5          6.91 ███████████████████████████████████▍
Claude Sonnet 5   6.30 ████████████████████████████████▎
DeepSeek V4 Pro   6.23 ████████████████████████████████
GLM 5.2           5.37 ███████████████████████████▌
Gemini 3.6 Flash  5.00 █████████████████████████▋
MiniMax M3        1.67 ████████▌
Solar Pro 3       2.39 ████████████▎
                       0        2        4        6        8       10
```

## What the numbers say

- **Claude Fable 5 leads the rubric; GPT-5.6 Sol leads Elo.** The two scoring systems disagreeing
  at the top is expected — rubric rewards consistent per-criterion quality, Elo rewards winning
  direct matchups. With 1 judge iteration and a capped pairwise pass, top-3 order should be
  treated as a statistical tie pending the full run.
- **Creative writing (문학) is the discriminating track**, exactly as the benchmark was designed:
  spread of 1.7–8.2 vs. EQ's 3.2–8.7 where 8 of 13 models sit within one point. MiniMax M3's
  collapse (1.67, with 25% of 문학 outputs code-switching out of Korean) shows the track catching
  a model that superficially handles Korean conversation but cannot sustain literary register.
- **The culture-pair delta caught real behavior**: GLM 5.2 scores **−1.86** (does *better* when
  Korea is unstated — i.e., its explicitly-Korean writing gets *worse*), and Grok 4.5 and
  MiniMax M3 also go negative. Every frontier Western model holds a positive but small delta
  (+0.2–0.4). This is the "영어를 한국어로 입고 있는 모델" detector working.
- **Grok 4.5's 결 (nuance) problem is code-switching**: 41.7% of its Track-3 outputs drifted into
  Latin script — the judge-free tripwire catching what LLM judges are documented to miss.
- **Solar Pro 3 (the only Korean flagship on OpenRouter) finishing last is the most surprising
  result** — last on every track, 30% code-switch on EQ roleplay. Worth a manual read of its
  transcripts before drawing conclusions; if it holds, "Korean-specialized" ≠ "good at Korean
  writing" is itself a headline finding.
- **Judge behavior sanity check**: Gemini Flash judges ~1.5 points more generously than Opus 4.6
  and Luna, but all three judges *rank* models near-identically (Flash-the-judge also scored
  Flash-the-contestant 9th — no self-rescue). Directionally the ensemble agrees; absolute scores
  inherit Flash's optimism, which the full run's 3-iteration σ will quantify.

## Caveats (read before quoting numbers)

1. **Subset run**: 47 of 158 items, stratified across every track/subtype. Full-set scores will move.
2. **1 rubric iteration** per judge (budget) — the ± columns in `leaderboard.md` reflect
   cross-item spread, not judge repeatability.
3. **Judges = Opus 4.6 (membership CLI) + Gemini 3.6 Flash + GPT-5.6 Luna (API)**, not the
   config-default trio; pairwise used the two API judges only.
4. **Muse Spark 1.1 was dropped**: OpenRouter returns a hard 403 — model is US-only, and this run
   was made from Korea.
5. **"4 other Korean flagships" couldn't be included**: HyperCLOVA X, EXAONE, A.X, Mi:dm are not
   on OpenRouter — they need direct provider integrations (future work). Solar Pro 3 is the only
   Korean flagship OpenRouter hosts.
6. Scores are **LLM-judged, not human-validated** (see README's KUDGE caveat). `human_score`
   hooks are in every record for a future native-rater pass.

Raw artifacts: `results/initial-run/` (generations, per-judge rubric scores, pairwise
comparisons, Elo fit, leaderboard.md/json) — local only, not committed; the tables above are the
faithful summary. Reproduce with:
`munbench run --config configs/initial-run.yaml`
