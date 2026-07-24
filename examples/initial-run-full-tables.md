# MunBench Leaderboard

Scores are LLM-judged (ensemble + bias controls, not yet human-validated — see README).

| Model | Overall | Elo | T1 rubric | T1 slop/1k | T1 switch% | T2 rubric | T2 slop/1k | T2 switch% | T3 rubric | T3 slop/1k | T3 switch% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| openrouter/anthropic/claude-fable-5@high | 8.59 | 1540 | 8.69 ± 0.40 | 0.06 | 10.0% | 8.19 ± 0.90 | 0.00 | 0.0% | 8.87 ± 0.77 | 0.00 | 4.2% |
| openrouter/anthropic/claude-opus-4.8@high | 8.48 | 1200 | 8.46 ± 0.51 | 0.00 | 0.0% | 8.15 ± 0.75 | 0.07 | 0.0% | 8.81 ± 0.41 | 0.00 | 4.2% |
| openrouter/openai/gpt-5.6-sol@high | 8.27 | 1515 | 8.57 ± 0.50 | 0.10 | 0.0% | 7.51 ± 2.63 | 0.10 | 0.0% | 8.73 ± 0.54 | 0.01 | 4.2% |
| openrouter/openai/gpt-5.6-luna | 8.12 | 1200 | 8.26 ± 0.66 | 0.00 | 0.0% | 7.87 ± 0.96 | 0.00 | 0.0% | 8.24 ± 1.18 | 0.00 | 0.0% |
| openrouter/openai/gpt-5.6-terra@high | 8.07 | 1444 | 8.73 ± 0.39 | 0.03 | 0.0% | 7.01 ± 3.03 | 0.00 | 0.0% | 8.46 ± 0.95 | 0.00 | 0.0% |
| openrouter/moonshotai/kimi-k3@max | 7.92 | 1278 | 8.16 ± 0.67 | 0.04 | 0.0% | 7.27 ± 2.28 | 0.00 | 0.0% | 8.34 ± 1.25 | 0.00 | 8.3% |
| openrouter/anthropic/claude-sonnet-5@high | 7.72 | 1341 | 8.27 ± 0.53 | 0.20 | 0.0% | 6.30 ± 2.82 | 0.00 | 0.0% | 8.58 ± 0.51 | 0.00 | 0.0% |
| openrouter/deepseek/deepseek-v4-pro@high | 7.39 | 1245 | 7.87 ± 0.60 | 0.00 | 10.0% | 6.23 ± 2.32 | 0.11 | 0.0% | 8.08 ± 1.06 | 0.05 | 4.2% |
| openrouter/google/gemini-3.6-flash@high | 6.84 | 1118 | 7.04 ± 0.80 | 0.35 | 0.0% | 5.00 ± 2.31 | 0.00 | 15.4% | 8.48 ± 0.49 | 0.42 | 0.0% |
| openrouter/x-ai/grok-4.5@high | 6.78 | 1241 | 6.43 ± 0.76 | 0.00 | 10.0% | 6.91 ± 1.29 | 0.00 | 7.7% | 6.99 ± 1.30 | 0.08 | 41.7% |
| openrouter/z-ai/glm-5.2@max | 6.59 | 1222 | 7.71 ± 0.88 | 0.25 | 0.0% | 5.37 ± 3.11 | 0.00 | 0.0% | 6.68 ± 2.76 | 0.16 | 9.5% |
| openrouter/minimax/minimax-m3 | 5.43 | 1171 | 7.84 ± 0.61 | 0.10 | 0.0% | 1.67 ± 2.78 | 0.00 | 25.0% | 6.79 ± 2.55 | 0.00 | 0.0% |
| openrouter/upstage/solar-pro-3 | 3.41 | 1200 | 3.21 ± 1.42 | 0.17 | 30.0% | 2.39 ± 0.96 | 0.14 | 7.7% | 4.62 ± 1.76 | 0.14 | 4.2% |

## Per-judge mean score (self-preference check)

| Model | claude-cli/claude-opus-4-6 | openrouter/google/gemini-3.6-flash | openrouter/openai/gpt-5.6-luna |
|---|---|---|---|
| openrouter/anthropic/claude-fable-5@high | 8.26 | 9.63 | 8.24 |
| openrouter/anthropic/claude-opus-4.8@high | 7.92 | 9.66 | 8.11 |
| openrouter/openai/gpt-5.6-sol@high | 7.54 | 9.44 | 8.16 |
| openrouter/openai/gpt-5.6-luna | 7.07 | 9.20 | 8.09 |
| openrouter/openai/gpt-5.6-terra@high | 7.36 | 9.20 | 7.96 |
| openrouter/moonshotai/kimi-k3@max | 7.58 | 9.09 | 7.68 |
| openrouter/anthropic/claude-sonnet-5@high | 7.07 | 9.20 | 7.58 |
| openrouter/deepseek/deepseek-v4-pro@high | 6.82 | 9.02 | 7.12 |
| openrouter/google/gemini-3.6-flash@high | 6.62 | 8.72 | 6.91 |
| openrouter/x-ai/grok-4.5@high | 6.23 | 7.81 | 6.81 |
| openrouter/z-ai/glm-5.2@max | 6.19 | 7.86 | 6.37 |
| openrouter/minimax/minimax-m3 | 5.46 | 7.07 | 5.52 |
| openrouter/upstage/solar-pro-3 | 2.98 | 3.61 | 4.09 |

## Completion / error rates

| Model | Generation errors | Judge-slot failures |
|---|---|---|
| openrouter/anthropic/claude-fable-5@high | 0/60 (0.0%) | 0/180 (0.0%) |
| openrouter/anthropic/claude-opus-4.8@high | 0/60 (0.0%) | 0/180 (0.0%) |
| openrouter/openai/gpt-5.6-sol@high | 0/60 (0.0%) | 0/180 (0.0%) |
| openrouter/openai/gpt-5.6-luna | 0/60 (0.0%) | 0/180 (0.0%) |
| openrouter/openai/gpt-5.6-terra@high | 0/60 (0.0%) | 0/180 (0.0%) |
| openrouter/moonshotai/kimi-k3@max | 0/60 (0.0%) | 0/180 (0.0%) |
| openrouter/anthropic/claude-sonnet-5@high | 0/60 (0.0%) | 0/180 (0.0%) |
| openrouter/deepseek/deepseek-v4-pro@high | 0/60 (0.0%) | 0/180 (0.0%) |
| openrouter/google/gemini-3.6-flash@high | 0/60 (0.0%) | 0/180 (0.0%) |
| openrouter/x-ai/grok-4.5@high | 0/60 (0.0%) | 0/180 (0.0%) |
| openrouter/z-ai/glm-5.2@max | 0/60 (0.0%) | 0/180 (0.0%) |
| openrouter/minimax/minimax-m3 | 0/60 (0.0%) | 0/180 (0.0%) |
| openrouter/upstage/solar-pro-3 | 0/60 (0.0%) | 0/180 (0.0%) |

## Track 3 culture-pair: specified vs. neutral (명시/비명시)

| Model | N pairs | Specified mean | Neutral mean | Delta (specified − neutral) |
|---|---|---|---|---|
| openrouter/anthropic/claude-fable-5@high | 6 | 9.12 | 9.05 | 0.07 |
| openrouter/anthropic/claude-opus-4.8@high | 6 | 8.66 | 8.53 | 0.13 |
| openrouter/openai/gpt-5.6-sol@high | 6 | 8.98 | 8.85 | 0.13 |
| openrouter/openai/gpt-5.6-luna | 6 | 8.14 | 8.31 | -0.17 |
| openrouter/openai/gpt-5.6-terra@high | 6 | 8.63 | 8.60 | 0.03 |
| openrouter/moonshotai/kimi-k3@max | 6 | 8.71 | 8.60 | 0.12 |
| openrouter/anthropic/claude-sonnet-5@high | 6 | 8.22 | 8.18 | 0.04 |
| openrouter/deepseek/deepseek-v4-pro@high | 6 | 8.13 | 8.37 | -0.24 |
| openrouter/google/gemini-3.6-flash@high | 6 | 8.42 | 8.30 | 0.12 |
| openrouter/x-ai/grok-4.5@high | 6 | 6.50 | 6.70 | -0.20 |
| openrouter/z-ai/glm-5.2@max | 6 | 4.73 | 7.57 | -2.84 |
| openrouter/minimax/minimax-m3 | 6 | 7.51 | 7.49 | 0.02 |
| openrouter/upstage/solar-pro-3 | 6 | 3.28 | 2.60 | 0.69 |
