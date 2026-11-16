# Teacher Candidate Screening — Phase 8

Generated 2026-10-31T07:22:32Z by `python -m src.training.screen_teachers`.

## Method

Tokenizers only — no weights are loaded, so this screening costs no GPU and cannot leak evaluation data.
Metrics are computed from real tokenizer runs over real Indic bytes:

| metric | definition | direction |
|---|---|---|
| fertility | subword tokens / whitespace word | lower is better |
| chars/token | characters consumed per subword token | higher is better |
| UNK rate | share of unknown tokens | lower is better |
| single-char share | share of 1-character tokens (word must be rebuilt from graphemes) | lower is better |
| truncation @ 64 | share of examples exceeding the project's CPU token budget | lower is better |

**Ranking criterion:** mean fertility over Hindi+Bengali, ascending; ties -> chars/token desc, truncation asc.

## Corpora

- Hindi: 1156 IndicSentiment `INDIC REVIEW` sentences (Devanagari script)
- Bengali: 1156 IndicSentiment `INDIC REVIEW` sentences (Bengali script)
- English: 2300 GWSD sentences — contrast only, **not** used for ranking

IndicSentiment is translated product reviews: it is a valid *tokenizer-coverage* probe because
the bytes are genuine Devanagari and Bengali, but it is not climate-domain text and is never
used here to claim stance performance.

## Ranked results

| # | model | fertility hi | fertility bn | mean | chars/tok | single-char | UNK | trunc@64 | p95 hi / bn | vocab |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `ai4bharat/IndicBERTv2-MLM-only` | 1.272 | 1.405 | **1.339** | 4.16 | 0.130 | 0.0000 | 0.045 | 66 / 57 | 250,000 |
| 2 | `google/muril-base-cased` | 1.310 | 1.451 | **1.381** | 4.04 | 0.152 | 0.0000 | 0.051 | 68 / 59 | 197,258 |
| 3 | `google/muril-large-cased` | 1.310 | 1.451 | **1.381** | 4.04 | 0.152 | 0.0000 | 0.051 | 68 / 59 | 197,258 |
| 4 | `l3cube-pune/hindi-bert-v2` | 1.310 | 1.451 | **1.381** | 4.04 | 0.152 | 0.0000 | 0.051 | 68 / 59 | 197,285 |
| 5 | `l3cube-pune/bengali-bert` | 1.310 | 1.451 | **1.381** | 4.04 | 0.152 | 0.0000 | 0.051 | 68 / 59 | 197,285 |
| 6 | `xlm-roberta-base` | 1.552 | 2.068 | **1.810** | 3.10 | 0.232 | 0.0000 | 0.131 | 80 / 84 | 250,002 |
| 7 | `xlm-roberta-large` | 1.552 | 2.068 | **1.810** | 3.10 | 0.232 | 0.0000 | 0.131 | 80 / 84 | 250,002 |
| 8 | `bert-base-multilingual-cased` | 2.089 | 2.677 | **2.383** | 2.35 | 0.447 | 0.0180 | 0.314 | 109 / 109 | 119,547 |
| 9 | `distilbert-base-multilingual-cased` | 2.089 | 2.677 | **2.383** | 2.35 | 0.447 | 0.0180 | 0.314 | 109 / 109 | 119,547 |

## Not screened

| model | status | error |
|---|---|---|
| `ai4bharat/indic-bert` | not_screened | `OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/ai4bharat/indic` |

## Selection

Rank 1 is `ai4bharat/IndicBERTv2-MLM-only` at mean fertility 1.339 (Hindi 1.272, Bengali 1.405). Selection is not automatic: the top-ranked tokenizer is adopted as **teacher** only if it is also trainable inside the measured CPU budget and licensed for this use, which is checked before any fine-tuning.

