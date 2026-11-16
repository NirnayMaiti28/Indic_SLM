# Phase status

| phase | scope | state | key measured result |
|---|---|---|---|
| 1-3 | scaffold, env, data audit | done | 2,300 GWStance / 2,290 Dhoroni / 1,156+1,156 IndicSentiment |
| 4 | annotation protocol + collection tooling | tooling done, **no data** | 0 rows collected |
| 5 | EDA | done | 26.23 tokens/item probe, 1,221 negation items |
| 6 | preprocessing A/B | done | char retention 1.0074 |
| 7 | classical baselines | done | best macro-F1 0.5774 |
| 8 | tokenizer screening | done | IndicBERTv2 1.3388 / MuRIL 1.3805 |
| 9-10 | teacher, student, KD, PEFT | done | KD 0.3433 vs 0.2619 |
| 11 | augmentation ablation | done | none 0.2069 / char+emoji 0.3149 |
| 12 | robustness | done | mean retention 0.9803 |
| 13 | explainability | done | 185 errors, IG working |
| 14 | efficiency | done, INT8 failed | 15.46 ms p50 b1 |
| 15 | report + paper | done | reports/final_report.md |
| 16 | Streamlit demo | done (report mode) | compiles, no live checkpoint |
| 17 | packaging | done | zip + model cards |
