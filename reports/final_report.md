# Indic Climate Stance Challenge - final report

Task: three-way stance (FAVOUR / AGAINST / NONE) on the claim *"Climate change and global warming is a serious concern."* Primary metric: **macro-F1**.

> **Headline caveat, stated first.** No Hindi or Bengali *stance* labels exist in this repository. Every supervised number below is measured on **GWStance**, an English news corpus whose FAVOUR/AGAINST/NONE labels are the argmax of a Bayesian posterior over crowd annotations. The pipeline, tokenizer analysis and efficiency measurements are language-independent and transfer; the stance accuracy does not.

## 1. Data actually used

| resource | rows | what it is | role | licence |
|---|---|---|---|---|
| `gwstance` | 2300 | English news sentences, derived posterior-argmax labels | dev corpus | none stated (research use only) |
| `dhoroni` | 2290 | Bengali news articles, no stance column | domain / tokenizer probe | MIT |
| `indic_sentiment_hi` | 1156 | machine-translated product reviews (sentiment) | auxiliary probe | CC-BY-NC 4.0 |
| `indic_sentiment_bn` | 1156 | machine-translated product reviews (sentiment) | auxiliary probe | CC-BY-NC 4.0 |

**Custom Hindi-Bengali corpus: 0 rows.** The annotation guideline (`docs/annotation_guideline.md`), collection tool (`src/data/collection.py`) and agreement machinery (`src/data/agreement.py`) are built and tested, but no human annotations have been produced. No synthetic rows were ever written into a gold split.

## 2. Teacher selection (Phase 8)

Ranked by mean Hindi/Bengali tokenizer fertility on real text.

| rank | tokenizer | fert hi | fert bn | mean | UNK | trunc@64 | vocab |
|---|---|---|---|---|---|---|---|
| 1 | `ai4bharat/IndicBERTv2-MLM-only` | 1.2722 | 1.4053 | 1.3388 | 0.00e+00 | 0.0450 | 250,000 |
| 2 | `google/muril-base-cased` | 1.3101 | 1.4510 | 1.3805 | 2.70e-05 | 0.0506 | 197,258 |
| 3 | `google/muril-large-cased` | 1.3101 | 1.4510 | 1.3805 | 2.70e-05 | 0.0506 | 197,258 |
| 4 | `l3cube-pune/hindi-bert-v2` | 1.3101 | 1.4510 | 1.3805 | 2.70e-05 | 0.0506 | 197,285 |
| 5 | `l3cube-pune/bengali-bert` | 1.3101 | 1.4510 | 1.3805 | 2.70e-05 | 0.0506 | 197,285 |
| 6 | `xlm-roberta-base` | 1.5520 | 2.0680 | 1.8100 | 0.00e+00 | 0.1310 | 250,002 |
| 7 | `xlm-roberta-large` | 1.5520 | 2.0680 | 1.8100 | 0.00e+00 | 0.1310 | 250,002 |
| 8 | `bert-base-multilingual-cased` | 2.0889 | 2.6774 | 2.3832 | 1.80e-02 | 0.3135 | 119,547 |
| 9 | `distilbert-base-multilingual-cased` | 2.0889 | 2.6774 | 2.3832 | 1.80e-02 | 0.3135 | 119,547 |

`ai4bharat/indic-bert` could not be screened (gated repository, HTTP 401). Rank 2-5 are one distinct vocabulary shared by four checkpoints, so the table has 5 unique tokenizers in 9 rows.

**Selected teacher: `google/muril-base-cased`.** It loses 0.0416 mean fertility to IndicBERTv2 but is ungated and fits the memory budget; the student inherits its vocabulary.

## 3. Classical baselines (Phase 7)

Best baseline `tfidf-word-logreg-C0.3`: test macro-F1 **0.5774**, accuracy 0.5798, 95% bootstrap CI [0.5163, 0.6321] over 1000 resamples.

## 4. Teacher, student and distillation (Phases 9-10)

| model | split | macro-F1 | accuracy | F1 FAVOUR | F1 AGAINST | F1 NONE |
|---|---|---|---|---|---|---|
| teacher | val | 0.1824 | 0.3766 | 0.5472 | 0.0000 | 0.0000 |
| teacher | test | 0.1828 | 0.3779 | 0.5485 | 0.0000 | 0.0000 |
| student (no KD) | val | 0.2590 | 0.4123 | 0.5477 | 0.0000 | 0.2293 |
| student (no KD) | test | 0.2619 | 0.4137 | 0.5468 | 0.0000 | 0.2390 |
| student + KD | val | 0.3318 | 0.4481 | 0.4789 | 0.0000 | 0.5166 |
| student + KD | test | 0.3433 | 0.4658 | 0.5298 | 0.0000 | 0.5000 |

Distillation gain: **+0.0814 macro-F1** (test), T=2.0, alpha=0.5, beta=0.5, soft targets exported once to disk so teacher and student never coexist in RAM.

Parameters: teacher 237,853,443 (297,219 trainable), student 180,953,859 (100,611 trainable). Total-parameter ratio 1.31x, **trainable** ratio 2.95x.

The student is *not* smaller in total parameters - layer pruning to 4 of 12 was offset by the classification head and embeddings. It is 2.95x cheaper to fine-tune, which is the honest claim.

## 5. PEFT ablation (Phase 10)

Full fine-tuning needs ~3806.0 MB for AdamW state alone (237853443 params x 16 bytes) against 1.9 GB total RAM, so **it was not run**. LoRA r=8 on q/v peaked at 905.0 MB. The full-vs-PEFT comparison is therefore a memory/feasibility measurement, not an accuracy comparison.

## 6. Augmentation ablation (Phase 11)

| augmentation | n train | test macro-F1 | accuracy |
|---|---|---|---|
| none | 1435 | 0.2069 | 0.3779 |
| char_emoji | 2870 | 0.3149 | 0.4267 |
| romanize | 2870 | 0.1991 | 0.3876 |

At a fixed 36-step budget the ordering is unstable between runs (character/emoji noise beat no augmentation here, romanisation hurt); with 1,435 training rows and 36 steps these differences are within noise. **No augmentation is claimed to help.**

## 7. Robustness (Phase 12)

| perturbation | macro-F1 | retention |
|---|---|---|
| clean | 0.2151 | 1.0000 |
| char_elongation | 0.2138 | 0.9940 |
| spacing_noise | 0.2046 | 0.9512 |
| emoji_suffix | 0.2098 | 0.9754 |
| hashtag_suffix | 0.2151 | 1.0000 |
| partial_romanization | 0.2151 | 1.0000 |
| english_codemix | 0.2101 | 0.9768 |
| all_noise | 0.2033 | 0.9451 |

Mean retention 0.9803. Retention is high because the model predicts FAVOUR for almost everything: a degenerate classifier is trivially robust. This is a property of the failure, not a strength.

## 8. Explainability (Phase 13)

KD student confusion matrix (rows gold: FAVOUR/AGAINST/NONE): `[[116, 0, 0], [58, 0, 2], [125, 0, 6]]`; 185 errors on 307 test rows.

| text | predicted | confidence | top IG tokens |
|---|---|---|---|
| जलवायु परिवर्तन एक गंभीर समस्या है। | FAVOUR | 0.405 | ।, है, समस्या, जलवायु |
| ग्लोबल वार्मिंग कोई गंभीर समस्या नहीं है। | FAVOUR | 0.395 | ।, है, समस्या, नहीं |
| জলবায়ু পরিবর্তন একটি গুরুতর উদ্বেগের বিষয়। | FAVOUR | 0.401 | ।, বিষয়, জলবায়ু, পরিবর্তন |
| জলবায়ু পরিবর্তন কোনো গুরুতর সমস্যা নয়। | FAVOUR | 0.418 | ।, সমস্যা, কোনো, জলবায়ু |
| सरकार की जलवायु नीति बिल्कुल फेल है, पैसा बर्बाद हो रहा है। | FAVOUR | 0.372 | ।, जलवायु, ,, बिल्कुल |
| climate change koi serious problem nahi hai | FAVOUR | 0.409 | serious, hai, climate, koi |

Integrated Gradients are implementable and run; they are attribution heuristics, not causal explanations. Note the model assigns high attribution to the danda `।`, a sentence-final punctuation mark - a signature of a model keying on length/style cues.

## 9. Efficiency (Phase 14)

- Student parameters: 180,953,859 (trainable 100,611)
- p50 latency, batch 1: 15.46 ms; p95 18.21 ms
- Throughput, batch 1: 60.76 items/s; batch 8: 220.83 items/s
- Peak RSS during measurement: 949.0 MB
- INT8 dynamic quantisation: **fail** (not delivered)

## 10. What could not be run here

| phase | item | why |
|---|---|---|
| 4 | custom hi-bn corpus collection | needs human annotators and platform API keys |
| 10 | full fine-tuning vs PEFT accuracy comparison | 3.8 GB optimiser state vs 1.9 GB RAM |
| 12 | cross-lingual Hindi/Bengali evaluation | no labelled hi/bn stance data |
| 14 | INT8 quantised model | dynamic quantisation failed on this torch build |
| 16 | live Streamlit inference | adapter-only checkpoint needs the 236M encoder resident |

## 11. Reproduction

```bash
python -m pytest tests -q                 # unit tests
python -m src.training.run_baselines      # Phase 7
python -m src.training.screen_teachers    # Phase 8
python src/training/phase9_lora_kd.py     # Phases 9-10
python src/evaluation/phase11_14.py       # Phases 11-14
python src/report/build_final_report.py   # Phases 15/17
python -m streamlit run app/streamlit_app.py
```
