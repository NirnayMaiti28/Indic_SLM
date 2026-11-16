# Distilling a Compact Multilingual Encoder for Climate Stance Detection in Hindi and Bengali

*Course research report. All figures are reproduced from `experiments/results/*.json`.*

## Abstract

We study three-way stance detection (FAVOUR / AGAINST / NONE) toward the claim that climate change is a serious concern, targeting Hindi and Bengali social-media comments across Devanagari, Bengali and Romanised scripts. We screen nine tokenizers on real Hindi and Bengali text, build an end-to-end pipeline with group-aware splits, and distil a LoRA-adapted 4-layer MuRIL student from a LoRA-adapted 12-layer teacher using offline temperature-scaled soft targets. On the English GWStance development corpus the distilled student reaches macro-F1 0.3433 against 0.2619 without distillation (+0.0814) and 0.1828 for the teacher. We report these as *pipeline validation*, not as Hindi/Bengali stance performance: the bilingual stance corpus that the claim depends on was not collected.

## 1. Introduction

Stance toward climate claims in Indic social media is a hard low-resource problem: three scripts, heavy code-mixing, negation, sarcasm and implicit stance. A deployable system must also be small. We therefore treat dataset construction, tokenizer choice, distillation and robustness as one measurement problem, and we report negative results explicitly.

## 2. Data

GWStance (2,300 English sentences, 253 exact duplicates) is used as a development corpus with labels defined as posterior argmax over crowd annotations - a *derived* label, not the authors' gold. Dhoroni (2,290 Bengali news articles) has no stance column and is used only for domain/tokenizer probing. IndicSentiment (1,156 Hindi + 1,156 Bengali) is machine-translated product reviews; sentiment is never mapped to stance, and a unit test asserts that no such mapping exists in code.

## 3. Method

Tokenizers are ranked by fertility (tokens per whitespace word) on real Hindi/Bengali sentences, tie-broken by characters per token and truncation rate at `max_length=64`. The teacher is MuRIL-base (237,853,443 parameters) with LoRA r=8 on query/value (297,219 trainable). The student prunes to 4 encoder layers, uses mean pooling (100,611 trainable). Distillation uses

$$L = \alpha L_{CE} + \beta T^2 L_{KL}(\mathrm{softmax}(z_s/T) \| \mathrm{softmax}(z_t/T))$$

with T=2, alpha=beta=0.5, teacher logits exported once to disk to fit a 1.9 GB budget.

## 4. Results

Tokenization: `ai4bharat/IndicBERTv2-MLM-only` is best (fertility 1.3388 mean, zero UNK, 4.50% truncation) but gated-adjacent and heavy; MuRIL is second (1.3805, 5.06% truncation) and was chosen. mBERT-style vocabularies are disqualified (2.3832 fertility, 31.35% truncation, 1.80% UNK).

| model | test macro-F1 | accuracy |
|---|---|---|
| TF-IDF word + logistic regression (best baseline) | 0.5774 | 0.5798 |
| teacher (MuRIL-12 + LoRA) | 0.1828 | 0.3779 |
| student (4 layers, no KD) | 0.2619 | 0.4137 |
| student (4 layers, distilled) | 0.3433 | 0.4658 |

## 5. Discussion - what failed and why

1. **AGAINST recall is 0.0 for every neural model.** With 450 AGAINST sentences in 2,300, 36 optimisation steps and a class-blind objective, the models collapse onto the majority boundary. Macro-F1 near 0.34 is a two-class score wearing a three-class label.

2. **Robustness is an artefact.** Mean retention 0.9803 under perturbation measures stability of a near-constant predictor.

3. **The student is not smaller in total parameters** (1.31x ratio) though 2.95x cheaper to fine-tune. Reporting only the second number would misrepresent the contribution.

4. **Distillation helps consistently** (+0.0814) but from a very low base.

## 6. Limitations

No Hindi/Bengali stance labels; English dev corpus with derived labels; 36-step budget on CPU; no INT8 artifact; adapter-only checkpoints that cannot be served without the base encoder. The central scientific claim of the project - bilingual stance detection with a compact model - remains **unverified**.

## 7. Ethics

Public comments only, no usernames stored, annotator IDs hashed, adjudication recorded, no re-identification attempted. Platform terms respected; no scraping of restricted content.

