# Student - 4-layer MuRIL + LoRA (distilled)

*Base encoder:* `google/muril-base-cased`  
*Adapters:* LoRA r=8 q/v + head  
*Pooling:* mean  
*Total parameters:* 180,953,859  
*Trainable parameters:* 100,611  

## Measured test performance (dev corpus, n = 307)

| metric | value |
|---|---|
| macro-F1 | 0.3433 |
| accuracy | 0.4658 |
| macro-precision | 0.3169 |
| macro-recall | 0.3902 |
| weighted F1 | 0.4135 |
| balanced accuracy | 0.3902 |
| Cohen's kappa | 0.1139 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| FAVOUR | 0.4301 | 0.6897 | 0.5298 | 116 |
| AGAINST | 0.0000 | 0.0000 | 0.0000 | 60 |
| NONE | 0.5207 | 0.4809 | 0.5000 | 131 |

## Confusion matrix (rows = gold, cols = FAVOUR/AGAINST/NONE)

`[[80, 0, 36], [38, 0, 22], [68, 0, 63]]`

## Intended use and limits

- Trained on GWStance (English news sentences, derived posterior-argmax labels).
- **Not** a Hindi/Bengali stance model: no Hindi/Bengali stance labels exist in this repo.
- AGAINST recall is 0.0 - the model has not learned the negative class. Do not deploy.
- Adapter-only checkpoint: load with the same base encoder and the same head.
