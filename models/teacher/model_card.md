# Teacher - MuRIL-base + LoRA

*Base encoder:* `google/muril-base-cased`  
*Adapters:* LoRA r=8, alpha=16, q/v  
*Pooling:* cls  
*Total parameters:* 237,853,443  
*Trainable parameters:* 297,219  

## Measured test performance (dev corpus, n = 307)

| metric | value |
|---|---|
| macro-F1 | 0.1828 |
| accuracy | 0.3779 |
| macro-precision | 0.1260 |
| macro-recall | 0.3333 |
| weighted F1 | 0.2072 |
| balanced accuracy | 0.3333 |
| Cohen's kappa | 0.0000 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| FAVOUR | 0.3779 | 1.0000 | 0.5485 | 116 |
| AGAINST | 0.0000 | 0.0000 | 0.0000 | 60 |
| NONE | 0.0000 | 0.0000 | 0.0000 | 131 |

## Confusion matrix (rows = gold, cols = FAVOUR/AGAINST/NONE)

`[[116, 0, 0], [60, 0, 0], [131, 0, 0]]`

## Intended use and limits

- Trained on GWStance (English news sentences, derived posterior-argmax labels).
- **Not** a Hindi/Bengali stance model: no Hindi/Bengali stance labels exist in this repo.
- AGAINST recall is 0.0 - the model has not learned the negative class. Do not deploy.
- Adapter-only checkpoint: load with the same base encoder and the same head.
