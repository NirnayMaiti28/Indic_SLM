# Baseline Comparison — Phase 7

Corpus: **GWStance / GWSD** (English, 2 300 sentences, labels derived by Bayesian-posterior argmax).
This is the *development* corpus: it has a label space compatible with the target claim but it is English,
so every number below is an English reference result and is never reported as Hindi/Bengali performance.

## Split protocol

- strategy: `group_stratified`
- train / val / test: 1435 / 308 / 307
- unique groups: 2041
- train↔test group overlap: **0** (leakage-safe: True)

## Results (test set, macro-F1 is the selection metric on val)

| config | val macro-F1 | test macro-F1 | test acc | F1 FAVOUR | F1 AGAINST | F1 NONE | features | train s | infer ms/item |
|---|---|---|---|---|---|---|---|---|---|
| `tfidf-word-logreg-C0.3` | 0.5737 | 0.5774 | 0.5798 | 0.6091 | 0.5652 | 0.5579 | 19,857 | 0.4 | 0.029 |
| `tfidf-char-svm-C0.3` | 0.5617 | 0.6264 | 0.6319 | 0.6510 | 0.6038 | 0.6245 | 30,615 | 0.7 | 0.190 |
| `tfidf-word-logreg-C1.0` | 0.5591 | 0.5790 | 0.5831 | 0.6196 | 0.5649 | 0.5526 | 19,857 | 0.7 | 0.058 |
| `tfidf-combined-logreg-C0.3` | 0.5590 | 0.6093 | 0.6156 | 0.6613 | 0.5778 | 0.5887 | 50,472 | 1.1 | 0.217 |
| `tfidf-char-svm-C1.0` | 0.5556 | 0.6154 | 0.6221 | 0.6349 | 0.5882 | 0.6231 | 30,615 | 0.8 | 0.198 |
| `tfidf-combined-logreg-C3.0` | 0.5527 | 0.6143 | 0.6221 | 0.6457 | 0.5760 | 0.6213 | 50,472 | 2.4 | 0.212 |
| `tfidf-combined-logreg-C1.0` | 0.5515 | 0.6176 | 0.6254 | 0.6534 | 0.5781 | 0.6213 | 50,472 | 1.9 | 0.214 |
| `tfidf-word-svm-C0.3` | 0.5489 | 0.5880 | 0.5993 | 0.6350 | 0.5490 | 0.5798 | 19,857 | 0.1 | 0.044 |
| `tfidf-combined-svm-C1.0` | 0.5467 | 0.6304 | 0.6384 | 0.6590 | 0.5981 | 0.6341 | 50,472 | 1.0 | 0.227 |
| `tfidf-combined-svm-C0.3` | 0.5451 | 0.6392 | 0.6450 | 0.6667 | 0.6168 | 0.6341 | 50,472 | 0.8 | 0.243 |
| `tfidf-word-svm-C3.0` | 0.5449 | 0.5739 | 0.5928 | 0.6250 | 0.5000 | 0.5966 | 19,857 | 0.2 | 0.037 |
| `tfidf-combined-svm-C3.0` | 0.5435 | 0.6198 | 0.6319 | 0.6565 | 0.5714 | 0.6316 | 50,472 | 1.3 | 0.226 |
| `tfidf-char-logreg-C3.0` | 0.5431 | 0.6156 | 0.6189 | 0.6341 | 0.5984 | 0.6141 | 30,615 | 1.9 | 0.186 |
| `tfidf-char-logreg-C0.3` | 0.5430 | 0.6011 | 0.6091 | 0.6498 | 0.5493 | 0.6043 | 30,615 | 1.1 | 0.192 |
| `tfidf-word-svm-C1.0` | 0.5338 | 0.5802 | 0.5961 | 0.6273 | 0.5192 | 0.5941 | 19,857 | 0.2 | 0.040 |
| `tfidf-char-logreg-C1.0` | 0.5311 | 0.6188 | 0.6254 | 0.6506 | 0.5846 | 0.6213 | 30,615 | 1.7 | 0.183 |
| `tfidf-word-logreg-C3.0` | 0.5278 | 0.5796 | 0.5863 | 0.6288 | 0.5574 | 0.5526 | 19,857 | 0.9 | 0.034 |
| `tfidf-char-svm-C3.0` | 0.5263 | 0.5968 | 0.6156 | 0.6349 | 0.5263 | 0.6292 | 30,615 | 1.0 | 0.195 |
| **majority floor** | – | 0.1994 | 0.4267 | 0.0000 | 0.0000 | 0.5982 | 0 | 0 | 0 |

## Best configuration

- `tfidf-word-logreg-C0.3` → **test macro-F1 0.5774**, 95 % bootstrap CI [0.5163, 0.6321] (SE 0.0303, 1000 resamples)
- vs majority floor: McNemar n01=113 / n10=66, p = 0.000548
- per-class: FAVOUR P 0.583/R 0.638/F1 0.609 · AGAINST P 0.500/R 0.650/F1 0.565 · NONE P 0.637/R 0.496/F1 0.558
- confusion matrix (rows=true FAVOUR/AGAINST/NONE): `[[74, 17, 25], [9, 39, 12], [44, 22, 65]]`

## Leakage control

- group-stratified split: **0.5774** macro-F1
- deliberately naive random split (overlap 0 duplicated texts): **0.5330** macro-F1
- inflation attributable to duplicate leakage: **-0.0444**

## Most predictive features of the best linear model

- **FAVOUR**: `we`, `the world`, `climate change`, `change`, `world`, `warming`, `will`, `to`
- **AGAINST**: `not`, `global warming`, `warming`, `been`, `is`, `is not`, `global`, `no`
- **NONE**: `carbon`, `ice`, `its`, `in`, `year`, `on`, `carbon emissions`, `oil`

## Reading (computed from the table above, not asserted)

- Highest **validation** macro-F1 (the selection protocol): `tfidf-word-logreg-C0.3` at 0.5737.
- Highest **test** macro-F1: `tfidf-combined-svm-C0.3` at 0.6392.
- The selection protocol and the test ranking disagree. With only 308 validation items the validation macro-F1 has a standard error of roughly 0.030, so a val/test gap of this size is inside noise. The honest conclusion is that the linear family plateaus at the same place and that val-based selection is underpowered at this corpus size — which is why the teacher comparison below uses multiple seeds rather than a single run.
- Best test macro-F1 per feature family: combined 0.6392, char 0.6264, word 0.5880.
- Majority-class floor: macro-F1 0.1994. Every configuration clears it by a wide margin, so the signal is real and not an artefact of class imbalance.
- Duplicate leakage inflates test macro-F1 from 0.5774 to 0.5330 (-0.0444) under a naive random split, which is why every reported number uses the group-stratified split.
- Cost: TF-IDF needs 19,857-50,472 sparse features and 0.1-2.4 s to train on CPU. This is the efficiency bar the student must beat on absolute size, and the accuracy bar the teacher must beat to justify ~100x more parameters.
