# Data directory — provenance, licences and permitted use

This file records exactly what is in `data/`, where it came from, and what each
resource is **allowed** to be used for in this project. Nothing here is merged
blindly: every dataset keeps its own schema until §"compatibility verdict"
says otherwise.

| Resource | Local path | Origin | Rows obtained | Licence | Permitted role |
|---|---|---|---|---|---|
| GWStance (GWSD) | `raw/gwstance/GWSD.tsv` | github.com/yiweiluo/GWStance (Luo, Card & Jurafsky, Findings of EMNLP 2020) | 2 300 sentences | No `LICENSE` file in the repository and no licence statement in its README — **research use only, redistribution not granted** | English reference baseline; framing analysis |
| Dhoroni | `raw/dhoroni/dhoroni_train.parquet` | huggingface.co/datasets/ciol-research/Dhoroni (arXiv:2410.17225) | 2 290 articles, single column `text` | **MIT** (declared on the HF dataset card) | Bengali domain-adaptation / linguistic probe |
| IndicSentiment | `raw/indic_sentiment/{train,validation,test}_{hi,bn}.json` | huggingface.co/datasets/ai4bharat/IndicSentiment | 1 156 hi + 1 156 bn (validation + test; `train/*.json` returns `Entry not found`) | AI4Bharat card terms; see dataset page | Auxiliary linguistic / representation probe only |
| Custom Hindi–Bengali stance corpus | `raw/custom_climate_dataset/` | to be collected per `docs/annotation_guideline.md` | 0 (not yet collected) | Self-collected | **Primary supervised signal** |

`interim/`, `processed/`, `augmented/` and `splits/` are build artefacts and are
git-ignored except for their `.gitkeep`.

## Compatibility verdicts (recorded before any experiment)

**GWStance → target task: partially compatible, English only.**
GWSD labels each sentence with 8 worker judgments plus Bayesian posteriors over
`agree` / `disagree` / `neutral` *toward the exact claim "climate change and
global warming is a serious concern"*. That is the same claim our project uses,
so the label semantics match. It is nevertheless **English**, so it is used for
the English reference baseline and for framing/lexicon analysis, never as Hindi
or Bengali supervision, and never as a translated substitute for Indic data.

**Dhoroni → target task: incompatible as supervision.**
The Hugging Face release exposes a single `text` column of Bengali news
articles; the parquet file we downloaded contains **no label column at all**
(checked: `rows 2290 cols ['text']`). The paper describes multi-perspective
annotations (political influence, scientific/statistical data, authenticity,
stance, stakeholder involvement) but those fields are not present in the public
parquet. Therefore Dhoroni **cannot** supply FAVOUR/AGAINST/NONE labels. Its
legitimate uses here: (a) Bengali domain-adaptation corpus for continued
pretraining, (b) linguistic/EDA reference for climate vocabulary, (c) a
zero-shot domain-transfer probe.

**IndicSentiment → target task: incompatible as supervision, by design.**
It is **translated product/service reviews** (the English review and its Indic
translation are both present), labelled `Positive`/`Negative`. Sentiment toward
a soundbar says nothing about stance toward a climate claim. It is used only as
an auxiliary representation-learning / tokenizer-coverage resource, and any
auxiliary training is reported separately from supervised stance training.

## What must never happen

1. GWStance sentences labelled as Hindi or Bengali.
2. Dhoroni articles given FAVOUR/AGAINST/NONE labels inferred from keywords.
3. IndicSentiment `Positive` → `FAVOUR`, `Negative` → `AGAINST` mapping.
4. Machine-translated GWStance used as if it were native Indic annotation.
5. Synthetic or model-generated stance rows counted as human annotation.

## Reproduction

```bash
python -m src.data.download --all          # fetches the three public resources
python -m src.data.dataset_validator --all --json experiments/results/audit.json
```
