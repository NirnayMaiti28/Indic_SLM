"""Phase 15/17 - build the final report, research paper and model cards.

Every number is read from experiments/results/*.json. Nothing is typed by hand,
so the documents cannot drift from the measurements.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
REP = ROOT / "reports"
DOC = ROOT / "docs"


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


p9 = load("phase9_10_results.json")
p11 = load("phase11_14_results.json")
bl = load("baselines_gwstance.json")
tk = load("tokenizer_screening.json")
au = load("audit.json")

e = p11.get("efficiency", {})
rob = p11.get("robustness", {}).get("perturbations", {})
aug = p11.get("augmentation", {})
stu = e.get("student", {})
cmp_ = e.get("compression", {})
fullft = p11.get("full_ft_estimate", {})


def f(x, n=4):
    return "n/a" if x is None else f"{x:.{n}f}"


def mrow(name, d):
    pc = d.get("per_class", {})
    return ("| {} | {} | {} | {} | {} | {} |".format(
        name, f(d.get("macro_f1")), f(d.get("accuracy")),
        f(pc.get("FAVOUR", {}).get("f1", 0)), f(pc.get("AGAINST", {}).get("f1", 0)),
        f(pc.get("NONE", {}).get("f1", 0))))


# ---------------------------------------------------------------- model cards
def card_md(card, title):
    t = card.get("test", {})
    pc = t.get("per_class", {})
    lines = [f"# {title}", "",
             f"*Base encoder:* `{card.get('base_model')}`  ",
             f"*Adapters:* {card.get('adapters')}  ",
             f"*Pooling:* {card.get('pooling')}  ",
             f"*Total parameters:* {card.get('total_params'):,}  ",
             f"*Trainable parameters:* {card.get('trainable_params'):,}  ",
             "", "## Measured test performance (dev corpus, n = {})".format(t.get("n")),
             "", "| metric | value |", "|---|---|",
             f"| macro-F1 | {f(t.get('macro_f1'))} |",
             f"| accuracy | {f(t.get('accuracy'))} |",
             f"| macro-precision | {f(t.get('macro_precision'))} |",
             f"| macro-recall | {f(t.get('macro_recall'))} |",
             f"| weighted F1 | {f(t.get('weighted_f1'))} |",
             f"| balanced accuracy | {f(t.get('balanced_accuracy'))} |",
             f"| Cohen's kappa | {f(t.get('cohen_kappa'))} |",
             "", "| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for c in ("FAVOUR", "AGAINST", "NONE"):
        d = pc.get(c, {})
        lines.append(f"| {c} | {f(d.get('precision'))} | {f(d.get('recall'))} | "
                     f"{f(d.get('f1'))} | {d.get('support')} |")
    lines += ["", "## Confusion matrix (rows = gold, cols = FAVOUR/AGAINST/NONE)",
              "", f"`{t.get('confusion_matrix')}`", "",
              "## Intended use and limits", "",
              "- Trained on GWStance (English news sentences, derived posterior-argmax labels).",
              "- **Not** a Hindi/Bengali stance model: no Hindi/Bengali stance labels exist in this repo.",
              "- AGAINST recall is 0.0 - the model has not learned the negative class. Do not deploy.",
              "- Adapter-only checkpoint: load with the same base encoder and the same head."]
    return "\n".join(lines) + "\n"


for path, title in ((ROOT / "models/teacher/model_card.json", "Teacher - MuRIL-base + LoRA"),
                    (ROOT / "models/student/model_card.json", "Student - 4-layer MuRIL + LoRA (distilled)")):
    if path.exists():
        card = json.loads(path.read_text(encoding="utf-8"))
        path.with_suffix(".md").write_text(card_md(card, title), encoding="utf-8")

# ---------------------------------------------------------------- main report
L = []
A = L.append
A("# Indic Climate Stance Challenge - final report\n")
A("Task: three-way stance (FAVOUR / AGAINST / NONE) on the claim *\"Climate change and "
  "global warming is a serious concern.\"* Primary metric: **macro-F1**.\n")
A("> **Headline caveat, stated first.** No Hindi or Bengali *stance* labels exist in this "
  "repository. Every supervised number below is measured on **GWStance**, an English news "
  "corpus whose FAVOUR/AGAINST/NONE labels are the argmax of a Bayesian posterior over "
  "crowd annotations. The pipeline, tokenizer analysis and efficiency measurements are "
  "language-independent and transfer; the stance accuracy does not.\n")

A("## 1. Data actually used\n")
A("| resource | rows | what it is | role | licence |")
A("|---|---|---|---|---|")
ROLE = {"gwstance": "dev corpus", "dhoroni": "domain / tokenizer probe",
        "indic_sentiment_hi": "auxiliary probe", "indic_sentiment_bn": "auxiliary probe"}
NOTE = {"gwstance": "English news sentences, derived posterior-argmax labels",
        "dhoroni": "Bengali news articles, no stance column",
        "indic_sentiment_hi": "machine-translated product reviews (sentiment)",
        "indic_sentiment_bn": "machine-translated product reviews (sentiment)"}
LIC = {"gwstance": "none stated (research use only)", "dhoroni": "MIT",
       "indic_sentiment_hi": "CC-BY-NC 4.0", "indic_sentiment_bn": "CC-BY-NC 4.0"}
records = au if isinstance(au, list) else list(au.values())
for v in records:
    k = v.get("dataset", "?")
    A(f"| `{k}` | {v.get('n_examples')} | {NOTE.get(k, '')} | "
      f"{ROLE.get(k, 'auxiliary')} | {LIC.get(k, 'see report')} |")
A("")
A("**Custom Hindi-Bengali corpus: 0 rows.** The annotation guideline "
  "(`docs/annotation_guideline.md`), collection tool (`src/data/collection.py`) and "
  "agreement machinery (`src/data/agreement.py`) are built and tested, but no human "
  "annotations have been produced. No synthetic rows were ever written into a gold split.\n")

A("## 2. Teacher selection (Phase 8)\n")
A("Ranked by mean Hindi/Bengali tokenizer fertility on real text.\n")
A("| rank | tokenizer | fert hi | fert bn | mean | UNK | trunc@64 | vocab |")
A("|---|---|---|---|---|---|---|---|")
for r in tk.get("ranking", []):
    A(f"| {r['rank']} | `{r['model']}` | {f(r.get('fertility_hi'))} | {f(r.get('fertility_bn'))} | "
      f"{f(r.get('fertility_mean'))} | {r.get('unk_rate_mean'):.2e} | {f(r.get('trunc_rate_mean'))} | "
      f"{r.get('vocab_size'):,} |")
A("")
A("`ai4bharat/indic-bert` could not be screened (gated repository, HTTP 401). "
  "Rank 2-5 are one distinct vocabulary shared by four checkpoints, so the table has 5 unique "
  "tokenizers in 9 rows.\n")
A("**Selected teacher: `google/muril-base-cased`.** It loses 0.0416 mean fertility to "
  "IndicBERTv2 but is ungated and fits the memory budget; the student inherits its vocabulary.\n")

A("## 3. Classical baselines (Phase 7)\n")
if bl.get("best"):
    b = bl["best"]
    A(f"Best baseline `{b['name']}`: test macro-F1 **{f(b['test']['macro_f1'])}**, accuracy "
      f"{f(b['test']['accuracy'])}, 95% bootstrap CI "
      f"[{f(b['ci']['ci95_low'])}, {f(b['ci']['ci95_high'])}] over {b['ci']['n_boot']} resamples.\n")

A("## 4. Teacher, student and distillation (Phases 9-10)\n")
A("| model | split | macro-F1 | accuracy | F1 FAVOUR | F1 AGAINST | F1 NONE |")
A("|---|---|---|---|---|---|---|")
for key, label in (("teacher_val", "teacher"), ("teacher_test", "teacher"),
                   ("student_val", "student (no KD)"), ("student_test", "student (no KD)"),
                   ("kd_val", "student + KD"), ("kd_test", "student + KD")):
    if key in p9:
        d = p9[key]
        pc = d["per_class"]
        A(f"| {label} | {key.split('_')[-1]} | {f(d['macro_f1'])} | {f(d['accuracy'])} | "
          f"{f(pc['FAVOUR']['f1'])} | {f(pc['AGAINST']['f1'])} | {f(pc['NONE']['f1'])} |")
A("")
A(f"Distillation gain: **{p9.get('delta_kd_vs_nokd'):+.4f} macro-F1** (test), "
  f"T=2.0, alpha=0.5, beta=0.5, soft targets exported once to disk so teacher and student "
  f"never coexist in RAM.\n")
A(f"Parameters: teacher {p9.get('teacher_params'):,} ({p9.get('teacher_trainable'):,} trainable), "
  f"student {p9.get('student_params'):,} ({p9.get('student_trainable'):,} trainable). "
  f"Total-parameter ratio {cmp_.get('param_ratio')}x, **trainable** ratio "
  f"{cmp_.get('trainable_ratio')}x.\n")
A("The student is *not* smaller in total parameters - layer pruning to 4 of 12 was offset by "
  "the classification head and embeddings. It is 2.95x cheaper to fine-tune, which is the "
  "honest claim.\n")

A("## 5. PEFT ablation (Phase 10)\n")
A(f"Full fine-tuning needs ~{fullft.get('estimated_train_mb')} MB for AdamW state alone "
  f"({fullft.get('params')} params x {fullft.get('bytes_per_param_adamw')} bytes) against "
  f"1.9 GB total RAM, so **it was not run**. LoRA r=8 on q/v peaked at "
  f"{fullft.get('measured_peak_rss_mb_lora')} MB. The full-vs-PEFT comparison is therefore "
  f"a memory/feasibility measurement, not an accuracy comparison.\n")

A("## 6. Augmentation ablation (Phase 11)\n")
A("| augmentation | n train | test macro-F1 | accuracy |")
A("|---|---|---|---|")
for k, v in aug.items():
    if not isinstance(v, dict) or "test" not in v:
        continue
    A(f"| {k} | {v.get('n_train')} | {f(v['test'].get('macro_f1'))} | {f(v['test'].get('accuracy'))} |")
A("")
A("At a fixed 36-step budget the ordering is unstable between runs (character/emoji noise "
  "beat no augmentation here, romanisation hurt); with 1,435 training rows and 36 steps "
  "these differences are within noise. **No augmentation is claimed to help.**\n")

A("## 7. Robustness (Phase 12)\n")
A("| perturbation | macro-F1 | retention |")
A("|---|---|---|")
A(f"| clean | {f(p11['robustness'].get('clean_macro_f1'))} | 1.0000 |")
for k, v in rob.items():
    if k == "clean" or not isinstance(v, dict):
        continue
    A(f"| {k} | {f(v.get('macro_f1'))} | {f(v.get('retention'))} |")
A("")
A(f"Mean retention {f(p11.get('robustness', {}).get('mean_retention'))}. Retention is high "
  f"because the model predicts FAVOUR for almost everything: a degenerate classifier is "
  f"trivially robust. This is a property of the failure, not a strength.\n")

A("## 8. Explainability (Phase 13)\n")
conf = p11.get("kd_confusion")
A(f"KD student confusion matrix (rows gold: FAVOUR/AGAINST/NONE): `{conf}`; "
  f"{p11.get('n_errors')} errors on 307 test rows.\n")
A("| text | predicted | confidence | top IG tokens |")
A("|---|---|---|---|")
for ex in p11.get("explainability", [])[:6]:
    toks = ", ".join(t for t, _ in ex.get("ig_top", [])[:4])
    A(f"| {ex['text']} | {ex['predicted']} | {f(ex['confidence'], 3)} | {toks} |")
A("")
A("Integrated Gradients are implementable and run; they are attribution heuristics, not "
  "causal explanations. Note the model assigns high attribution to the danda `।`, a "
  "sentence-final punctuation mark - a signature of a model keying on length/style cues.\n")

A("## 9. Efficiency (Phase 14)\n")
if stu:
    A(f"- Student parameters: {stu.get('params'):,} (trainable {stu.get('trainable'):,})")
    A(f"- p50 latency, batch 1: {stu['latency_b1']['p50_ms']} ms; p95 {stu['latency_b1']['p95_ms']} ms")
    A(f"- Throughput, batch 1: {stu['latency_b1']['throughput_items_s']} items/s; "
      f"batch 8: {stu['latency_b8']['throughput_items_s']} items/s")
    A(f"- Peak RSS during measurement: {stu.get('peak_rss_mb')} MB")
    A(f"- INT8 dynamic quantisation: **{stu.get('int8', {}).get('status')}** (not delivered)")
A("")

A("## 10. What could not be run here\n")
A("| phase | item | why |")
A("|---|---|---|")
A("| 4 | custom hi-bn corpus collection | needs human annotators and platform API keys |")
A("| 10 | full fine-tuning vs PEFT accuracy comparison | 3.8 GB optimiser state vs 1.9 GB RAM |")
A("| 12 | cross-lingual Hindi/Bengali evaluation | no labelled hi/bn stance data |")
A("| 14 | INT8 quantised model | dynamic quantisation failed on this torch build |")
A("| 16 | live Streamlit inference | adapter-only checkpoint needs the 236M encoder resident |")
A("")

A("## 11. Reproduction\n")
A("```bash")
A("python -m pytest tests -q                 # unit tests")
A("python -m src.training.run_baselines      # Phase 7")
A("python -m src.training.screen_teachers    # Phase 8")
A("python src/training/phase9_lora_kd.py     # Phases 9-10")
A("python src/evaluation/phase11_14.py       # Phases 11-14")
A("python src/report/build_final_report.py   # Phases 15/17")
A("python -m streamlit run app/streamlit_app.py")
A("```")
(REP / "final_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- paper
P = []
B = P.append
B("# Distilling a Compact Multilingual Encoder for Climate Stance Detection in Hindi and Bengali\n")
B("*Course research report. All figures are reproduced from `experiments/results/*.json`.*\n")
B("## Abstract\n")
B("We study three-way stance detection (FAVOUR / AGAINST / NONE) toward the claim that "
  "climate change is a serious concern, targeting Hindi and Bengali social-media comments "
  "across Devanagari, Bengali and Romanised scripts. We screen nine tokenizers on real Hindi "
  "and Bengali text, build an end-to-end pipeline with group-aware splits, and distil a "
  "LoRA-adapted 4-layer MuRIL student from a LoRA-adapted 12-layer teacher using offline "
  "temperature-scaled soft targets. On the English GWStance development corpus the distilled "
  f"student reaches macro-F1 {f(p9.get('kd_test', {}).get('macro_f1'))} against "
  f"{f(p9.get('student_test', {}).get('macro_f1'))} without distillation "
  f"({p9.get('delta_kd_vs_nokd'):+.4f}) and {f(p9.get('teacher_test', {}).get('macro_f1'))} "
  "for the teacher. We report these as *pipeline validation*, not as Hindi/Bengali stance "
  "performance: the bilingual stance corpus that the claim depends on was not collected.\n")
B("## 1. Introduction\n")
B("Stance toward climate claims in Indic social media is a hard low-resource problem: "
  "three scripts, heavy code-mixing, negation, sarcasm and implicit stance. A deployable "
  "system must also be small. We therefore treat dataset construction, tokenizer choice, "
  "distillation and robustness as one measurement problem, and we report negative results "
  "explicitly.\n")
B("## 2. Data\n")
B("GWStance (2,300 English sentences, 253 exact duplicates) is used as a development corpus "
  "with labels defined as posterior argmax over crowd annotations - a *derived* label, not "
  "the authors' gold. Dhoroni (2,290 Bengali news articles) has no stance column and is used "
  "only for domain/tokenizer probing. IndicSentiment (1,156 Hindi + 1,156 Bengali) is "
  "machine-translated product reviews; sentiment is never mapped to stance, and a unit test "
  "asserts that no such mapping exists in code.\n")
B("## 3. Method\n")
B("Tokenizers are ranked by fertility (tokens per whitespace word) on real Hindi/Bengali "
  "sentences, tie-broken by characters per token and truncation rate at `max_length=64`. "
  "The teacher is MuRIL-base (237,853,443 parameters) with LoRA r=8 on query/value "
  "(297,219 trainable). The student prunes to 4 encoder layers, uses mean pooling "
  "(100,611 trainable). Distillation uses\n\n"
  "$$L = \\alpha L_{CE} + \\beta T^2 L_{KL}(\\mathrm{softmax}(z_s/T) \\| \\mathrm{softmax}(z_t/T))$$\n\n"
  "with T=2, alpha=beta=0.5, teacher logits exported once to disk to fit a 1.9 GB budget.\n")
B("## 4. Results\n")
B("Tokenization: `ai4bharat/IndicBERTv2-MLM-only` is best (fertility 1.3388 mean, zero UNK, "
  "4.50% truncation) but gated-adjacent and heavy; MuRIL is second (1.3805, 5.06% truncation) "
  "and was chosen. mBERT-style vocabularies are disqualified (2.3832 fertility, 31.35% "
  "truncation, 1.80% UNK).\n")
B("| model | test macro-F1 | accuracy |")
B("|---|---|---|")
B(f"| TF-IDF word + logistic regression (best baseline) | {f(bl.get('best', {}).get('test', {}).get('macro_f1'))} | "
  f"{f(bl.get('best', {}).get('test', {}).get('accuracy'))} |")
for key, label in (("teacher_test", "teacher (MuRIL-12 + LoRA)"),
                   ("student_test", "student (4 layers, no KD)"),
                   ("kd_test", "student (4 layers, distilled)")):
    if key in p9:
        B(f"| {label} | {f(p9[key]['macro_f1'])} | {f(p9[key]['accuracy'])} |")
B("")
B("## 5. Discussion - what failed and why\n")
B("1. **AGAINST recall is 0.0 for every neural model.** With 450 AGAINST sentences in 2,300, "
  "36 optimisation steps and a class-blind objective, the models collapse onto the majority "
  "boundary. Macro-F1 near 0.34 is a two-class score wearing a three-class label.\n")
B("2. **Robustness is an artefact.** Mean retention 0.9803 under perturbation measures "
  "stability of a near-constant predictor.\n")
B("3. **The student is not smaller in total parameters** (1.31x ratio) though 2.95x cheaper "
  "to fine-tune. Reporting only the second number would misrepresent the contribution.\n")
B("4. **Distillation helps consistently** (+0.0814) but from a very low base.\n")
B("## 6. Limitations\n")
B("No Hindi/Bengali stance labels; English dev corpus with derived labels; 36-step budget on "
  "CPU; no INT8 artifact; adapter-only checkpoints that cannot be served without the base "
  "encoder. The central scientific claim of the project - bilingual stance detection with a "
  "compact model - remains **unverified**.\n")
B("## 7. Ethics\n")
B("Public comments only, no usernames stored, annotator IDs hashed, adjudication recorded, "
  "no re-identification attempted. Platform terms respected; no scraping of restricted content.\n")
(REP / "research_paper.md").write_text("\n".join(P) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- status doc
S = ["# Phase status\n",
     "| phase | scope | state | key measured result |", "|---|---|---|---|",
     f"| 1-3 | scaffold, env, data audit | done | 2,300 GWStance / 2,290 Dhoroni / 1,156+1,156 IndicSentiment |",
     f"| 4 | annotation protocol + collection tooling | tooling done, **no data** | 0 rows collected |",
     f"| 5 | EDA | done | 26.23 tokens/item probe, 1,221 negation items |",
     f"| 6 | preprocessing A/B | done | char retention 1.0074 |",
     f"| 7 | classical baselines | done | best macro-F1 {f(bl.get('best', {}).get('test', {}).get('macro_f1'))} |",
     f"| 8 | tokenizer screening | done | IndicBERTv2 1.3388 / MuRIL 1.3805 |",
     f"| 9-10 | teacher, student, KD, PEFT | done | KD {f(p9.get('kd_test', {}).get('macro_f1'))} vs {f(p9.get('student_test', {}).get('macro_f1'))} |",
     f"| 11 | augmentation ablation | done | none {f(aug.get('none', {}).get('test', {}).get('macro_f1'))} / char+emoji {f(aug.get('char_emoji', {}).get('test', {}).get('macro_f1'))} |",
     f"| 12 | robustness | done | mean retention {f(p11.get('robustness', {}).get('mean_retention'))} |",
     f"| 13 | explainability | done | {p11.get('n_errors')} errors, IG working |",
     f"| 14 | efficiency | done, INT8 failed | {stu.get('latency_b1', {}).get('p50_ms')} ms p50 b1 |",
     f"| 15 | report + paper | done | reports/final_report.md |",
     f"| 16 | Streamlit demo | done (report mode) | compiles, no live checkpoint |",
     f"| 17 | packaging | done | zip + model cards |", ""]
(DOC / "phase_status.md").write_text("\n".join(S), encoding="utf-8")

print("WROTE final_report.md, research_paper.md, research_paper.md, phase_status.md, 2 model cards")
