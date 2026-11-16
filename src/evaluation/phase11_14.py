"""Phase 11-14: augmentation ablation, robustness, explainability, efficiency.

All runs reuse the LoRA recipe from phase9_lora_kd (the only one that fits this
box) with a fixed 36-step budget, so comparisons are budget-matched.
"""
from __future__ import annotations
import gc, json, os, statistics, sys, time
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None
    import psutil

os.environ.setdefault("OMP_NUM_THREADS","2"); os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import numpy as np, torch
torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from src.training.phase9_lora_kd import LoRAStance, encode, infer, train, evaluate, LABELS, ENCODER, RES
from src.data.dataset_loader import load_gwstance
from src.data.preprocessing import preprocess
from src.data.splitting import group_stratified_split
from src.data.augmentation import augment, char_repeat, emoji_append, hashtag_append, partial_romanize, spacing_noise, english_codemix
from src.evaluation.metrics import compute_metrics
import random

def peak():
    if resource is not None:
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,0)
    return round(psutil.Process().memory_info().rss/(1024*1024),0)
out={}
recs=load_gwstance(drop_screening=True)
for r in recs: r["text"]=preprocess(r["text"],variant="minimal").text
split,rep=group_stratified_split(recs,fractions=(0.70,0.15,0.15),seed=42)
tr,va,te=split["train"],split["val"],split["test"]
tok=AutoTokenizer.from_pretrained(ENCODER)
print(f"split {len(tr)}/{len(va)}/{len(te)}",flush=True)

# ---------- PHASE 11: augmentation ablation + LoRA full-FT proxy ----------
print("[P11] augmentation ablation (budget-matched 36 steps)",flush=True)
aug_runs={}
for tag,ops in (("none",None),("char_emoji",["char_repeat","emoji"]),("romanize",["romanize"])):
    rows=tr if ops is None else augment(tr,ops,copies=1,seed=42,keep_original=True)
    m=LoRAStance(ENCODER,pooling="mean",keep_layers=4,r=8)
    h=train(m,rows,tok,steps=36,lr=3e-4,bs=2,accum=2)
    te_m=evaluate(m,te,tok,f"aug={tag}/test")
    aug_runs[tag]={"n_train":len(rows),"test":{k:v for k,v in te_m.items() if not k.startswith('_')},"hist":h}
    del m; gc.collect()
out["augmentation"]=aug_runs
out["augmentation"]["delta_char_emoji"]=round(aug_runs["char_emoji"]["test"]["macro_f1"]-aug_runs["none"]["test"]["macro_f1"],4)
out["augmentation"]["delta_romanize"]=round(aug_runs["romanize"]["test"]["macro_f1"]-aug_runs["none"]["test"]["macro_f1"],4)

# PEFT vs full-FT: measure what full FT would cost, and whether it fits
print("[P11] full fine-tuning feasibility measurement",flush=True)
n_params=237853443
bytes_per_param_adam=16
out["full_ft_estimate"]={"params":n_params,"bytes_per_param_adamw":bytes_per_param_adam,
    "estimated_train_mb":round(n_params*bytes_per_param_adam/1e6,0),
    "measured_peak_rss_mb_lora":peak(),
    "verdict":"full fine-tuning requires ~3.8 GB for optimizer state alone vs 1.9 GB total RAM -> not runnable on this host"}
print(f"  full-FT estimate {out['full_ft_estimate']['estimated_train_mb']} MB > 1900 MB RAM -> not runnable",flush=True)

# ---------- PHASE 12: robustness ----------
print("[P12] robustness perturbations",flush=True)
kd=LoRAStance(ENCODER,pooling="mean",keep_layers=4,r=8)
train(kd,tr,tok,steps=36,lr=3e-4,bs=2,accum=2)
PERT={
 "clean":lambda t,r:t,
 "char_elongation":lambda t,r:char_repeat(t,0.35,r),
 "spacing_noise":lambda t,r:spacing_noise(t,0.30,r),
 "emoji_suffix":lambda t,r:emoji_append(t,rng=r),
 "hashtag_suffix":lambda t,r:hashtag_append(t,rng=r),
 "partial_romanization":lambda t,r:partial_romanize(t,0.6,r),
 "english_codemix":lambda t,r:english_codemix(t,rng=r),
 "all_noise":lambda t,r:partial_romanize(emoji_append(hashtag_append(spacing_noise(char_repeat(t,0.3,r),0.25,r),rng=r),rng=r),0.5,r),
}
rob={}
for name,fn in PERT.items():
    rng=random.Random(7)
    rows=[{"text":fn(str(r["text"]),rng),"label":r["label"]} for r in te]
    p=infer(kd,rows,tok)
    m=compute_metrics([r["label"] for r in rows],[LABELS[i] for i in p.argmax(1)])
    rob[name]={"macro_f1":m["macro_f1"],"accuracy":m["accuracy"],"n":m["n"],
               "per_class":{k:v["f1"] for k,v in m["per_class"].items()}}
    print(f"  {name:22s} macroF1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}",flush=True)
clean=rob["clean"]["macro_f1"]
for k,v in rob.items():
    v["delta_vs_clean"]=round(v["macro_f1"]-clean,4); v["retention"]=round(v["macro_f1"]/clean,4) if clean else 0
out["kd_confusion"]=compute_metrics([r["label"] for r in te],[LABELS[i] for i in infer(kd,te,tok).argmax(1)])["confusion_matrix"]
_p=infer(kd,te,tok); _pr=[LABELS[i] for i in _p.argmax(1)]
out["error_examples"]=[{"text":r["text"][:180],"gold":r["label"],"pred":_pr[i],
  "confidence":round(float(_p[i].max()),3)} for i,r in enumerate(te) if _pr[i]!=r["label"]][:25]
out["n_errors"]=sum(1 for i,r in enumerate(te) if _pr[i]!=r["label"])
out["robustness"]={"perturbations":rob,"mean_retention":round(statistics.mean(v["retention"] for v in rob.values()),4),
                   "clean_macro_f1":clean}

# ---------- PHASE 13: explainability ----------
print("[P13] integrated gradients + occlusion",flush=True)
EX=["जलवायु परिवर्तन एक गंभीर समस्या है।","ग्लोबल वार्मिंग कोई गंभीर समस्या नहीं है।",
    "জলবায়ু পরিবর্তন একটি গুরুতর উদ্বেগের বিষয়।","জলবায়ু পরিবর্তন কোনো গুরুতর সমস্যা নয়।",
    "सरकार की जलवायु नीति बिल्कुल फेल है, पैसा बर्बाद हो रहा है।",
    "climate change koi serious problem nahi hai"]
def ig(model,text,target,steps=20):
    """Integrated Gradients on the input embeddings.

    peft injects lora_A/lora_B into the *existing* Linear modules, so unwrapping to
    base_model.model keeps the adapters active while letting us pass inputs_embeds.
    """
    enc=tok(text,truncation=True,max_length=64,return_tensors="pt")
    ids,mask=enc["input_ids"],enc["attention_mask"]
    base=getattr(model.encoder,"base_model",model.encoder)
    base=getattr(base,"model",base)
    emb=base.get_input_embeddings().weight[ids].detach()
    emb=emb.clone()
    tot=torch.zeros_like(emb)
    for a in range(1,steps+1):
        x=(a/steps*emb).requires_grad_(True)
        h=base(inputs_embeds=x,attention_mask=mask).last_hidden_state
        sc=model.head(h,mask)[0,target]
        sc.backward()
        tot+=x.grad.detach()
    at=(emb*tot/steps).sum(-1)[0].tolist()
    toks=tok.convert_ids_to_tokens(ids[0].tolist())
    pairs=[(t,a) for t,a in zip(toks,at) if t not in (tok.cls_token,tok.sep_token,tok.pad_token)]
    pairs.sort(key=lambda x:-abs(x[1]))
    return pairs
def occ(model,text,target):
    enc=tok(text,truncation=True,max_length=64,return_tensors="pt")
    ids,mask=enc["input_ids"][0],enc["attention_mask"][0]
    toks=tok.convert_ids_to_tokens(ids.tolist())
    keep=[i for i,t in enumerate(toks) if t not in (tok.cls_token,tok.sep_token,tok.pad_token)]
    with torch.no_grad():
        p0=torch.softmax(model(input_ids=ids.unsqueeze(0),attention_mask=mask.unsqueeze(0)),-1)[0,target].item()
        d=[]
        for i in keep:
            ni=[x for j,x in enumerate(ids.tolist()) if j!=i]
            p=torch.softmax(model(input_ids=torch.tensor([ni]),attention_mask=torch.ones(1,len(ni),dtype=torch.long)),-1)[0,target].item()
            d.append((toks[i],round(p0-p,5)))
    d.sort(key=lambda x:-abs(x[1])); return p0,d
expl=[]
for text in EX:
    with torch.no_grad():
        enc=tok(text,truncation=True,max_length=64,return_tensors="pt")
        pr=torch.softmax(kd(input_ids=enc["input_ids"],attention_mask=enc["attention_mask"]),-1)[0]
    pred=int(pr.argmax())
    try:
        g=ig(kd,text,pred)
    except Exception as e:
        print("   IG failed:",type(e).__name__,str(e)[:80]); g=[("n/a",0.0)]
    o=occ(kd,text,pred)
    expl.append({"text":text,"predicted":LABELS[pred],"confidence":round(float(pr[pred]),4),
                 "probabilities":{LABELS[i]:round(float(pr[i]),4) for i in range(3)},
                 "ig_top":[(t,round(a,5)) for t,a in g[:6]],
                 "occlusion_top":[(t,d) for t,d in o[1][:5]]})
    print(f"  {LABELS[pred]:7s} {float(pr[pred]):.3f} {text[:44]:44s} IG={[t for t,_ in g[:4]]}",flush=True)
out["explainability"]=expl

# ---------- PHASE 14: efficiency ----------
print("[P14] efficiency: params, storage, latency, throughput, INT8",flush=True)
SENT=["जलवायु परिवर्तन एक गंभीर समस्या है।","গ্লোবাল ওয়ার্মিং একটি গুরুতর উদ্বেগের বিষয়।",
      "ग्लोबल वार्मिंग कोई गंभीर समस्या नहीं है।","জলবায়ু পরিবর্তন কোনো গুরুতর সমস্যা নয়।",
      "climate change koi serious problem nahi hai"]
def bench(model,n=60,batch=1):
    model.eval(); lat=[]
    texts=[SENT[i%len(SENT)] for i in range(n)]
    with torch.no_grad():
        for i in range(n):
            e=tok(texts[i],truncation=True,max_length=64,return_tensors="pt")
            t0=time.perf_counter(); model(input_ids=e["input_ids"],attention_mask=e["attention_mask"]); lat.append((time.perf_counter()-t0)*1000)
        lat.sort()
        t0=time.perf_counter()
        for i in range(0,n,batch):
            e=tok(texts[i:i+batch],truncation=True,max_length=64,padding=True,return_tensors="pt")
            model(input_ids=e["input_ids"],attention_mask=e["attention_mask"])
        tot=time.perf_counter()-t0
    return {"n":n,"batch":batch,"mean_ms":round(statistics.mean(lat),2),"p50_ms":round(lat[len(lat)//2],2),
            "p95_ms":round(lat[int(0.95*(len(lat)-1))],2),"throughput_items_s":round(n/tot,2)}
def store(path):
    return round(sum(os.path.getsize(os.path.join(path,f)) for f in os.listdir(path))/1e6,2) if os.path.isdir(path) else 0.0
eff={"student":{"params":sum(p.numel() for p in kd.parameters()),
                "trainable":sum(p.numel() for p in kd.parameters() if p.requires_grad),
                "adapter_storage_mb":store(os.path.join(ROOT,"models","student")),
                "latency_b1":bench(kd,60,1),"latency_b8":bench(kd,64,8),
                "peak_rss_mb":peak()}}
try:
    q=torch.quantization.quantize_dynamic(kd.eval(),{torch.nn.Linear},dtype=torch.qint8)
    eff["student"]["int8"]={"latency_b1":bench(q,60,1),"status":"ok","peak_rss_mb":peak()}
    eff["student"]["int8"]["speedup_p50"]=round(eff["student"]["latency_b1"]["p50_ms"]/eff["student"]["int8"]["latency_b1"]["p50_ms"],2)
except Exception as e:
    eff["student"]["int8"]={"status":"fail","error":f"{type(e).__name__}: {str(e)[:120]}"}
# teacher reference numbers from the Phase 9/10 run (measured, same box)
p9=json.load(open(os.path.join(RES,"phase9_10_results.json")))
eff["teacher"]={"params":p9["teacher_params"],"trainable":p9["teacher_trainable"],
                "adapter_storage_mb":store(os.path.join(ROOT,"models","teacher")),
                "note":"teacher latency measured separately in Phase 14b (teacher+student cannot co-reside in 1.9GB)"}
eff["compression"]={"param_ratio":round(p9["teacher_params"]/eff["student"]["params"],2),
                    "trainable_ratio":round(p9["teacher_trainable"]/eff["student"]["trainable"],2)}
out["efficiency"]=eff
print(json.dumps({"student_params":eff["student"]["params"],"p50":eff["student"]["latency_b1"]["p50_ms"],
                  "p95":eff["student"]["latency_b1"]["p95_ms"],"thr8":eff["student"]["latency_b8"]["throughput_items_s"],
                  "int8":eff["student"]["int8"].get("status")},indent=1),flush=True)
del kd; gc.collect()
json.dump(out,open(os.path.join(RES,"phase11_14_results.json"),"w"),ensure_ascii=False,indent=2)
print("WROTE phase11_14_results.json peak",peak(),flush=True)
