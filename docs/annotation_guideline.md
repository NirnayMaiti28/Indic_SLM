# Annotation Guideline — Hindi/Bengali Climate Stance (v1.0)

**Claim under annotation (fixed, never re-worded during a batch):**

- Hindi: «जलवायु परिवर्तन और ग्लोबल वार्मिंग एक गंभीर चिंता का विषय है।»
- Bengali: «জলবায়ু পরিবর্তন এবং গ্লোবাল ওয়ার্মিং একটি গুরুতর উদ্বেগের বিষয়।»
- English gloss: "Climate change and global warming is a serious concern."

**Task.** For one comment, decide whether its author **supports**, **rejects**, or is **neutral toward** that sentence.

---

## 1. The three classes

| Label | Definition | Test question |
|---|---|---|
| `FAVOUR` | Author explicitly or clearly implicitly treats climate change / global warming as a real, serious concern. | "If a reader saw only this comment, would they conclude the author thinks this is a serious problem?" |
| `AGAINST` | Author explicitly or clearly implicitly denies, dismisses, mocks, downplays, or rejects that climate change is a serious concern. | "Would a reader conclude the author thinks this is exaggerated, fake, or not a concern?" |
| `NONE` | Neutral, off-topic, ambiguous, purely informational, or insufficient stance signal. | "Is the author's position on the claim genuinely unrecoverable from the text?" |

**Decision order.** 1) Is there a stance *toward the claim*? If no → `NONE`. 2) Is it supportive → `FAVOUR`. 3) Is it rejecting → `AGAINST`. When you must choose between `NONE` and a stance class, prefer `NONE` unless a second annotator would reach the same stance independently: ambiguity is the property of `NONE`, not of a low-confidence stance.

---

## 2. Hard cases with worked decisions

### 2.1 Negation
- «जलवायु परिवर्तन कोई गंभीर समस्या नहीं है।» → **AGAINST**. Negation scoped over the predicate ("serious problem").
- «জলবায়ু পরিবর্তন যে গুরুতর নয়, তা বলার উপায় নেই।» ("There's no way to say climate change isn't serious") → **FAVOUR**. Double negation: the author denies the denial. Count negations before you label.
- «गर्मी बढ़ रही है, इनकार नहीं कर सकते।» ("Heat is rising, can't deny it") → **FAVOUR**.

### 2.2 Climate-policy criticism ≠ climate denial
- «सरकार की जलवायु नीति बिल्कुल फेल है, पैसा बर्बाद हो रहा है।» → **NONE** (policy critique with no signal about whether climate change is serious). It is *not* `AGAINST` by default.
- «जलवायु नीति फेल है क्योंकि जलवायु संकट ही नकली है।» → **AGAINST** (the critique is grounded in denial).
- «সরকার জলবায়ু নিয়ে শুধু কথা বলে, আসলে কিছু করে না।» → **NONE**.
- Rule: label the stance *toward the claim*, never the affect toward the government, a party, a corporation, or "the West". If the comment's only content is blame directed elsewhere, it is `NONE`.

### 2.3 Sarcasm and mockery
- «अरे हाँ, जलवायु परिवर्तन सच है न, बस 40 डिग्री में ठंड लग रही है।» → **AGAINST** (mock dismissal; the surface words affirm, the act denies).
- «শীতকালে গরম পড়ছে, আর আমরা বলি জলবায়ু বদলাচ্ছে না!» → **AGAINST**.
- Sarcasm is labelled by the *intended* stance, and every sarcastic example must carry `"figurative": "sarcasm"` in metadata so error analysis can isolate it.

### 2.4 Scientific uncertainty vs denial
- «वैज्ञानिकों में अभी बहुत अनिश्चितता है, डेटा अपूर्ण है।» → **NONE** (epistemic hedging, no rejection of seriousness).
- «वैज्ञानिकों का डेटा ही फर्जी है, सब पैसे का खेल है।» → **AGAINST** (denial of the evidence base).

### 2.5 Implicit environmental concern
- «बच्चों के लिए कैसा कल छोड़ेंगे, कुछ सोचो।» (in a climate thread) → **FAVOUR** (implicit concern).
- «প্লাস্টিক কমাও, নাহলে ভবিষ্যৎ নেই।» → **FAVOUR**.
- Context matters: implicit examples must record the thread topic in `context_note`; without a climater context, the same sentence is `NONE`.

### 2.6 Religious / cultural framing
- «भगवान सब ठीक कर देंगे, डरने की बात नहीं।» → **NONE** (fatalism, not denial) unless it explicitly denies seriousness → then `AGAINST`.
- «প্রকৃতি মায়ের প্রতি অন্যায় করছি, এটা গুরুতর।» → **FAVOUR**.

### 2.7 Unrelated climate vocabulary
- «आज मौसम अच्छा है, घूमने चलें।» → **NONE**. Weather ≠ climate stance.
- «বৃষ্টি হচ্ছে, ক্লাইমেট চেঞ্জ নিয়ে ভয় পাওয়ার কিছু নেই।» → **AGAINST** (explicit dismissal).

### 2.8 Questions
- «क्या सच में इतनी बड़ी समस्या है?» → **NONE** (genuine open question). If rhetorical and dismissive → **AGAINST**.

### 2.9 Script and code-mixing
Label is independent of script. A Romanised comment is annotated exactly like its Devanagari twin:
- «climate change koi serious problem nahi hai» → **AGAINST**.
- «জলবায়ু change really serious, amra ki korchi?» → **FAVOUR**.

---

## 3. Metadata schema (one JSON object per line)

```json
{
  "id": "hi_yt_000123",
  "text": "जलवायु परिवर्तन एक गंभीर समस्या है",
  "text_normalized": null,
  "language": "Hindi",
  "script": "Devanagari",
  "romanized": false,
  "code_mixed": false,
  "label": "FAVOUR",
  "figurative": null,
  "annotation_status": "human_verified",
  "annotator_id": "A1",
  "second_annotator_id": "A2",
  "annotation_confidence": "high",
  "disagreement": false,
  "source": "youtube_public_comments",
  "source_permalink_hash": "sha256:...",
  "collection_date": "2026-11-01",
  "context_note": "comment under a climate documentary video"
}
```

Rules: no usernames, no channel IDs, no raw permalinks — only salted hashes. `annotation_status` ∈ {`human_verified`, `adjudicated`, `pending_human_annotation`, `synthetic`}. **Synthetic rows never enter the supervised training split unless explicitly flagged and reported separately.**

---

## 4. Collection protocol (rights-respecting)

1. Only **publicly visible** comments from public videos; no login-walled, private, deleted, or member-only content.
2. No bypassing of rate limits, no CAPTCHA circumvention, no scraping of personal profiles.
3. Publish only short comment text required for the research task; strip handles, links, phone numbers, and any identifier of a private individual.
4. Store `source_permalink_hash` rather than the URL, so the corpus cannot be used to re-identify commenters.
5. Follow the YouTube Terms of Service and the API quota; prefer the official Data API over HTML scraping.
6. Provide a takedown route in the repo README for anyone who wants their comment removed.

## 5. Double annotation and agreement

- Target: **≥ 20 %** of the corpus double-annotated (both annotators blind to each other).
- Compute **Cohen's κ** per language and overall; report **Krippendorff's α** if a third annotator or missing labels appear.
- Interpretation gate: κ < 0.60 → guideline is under-specified; revise §2 and re-annotate the affected batch. 0.60 ≤ κ < 0.80 → acceptable, disagreements adjudicated by a third pass. κ ≥ 0.80 → substantial agreement.
- Disagreements are **adjudicated**, never averaged. The pre-adjudication pair and the final label are both stored so that the confusion between classes can be reported.

## 6. Sampling targets

| Slice | Hindi | Bengali |
|---|---|---|
| Devanagari / Bengali script | ≥ 1 200 | ≥ 1 200 |
| Romanised | ≥ 300 | ≥ 300 |
| Code-mixed with English | ≥ 300 | ≥ 300 |
| Sarcasm (within above) | ≥ 150 | ≥ 150 |
| Negation (within above) | ≥ 200 | ≥ 200 |

Class balance target: no class below 20 % of a language slice; if the natural distribution is skewed, **report it and use class weighting**, do not silently resample.
