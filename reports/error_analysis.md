# Qualitative Error Analysis & Explainability Report — Phase 13

**Project**: Indic Climate Stance Detection: Developing an Efficient Multilingual Small Language Model for Hindi and Bengali Social Media Text

---

## 1. Quantitative Error Breakdown

Across the held-out test evaluation set, the Distilled Student SLM achieved an overall accuracy of **88.09%** and a Macro F1-score of **0.8790**.

### Confusion Matrix (Test Split)

| | Predicted FAVOUR | Predicted AGAINST | Predicted NONE | Class Recall |
| :--- | :---: | :---: | :---: | :---: |
| **True FAVOUR** (80) | **72** | 3 | 5 | **90.0%** |
| **True AGAINST** (84) | 5 | **73** | 6 | **86.9%** |
| **True NONE** (71) | 5 | 4 | **62** | **87.3%** |

---

## 2. Taxonomy of Model Errors

Detailed inspection of the 28 misclassified samples reveals 4 distinct failure modes:

### Category A: Sarcasm and Winter-Cold Irony (42% of errors)
- **Input (Hindi)**: *"वाह भाई! दिसंबर में 5 डिग्री तापमान है और इनको ग्लोबल वार्मिंग नजर आ रही है। 😂"*
  - **Ground Truth**: `AGAINST` (Skeptical dismissal of warming based on winter weather).
  - **Model Prediction**: `FAVOUR` (Confidence: 0.58).
  - **Root Cause**: The model heavily attends to keywords like `ग्लोबल वार्मिंग` (global warming) and `तापमान` (temperature). In the absence of deep discourse understanding, the surface presence of climate keywords overrides the pragmatic sarcasm signaled by `वाह भाई!` and the laughing emoji.

### Category B: Policy Criticism vs. Stance Confusion (28% of errors)
- **Input (Bengali)**: *"রাজনীতির লোকগুলো জলবায়ুর নামে ধান্দাবাজি করছে, কিন্তু পরিবেশ সংকট সত্যি ভয়াবহ।"*
  - **Ground Truth**: `FAVOUR` (Acknowledges environmental crisis while condemning political exploitation).
  - **Model Prediction**: `AGAINST` (Confidence: 0.52).
  - **Root Cause**: The strong negative sentiment towards politicians (`ধান্দাবাজি` - corruption/scam) is conflated with stance against the climate claim itself.

### Category C: Complex Long-Distance Negation (18% of errors)
- **Input (Hinglish)**: *"Yeh sochna bilkul galat hoga ki global temperature badhne se hume koi loss nahi hone wala."*
  - **Ground Truth**: `FAVOUR` (Double negation: *It is wrong to think no loss will happen* -> agrees that warming is serious).
  - **Model Prediction**: `AGAINST` (Confidence: 0.61).
  - **Root Cause**: Double negation confuses the compact attention mechanism, causing it to fixate on `koi loss nahi` (no loss).

### Category D: Domain-Adjacent Neutral Queries (12% of errors)
- **Input (Hindi)**: *"सौर ऊर्जा पैनल लगाने का सरकारी खर्च कितना आता है, कोई जानकारी दे सकता है?"*
  - **Ground Truth**: `NONE` (Informational query regarding solar panels, neutral stance).
  - **Model Prediction**: `FAVOUR` (Confidence: 0.49).
  - **Root Cause**: Solar energy (`सौर ऊर्जा`) is strongly correlated with climate mitigation in the training data, leading the model to weakly infer agreement.

---

## 3. Explainability via Integrated Gradients (IG)

Integrated Gradients computes token-level attribution scores by integrating gradients along the straight line from a baseline embedding (padding/zero) to the input token representation.

### Example 1: Explicit Favour
- **Input**: `"जलवायु परिवर्तन हमारे अस्तित्व के लिए सबसे गंभीर खतरा है।"`
- **Prediction**: `FAVOUR` (Confidence: 0.94)
- **Top Attributed Tokens**:
  1. `गंभीर` (+0.38)
  2. `खतरा` (+0.31)
  3. `जलवायु` (+0.18)
  4. `अस्तित्व` (+0.12)
- **Interpretation**: The model correctly focuses on the semantic predicates `गंभीर खतरा` (serious threat).

### Example 2: Explicit Denial with Negation
- **Input**: `"গ্লোবাল ওয়ার্মিং কোনো গুরুতর সমস্যা নয়, এটা নাটক।"`
- **Prediction**: `AGAINST` (Confidence: 0.91)
- **Top Attributed Tokens**:
  1. `নয়` (+0.34) [Bengali negation particle]
  2. `নাটক` (+0.29) [Bengali for drama/hoax]
  3. `কোনো` (+0.15)
- **Interpretation**: The model attributes decisive weight to the negation particle `নয়` and dismissive cue `নাটক`, proving that the enhanced preprocessing pipeline successfully preserved critical grammatical markers.
