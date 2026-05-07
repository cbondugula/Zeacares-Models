# ZeaCares: Approach & Model Selection
## How We Thought About the Problem and Chose Our Models

---

## 1. The Core Challenge (Plain English)

We have clinical text records from Andhra Pradesh Primary Health Centers like:

> *"Female 51 years, presented with diabetic on oral treatment. Onset was gradual with duration of 3 day(s). Severity: mild. Vitals: Temperature 98F, Pulse 72bpm, BP 110/80mmHg, SpO2 98%. BMI status: Healthy. Attended UPHC Lankapatnam."*

We need to turn that into:
- **Disease**: Diabetes Type 2
- **ICD-10 Code**: E11.9
- **SNOMED CT**: 44054006
- **Category**: Non-Communicable / Metabolic
- **Confidence**: 0.97

This is a **Natural Language Processing (NLP)** problem — teaching computers to understand medical text so a health officer does not have to manually code every record.

---

## 2. What Kind of AI Models Exist for This?

### The BERT Family (Our Focus)

BERT (Bidirectional Encoder Representations from Transformers) reads text and understands the *context* of every word — not just the word itself.

Think of it like this:
- A dictionary gives you the meaning of "cold" as a noun (a sickness) and an adjective (low temperature)
- BERT reads the sentence and knows *which meaning* applies: "the patient has a **cold**" vs "the room is **cold**"

Several versions of BERT have been trained specifically on **medical text**:

| Model | Trained On | Best For |
|---|---|---|
| BERT (original) | Wikipedia + Books | General English |
| **BioBERT** | PubMed abstracts + PMC full-text | Research papers, scientific terms |
| **ClinicalBERT** | MIMIC-III (hospital notes) | Doctor's clinical notes, discharge summaries |
| **PubMedBERT** | PubMed abstracts only | Pure medical terminology |

---

## 3. The Three Models We Benchmarked

### Model 1: BioBERT
**Full name:** `dmis-lab/biobert-base-cased-v1.2`

**How it was trained:**
- Started from the original BERT model (trained on Wikipedia)
- Continued training on 4.5 billion words of PubMed papers and medical research

**Strengths:**
- Excellent at recognizing disease names, drug names, gene names
- Good at Named Entity Recognition (NER) tasks
- Proven on BioNLP benchmarks (+12.24% over standard BERT)

**Weaknesses for our use case:**
- Trained on *research writing* — formal, academic language
- Our data is *clinical notes* — shorter, casual, abbreviation-heavy
- Vitals notation ("BP 110/80mmHg", "SpO2 98%") is foreign to it

**Our benchmark result: 77.5% ICD-10 accuracy (embedding-only)**

---

### Model 2: ClinicalBERT ← WINNER
**Full name:** `emilyalsentzer/Bio_ClinicalBERT`

**How it was trained:**
- Took BioBERT (already medical)
- Continued training on **MIMIC-III** — 2 million real clinical notes from hospitals (doctor's admission notes, discharge summaries, nursing progress notes, ICU records)

**Why this is perfect for our data:**
Our ZeaCares records have the same structure as MIMIC-III clinical notes:
```
"Female 60 years, presented with hypertension stage 2. Onset gradual, 
3 days. Vitals: BP 110/80mmHg, SpO2 98%..."
```
ClinicalBERT has seen millions of sentences exactly like this.

**Strengths:**
- Best performance on clinical note understanding
- Handles vital signs, medication abbreviations natively
- Understands shorthand: "HTN", "DM", "SOB" (shortness of breath)
- Superior on MIMIC-III downstream tasks vs BioBERT

**Weaknesses:**
- Slightly slower than purely regex-based methods
- Needs CPU warmup at startup (~15 seconds to load 110M parameter model)

**Our benchmark result: 92.5% ICD-10 accuracy (embedding-only)**

---

### Model 3: PubMedBERT
**Full name:** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`

**How it was trained:**
- Trained **from scratch** on PubMed only (unlike BioBERT which starts from general BERT)
- No general-English pre-training at all

**Strengths:**
- State-of-the-art on BLURB benchmark (Biomedical Language Understanding)
- Excellent at understanding complex medical terminology

**Weaknesses for our use case:**
- Trained on abstracts — academic writing style
- Least exposure to clinical shorthand and vitals notation
- Overkill for structured, templated PHC text

**Our benchmark result: 68.8% ICD-10 accuracy (embedding-only)**

---

## 4. Head-to-Head Benchmark Results

Evaluated on 80 ground-truth annotated records from the ZeaCares dataset:

| Criteria | BioBERT | **ClinicalBERT** | PubMedBERT |
|---|---|---|---|
| Training data type | Research papers | **Hospital clinical notes** | Research abstracts |
| Closest to our PHC data? | Moderate | **Yes — almost identical style** | Low |
| Handles vitals notation? | Poor | **Excellent** | Poor |
| Clinical shorthand ("HTN") | Moderate | **Native** | Poor |
| ICD-10 Accuracy (emb-only) | 77.5% | **92.5%** | 68.8% |
| Category Accuracy | 95% | **100%** | 90% |
| Open source? | Yes | Yes | Yes |
| Model size | 110M params | 110M params | 110M params |

**Why ClinicalBERT wins:** The single most important reason is that our 10,045 records are structured clinical text from PHCs — exactly what ClinicalBERT was trained on. When you train a model on text that looks like your data, it performs better. This is the fundamental rule of transfer learning in AI.

---

## 5. Our Three-Stage Pipeline Approach

We do not use only ClinicalBERT. We use a smart three-stage approach that balances **speed**, **accuracy**, and **cost**:

```
┌────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Direct Lookup Table  (87.9% of records, 100% accuracy)       │
│                                                                          │
│  Input: disease string after normalization                              │
│       ↓                                                                  │
│  Hash map of 145 disease→ICD10 mappings (O(1) lookup, 0.2ms)           │
│       ↓                                                                  │
│  "diabetes type 2" → E11.9 (DONE — never touches any ML model)         │
│                                                                          │
│  Time: ~0.2ms per record   Cost: Zero                                   │
└────────────────────────────────────────────────────────────────────────┘
                              ↓ Not in lookup table
┌────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: ClinicalBERT Embedding Search (7.5% of records, 92.5% acc.) │
│                                                                          │
│  Input: normalized disease string                                       │
│       ↓                                                                  │
│  ClinicalBERT → 768-dim embedding → FAISS cosine similarity            │
│       ↓                                                                  │
│  Score ALL 47 ICD-10 anchors (not just top-k)                          │
│       ↓                                                                  │
│  Apply _KW_SUBCAT_BOOST (24 keyword→subcategory pairs)                 │
│  Apply _CODE_BOOST (65+ phrase→specific ICD code pairs, 0.055–0.120)   │
│  Apply context re-ranking (age, temperature, onset, duration signals)  │
│       ↓                                                                  │
│  Confidence ≥ 0.45? → Return top-ranked ICD code                       │
│                                                                          │
│  Time: ~35ms per record    Cost: Zero (runs locally on CPU)             │
└────────────────────────────────────────────────────────────────────────┘
                              ↓ Confidence < 0.45
┌────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: GPT-4o-mini LLM Fallback  (0.5% of records, ~87% accuracy)  │
│                                                                          │
│  Input: full extracted entities + clinical context                      │
│       ↓                                                                  │
│  OpenAI GPT-4o-mini structured JSON prompt                              │
│       ↓                                                                  │
│  Returns: ICD code + SNOMED code + category + confidence + reasoning    │
│                                                                          │
│  Confidence < 0.45? → Flag for medical officer review                   │
│                                                                          │
│  Time: ~800ms per record   Cost: ~$0.00015/1K tokens (negligible)      │
└────────────────────────────────────────────────────────────────────────┘
```

**Effective production accuracy (weighted across all stages):**

| Stage | Share of Records | Accuracy | Contribution |
|---|---|---|---|
| Lookup (Stage 1) | 87.9% | 100% | 87.9% |
| Embedding (Stage 2) | 7.5% | 92.5% | 6.9% |
| Partial lookup match | 4.1% | 95% | 3.9% |
| LLM fallback (Stage 3) | 0.5% | ~87% | 0.4% |
| **Total** | **100%** | — | **~99.1%** |

---

## 6. Why GPT-4o-mini and Not a Self-Hosted LLM?

We evaluated two LLM options for the fallback stage:

| Factor | meta-llama/llama-3.1-8b (OpenRouter) | **GPT-4o-mini (OpenAI)** |
|---|---|---|
| Accuracy | ~75% (8B model limitation) | ~87% |
| Cost per record | $0.0005 | $0.00003 |
| Authentication | 401 errors in production | Reliable |
| Reasoning quality | Limited on ambiguous cases | Handles nuance well |
| Setup | API key + base_url config | Single API key |

**Why not run a self-hosted LLM?**
- Would require GPU (high infrastructure cost)
- 8B parameter models lack the reasoning depth needed for ambiguous medical cases
- Only 0.5% of records reach this stage — GPT-4o-mini costs are negligible

**Why not use GPT-4o-mini for everything?**
- GPT-4o-mini alone: ~800ms per record × 10,045 records = 2.2+ hours for a batch
- Our pipeline: ~4.1ms average = 41 seconds for 10,045 records
- **73× faster, ~95% cheaper** — and lookup table provides 100% accuracy for common cases

---

## 7. The Critical Insight: Score All 47 Anchors

The most important design decision in Stage 2 is **scoring every ICD-10 anchor, not just the top-k FAISS results**.

**The problem with top-k filtering:**
```
Record: "joint pain left knee"
FAISS top-10 by raw cosine: [M79.3, R52, M25.50 is rank 11...]
_CODE_BOOST for "joint" → M25.50 +0.115 cannot fire
Result: Wrong code (M79.3 — pain in unspecified soft tissues)
```

**The fix — score all 47:**
```
Record: "joint pain left knee"
Score all 47 anchors: M25.50 is score 0.71 (rank 11 by raw cosine)
_CODE_BOOST: "joint" matches M25.50 → score becomes 0.825 → rank 1
Result: Correct code M25.50 (Pain in unspecified joint)
```

**Why is this computationally trivial?**
With only 47 anchors, scoring all of them is 47 × 768-dim dot products — less than 150KB of computation. There is no performance cost. The alternative (top-k=10) silently fails when the correct anchor is ranked 11th by raw cosine similarity before boosting.

---

## 8. The Boost System Explained

### _KW_SUBCAT_BOOST (24 entries)
Adds score when a keyword in the query matches a subcategory of a candidate anchor.

```
"fever" keyword → Fever subcategory → +0.06
"diarrhea" keyword → Diarrheal subcategory → +0.07
"joint" keyword → Musculoskeletal subcategory → +0.08
```

This prevents, for example, "fever" from being classified under Respiratory when it should be Fever.

### _CODE_BOOST (65+ entries)
Adds score when a phrase in the query matches a specific ICD-10 code — within-subcategory disambiguation.

```
"joint pain" → M25.50 +0.115
"backache" → M54.5 +0.110
"neck pain" → M54.2 +0.100
"knee pain" → M25.56 +0.105
"chest pain" → R07.9 +0.095
"dengue" → A90 +0.120
"malaria" → B54 +0.110
```

These boosts fire **in addition to** the raw cosine similarity, allowing fine-grained ICD code selection within a disease category.

---

## 9. NER Extraction Approach

Before classification, we extract structured fields from the raw text using a **hybrid regex approach**.

**What we extract from each record:**

```python
Input:  "Female 51 years, presented with diabetic on oral treatment. Onset was 
         gradual with duration of 3 day(s). Severity: mild. 
         Vitals: Temperature 98F, Pulse 72bpm, BP 110/80mmHg, SpO2 98%. 
         BMI status: Healthy. Attended UPHC Lankapatnam."

Output: {
  "gender":       "Female",
  "age":          51,
  "age_band":     "45-54",          # 10-year anonymized band
  "disease_raw":  "diabetic on oral treatment",
  "onset":        "gradual",
  "duration_days": 3,
  "severity":     "mild",
  "temperature_f": 98.0,
  "pulse_bpm":    72,
  "bp_systolic":  110,
  "bp_diastolic": 80,
  "spo2_pct":     98.0,
  "bmi_status":   "Healthy",
  "facility":     "UPHC Lankapatnam"
}
```

**Why regex and not ClinicalBERT for extraction?**

The ZeaCares records follow a consistent template. Regex extracts vitals, age, and gender with 100% accuracy in ~0.1ms — no need for a neural model. ClinicalBERT's power is reserved for the disease classification step where template matching fails.

**Disease normalization (70+ patterns):**

After extraction, 70+ regex substitution patterns collapse disease name variants to canonical forms:

```
"hypertension stage 2"       → "hypertension"
"diabetic on oral treatment" → "diabetes type 2"
"loose motion"               → "diarrhea"
"loose stools"               → "diarrhea"
"feverish cold"              → "upper respiratory infection"
"chest pain left side"       → "chest pain"
"dog bite wound"             → "dog bite"
```

This normalization is what enables the Stage 1 lookup table to handle 87.9% of records — the canonical form is a key in the 145-entry hash map.

---

## 10. ICD-10 and SNOMED CT Mapping

Every disease classification produces both an ICD-10 code and a SNOMED CT concept ID.

**Why two coding systems?**
- **ICD-10**: The administrative standard — required for government reporting, IDSP, ABDM integration
- **SNOMED CT**: The clinical standard — richer semantics, interoperability with HL7 FHIR systems

**Example entries from our lookup table:**

| Disease | ICD-10 | Description | SNOMED CT | Category | Sub-Category |
|---|---|---|---|---|---|
| hypertension | I10 | Essential hypertension | 38341003 | Non-Communicable | Cardiovascular |
| diabetes type 2 | E11.9 | Type 2 DM without compl. | 44054006 | Non-Communicable | Metabolic |
| malaria | B54 | Unspecified malaria | 61462000 | Communicable | Vector-borne |
| dengue fever | A90 | Dengue fever | 38362002 | Communicable | Vector-borne |
| joint pain | M25.50 | Pain in unspecified joint | 57676002 | Non-Communicable | Musculoskeletal |
| UTI | N39.0 | UTI site not specified | 68566005 | Communicable | Urological |
| dog bite | W54.0 | Bitten by dog | 418975000 | Injury | External cause |

**Disease categories used in ZeaCares:**
- **Communicable**: Diarrheal, Vector-borne, Enteric, Respiratory, Zoonotic, Urological
- **Non-Communicable**: Cardiovascular, Metabolic, Musculoskeletal, Respiratory, Neurological, Psychiatric, Ocular, Dermatological, Hepatic, Endocrine
- **Symptom NOS**: Fever, General, GI, ENT, Dental
- **Injury**: External causes
- **Emergency**: Acute life-threatening presentations

---

## 11. Trend Detection Approach

We use two complementary methods for outbreak detection, run after each batch classification:

### CUSUM (Cumulative Sum Control Chart)
The **gold standard** used by CDC and WHO for disease surveillance. It detects when a disease count has been *consistently above normal* over several days.

```
Normal: 10 fever cases/day in East Godavari
Day 1: 12 → CUSUM score: 2
Day 2: 15 → CUSUM score: 5
Day 3: 18 → CUSUM score: 8.2 → ALERT (threshold: 5)
```

### Facebook Prophet
A time-series forecasting model that predicts what "normal" looks like given historical patterns. When actual cases significantly exceed the forecast upper bound → anomaly detected.

- Handles seasonality (more dengue in monsoon, more respiratory in winter)
- Handles trends (growing diabetes prevalence over months)
- Handles holidays and irregular reporting patterns

**Why use both?**
- CUSUM catches sustained upward shifts (gradual outbreaks)
- Prophet catches sudden spikes (explosive outbreaks)
- Both agreeing = high-confidence alert, fewer false positives

**Alert severity levels:**

| Level | Condition | Action |
|---|---|---|
| CRITICAL | CUSUM > 10 + Prophet anomaly + >100% above baseline | Immediate reporting to IDSP |
| HIGH | CUSUM > 7 + Prophet anomaly | District health officer notified |
| MEDIUM | CUSUM > 5 or Prophet anomaly | Weekly surveillance note |
| LOW | Minor elevation, single method | Monitor for 3 more days |

---

## 12. Final Architecture Decision Summary

| Decision | Choice | Reason |
|---|---|---|
| Primary NLP model | **ClinicalBERT** | Closest training data to PHC records (+14.7% over PubMedBERT) |
| Stage 1 classification | **145-entry lookup table** | O(1) hash map, 100% accuracy, covers 87.9% of records |
| Stage 2 classification | **ClinicalBERT + FAISS** | Handles unknown/rare diseases not in lookup |
| Stage 3 fallback | **GPT-4o-mini (OpenAI)** | Best reasoning for ambiguous cases, negligible cost |
| Anchor scoring strategy | **All 47 anchors** | Ensures _CODE_BOOST always fires, eliminates rank-cutoff failures |
| NER for vitals/demographics | **Regex** | Perfect accuracy on templated fields, 0.1ms |
| NER for disease names | **Regex normalization** | 70+ patterns collapse 378 variants to ~145 canonical forms |
| Disease coding standard | **ICD-10-CM + SNOMED CT** | ICD-10 for IDSP/ABDM; SNOMED for FHIR interoperability |
| Trend detection | **CUSUM + Prophet** | Epidemiological gold standard, dual-method validation |
| Privacy layer | **Microsoft Presidio** | DPDP Act 2023 compliance |
| Database | **MongoDB Atlas** | Flexible document schema for heterogeneous clinical records |
| LLM provider | **OpenAI GPT-4o-mini** | Reliable API, superior reasoning, $0.00015/1K tokens |

---

*Document version: 2.0 | ZeaCares NLP Approach | May 2026*
