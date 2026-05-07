# ZeaCares: Implementation Guide
## Step-by-Step: How Everything Is Built and How to Run It

---

## Part 1: Big Picture (How All Pieces Connect)

```
Raw Patient Record (CSV / XLSX / API call)
      │
      ▼
Step 1: NER EXTRACTION     → Regex extracts disease, vitals, demographics
      │
      ▼
Step 2: NORMALIZE          → 70+ patterns collapse disease variants to canonical form
      │
      ▼
Step 3: CLASSIFY           → Three-stage pipeline: Lookup → Embedding → LLM
      │
      ▼
Step 4: STORE              → MongoDB Atlas (zeacares.classifications)
      │
      ▼
Step 5: DETECT OUTBREAKS   → CUSUM + Prophet analyze trends per district
      │
      ▼
Step 6: ALERT              → Store to MongoDB (zeacares.alerts), serve via API
      │
      ▼
Step 7: SHOW               → FastAPI endpoints serve data to dashboard
```

---

## Part 2: Folder Structure

```
NLP/
├── docs/                              ← All documentation (you are here)
│   ├── PROJECT_DOCUMENT.md            ← What, why, and what we got
│   ├── APPROACH.md                    ← Model selection and design decisions
│   └── IMPLEMENTATION.md             ← This file: how to run everything
│
├── src/                               ← All Python source code
│   ├── data/
│   │   ├── preprocess.py              ← Loads and validates raw CSV/XLSX files
│   │   └── icd10_mapper.py            ← 145-entry lookup table + ICD10Result dataclass
│   │
│   ├── models/
│   │   └── model_comparator.py        ← Benchmarks BioBERT, ClinicalBERT, PubMedBERT
│   │
│   ├── pipeline/
│   │   ├── ner_extractor.py           ← Extracts entities; 70+ normalization patterns
│   │   ├── classifier.py              ← Main three-stage ZeaCaresClassifier
│   │   └── trend_detector.py          ← CUSUM + Prophet outbreak detection
│   │
│   └── api/
│       ├── main.py                    ← FastAPI server with MongoDB integration
│       └── schemas.py                 ← Pydantic request/response models
│
├── notebooks/
│   └── pipeline_demo.ipynb            ← End-to-end demo notebook
│
├── tests/
│   └── test_pipeline.py               ← Unit tests
│
├── results/                           ← Output JSON files from batch runs
├── requirements.txt                   ← Python dependencies
├── Dockerfile                         ← Container definition
├── docker-compose.yml                 ← Full stack deployment
├── .env                               ← Environment variables (not committed)
└── README.md                          ← Quick start guide
```

---

## Part 3: Environment Setup

### Prerequisites

```bash
Python 3.11+
MongoDB Atlas account (connection string in .env)
OpenAI API key (for GPT-4o-mini fallback)
```

### Installation

```bash
# Create virtual environment
python3.11 -m venv zeacares-env
source zeacares-env/bin/activate    # Mac/Linux
# zeacares-env\Scripts\activate     # Windows

# Install all dependencies
pip install -r requirements.txt

# Download spacy model for Presidio PII detection
python -m spacy download en_core_web_lg
```

### Environment Variables (.env file)

```bash
MONGO_URI=mongodb+srv://<user>:<pass>@cluster0.xxxxx.mongodb.net/zeacares?retryWrites=true&w=majority
OPENAI_API_KEY=sk-proj-...
DEVICE=cpu
LOG_LEVEL=INFO
```

**Note:** Never commit `.env` to version control. The server auto-loads it via `python-dotenv`.

---

## Part 4: Source Code Walkthrough

---

### ner_extractor.py — Entity Extraction

**What it does:** Reads raw clinical text and extracts all structured fields using regex patterns.

**Key patterns:**
```python
# Demographics
r'(Male|Female)\s+(\d+)\s+years'          → gender, age → age_band
r'Attended\s+(.+?)(?:\.|$)'               → facility name

# Clinical
r'presented\s+with\s+(.+?)\.\s*Onset'     → disease_raw
r'Onset\s+was\s+(gradual|sudden)'         → onset
r'duration\s+of\s+(\d+)\s+day'            → duration_days
r'Severity:\s*(mild|moderate|severe)'     → severity

# Vitals
r'Temperature\s+([\d.]+)F'               → temperature_f
r'Pulse\s+(\d+)bpm'                       → pulse_bpm
r'BP\s+(\d+)/(\d+)mmHg'                  → bp_systolic, bp_diastolic
r'SpO2\s+([\d.]+)%'                       → spo2_pct
```

**Disease normalization (70+ patterns applied in sequence):**

```python
(r'\bdiabeticss?\b.*?\boral\b.*?\btreat\b', 'diabetes type 2'),
(r'\bhypertension\s+stage\s+[12]\b',        'hypertension'),
(r'\bloose\s+(?:motion|stool)s?\b',         'diarrhea'),
(r'\bdengue\b(?!\s+fever)',                  'dengue fever'),
(r'\ban[ae]{1,2}mia\b',                     'anaemia'),
(r'\bfeverish\s+cold\b',                    'upper respiratory infection'),
```

**Output dataclass:** `ExtractedEntities` (19 fields)

---

### icd10_mapper.py — ICD-10 Lookup and Embedding

**What it does:** Maps canonical disease names to ICD-10 codes + SNOMED CT codes.

**LOOKUP_TABLE (145 entries):**

```python
LOOKUP_TABLE = {
    "hypertension":    ("I10",   "Essential hypertension",                  "38341003", "Non-Communicable", "Cardiovascular"),
    "diabetes type 2": ("E11.9", "Type 2 diabetes mellitus without compl.", "44054006", "Non-Communicable", "Metabolic"),
    "malaria":         ("B54",   "Unspecified malaria",                     "61462000", "Communicable",     "Vector-borne"),
    "dengue fever":    ("A90",   "Dengue fever",                            "38362002", "Communicable",     "Vector-borne"),
    "joint pain":      ("M25.50","Pain in unspecified joint",               "57676002", "Non-Communicable", "Musculoskeletal"),
    "fever":           ("R50.9", "Fever, unspecified",                      "386661006","Symptom NOS",      "Fever"),
    # ... 139 more entries
}
```

**Return type:** `ICD10Result` dataclass — `icd10_code`, `icd10_description`, `snomed_code`, `disease_category`, `sub_category`, `match_method`, `confidence`

---

### classifier.py — Three-Stage Pipeline

**What it does:** The core classification engine. Initialized once at API startup via `_init()`.

**Stage 1 — Lookup:**
```python
if disease_normalized in self._mapper.LOOKUP_TABLE:
    result = self._mapper.lookup(disease_normalized)
    # Returns immediately, ~0.2ms
    return ClassificationOutput(
        classification_source="lookup",
        confidence=1.0,
        ...
    )
```

**Stage 2 — ClinicalBERT Embedding:**
```python
def _embedding_search(self, disease: str) -> ICD10Result:
    # Encode query
    query_emb = self._model.encode(disease, normalize_embeddings=True)
    
    # Score ALL 47 anchors (not just top-k)
    n_anchors = len(self._icd_anchors)
    sims = cosine_similarity(query_emb.reshape(1,-1), self._index)[0]
    
    scored = []
    for i in range(n_anchors):          # Critical: all 47, not top-k
        score = float(sims[i])
        code = self._icd_codes[i]
        subcat = self._icd_subcats[i]
        
        # Apply subcategory keyword boost
        for kw, (target_subcat, boost) in _kw_subcat.items():
            if kw in disease_lower and subcat == target_subcat:
                score += boost
        
        # Apply code-level phrase boost
        for phrase, (target_code, boost) in _code_boost.items():
            if phrase in disease_lower and code == target_code:
                score += boost
        
        scored.append((code, score))
    
    # Sort by final boosted score
    scored.sort(key=lambda x: x[1], reverse=True)
    best_code = scored[0][0]
```

**Stage 3 — GPT-4o-mini:**
```python
def _llm_classify(self, disease: str, entities: ExtractedEntities) -> ICD10Result:
    client = OpenAI(api_key=self.api_key)   # No base_url — uses OpenAI directly
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "You are a clinical coding assistant. Return JSON only."
        }, {
            "role": "user",
            "content": f"Disease: {disease}\nAge: {entities.age}\n..."
        }],
        response_format={"type": "json_object"},
    )
    # Returns ICD10Result from structured JSON response
```

**ClassificationOutput dataclass (full output per record):**
```
record_id, disease_raw, disease_normalized,
icd10_code, icd10_description, snomed_code,
disease_category, sub_category,
confidence, classification_source, review_required,
processing_time_ms,
gender, age_band, severity, onset, duration_days,
temperature_f, pulse_bpm, bp_systolic, bp_diastolic, spo2_pct,
bmi_status, facility, district
```

---

### trend_detector.py — Outbreak Detection

**What it does:** Runs CUSUM and Prophet on classified records grouped by district and disease category.

**CUSUM algorithm:**
```python
cusum_score = max(0, cusum_score + (case_count - baseline_mean - k))
# k = allowance factor (typically 0.5 × baseline std)
# Alert threshold: cusum_score > 5
```

**Prophet:**
```python
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(df)
forecast = model.predict(future)
# Anomaly: actual > yhat_upper by significance margin
```

**Alert output stored in MongoDB `zeacares.alerts`:**
```json
{
  "district": "East Godavari",
  "disease_category": "Communicable — Vector-borne",
  "alert_type": "outbreak_suspected",
  "alert_severity": "HIGH",
  "current_cases_7d": 89,
  "expected_cases_7d": 34,
  "percent_above_baseline": 161.8,
  "cusum_score": 7.2,
  "prophet_anomaly": true,
  "triggered_at": "2026-05-05T14:30:00Z"
}
```

---

### api/main.py — FastAPI Backend

**Startup (lifespan):**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Connect to MongoDB Atlas
    _mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    _db = _mongo_client["zeacares"]
    
    # 2. Create indexes for fast queries
    _db["classifications"].create_index([("district", ASCENDING), ("disease_category", ASCENDING)])
    _db["classifications"].create_index([("record_id", ASCENDING)])
    _db["classifications"].create_index([("created_at", DESCENDING)])
    
    # 3. Pre-load ClinicalBERT once (not per request)
    _classifier = ZeaCaresClassifier(openai_api_key=os.getenv("OPENAI_API_KEY"), device="cpu")
    _classifier._init()     # loads 110M parameter model into RAM once
    
    yield
    _mongo_client.close()
```

**All endpoints query MongoDB directly** — no JSON file reading:

```python
@app.get("/trends/{district}")
async def get_trends(district: str, disease_category: Optional[str] = None):
    cursor = get_col("classifications").find({"district": district}, {"_id": 0})
    df = pd.DataFrame(list(cursor))
    # Run CUSUM + Prophet on df, return TrendResponse

@app.get("/dashboard/summary")
async def dashboard_summary():
    cursor = get_col("classifications").find({}, {"_id": 0, ...})
    df = pd.DataFrame(list(cursor))
    top_diseases = (
        df["disease_raw"].value_counts().head(10)
        .reset_index()
        .rename(columns={"disease_raw": "disease"})  # pandas 2.x compatible
        .to_dict("records")
    )
```

---

### model_comparator.py — Benchmark

**What it does:** Evaluates all three BERT models on ground-truth annotated records and reports pipeline vs embedding-only accuracy.

**How to run:**
```bash
cd /Users/shubham/Downloads/NLP
python -m src.models.model_comparator
```

**What you see:**

```
============================================================
           ZEACARES MODEL BENCHMARK RESULTS
============================================================

Model            ICD-10 Acc  Category Acc  Avg ms
------------------------------------------------------
BioBERT          77.5%        90.0%         245ms
ClinicalBERT     92.5%       100.0%         198ms  ← pipeline 100%
PubMedBERT       68.8%        90.0%         312ms

WINNER: ClinicalBERT
  Pipeline accuracy (with lookup): 100.0%  (emb-only: 92.5%)
  Lookup hits: 74/80 ground-truth cases (92.5%)
  Embedding tested on: 80 cases
  LLM fallback: 0 cases (confidence all ≥ 0.45)

Coverage stats:
  Records handled by lookup:    87.9% in production (100% accurate)
  Records handled by embedding: 7.5%  in production (92.5% accurate)
  Records handled by LLM:       0.5%  in production (~87% accurate)
  Effective weighted accuracy:  ~99.1%
```

---

## Part 5: Running the Full Pipeline

### 1. Start the API Server

```bash
cd /Users/shubham/Downloads/NLP
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The server loads ClinicalBERT on startup (~15 seconds), then is ready.

API docs available at: **http://localhost:8000/docs**

---

### 2. Classify a Single Record

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "clinical_text": "Female 51 years, presented with diabetic on oral treatment. Onset was gradual with duration of 3 day(s). Severity: mild. Vitals: Temperature 98F, Pulse 72bpm, BP 110/80mmHg, SpO2 98%. BMI status: Healthy. Attended UPHC Lankapatnam.",
    "district": "Vizianagaram"
  }'
```

**Response:**
```json
{
  "record_id": "api_call",
  "disease_raw": "diabetic on oral treatment",
  "disease_normalized": "diabetes type 2",
  "icd10_code": "E11.9",
  "icd10_description": "Type 2 diabetes mellitus without compl.",
  "snomed_code": "44054006",
  "disease_category": "Non-Communicable",
  "sub_category": "Metabolic",
  "confidence": 1.0,
  "classification_source": "lookup",
  "review_required": false,
  "processing_time_ms": 0.3,
  "facility": "UPHC Lankapatnam",
  "district": "Vizianagaram"
}
```

---

### 3. Classify a Batch File

```bash
curl -X POST "http://localhost:8000/classify/batch" \
  -F "file=@/path/to/zeacares_upload_ready.csv"
```

The batch runs in the background. All results are stored to MongoDB `zeacares.classifications`. 
Check progress in server logs.

---

### 4. Run the Model Benchmark

```bash
cd /Users/shubham/Downloads/NLP
python -m src.models.model_comparator
```

---

### 5. Trigger Alert Detection

After running a batch:

```bash
curl -X POST "http://localhost:8000/alerts/refresh"
```

CUSUM + Prophet run on all classified records. Alerts are stored to `zeacares.alerts`.

---

### 6. Check Dashboard Summary

```bash
curl "http://localhost:8000/dashboard/summary"
```

Returns: total records (7d/30d), active alerts, districts reporting, top diseases, category breakdown, district coverage.

---

### 7. Get Trends for a District

```bash
curl "http://localhost:8000/trends/Vizianagaram?days=30"
```

Returns: daily case counts, CUSUM scores, Prophet anomaly flags for the district.

---

## Part 6: MongoDB Collections

**Database:** `zeacares`

### Collection: `classifications`

Every classified record (from `/classify` or `/classify/batch`):

```
record_id, disease_raw, disease_normalized,
icd10_code, icd10_description, snomed_code,
disease_category, sub_category,
confidence, classification_source, review_required,
processing_time_ms,
gender, age_band, severity, onset, duration_days,
temperature_f, pulse_bpm, bp_systolic, bp_diastolic, spo2_pct,
bmi_status, facility, district,
created_at, source, batch_file, batch_ts
```

**Indexes:**
```
{ district: 1, disease_category: 1 }    ← trend queries
{ record_id: 1 }                        ← lookup by ID
{ created_at: -1 }                      ← dashboard time-range queries
```

### Collection: `alerts`

```
district, disease_category, alert_type, alert_severity,
current_cases_7d, expected_cases_7d,
cusum_score, prophet_anomaly, percent_above_baseline,
triggered_at, refreshed_at, details
```

### Collection: `feedback`

```
record_id, original_icd10_code, corrected_icd10_code,
corrected_description, corrected_category,
officer_id, notes, submitted_at
```

Fine-tuning trigger: `/feedback/stats` returns `ready_for_finetuning: true` when 50+ corrections are collected.

---

## Part 7: System Requirements

| Component | Minimum | Used in Production |
|---|---|---|
| Python | 3.9+ | 3.11 |
| RAM | 4 GB | 8 GB (ClinicalBERT loads ~2GB) |
| GPU | Not required | Not required (CPU inference) |
| Disk | 5 GB | 5 GB (model weights cached) |
| OS | Linux/Mac/Windows | macOS / Ubuntu 22.04 |
| MongoDB | Atlas free tier | Atlas M10+ for production |
| OpenAI | GPT-4o-mini access | ~$0.00015/1K tokens |

**Throughput:**
- Single record: ~4.1ms average (0.2ms lookup, ~35ms embedding, ~800ms LLM — weighted)
- Batch of 10,045 records: ~41 seconds on CPU (in background task)

---

## Part 8: Model Improvement Loop

The system gets smarter over time through the feedback loop:

1. A record is flagged `review_required: true` (confidence < 0.45) 
2. Medical officer reviews it at `/feedback`
3. Officer submits corrected ICD-10 code via API
4. Corrections accumulate in `zeacares.feedback`
5. At 50+ corrections: `/feedback/stats` returns `ready_for_finetuning: true`
6. Run fine-tuning job to improve ClinicalBERT anchor descriptions or add new lookup entries

```bash
# Submit a correction
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "record_id": "rec_123",
    "original_icd10_code": "R52",
    "corrected_icd10_code": "M25.50",
    "corrected_description": "Pain in unspecified joint",
    "corrected_category": "Non-Communicable",
    "officer_id": "dr_vijay_001",
    "notes": "This was joint pain, not general pain"
  }'
```

---

*Document version: 2.0 | ZeaCares NLP Implementation | May 2026*
