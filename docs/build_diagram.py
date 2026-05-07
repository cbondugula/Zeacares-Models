"""
ZeaCares Technical Architecture Diagram — Draw.io style PNG
Generates: docs/ZeaCares_Architecture_Diagram.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(28, 20))
ax.set_xlim(0, 28)
ax.set_ylim(0, 20)
ax.axis("off")
fig.patch.set_facecolor("#F8F9FA")

# ── Color Palette ─────────────────────────────────────────────────────────────
C = {
    "title_bg":   "#1A2A4A",
    "input":      "#1565C0",
    "ner":        "#6A1B9A",
    "stage1":     "#1B5E20",
    "stage2":     "#E65100",
    "stage3":     "#880E4F",
    "storage":    "#0277BD",
    "surveillance":"#AD1457",
    "api":        "#004D40",
    "client":     "#37474F",
    "arrow":      "#455A64",
    "white":      "#FFFFFF",
    "light_gray": "#ECEFF1",
    "border":     "#90A4AE",
    "mongo":      "#13AA52",
    "openai":     "#412991",
    "warn":       "#F57F17",
}

def box(ax, x, y, w, h, facecolor, edgecolor="#FFFFFF", linewidth=2,
        radius=0.25, alpha=1.0, zorder=3, style="round,pad=0.05"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={radius*0.2}",
                       facecolor=facecolor, edgecolor=edgecolor,
                       linewidth=linewidth, alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p

def label(ax, x, y, text, fontsize=9, color="white", bold=False,
          ha="center", va="center", zorder=5, wrap=False):
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, fontsize=fontsize, color=color,
            fontweight=weight, ha=ha, va=va, zorder=zorder,
            wrap=wrap, multialignment="center",
            fontfamily="DejaVu Sans")

def arrow(ax, x1, y1, x2, y2, color="#455A64", lw=2.0,
          connectionstyle="arc3,rad=0.0", zorder=4, label_text=None,
          label_color="#333333", label_size=7.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, connectionstyle=connectionstyle),
                zorder=zorder)
    if label_text:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.13, label_text, fontsize=label_size,
                color=label_color, ha="center", va="bottom",
                zorder=zorder+1, style="italic",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.75, pad=1.5))

def divider(ax, y, color="#B0BEC5", lw=0.8):
    ax.axhline(y=y, xmin=0.01, xmax=0.99, color=color, lw=lw,
               linestyle="--", zorder=2)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE BAR
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 0.2, 19.0, 27.6, 0.85, C["title_bg"], "#FFFFFF", 1.5, zorder=5)
label(ax, 14, 19.42, "ZeaCares — AI Disease Surveillance Platform",
      fontsize=18, bold=True, color="#FFFFFF")
label(ax, 14, 19.10, "Technical Architecture Diagram  |  Andhra Pradesh IDSP  |  May 2026",
      fontsize=9.5, color="#90CAF9")

# ══════════════════════════════════════════════════════════════════════════════
# LAYER LABELS (left rail)
# ══════════════════════════════════════════════════════════════════════════════
layers = [
    (18.55, "INPUT",          C["input"]),
    (16.70, "NER & NORMALIZE",C["ner"]),
    (13.50, "CLASSIFICATION", C["stage2"]),
    (10.20, "STORAGE",        C["storage"]),
    (8.30,  "SURVEILLANCE",   C["surveillance"]),
    (5.20,  "API LAYER",      C["api"]),
    (2.20,  "CLIENTS",        C["client"]),
]
for ly, ltext, lc in layers:
    box(ax, 0.20, ly-0.38, 2.05, 0.76, lc, "#FFFFFF", 1, 0.15, zorder=4)
    ax.text(1.22, ly, ltext, fontsize=7.2, color="white",
            fontweight="bold", ha="center", va="center",
            rotation=0, zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# 1. INPUT LAYER  (y ≈ 18.2 – 18.9)
# ══════════════════════════════════════════════════════════════════════════════
# CSV/XLSX upload
box(ax, 2.5, 18.15, 4.0, 0.80, C["input"], "#BBDEFB", 1.5)
label(ax, 4.5, 18.72, "CSV / XLSX Batch Upload", 9.5, bold=True)
label(ax, 4.5, 18.38, "clinicalText + district columns", 8, color="#BBDEFB")

# Single API call
box(ax, 7.2, 18.15, 4.0, 0.80, C["input"], "#BBDEFB", 1.5)
label(ax, 9.2, 18.72, "POST /classify  (single)", 9.5, bold=True)
label(ax, 9.2, 18.38, "JSON { clinical_text, district }", 8, color="#BBDEFB")

# Raw PHC record example
box(ax, 11.9, 17.90, 11.5, 1.05, "#263238", "#546E7A", 1.5)
label(ax, 17.65, 18.61, "Sample PHC Record (Andhra Pradesh)", 8.5, bold=True, color="#80CBC4")
label(ax, 17.65, 18.32,
      '"Female 51 yrs, presented with diabetic on oral treatment.', 8.0, color="#B2EBF2")
label(ax, 17.65, 18.08,
      'Vitals: BP 110/80mmHg, SpO2 98%. Attended UPHC Lankapatnam."', 8.0, color="#B2EBF2")

divider(ax, 17.85)

# ══════════════════════════════════════════════════════════════════════════════
# 2. NER & NORMALISATION LAYER  (y ≈ 16.2 – 17.75)
# ══════════════════════════════════════════════════════════════════════════════
# NER box
box(ax, 2.5, 16.30, 7.0, 1.40, "#4A148C", "#CE93D8", 1.5)
label(ax, 6.0, 17.40, "[NER] Extractor  (ner_extractor.py)", 10, bold=True, color="#F3E5F5")
fields = [
    "gender • age → age_band",
    "disease_raw  •  onset  •  duration_days  •  severity",
    "temperature_f • pulse_bpm • bp_sys/dia • spo2_pct",
    "facility  •  district  •  bmi_status",
]
for i, f in enumerate(fields):
    label(ax, 6.0, 17.10 - i*0.20, f, 7.8, color="#E1BEE7")

# Normalization box
box(ax, 10.2, 16.30, 7.0, 1.40, "#6A1B9A", "#BA68C8", 1.5)
label(ax, 13.7, 17.40, "Disease Normaliser  (70+ regex patterns)", 10, bold=True, color="#F3E5F5")
norm_examples = [
    '"diabetic on oral treatment"  →  "diabetes type 2"',
    '"loose motion / loose stools"  →  "diarrhea"',
    '"hypertension stage 2"  →  "hypertension"',
    '"dengue"  →  "dengue fever"   |   "anaemia"  →  "anaemia"',
]
for i, n in enumerate(norm_examples):
    label(ax, 13.7, 17.10 - i*0.20, n, 7.5, color="#E1BEE7")

# Output note
box(ax, 17.9, 16.30, 5.5, 1.40, "#4A148C", "#CE93D8", 1.5)
label(ax, 20.65, 17.40, "Output: ExtractedEntities", 9.5, bold=True, color="#F3E5F5")
out_fields = [
    "disease_normalized  •  icd10_code",
    "disease_category  •  sub_category",
    "confidence  •  classification_source",
    "review_required  •  processing_time_ms",
]
for i, f in enumerate(out_fields):
    label(ax, 20.65, 17.10 - i*0.20, f, 7.8, color="#E1BEE7")

divider(ax, 16.20)

# ══════════════════════════════════════════════════════════════════════════════
# 3. THREE-STAGE CLASSIFICATION  (y ≈ 10.4 – 16.10)
# ══════════════════════════════════════════════════════════════════════════════
# Stage header band
box(ax, 2.5, 15.60, 20.9, 0.50, "#BF360C", "#FFCCBC", 0.8, 0.10)
label(ax, 12.95, 15.85,
      "THREE-STAGE CLASSIFICATION PIPELINE  (classifier.py)  —  avg 4.1 ms/record",
      10.5, bold=True, color="#FFF3E0")

# ── Stage 1 ───────────────────────────────────────────────────────────────────
box(ax, 2.5, 12.50, 6.2, 3.00, C["stage1"], "#A5D6A7", 1.5)
label(ax, 5.6, 15.22, "STAGE 1", 8.5, bold=True, color="#C8E6C9")
label(ax, 5.6, 14.95, "Lookup Table", 11, bold=True, color="#FFFFFF")
label(ax, 5.6, 14.65, "(icd10_mapper.py)", 8, color="#A5D6A7")
label(ax, 5.6, 14.38, "145 canonical disease → ICD-10 entries", 8.2, color="#E8F5E9")
label(ax, 5.6, 14.15, "Hash map  |  O(1)  |  0.2 ms per record", 8.2, color="#E8F5E9")
label(ax, 5.6, 13.90, "Alias dict for spelling variants", 8.2, color="#E8F5E9")
label(ax, 5.6, 13.62, "87.9% of records — 100% accuracy", 9, bold=True, color="#FFEB3B")
label(ax, 5.6, 13.35, "Never touches any ML model", 8.0, color="#C8E6C9")
label(ax, 5.6, 13.10, "\"diabetes type 2\" → E11.9 instantly", 8.0, color="#C8E6C9", bold=True)
label(ax, 5.6, 12.82, "[fast] Returns immediately on match", 8.5, color="#69F0AE", bold=True)

# Stage 1 badge
box(ax, 3.2, 15.32, 1.80, 0.38, "#2E7D32", "#FFFFFF", 1)
label(ax, 4.10, 15.51, "[MATCH] EXACT MATCH", 7.5, bold=True, color="#F1F8E9")

# ── Stage 2 ───────────────────────────────────────────────────────────────────
box(ax, 9.4, 12.50, 7.0, 3.00, C["stage2"], "#FFCC80", 1.5)
label(ax, 12.9, 15.22, "STAGE 2", 8.5, bold=True, color="#FFE0B2")
label(ax, 12.9, 14.95, "ClinicalBERT + FAISS", 11, bold=True, color="#FFFFFF")
label(ax, 12.9, 14.65, "(emilyalsentzer/Bio_ClinicalBERT)", 7.8, color="#FFE0B2")
label(ax, 12.9, 14.38, "768-dim embeddings  |  FAISS IndexFlatIP", 8.2, color="#FFF3E0")
label(ax, 12.9, 14.15, "Score ALL 47 ICD-10 anchors (not top-k)", 8.2, color="#FFF3E0")
label(ax, 12.9, 13.90, "_KW_SUBCAT_BOOST: 24 keyword→subcat pairs", 8.2, color="#FFF3E0")
label(ax, 12.9, 13.65, "_CODE_BOOST: 65+ phrase→ICD pairs (0.055–0.120)", 8.2, color="#FFF3E0")
label(ax, 12.9, 13.40, "Context re-rank: age + temp + onset signals", 8.2, color="#FFF3E0")
label(ax, 12.9, 13.15, "7.5% of records — 92.5% accuracy", 9, bold=True, color="#FFEB3B")
label(ax, 12.9, 12.90, "Confidence ≥ 0.45 → Return result", 8.0, color="#FFE0B2")
label(ax, 12.9, 12.66, "[core] ~35ms per record (local CPU)", 8.5, color="#FFD54F", bold=True)

# Stage 2 badge
box(ax, 9.85, 15.32, 1.80, 0.38, "#E65100", "#FFFFFF", 1)
label(ax, 10.75, 15.51, "[EMB] EMBEDDING", 7.5, bold=True, color="#FFF8E1")

# ── Stage 3 ───────────────────────────────────────────────────────────────────
box(ax, 17.1, 12.50, 6.3, 3.00, C["stage3"], "#F48FB1", 1.5)
label(ax, 20.25, 15.22, "STAGE 3", 8.5, bold=True, color="#FCE4EC")
label(ax, 20.25, 14.95, "GPT-4o-mini Fallback", 11, bold=True, color="#FFFFFF")
label(ax, 20.25, 14.65, "(OpenAI API)", 7.8, color="#F8BBD9")
label(ax, 20.25, 14.38, "Fires only when confidence < 0.45", 8.2, color="#FCE4EC")
label(ax, 20.25, 14.15, "Full clinical context in prompt", 8.2, color="#FCE4EC")
label(ax, 20.25, 13.90, "Structured JSON output schema", 8.2, color="#FCE4EC")
label(ax, 20.25, 13.65, "Returns: ICD + SNOMED + reasoning", 8.2, color="#FCE4EC")
label(ax, 20.25, 13.40, "0.5% of records — ~87% accuracy", 9, bold=True, color="#FFEB3B")
label(ax, 20.25, 13.15, "$0.00015/1K tokens  (negligible cost)", 8.0, color="#F8BBD9")
label(ax, 20.25, 12.90, "Low conf → review_required = true", 8.0, color="#F8BBD9")
label(ax, 20.25, 12.66, "[LLM] ~800ms (only 55/10,045 records)", 8.5, color="#FF80AB", bold=True)

# Stage 3 badge
box(ax, 17.55, 15.32, 1.80, 0.38, "#880E4F", "#FFFFFF", 1)
label(ax, 18.45, 15.51, "[LLM]  LLM FALLBACK", 7.5, bold=True, color="#FCE4EC")

# Accuracy summary band
box(ax, 2.5, 10.50, 20.9, 1.85, "#1A237E", "#3F51B5", 1.5)
label(ax, 12.95, 12.16, "EFFECTIVE PRODUCTION ACCURACY", 10, bold=True, color="#C5CAE9")
label(ax, 12.95, 11.86,
      "Lookup 87.9% × 100%  +  Embedding 7.5% × 92.5%  +  Partial 4.1% × 95%  +  LLM 0.5% × 87%",
      9.5, color="#9FA8DA")
label(ax, 12.95, 11.58,
      "=  87.9%  +  6.9%  +  3.9%  +  0.4%   =   99.1%  Overall Pipeline Accuracy",
      11, bold=True, color="#FFEB3B")
label(ax, 12.95, 11.05,
      "Pipeline: 100% (ClinicalBERT benchmark)    |    Embedding-only: 92.5%    |    BioBERT: 77.5%    |    PubMedBERT: 68.8%",
      8.5, color="#7986CB")

divider(ax, 10.45)

# ══════════════════════════════════════════════════════════════════════════════
# 4. STORAGE — MongoDB Atlas  (y ≈ 9.0 – 10.35)
# ══════════════════════════════════════════════════════════════════════════════
# classifications collection
box(ax, 2.5, 9.10, 7.0, 1.20, C["mongo"], "#A5D6A7", 1.5)
label(ax, 6.0, 10.05, "[DB]  zeacares.classifications", 10, bold=True, color="#E8F5E9")
label(ax, 6.0, 9.78, "record_id • disease_raw • disease_normalized", 7.8, color="#DCEDC8")
label(ax, 6.0, 9.58, "icd10_code • snomed_code • disease_category", 7.8, color="#DCEDC8")
label(ax, 6.0, 9.38, "confidence • classification_source • review_required", 7.8, color="#DCEDC8")
label(ax, 6.0, 9.18, "vitals • facility • district • created_at • batch_ts", 7.8, color="#DCEDC8")

# alerts collection
box(ax, 10.2, 9.10, 5.5, 1.20, C["mongo"], "#A5D6A7", 1.5)
label(ax, 12.95, 10.05, "[ALERT]  zeacares.alerts", 10, bold=True, color="#E8F5E9")
label(ax, 12.95, 9.78, "district • disease_category • alert_type", 7.8, color="#DCEDC8")
label(ax, 12.95, 9.58, "alert_severity • current_cases_7d", 7.8, color="#DCEDC8")
label(ax, 12.95, 9.38, "cusum_score • prophet_anomaly", 7.8, color="#DCEDC8")
label(ax, 12.95, 9.18, "percent_above_baseline • triggered_at", 7.8, color="#DCEDC8")

# feedback collection
box(ax, 16.4, 9.10, 6.5, 1.20, C["mongo"], "#A5D6A7", 1.5)
label(ax, 19.65, 10.05, "[LOG]  zeacares.feedback", 10, bold=True, color="#E8F5E9")
label(ax, 19.65, 9.78, "record_id • original_icd10_code", 7.8, color="#DCEDC8")
label(ax, 19.65, 9.58, "corrected_icd10_code • corrected_category", 7.8, color="#DCEDC8")
label(ax, 19.65, 9.38, "officer_id • notes • submitted_at", 7.8, color="#DCEDC8")
label(ax, 19.65, 9.18, "Threshold: 50 corrections → fine-tune trigger", 7.8, color="#A5D6A7")

# MongoDB header label
box(ax, 2.5, 10.32, 20.4, 0.28, "#0A7B3C", "#FFFFFF", 1, 0.05)
label(ax, 12.7, 10.46,
      "MongoDB Atlas  |  db: zeacares  |  Indexes: district+category, record_id, created_at",
      8.5, bold=True, color="#C8E6C9")

divider(ax, 9.05)

# ══════════════════════════════════════════════════════════════════════════════
# 5. SURVEILLANCE  (y ≈ 7.4 – 8.95)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 2.5, 7.45, 6.2, 1.45, "#880E4F", "#F48FB1", 1.5)
label(ax, 5.6, 8.66, "CUSUM", 10, bold=True, color="#FCE4EC")
label(ax, 5.6, 8.40, "Cumulative Sum Control Chart", 8.2, color="#F8BBD9")
label(ax, 5.6, 8.18, "Detects sustained upward trend", 8.2, color="#F8BBD9")
label(ax, 5.6, 7.96, "Alert threshold: CUSUM score > 5", 8.2, color="#F8BBD9")
label(ax, 5.6, 7.72, "WHO/CDC gold standard method", 8.0, color="#F48FB1")

box(ax, 9.4, 7.45, 7.0, 1.45, "#880E4F", "#F48FB1", 1.5)
label(ax, 12.9, 8.66, "Facebook Prophet", 10, bold=True, color="#FCE4EC")
label(ax, 12.9, 8.40, "Time-series anomaly detection", 8.2, color="#F8BBD9")
label(ax, 12.9, 8.18, "Forecasts expected case counts", 8.2, color="#F8BBD9")
label(ax, 12.9, 7.96, "Anomaly: actual > yhat_upper bound", 8.2, color="#F8BBD9")
label(ax, 12.9, 7.72, "Handles seasonality + trends", 8.0, color="#F48FB1")

box(ax, 17.1, 7.45, 6.3, 1.45, "#C62828", "#FFCDD2", 1.5)
label(ax, 20.25, 8.66, "Alert Severity Levels", 10, bold=True, color="#FFEBEE")
label(ax, 20.25, 8.40, "CRITICAL: CUSUM>10 + Prophet + >100% baseline", 7.8, color="#EF9A9A")
label(ax, 20.25, 8.18, "HIGH:     CUSUM>7 + Prophet anomaly", 7.8, color="#EF9A9A")
label(ax, 20.25, 7.96, "MEDIUM:   CUSUM>5 OR Prophet anomaly", 7.8, color="#FFCDD2")
label(ax, 20.25, 7.72, "LOW:      Minor elevation, single method", 7.8, color="#FFCDD2")

# Surveillance trigger label
box(ax, 2.5, 8.88, 20.9, 0.24, "#6A0032", "#FFFFFF", 1, 0.05)
label(ax, 12.95, 9.00,
      "POST /alerts/refresh  →  reads classified records from MongoDB  →  runs CUSUM + Prophet  →  writes to zeacares.alerts",
      8.0, bold=True, color="#F8BBD9")

divider(ax, 7.38)

# ══════════════════════════════════════════════════════════════════════════════
# 6. API LAYER  (y ≈ 2.9 – 7.28)
# ══════════════════════════════════════════════════════════════════════════════
# FastAPI header
box(ax, 2.5, 6.85, 20.9, 0.38, "#004D40", "#80CBC4", 1.5)
label(ax, 12.95, 7.05,
      "FastAPI Backend  (src/api/main.py)  —  uvicorn  |  port 8000  |  CORS *  |  Background Tasks",
      9.5, bold=True, color="#E0F2F1")

# API endpoint boxes
endpoints = [
    (2.5,  5.65, 5.8,  "POST  /classify",          "#00695C",
     ["Single record classification",
      "Body: { clinical_text, district }",
      "Runs 3-stage pipeline",
      "Stores to classifications",
      "Returns full ClassificationResponse"]),
    (9.0,  5.65, 5.8,  "POST  /classify/batch",     "#00695C",
     ["CSV / XLSX upload",
      "Runs in BackgroundTask",
      "classify_batch() on all rows",
      "insert_many to MongoDB",
      "Returns BatchClassificationResponse"]),
    (15.5, 5.65, 5.0,  "GET   /trends/{district}",  "#006064",
     ["Query MongoDB by district",
      "Group by date → daily counts",
      "Run CUSUM + Prophet",
      "Returns TrendResponse",
      "Includes has_alert flag"]),
    (21.2, 5.65, 2.2,  "GET   /alerts/active",      "#880E4F",
     ["Read zeacares.alerts",
      "Sort by triggered_at",
      "Returns ActiveAlertsResponse",
      "(CRITICAL/HIGH/MEDIUM/LOW)"]),
    (2.5,  3.85, 3.5,  "POST  /alerts/refresh",     "#880E4F",
     ["Trigger CUSUM+Prophet",
      "Background task",
      "Writes to zeacares.alerts"]),
    (6.7,  3.85, 4.0,  "GET   /dashboard/summary",  "#0277BD",
     ["Statewide aggregation",
      "Top 10 diseases",
      "Category breakdown",
      "District coverage",
      "Returns DashboardSummary"]),
    (11.4, 3.85, 4.0,  "POST  /feedback",           "#37474F",
     ["Submit ICD correction",
      "Stored in zeacares.feedback",
      "Fine-tune trigger at 50+",
      "Returns accepted status"]),
    (16.1, 3.85, 3.5,  "GET   /feedback/stats",     "#37474F",
     ["Count corrections collected",
      "ready_for_finetuning flag",
      "Returns {total, threshold}"]),
    (20.3, 3.85, 3.1,  "GET   /health",             "#1B5E20",
     ["Ping MongoDB",
      "Check model loaded",
      "Returns HealthCheckResponse"]),
]

for ex, ey, ew, etitle, ecolor, elines in endpoints:
    box(ax, ex, ey, ew, 1.75, ecolor, "#B2DFDB", 1.2)
    label(ax, ex+ew/2, ey+1.60, etitle, 8.2, bold=True, color="#E0F7FA")
    for i, line in enumerate(elines):
        label(ax, ex+ew/2, ey+1.35-i*0.22, line, 7.2, color="#B2DFDB")

# Pydantic schemas note
box(ax, 2.5, 3.25, 20.9, 0.48, "#004D40", "#80CBC4", 1)
label(ax, 12.95, 3.50,
      "Pydantic Schemas (schemas.py):  ClassifyRequest  •  ClassificationResponse  •  BatchClassificationResponse",
      8.0, color="#B2DFDB")
label(ax, 12.95, 3.32,
      "TrendResponse  •  TrendDataPoint  •  ActiveAlertsResponse  •  AlertResponse  •  DashboardSummary  •  FeedbackRequest",
      8.0, color="#B2DFDB")

divider(ax, 3.18)

# ══════════════════════════════════════════════════════════════════════════════
# 7. CLIENTS  (y ≈ 2.0 – 3.08)
# ══════════════════════════════════════════════════════════════════════════════
clients = [
    (2.5,  "React Dashboard\n(Health Officers)",    "#37474F"),
    (7.8,  "Mobile App\n(Field Workers)",            "#37474F"),
    (13.1, "IDSP / ABDM\nNational Systems",         "#37474F"),
    (18.4, "Postman / API\nDirect Testing",          "#37474F"),
    (22.8, "Jupyter\nNotebook Demo",                 "#37474F"),
]
for cx, ctxt, cc in clients:
    box(ax, cx, 2.10, 3.8, 0.88, cc, "#90A4AE", 1.2)
    label(ax, cx+1.9, 2.54, ctxt, 8.5, color="#ECEFF1", bold=False)

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS (data flow)
# ══════════════════════════════════════════════════════════════════════════════
# Input → NER
arrow(ax, 6.5,  18.15, 6.5,  17.70, C["input"],  2.5, label_text="raw clinical text")
# Input single → NER
arrow(ax, 9.2,  18.15, 9.2,  17.70, C["input"],  2.0)
# NER → Stage 1
arrow(ax, 6.0,  16.30, 5.8,  15.50, C["ner"],    2.0, "arc3,rad=0.1", label_text="normalized disease")
# NER → Stage 2
arrow(ax, 9.5,  16.30, 11.5, 15.50, C["ner"],    2.0, "arc3,rad=0.0", label_text="if not in lookup")
# NER → Stage 3
arrow(ax, 11.5, 16.30, 19.0, 15.50, C["ner"],    1.8, "arc3,rad=0.0", label_text="confidence<0.45")
# Stage 1 → MongoDB
arrow(ax, 5.6,  12.50, 5.6,  10.30, C["stage1"], 2.5, label_text="ICD-10 + SNOMED CT")
# Stage 2 → MongoDB
arrow(ax, 12.9, 12.50, 10.5, 10.30, C["stage2"], 2.0, "arc3,rad=-0.1")
# Stage 3 → MongoDB
arrow(ax, 20.0, 12.50, 18.0, 10.30, C["stage3"], 2.0, "arc3,rad=0.1")
# MongoDB → Surveillance
arrow(ax, 10.0, 9.10,  10.0, 8.90,  C["mongo"],  2.0, label_text="/alerts/refresh")
# Surveillance → MongoDB alerts
arrow(ax, 13.5, 7.45,  13.5, 9.10,  C["surveillance"], 2.0, "arc3,rad=0.0",
      label_text="write alerts")
# MongoDB → API
arrow(ax, 8.0,  9.10,  8.0,  7.23,  C["mongo"],  2.5, label_text="read collections")
# API → Clients
arrow(ax, 12.95, 3.25, 12.95, 2.98, C["api"],    2.5, label_text="JSON responses")

# ══════════════════════════════════════════════════════════════════════════════
# TECH STACK LEGEND  (bottom-right corner)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 23.4, 0.15, 4.35, 3.68, "#263238", "#546E7A", 1.0)
label(ax, 25.6, 3.60, "TECH STACK", 9, bold=True, color="#80CBC4")
stack_items = [
    ("ClinicalBERT", "emilyalsentzer/Bio_ClinicalBERT"),
    ("Vector Search", "FAISS IndexFlatIP (cosine sim)"),
    ("LLM Fallback",  "GPT-4o-mini (OpenAI)"),
    ("API Framework", "FastAPI + uvicorn"),
    ("Database",      "MongoDB Atlas (pymongo)"),
    ("Trends",        "CUSUM + Facebook Prophet"),
    ("Privacy",       "Microsoft Presidio (DPDP)"),
    ("Language",      "Python 3.11"),
    ("Inference",     "CPU (no GPU required)"),
    ("Data",          "pandas + numpy + FAISS"),
    ("Benchmark",     "80 GT records → 99.1% acc"),
    ("Records",       "10,045 AP PHC (13 districts)"),
]
for i, (k, v) in enumerate(stack_items):
    y_ = 3.35 - i*0.27
    ax.text(23.7, y_, f"▸ {k}:", fontsize=7.0, color="#80CBC4",
            fontweight="bold", va="center")
    ax.text(26.1, y_, v, fontsize=6.8, color="#B2DFDB", va="center")

# ══════════════════════════════════════════════════════════════════════════════
# DATA FLOW LEGEND  (bottom-left)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 2.5, 0.15, 9.5, 1.68, "#263238", "#546E7A", 1.0)
label(ax, 7.25, 1.62, "DATA FLOW SUMMARY", 9, bold=True, color="#80CBC4")
flow_steps = [
    "1. PHC Doctor submits CSV or single API call",
    "2. NER extracts 19 structured fields from clinical text",
    "3. 70+ regex patterns normalize disease name",
    "4. Stage 1: hash-map lookup (87.9% resolved instantly)",
    "5. Stage 2: ClinicalBERT embedding + boost scoring (7.5%)",
    "6. Stage 3: GPT-4o-mini structured JSON (0.5%)",
    "7. All results stored to MongoDB zeacares.classifications",
    "8. /alerts/refresh: CUSUM+Prophet → zeacares.alerts",
    "9. API serves trends, alerts, dashboard to health officers",
]
for i, step in enumerate(flow_steps):
    ax.text(2.7, 1.42 - i*0.145, step, fontsize=7.0, color="#B2DFDB", va="center")

# ══════════════════════════════════════════════════════════════════════════════
# ACCURACY LEGEND  (bottom-center)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 12.5, 0.15, 10.5, 1.68, "#1A237E", "#3F51B5", 1.0)
label(ax, 17.75, 1.62, "ACCURACY & PERFORMANCE", 9, bold=True, color="#9FA8DA")
acc_items = [
    ("Pipeline accuracy",         "~99.1%  (weighted across all stages)"),
    ("Stage 1 — Lookup",          "87.9% of records  |  100.0% accuracy"),
    ("Stage 2 — ClinicalBERT",    "7.5% of records   |  92.5% accuracy"),
    ("Stage 3 — GPT-4o-mini",     "0.5% of records   |  ~87% accuracy"),
    ("ClinicalBERT benchmark",    "92.5%  (vs BioBERT 77.5%  /  PubMed 68.8%)"),
    ("Avg processing time",       "~4.1ms per record (batch of 10,045 = 41s)"),
    ("LLM fallback rate",         "55 out of 10,045 records used LLM"),
    ("FAISS anchor count",        "47 ICD-10 anchors scored per query"),
    ("_CODE_BOOST entries",       "65+ phrase→ICD-code pairs"),
]
for i, (k, v) in enumerate(acc_items):
    y_ = 1.42 - i*0.145
    ax.text(12.7, y_, f"▸ {k}:", fontsize=7.0, color="#9FA8DA",
            fontweight="bold", va="center")
    ax.text(16.4, y_, v, fontsize=6.8, color="#C5CAE9", va="center")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 0.2, 0.02, 27.6, 0.14, "#1A2A4A", "#FFFFFF", 0.5, 0.05)
ax.text(14, 0.09,
        "ZeaCares AI Disease Surveillance Platform  •  Andhra Pradesh IDSP  •  May 2026  •  ClinicalBERT + FAISS + GPT-4o-mini + MongoDB Atlas",
        fontsize=6.5, color="#90CAF9", ha="center", va="center")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0)
out = "/Users/shubham/Downloads/NLP/docs/ZeaCares_Architecture_Diagram.png"
plt.savefig(out, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print(f"Diagram saved → {out}")
