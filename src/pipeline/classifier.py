"""
Main Classification Pipeline
Two-stage: PubMedBERT embedding similarity (fast) → Llama 3.3 70B (fallback)
Processes raw clinical text → ICD-10 code + disease category in <3 seconds.
"""
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# Ensure project root is on path when running as script
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD_HIGH = 0.72   # Auto-assign if above this (tuned for PubMedBERT + 45 anchors)
CONFIDENCE_THRESHOLD_LLM  = 0.45   # Flag for review if below this


@dataclass
class ClassificationOutput:
    record_id: str
    disease_raw: Optional[str]
    disease_normalized: Optional[str]
    icd10_code: str
    icd10_description: str
    snomed_code: Optional[str]
    disease_category: str
    sub_category: str
    confidence: float
    classification_source: str   # "lookup" | "embedding" | "llm" | "unspecified"
    review_required: bool
    processing_time_ms: float
    district: Optional[str]
    severity: Optional[str]
    age_band: Optional[str]
    gender: Optional[str]
    temperature_f: Optional[float] = None
    pulse_bpm: Optional[int] = None
    bp_systolic: Optional[int] = None
    bp_diastolic: Optional[int] = None
    spo2_pct: Optional[float] = None
    bmi_status: Optional[str] = None


class ZeaCaresClassifier:
    """
    End-to-end classifier for ZeaCares clinical records.

    Pipeline:
    1. NER extraction (regex + optional ClinicalBERT)
    2. ICD-10 lookup table (fast path, handles ~70% of records)
    3. ClinicalBERT embedding similarity (handles ~15% more)
    4. Llama 3.3 70B API (handles ambiguous ~15%)
    5. Flag for review if LLM confidence < threshold
    """

    def __init__(self, openrouter_api_key: Optional[str] = None,
                 device: Optional[str] = None,
                 model_cache_dir: str = "model_cache"):
        self.api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.device = device
        self.model_cache_dir = model_cache_dir
        self._ner = None
        self._icd_mapper = None
        self._embedding_model = None
        self._faiss_index = None
        self._icd_anchors = None
        self._initialized = False

    def _init(self) -> None:
        if self._initialized:
            return

        from src.pipeline.ner_extractor import NERExtractor
        from src.data.icd10_mapper import ICD10Mapper
        from src.models.pubmedbert_classifier import PubMedBERTClassifier
        from src.models.model_comparator import ICD10_ANCHORS

        self._ner = NERExtractor(use_model=False)
        self._icd_mapper = ICD10Mapper()
        # PubMedBERT wins raw embedding accuracy: 85.7% vs ClinicalBERT 66.2%
        self._embedding_model = PubMedBERTClassifier(device=self.device)
        self._icd_anchors = ICD10_ANCHORS
        self._build_faiss_index()
        self._initialized = True
        logger.info("ZeaCaresClassifier initialized with PubMedBERT embedding")

    def _build_faiss_index(self) -> None:
        try:
            import faiss
            anchor_texts = [desc for _, desc, _, _ in self._icd_anchors]
            logger.info(f"Building FAISS index over {len(anchor_texts)} ICD-10 anchors...")
            self._embedding_model.load()
            results = self._embedding_model.embed_batch(anchor_texts)
            vectors = np.array([r.embedding for r in results], dtype=np.float32)
            dim = vectors.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)  # Inner product = cosine on normalized vectors
            self._faiss_index.add(vectors)
            logger.info("FAISS index built")
        except ImportError:
            logger.warning("FAISS not available — using sklearn cosine_similarity fallback")
            anchor_texts = [desc for _, desc, _, _ in self._icd_anchors]
            self._embedding_model.load()
            results = self._embedding_model.embed_batch(anchor_texts)
            self._anchor_vectors = np.array([r.embedding for r in results], dtype=np.float32)

    def _embedding_search(self, disease_text: str, top_k: int = 3) -> tuple[str, str, str, str, float]:
        """
        Search ICD-10 anchors by embedding similarity.
        Returns top-k candidates then re-ranks using category consistency.
        Returns (code, desc, category, sub_category, score).
        """
        emb_result = self._embedding_model.embed(disease_text)
        query = emb_result.embedding.reshape(1, -1).astype(np.float32)

        if self._faiss_index is not None:
            import faiss
            scores, indices = self._faiss_index.search(query, top_k)
            top_scores = [float(s) for s in scores[0]]
            top_indices = [int(i) for i in indices[0]]
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(query, self._anchor_vectors)[0]
            top_indices = list(np.argsort(sims)[-top_k:][::-1])
            top_scores = [float(sims[i]) for i in top_indices]

        best_idx = top_indices[0]
        best_score = top_scores[0]

        # Re-ranking: if top-2 and top-3 score within 5% of top-1,
        # prefer the candidate whose ICD chapter is more common in our data.
        # Priority order for AP PHC data: NCD > Symptom NOS > Communicable > Injury
        chapter_priority = {"Non-Communicable": 3, "Symptom NOS": 2,
                            "Communicable": 1, "Injury": 0}
        if len(top_indices) > 1 and top_scores[0] > 0:
            gap = top_scores[0] - top_scores[1]
            if gap < 0.05:  # within 5% — re-rank by chapter priority
                candidates = list(zip(top_indices, top_scores))
                candidates.sort(
                    key=lambda x: (top_scores[0] - x[1] < 0.05,
                                   chapter_priority.get(self._icd_anchors[x[0]][2], 0)),
                    reverse=True,
                )
                best_idx, best_score = candidates[0]

        code, desc, cat, subcat = self._icd_anchors[best_idx]
        return code, desc, cat, subcat, best_score

    def _llm_classify(self, clinical_text: str, extracted_disease: str) -> dict:
        """Llama 3.3 70B fallback via OpenRouter for ambiguous cases."""
        if not self.api_key:
            logger.warning("No OpenRouter API key — skipping LLM classification")
            return {"icd10_code": "R69", "description": "Illness, unspecified",
                    "category": "Symptom NOS", "sub_category": "General", "confidence": 0.0}
        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)

            prompt = f"""You are a clinical coding expert. Classify the following patient presentation.

Patient record: {clinical_text[:500]}
Extracted disease/condition: {extracted_disease}

Return ONLY valid JSON with these fields:
{{
  "icd10_code": "<ICD-10-CM code>",
  "icd10_description": "<full ICD-10 description>",
  "snomed_code": "<SNOMED CT code or null>",
  "disease_category": "<Communicable|Non-Communicable|Symptom NOS|Injury|Emergency>",
  "sub_category": "<specific sub-type>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence explanation>"
}}"""

            response = client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            # Extract JSON from response
            import re
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")

        return {"icd10_code": "R69", "icd10_description": "Illness, unspecified",
                "snomed_code": None, "disease_category": "Symptom NOS",
                "sub_category": "General", "confidence": 0.0}

    def classify(self, clinical_text: str, district: str = "Unknown",
                 record_id: str = "0") -> ClassificationOutput:
        self._init()
        start = time.time()

        # Step 1: Extract entities
        entity = self._ner.extract(clinical_text, district)
        disease_raw = entity.disease_raw or ""
        disease_norm = entity.disease_normalized or disease_raw

        # Step 2: Direct lookup (fastest)
        lookup_result = self._icd_mapper.map(disease_norm, use_embeddings=False)
        if lookup_result.match_method in ("lookup", "partial_lookup"):
            elapsed_ms = (time.time() - start) * 1000
            return ClassificationOutput(
                record_id=record_id,
                disease_raw=disease_raw,
                disease_normalized=disease_norm,
                icd10_code=lookup_result.icd10_code,
                icd10_description=lookup_result.icd10_description,
                snomed_code=lookup_result.snomed_code,
                disease_category=lookup_result.disease_category,
                sub_category=lookup_result.sub_category,
                confidence=lookup_result.confidence,
                classification_source=lookup_result.match_method,
                review_required=False,
                processing_time_ms=elapsed_ms,
                district=entity.district,
                severity=entity.severity,
                age_band=entity.age_band,
                gender=entity.gender,
                temperature_f=entity.temp_f,
                pulse_bpm=entity.pulse,
                bp_systolic=entity.bp_sys,
                bp_diastolic=entity.bp_dia,
                spo2_pct=entity.spo2,
                bmi_status=entity.bmi_status,
            )

        # Step 3: Embedding similarity via ClinicalBERT + FAISS
        try:
            code, desc, cat, subcat, score = self._embedding_search(disease_norm or disease_raw)
            if score >= CONFIDENCE_THRESHOLD_HIGH:
                elapsed_ms = (time.time() - start) * 1000
                return ClassificationOutput(
                    record_id=record_id,
                    disease_raw=disease_raw,
                    disease_normalized=disease_norm,
                    icd10_code=code,
                    icd10_description=desc,
                    snomed_code=None,
                    disease_category=cat,
                    sub_category=subcat,
                    confidence=score,
                    classification_source="embedding",
                    review_required=False,
                    processing_time_ms=elapsed_ms,
                    district=entity.district,
                    severity=entity.severity,
                    age_band=entity.age_band,
                    gender=entity.gender,
                    temperature_f=entity.temp_f,
                    pulse_bpm=entity.pulse,
                    bp_systolic=entity.bp_sys,
                    bp_diastolic=entity.bp_dia,
                    spo2_pct=entity.spo2,
                    bmi_status=entity.bmi_status,
                )
        except Exception as e:
            logger.warning(f"Embedding search failed: {e}")
            score = 0.0

        # Step 4: LLM fallback
        llm_result = self._llm_classify(clinical_text, disease_raw)
        confidence = float(llm_result.get("confidence", 0.0))
        review_required = confidence < CONFIDENCE_THRESHOLD_LLM

        elapsed_ms = (time.time() - start) * 1000
        return ClassificationOutput(
            record_id=record_id,
            disease_raw=disease_raw,
            disease_normalized=disease_norm,
            icd10_code=llm_result.get("icd10_code", "R69"),
            icd10_description=llm_result.get("icd10_description", "Illness, unspecified"),
            snomed_code=llm_result.get("snomed_code"),
            disease_category=llm_result.get("disease_category", "Symptom NOS"),
            sub_category=llm_result.get("sub_category", "General"),
            confidence=confidence,
            classification_source="llm",
            review_required=review_required,
            processing_time_ms=elapsed_ms,
            district=entity.district,
            severity=entity.severity,
            age_band=entity.age_band,
            gender=entity.gender,
            temperature_f=entity.temp_f,
            pulse_bpm=entity.pulse,
            bp_systolic=entity.bp_sys,
            bp_diastolic=entity.bp_dia,
            spo2_pct=entity.spo2,
            bmi_status=entity.bmi_status,
        )

    def classify_batch(self, input_path: str, output_path: str = "results/classified.json",
                       max_records: Optional[int] = None) -> list[ClassificationOutput]:
        df = pd.read_excel(input_path) if input_path.endswith(".xlsx") else pd.read_csv(input_path)
        if max_records:
            df = df.head(max_records)

        logger.info(f"Classifying {len(df)} records...")
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                out = self.classify(
                    clinical_text=str(row["clinicalText"]),
                    district=str(row.get("district", "Unknown")),
                    record_id=str(i),
                )
                results.append(out)
            except Exception as e:
                logger.warning(f"Record {i} failed: {e}")

            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i+1}/{len(df)} records")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)

        self._print_summary(results)
        return results

    def _print_summary(self, results: list[ClassificationOutput]) -> None:
        total = len(results)
        by_source: dict[str, int] = {}
        categories: dict[str, int] = {}
        review_count = 0
        times = []

        for r in results:
            by_source[r.classification_source] = by_source.get(r.classification_source, 0) + 1
            categories[r.disease_category] = categories.get(r.disease_category, 0) + 1
            if r.review_required:
                review_count += 1
            times.append(r.processing_time_ms)

        print(f"\n=== Classification Summary ===")
        print(f"Total records:      {total}")
        print(f"By source:")
        for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
            print(f"  {src:<20} {count:>5} ({count/total*100:.1f}%)")
        print(f"By category:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"  {cat:<35} {count:>5} ({count/total*100:.1f}%)")
        print(f"Review required:    {review_count} ({review_count/total*100:.1f}%)")
        print(f"Avg time:           {sum(times)/len(times):.1f}ms/record")
        print(f"Results saved to results/classified.json")


def demo_mode() -> None:
    """Quick demo on 10 sample records."""
    classifier = ZeaCaresClassifier()
    samples = [
        ("Female 60 years, presented with hypertension stage 2. Onset was gradual with duration of 3 day(s). Severity: mild. Vitals: Temperature 98F, Pulse 72bpm, BP 110/80mmHg, SpO2 98%. BMI status: Healthy. Attended UPHC Lankapatnam.", "Vizianagaram"),
        ("Male 35 years, presented with fever. Onset was sudden with duration of 2 day(s). Severity: moderate. Vitals: Temperature 101F, Pulse 88bpm, BP 120/80mmHg, SpO2 97%. BMI status: Normal. Attended PHC Kurnool.", "Kurnool"),
        ("Female 51 years, presented with diabetic on oral treatment. Onset was gradual with duration of 3 day(s). Severity: mild. Vitals: Temperature 98F, Pulse 72bpm, BP 110/80mmHg, SpO2 98%. BMI status: Healthy. Attended UPHC Lankapatnam.", "Vizianagaram"),
        ("Male 45 years, presented with dog bite. Onset was sudden with duration of 1 day(s). Severity: moderate. Vitals: Temperature 98F, Pulse 80bpm, BP 125/82mmHg, SpO2 98%. BMI status: Normal. Attended CHC Guntur.", "Guntur"),
        ("Female 28 years, presented with general illness. Onset was gradual with duration of 2 day(s). Severity: mild. Vitals: Temperature 99F, Pulse 75bpm, BP 118/76mmHg, SpO2 98%. BMI status: Healthy. Attended PHC Nellore.", "Sri Potti Sriramulu Nellore"),
    ]

    print("=== ZeaCares Classification Demo ===\n")
    for i, (text, district) in enumerate(samples):
        result = classifier.classify(text, district, record_id=str(i))
        print(f"Record {i+1}:")
        print(f"  Disease (raw):    {result.disease_raw}")
        print(f"  ICD-10 Code:      {result.icd10_code}")
        print(f"  ICD-10 Desc:      {result.icd10_description}")
        print(f"  Category:         {result.disease_category} / {result.sub_category}")
        print(f"  Confidence:       {result.confidence:.2f}")
        print(f"  Source:           {result.classification_source}")
        print(f"  Review needed:    {result.review_required}")
        print(f"  Time:             {result.processing_time_ms:.1f}ms")
        print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", default="results/classified.json")
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    if args.demo or not args.input:
        demo_mode()
    else:
        classifier = ZeaCaresClassifier()
        classifier.classify_batch(args.input, args.output, args.max_records)
