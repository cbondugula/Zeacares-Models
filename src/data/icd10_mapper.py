"""
ICD-10 / SNOMED CT Mapper
Maps raw disease strings from ZeaCares data to standard codes.
Two approaches: direct lookup table + embedding similarity fallback.
"""
import json
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ICD10Result:
    icd10_code: str
    icd10_description: str
    snomed_code: Optional[str]
    disease_category: str
    sub_category: str
    match_method: str   # "lookup" | "embedding" | "llm" | "unspecified"
    confidence: float


# ─────────────────────────────────────────────────────────────────────────────
# Direct Lookup Table (covers top ~95% of ZeaCares conditions by frequency)
# Format: "raw string" → (ICD-10 code, description, SNOMED, category, sub-cat)
# ─────────────────────────────────────────────────────────────────────────────
LOOKUP_TABLE: dict[str, tuple] = {
    # ── Communicable — Respiratory ────────────────────────────────────────────
    "cough":                           ("R05.9",  "Cough, unspecified",                    "49727002",  "Communicable", "Respiratory"),
    "dry cough":                       ("R05.9",  "Cough, unspecified",                    "49727002",  "Communicable", "Respiratory"),
    "productive cough":                ("R05.9",  "Cough, unspecified",                    "49727002",  "Communicable", "Respiratory"),
    "fever":                           ("R50.9",  "Fever, unspecified",                    "386661006", "Symptom NOS",  "Fever"),
    "high fever":                      ("R50.9",  "Fever, unspecified",                    "386661006", "Symptom NOS",  "Fever"),
    "mild fever":                      ("R50.9",  "Fever, unspecified",                    "386661006", "Symptom NOS",  "Fever"),
    "viral fever":                     ("B34.9",  "Viral infection, unspecified",          "34014006",  "Communicable", "Viral"),
    "feverish cold":                   ("J06.9",  "Acute upper respiratory infection",     "54150009",  "Communicable", "Respiratory"),
    "upper respiratory infection":     ("J06.9",  "Acute upper respiratory infection",     "54150009",  "Communicable", "Respiratory"),
    "cold":                            ("J06.9",  "Acute upper respiratory infection",     "54150009",  "Communicable", "Respiratory"),
    "common cold":                     ("J06.9",  "Acute upper respiratory infection",     "54150009",  "Communicable", "Respiratory"),
    "running nose":                    ("J06.9",  "Acute upper respiratory infection",     "54150009",  "Communicable", "Respiratory"),
    "nasal discharge":                 ("J06.9",  "Acute upper respiratory infection",     "54150009",  "Communicable", "Respiratory"),
    "nasal congestion":                ("J06.9",  "Acute upper respiratory infection",     "54150009",  "Communicable", "Respiratory"),
    "throat pain":                     ("J02.9",  "Acute pharyngitis, unspecified",        "162397003", "Communicable", "Respiratory"),
    "sore throat":                     ("J02.9",  "Acute pharyngitis, unspecified",        "405737000", "Communicable", "Respiratory"),
    "throat infection":                ("J02.9",  "Acute pharyngitis, unspecified",        "405737000", "Communicable", "Respiratory"),
    "cold skin":                       ("R23.8",  "Other skin changes",                    "271807003", "Symptom NOS",  "Dermatological"),
    "tuberculosis":                    ("A15.9",  "Respiratory tuberculosis, unspecified", "56717001",  "Communicable", "Respiratory"),
    "tb":                              ("A15.9",  "Respiratory tuberculosis, unspecified", "56717001",  "Communicable", "Respiratory"),
    "pneumonia":                       ("J18.9",  "Pneumonia, unspecified organism",       "233604007", "Communicable", "Respiratory"),
    "bronchitis":                      ("J40",    "Bronchitis, not specified as acute or chronic", "32398004", "Communicable", "Respiratory"),

    # ── Communicable — Vector-borne ───────────────────────────────────────────
    "dengue fever":                    ("A90",    "Dengue fever",                          "38362002",  "Communicable", "Vector-borne"),
    "dengue":                          ("A90",    "Dengue fever",                          "38362002",  "Communicable", "Vector-borne"),
    "malaria":                         ("B54",    "Malaria, unspecified",                  "61462000",  "Communicable", "Vector-borne"),
    "malarial fever":                  ("B54",    "Malaria, unspecified",                  "61462000",  "Communicable", "Vector-borne"),

    # ── Communicable — Enteric ────────────────────────────────────────────────
    "typhoid":                         ("A01.00", "Typhoid fever, unspecified",            "4834000",   "Communicable", "Enteric"),
    "typhoid fever":                   ("A01.00", "Typhoid fever, unspecified",            "4834000",   "Communicable", "Enteric"),
    "enteric fever":                   ("A01.00", "Typhoid fever, unspecified",            "4834000",   "Communicable", "Enteric"),

    # ── Communicable — GI / Diarrheal ─────────────────────────────────────────
    "loose motion":                    ("A09",    "Infectious gastroenteritis and colitis", "409587002", "Communicable", "Diarrheal"),
    "diarrhea":                        ("A09",    "Infectious gastroenteritis and colitis", "409587002", "Communicable", "Diarrheal"),
    "gastroenteritis":                 ("A09",    "Infectious gastroenteritis and colitis", "409587002", "Communicable", "Diarrheal"),
    "vomiting food":                   ("R11.10", "Vomiting, unspecified",                 "422400008", "Symptom NOS",  "GI"),
    "vomiting":                        ("R11.10", "Vomiting, unspecified",                 "422400008", "Symptom NOS",  "GI"),
    "nausea":                          ("R11.10", "Vomiting, unspecified",                 "422400008", "Symptom NOS",  "GI"),
    "nausea vomiting":                 ("R11.10", "Vomiting, unspecified",                 "422400008", "Symptom NOS",  "GI"),

    # ── Communicable — Zoonotic ───────────────────────────────────────────────
    "dog bite":                        ("W54.0",  "Bitten by dog",                         "418975000", "Communicable", "Zoonotic"),
    "animal bite":                     ("W54.0",  "Bitten by dog",                         "418975000", "Communicable", "Zoonotic"),
    "snake bite":                      ("T63.04", "Toxic effect of venom, accidental",     "260413007", "Injury",       "Envenomation"),
    "insect bite":                     ("T14.0",  "Open wound, unspecified body region",   "276018003", "Injury",       "Trauma"),
    "injury risk":                     ("T14.9",  "Injury, unspecified",                   "416940007", "Injury",       "External"),

    # ── Communicable — Viral (exanthems) ─────────────────────────────────────
    "chickenpox":                      ("B01.9",  "Varicella without complication",        "38907003",  "Communicable", "Viral"),
    "chicken pox":                     ("B01.9",  "Varicella without complication",        "38907003",  "Communicable", "Viral"),
    "varicella":                       ("B01.9",  "Varicella without complication",        "38907003",  "Communicable", "Viral"),
    "measles":                         ("B05.9",  "Measles without complication",          "14189004",  "Communicable", "Viral"),
    "mumps":                           ("B26.9",  "Mumps without complication",            "240526004", "Communicable", "Viral"),

    # ── Communicable — Hepatic ────────────────────────────────────────────────
    "hepatitis":                       ("B19.9",  "Unspecified viral hepatitis",           "40468003",  "Communicable", "Hepatic"),
    "viral hepatitis":                 ("B19.9",  "Unspecified viral hepatitis",           "40468003",  "Communicable", "Hepatic"),
    "jaundice":                        ("R17",    "Unspecified jaundice",                  "18165001",  "Symptom NOS",  "Hepatic"),

    # ── Communicable — Ocular ─────────────────────────────────────────────────
    "red eye":                         ("H10.9",  "Unspecified conjunctivitis",            "9968009",   "Communicable", "Ocular"),
    "conjunctivitis":                  ("H10.9",  "Unspecified conjunctivitis",            "9968009",   "Communicable", "Ocular"),
    "eye infection":                   ("H10.9",  "Unspecified conjunctivitis",            "9968009",   "Communicable", "Ocular"),
    "eye redness":                     ("H10.9",  "Unspecified conjunctivitis",            "9968009",   "Communicable", "Ocular"),

    # ── Communicable — Urological ─────────────────────────────────────────────
    "urinary tract infection":         ("N39.0",  "Urinary tract infection, site not specified", "68566005", "Communicable", "Urological"),
    "uti":                             ("N39.0",  "Urinary tract infection, site not specified", "68566005", "Communicable", "Urological"),
    "burning urination":               ("N39.0",  "Urinary tract infection, site not specified", "68566005", "Communicable", "Urological"),
    "burning micturition":             ("N39.0",  "Urinary tract infection, site not specified", "68566005", "Communicable", "Urological"),
    "dysuria":                         ("N39.0",  "Urinary tract infection, site not specified", "68566005", "Communicable", "Urological"),

    # ── Communicable — Dermatological ────────────────────────────────────────
    "skin infection":                  ("L08.9",  "Local infection of skin, unspecified",  "72621006",  "Communicable", "Dermatological"),
    "wound infection":                 ("L08.9",  "Local infection of skin, unspecified",  "72621006",  "Communicable", "Dermatological"),
    "infected wound":                  ("L08.9",  "Local infection of skin, unspecified",  "72621006",  "Communicable", "Dermatological"),

    # ── Non-Communicable — Cardiovascular ────────────────────────────────────
    "hypertension stage 1":            ("I10",    "Essential (primary) hypertension",      "59621000",  "Non-Communicable", "Cardiovascular"),
    "hypertension stage 2":            ("I10",    "Essential (primary) hypertension",      "59621000",  "Non-Communicable", "Cardiovascular"),
    "stage 2 hypertension":            ("I10",    "Essential (primary) hypertension",      "59621000",  "Non-Communicable", "Cardiovascular"),
    "hypertension monitored":          ("I10",    "Essential (primary) hypertension",      "59621000",  "Non-Communicable", "Cardiovascular"),
    "fear of hypertension":            ("I10",    "Essential (primary) hypertension",      "59621000",  "Non-Communicable", "Cardiovascular"),
    "hypertension":                    ("I10",    "Essential (primary) hypertension",      "59621000",  "Non-Communicable", "Cardiovascular"),
    "chest pain":                      ("R07.9",  "Chest pain, unspecified",               "29857009",  "Symptom NOS",  "Cardiovascular"),
    "palpitations":                    ("R00.2",  "Palpitations",                          "80313002",  "Symptom NOS",  "Cardiovascular"),
    "angina":                          ("I20.9",  "Angina pectoris, unspecified",          "194828000", "Non-Communicable", "Cardiovascular"),
    "heart attack":                    ("I21.9",  "Acute myocardial infarction, unspecified","22298006", "Non-Communicable", "Cardiovascular"),

    # ── Non-Communicable — Metabolic / Endocrine ─────────────────────────────
    "diabetes monitored":              ("E11.9",  "Type 2 diabetes mellitus without compl.", "44054006", "Non-Communicable", "Metabolic"),
    "diabetic diet":                   ("E11.9",  "Type 2 diabetes mellitus without compl.", "44054006", "Non-Communicable", "Metabolic"),
    "diabetic on oral treatment":      ("E11.9",  "Type 2 diabetes mellitus without compl.", "44054006", "Non-Communicable", "Metabolic"),
    "diabetic urine":                  ("E11.65", "Type 2 diabetes mellitus with hyperglycemia", "44054006", "Non-Communicable", "Metabolic"),
    "diabetic - good control":         ("E11.9",  "Type 2 diabetes mellitus without compl.", "44054006", "Non-Communicable", "Metabolic"),
    "diabetes type 2":                 ("E11.9",  "Type 2 diabetes mellitus without compl.", "44054006", "Non-Communicable", "Metabolic"),
    "thyroid":                         ("E04.9",  "Nontoxic goiter, unspecified",           "14304000", "Non-Communicable", "Endocrine"),
    "hypothyroidism":                  ("E03.9",  "Hypothyroidism, unspecified",            "40930008", "Non-Communicable", "Endocrine"),
    "hyperthyroidism":                 ("E05.90", "Thyrotoxicosis, unspecified",            "34486009", "Non-Communicable", "Endocrine"),

    # ── Non-Communicable — Hematological ─────────────────────────────────────
    "anemia":                          ("D64.9",  "Anemia, unspecified",                   "271737000", "Non-Communicable", "Hematological"),
    "anaemia":                         ("D64.9",  "Anemia, unspecified",                   "271737000", "Non-Communicable", "Hematological"),
    "low hemoglobin":                  ("D64.9",  "Anemia, unspecified",                   "271737000", "Non-Communicable", "Hematological"),

    # ── Non-Communicable — Respiratory ───────────────────────────────────────
    "asthma":                          ("J45.909","Asthma, unspecified",                   "195967001", "Non-Communicable", "Respiratory"),
    "bronchial asthma":                ("J45.909","Asthma, unspecified",                   "195967001", "Non-Communicable", "Respiratory"),
    "breathlessness":                  ("R06.09", "Other forms of dyspnea",                "230145002", "Symptom NOS",  "Respiratory"),
    "shortness of breath":             ("R06.09", "Other forms of dyspnea",                "230145002", "Symptom NOS",  "Respiratory"),
    "dyspnoea":                        ("R06.09", "Other forms of dyspnea",                "230145002", "Symptom NOS",  "Respiratory"),

    # ── Non-Communicable — Musculoskeletal ────────────────────────────────────
    "backache":                        ("M54.5",  "Low back pain",                         "279039007", "Non-Communicable", "Musculoskeletal"),
    "back pain":                       ("M54.5",  "Low back pain",                         "279039007", "Non-Communicable", "Musculoskeletal"),
    "back ache":                       ("M54.5",  "Low back pain",                         "279039007", "Non-Communicable", "Musculoskeletal"),
    "joint pain":                      ("M25.50", "Pain in unspecified joint",             "57676002",  "Non-Communicable", "Musculoskeletal"),
    "knee pain":                       ("M25.561","Pain in right knee",                    "57676002",  "Non-Communicable", "Musculoskeletal"),
    "hip pain":                        ("M25.551","Pain in right hip",                     "57676002",  "Non-Communicable", "Musculoskeletal"),
    "ankle pain":                      ("M25.571","Pain in right ankle",                   "57676002",  "Non-Communicable", "Musculoskeletal"),
    "leg pain":                        ("M79.671","Pain in right foot",                    "10601006",  "Non-Communicable", "Musculoskeletal"),
    "whole body pain":                 ("M79.3",  "Panniculitis, unspecified",             "57676002",  "Non-Communicable", "Musculoskeletal"),
    "generalized pain":                ("M79.3",  "Panniculitis, unspecified",             "57676002",  "Non-Communicable", "Musculoskeletal"),
    "body ache":                       ("M79.10", "Myalgia, unspecified site",             "68962001",  "Non-Communicable", "Musculoskeletal"),
    "myalgia":                         ("M79.10", "Myalgia, unspecified site",             "68962001",  "Non-Communicable", "Musculoskeletal"),
    "neck pain":                       ("M54.2",  "Cervicalgia",                           "81680005",  "Non-Communicable", "Musculoskeletal"),
    "cervical pain":                   ("M54.2",  "Cervicalgia",                           "81680005",  "Non-Communicable", "Musculoskeletal"),
    "shoulder pain":                   ("M75.10", "Rotator cuff syndrome, unspecified",    "45326000",  "Non-Communicable", "Musculoskeletal"),
    "hand pain":                       ("M79.641","Pain in right hand",                    "57676002",  "Non-Communicable", "Musculoskeletal"),
    "arm pain":                        ("M79.629","Pain in unspecified upper arm",         "57676002",  "Non-Communicable", "Musculoskeletal"),

    # ── Non-Communicable — Neurological / Psychiatric ─────────────────────────
    "seizure":                         ("G40.909","Epilepsy, unspecified, not intractable", "313307000", "Non-Communicable", "Neurological"),
    "epilepsy":                        ("G40.909","Epilepsy, unspecified, not intractable", "313307000", "Non-Communicable", "Neurological"),
    "fits":                            ("G40.909","Epilepsy, unspecified, not intractable", "313307000", "Non-Communicable", "Neurological"),
    "convulsions":                     ("G40.909","Epilepsy, unspecified, not intractable", "313307000", "Non-Communicable", "Neurological"),
    "depression":                      ("F32.9",  "Major depressive disorder, unspecified","35489007",  "Non-Communicable", "Psychiatric"),
    "anxiety":                         ("F41.9",  "Anxiety disorder, unspecified",         "197480006", "Non-Communicable", "Psychiatric"),
    "insomnia":                        ("G47.00", "Insomnia, unspecified",                 "193462001", "Non-Communicable", "Neurological"),

    # ── Non-Communicable — GI ─────────────────────────────────────────────────
    "gastric reflux":                  ("K21.0",  "GERD with esophagitis",                 "235595009", "Non-Communicable", "GI"),
    "normal gastric acidity":          ("K21.9",  "GERD without esophagitis",              "235595009", "Non-Communicable", "GI"),
    "gastroesophageal reflux":         ("K21.0",  "GERD with esophagitis",                 "235595009", "Non-Communicable", "GI"),
    "acidity":                         ("K21.9",  "GERD without esophagitis",              "235595009", "Non-Communicable", "GI"),
    "gastric":                         ("K21.9",  "GERD without esophagitis",              "235595009", "Non-Communicable", "GI"),
    "constipation":                    ("K59.00", "Constipation, unspecified",             "14760008",  "Non-Communicable", "GI"),

    # ── Non-Communicable — Obstetric ──────────────────────────────────────────
    "antenatal":                       ("Z34.90", "Supervision of normal pregnancy",       "72892002",  "Non-Communicable", "Obstetric"),
    "anc visit":                       ("Z34.90", "Supervision of normal pregnancy",       "72892002",  "Non-Communicable", "Obstetric"),
    "pregnancy":                       ("Z34.90", "Supervision of normal pregnancy",       "72892002",  "Non-Communicable", "Obstetric"),

    # ── Symptom NOS ───────────────────────────────────────────────────────────
    "general illness":                 ("R68.89", "Other specified general symptoms",      "416940007", "Symptom NOS", "General"),
    "general weakness":                ("R53.1",  "Weakness",                              "13791008",  "Symptom NOS",  "General"),
    "weakness":                        ("R53.1",  "Weakness",                              "13791008",  "Symptom NOS",  "General"),
    "headache":                        ("R51.9",  "Headache, unspecified",                 "25064002",  "Symptom NOS",  "Neurological"),
    "giddiness":                       ("R42",    "Dizziness and giddiness",               "404640003", "Symptom NOS",  "Neurological"),
    "vertigo":                         ("R42",    "Dizziness and giddiness",               "404640003", "Symptom NOS",  "Neurological"),
    "abdominal pain":                  ("R10.9",  "Unspecified abdominal pain",            "21522001",  "Symptom NOS",  "GI"),
    "stomach pain":                    ("R10.9",  "Unspecified abdominal pain",            "21522001",  "Symptom NOS",  "GI"),
    "gas pain":                        ("R14.0",  "Abdominal distension (gaseous)",        "102614006", "Symptom NOS",  "GI"),
    "gas trouble":                     ("R14.0",  "Abdominal distension (gaseous)",        "102614006", "Symptom NOS",  "GI"),
    "allergy":                         ("T78.40", "Allergy, unspecified",                  "408439002", "Non-Communicable", "Immunological"),
    "ear pain":                        ("H92.09", "Otalgia, unspecified ear",              "16001004",  "Symptom NOS",  "ENT"),
    "earache":                         ("H92.09", "Otalgia, unspecified ear",              "16001004",  "Symptom NOS",  "ENT"),
    "toothache":                       ("K08.89", "Other specified disorders of teeth",    "27355003",  "Symptom NOS",  "Dental"),
    "tooth pain":                      ("K08.89", "Other specified disorders of teeth",    "27355003",  "Symptom NOS",  "Dental"),
    "dental pain":                     ("K08.89", "Other specified disorders of teeth",    "27355003",  "Symptom NOS",  "Dental"),
    "skin rash":                       ("R23.8",  "Other skin changes",                    "271807003", "Symptom NOS",  "Dermatological"),
    "rash":                            ("R23.8",  "Other skin changes",                    "271807003", "Symptom NOS",  "Dermatological"),
    "itching":                         ("L29.9",  "Pruritus, unspecified",                 "418290006", "Symptom NOS",  "Dermatological"),
    "pruritus":                        ("L29.9",  "Pruritus, unspecified",                 "418290006", "Symptom NOS",  "Dermatological"),
    "wound pain":                      ("T14.9",  "Injury, unspecified",                   "416940007", "Injury",       "External"),
    "wound red":                       ("L08.9",  "Local infection of skin, unspecified",  "416940007", "Communicable", "Dermatological"),
    "pain":                            ("R52",    "Pain, unspecified",                     "22253000",  "Symptom NOS",  "General"),
    "painful":                         ("R52",    "Pain, unspecified",                     "22253000",  "Symptom NOS",  "General"),
}

# Normalize aliases that are minor variations
_ALIASES: dict[str, str] = {
    "htn stage 1":          "hypertension stage 1",
    "htn stage 2":          "hypertension stage 2",
    "htn-1":                "hypertension stage 1",
    "htn-2":                "hypertension stage 2",
    "dm monitored":         "diabetes monitored",
    "dm on oral treatment": "diabetic on oral treatment",
    "loose stools":         "diarrhea",
    "bp high":              "hypertension",
    "high bp":              "hypertension",
    "cat bite":             "dog bite",
    "rat bite":             "dog bite",
    "body aches":           "myalgia",
    "generalised pain":     "generalized pain",
    "cold and cough":       "upper respiratory infection",
    "cough and cold":       "upper respiratory infection",
    "fever and cough":      "fever",
}

UNSPECIFIED_RESULT = ICD10Result(
    icd10_code="R69",
    icd10_description="Illness, unspecified",
    snomed_code=None,
    disease_category="Symptom NOS",
    sub_category="General",
    match_method="unspecified",
    confidence=0.0,
)


class ICD10Mapper:
    def __init__(self):
        self._lookup = {k.lower().strip(): v for k, v in LOOKUP_TABLE.items()}
        self._aliases = {k.lower().strip(): v for k, v in _ALIASES.items()}
        # Lazy-load embedding index only when needed
        self._embedding_index = None
        logger.info(f"ICD10Mapper initialized with {len(self._lookup)} direct entries")

    def lookup(self, disease_raw: str) -> Optional[ICD10Result]:
        key = disease_raw.lower().strip()
        # Check aliases first
        key = self._aliases.get(key, key)
        entry = self._lookup.get(key)
        if entry:
            code, desc, snomed, cat, subcat = entry
            return ICD10Result(
                icd10_code=code,
                icd10_description=desc,
                snomed_code=snomed,
                disease_category=cat,
                sub_category=subcat,
                match_method="lookup",
                confidence=1.0,
            )
        return None

    def map(self, disease_raw: str, use_embeddings: bool = True) -> ICD10Result:
        if not disease_raw:
            return UNSPECIFIED_RESULT

        # 1. Direct lookup (fast path)
        result = self.lookup(disease_raw)
        if result:
            return result

        # 2. Partial match — check if any lookup key is contained in the input
        key = disease_raw.lower().strip()
        for lookup_key, entry in self._lookup.items():
            if lookup_key in key or key in lookup_key:
                code, desc, snomed, cat, subcat = entry
                return ICD10Result(
                    icd10_code=code,
                    icd10_description=desc,
                    snomed_code=snomed,
                    disease_category=cat,
                    sub_category=subcat,
                    match_method="partial_lookup",
                    confidence=0.75,
                )

        # 3. Embedding similarity (requires model to be loaded)
        if use_embeddings and self._embedding_index:
            return self._embedding_lookup(disease_raw)

        return UNSPECIFIED_RESULT

    def _embedding_lookup(self, disease_raw: str) -> ICD10Result:
        # Implemented in classifier.py after model loading
        return UNSPECIFIED_RESULT

    def bulk_map(self, diseases: list[str]) -> list[ICD10Result]:
        return [self.map(d) for d in diseases]

    def coverage_stats(self, diseases: list[str]) -> dict:
        results = [self.map(d, use_embeddings=False) for d in diseases]
        total = len(results)
        by_method: dict[str, int] = {}
        for r in results:
            by_method[r.match_method] = by_method.get(r.match_method, 0) + 1
        return {
            "total": total,
            "coverage_pct": (total - by_method.get("unspecified", 0)) / total * 100,
            "by_method": by_method,
        }


def save_lookup_table(path: str = "data/icd10_lookup.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(LOOKUP_TABLE, f, indent=2)
    logger.info(f"Saved lookup table to {path}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    mapper = ICD10Mapper()

    # Test a few cases
    test_cases = [
        "hypertension stage 2",
        "stage 2 hypertension",
        "fever",
        "dog bite",
        "diabetes monitored",
        "general illness",
        "mysterious new disease",
    ]

    print("=== ICD-10 Mapping Test ===\n")
    for case in test_cases:
        r = mapper.map(case)
        print(f"Input:    {case!r}")
        print(f"  Code:   {r.icd10_code} — {r.icd10_description}")
        print(f"  Cat:    {r.disease_category} / {r.sub_category}")
        print(f"  Method: {r.match_method} (confidence: {r.confidence:.2f})")
        print()

    # Coverage on actual data
    if len(sys.argv) > 1:
        import pandas as pd
        df = pd.read_excel(sys.argv[1])
        import re
        diseases = []
        for text in df["clinicalText"]:
            m = re.search(r"presented with (.+?)\. Onset", str(text), re.IGNORECASE)
            diseases.append(m.group(1).strip().lower() if m else "")

        stats = mapper.coverage_stats(diseases)
        print("=== Dataset Coverage ===")
        print(f"Total records:    {stats['total']}")
        print(f"Coverage:         {stats['coverage_pct']:.1f}%")
        print(f"By method:        {stats['by_method']}")
