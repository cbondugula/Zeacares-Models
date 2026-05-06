"""
NER (Named Entity Recognition) Extractor
Hybrid approach: Regex for structured fields + ClinicalBERT for disease normalization.
Extracts: disease, severity, onset, vitals, demographics from ZeaCares clinical text.
"""
import re
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntities:
    gender: Optional[str]
    age: Optional[int]
    age_band: Optional[str]
    disease_raw: Optional[str]
    disease_normalized: Optional[str]
    onset: Optional[str]
    duration_days: Optional[int]
    severity: Optional[str]
    temp_f: Optional[float]
    pulse: Optional[int]
    bp_sys: Optional[int]
    bp_dia: Optional[int]
    spo2: Optional[float]
    bmi_status: Optional[str]
    facility: Optional[str]
    district: Optional[str]


# Age-band conversion
def age_to_band(age: int) -> str:
    bands = [(0, 14), (15, 24), (25, 34), (35, 44), (45, 54),
             (55, 64), (65, 74), (75, 84), (85, 120)]
    for lo, hi in bands:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return "Unknown"


# Disease normalization — collapse variants to canonical form
_NORMALIZE_MAP = {
    # Hypertension variants
    r"hypertension\s*stage\s*[12i]+":                  "hypertension",
    r"stage\s*[12]\s*hypertension":                    "hypertension",
    r"htn[-\s]?[12]?":                                 "hypertension",
    r"high\s*bp":                                      "hypertension",
    r"bp\s+high":                                      "hypertension",
    r"fear\s+of\s+hypertension":                       "hypertension",
    # Diabetes variants
    r"diabetic\s+(on\s+oral\s+treatment|monitored|diet|urine|good\s+control)": "diabetes type 2",
    r"diabetes\s+monitored":                           "diabetes type 2",
    r"dm\s*(monitored|type\s*2)?":                     "diabetes type 2",
    r"type\s*2\s+diabetes":                            "diabetes type 2",
    # Respiratory
    r"viral\s+fever":                                  "viral fever",
    r"feverish\s+cold":                                "upper respiratory infection",
    r"common\s+cold":                                  "upper respiratory infection",
    r"running\s+nose":                                 "upper respiratory infection",
    r"nasal\s+(discharge|congestion|drip)":            "upper respiratory infection",
    r"sore\s+throat":                                  "throat pain",
    r"throat\s+infection":                             "throat pain",
    r"dry\s+cough":                                    "cough",
    r"bronchial\s+asthma":                             "asthma",
    r"asthma\s+(attack|exacerbation)?":                "asthma",
    r"pneumonia\w*":                                   "pneumonia",
    r"breathlessness|short\w*\s+of\s+breath":          "breathlessness",
    r"dyspno?ea":                                      "breathlessness",
    # GI
    r"loose\s+mot?i?on":                               "diarrhea",
    r"loose\s+stools?":                                "diarrhea",
    r"vomiting\s+\w+":                                 "vomiting",
    r"nausea\s+(and\s+vomiting|vomiting)?":            "vomiting",
    r"gastric\s+reflux":                               "gastroesophageal reflux",
    r"normal\s+gastric\s+acidity":                     "gastroesophageal reflux",
    r"gas\s+trouble":                                  "gas pain",
    r"acidity\b":                                      "gastroesophageal reflux",
    r"stomach\s+pain":                                 "abdominal pain",
    # Musculoskeletal / pain
    r"general\s+(illness|weakness|malaise)":           "general illness",
    r"whole\s+body\s+pain":                            "generalized pain",
    r"body\s+(pain|ache)":                             "generalized pain",
    r"back\s+(pain|ache)":                             "backache",
    r"cervical\s+(pain|spondylosis)":                  "neck pain",
    # Communicable
    r"dog\s+bite":                                     "dog bite",
    r"dengue\s+(hemorrhagic\s+)?fever":               "dengue fever",
    r"\bdengue\b":                                     "dengue fever",
    r"malarial\s+fever":                               "malaria",
    r"\bmalaria\b":                                    "malaria",
    r"typhoid\s*(fever)?":                             "typhoid",
    r"enteric\s+fever":                                "typhoid",
    r"pulmonary\s+tb":                                 "tuberculosis",
    r"\btb\b(?!\s*(examination|test))":                "tuberculosis",
    r"chicken\s*pox":                                  "chickenpox",
    r"\bvaricella\b":                                  "chickenpox",
    r"viral\s+hepatitis|hepatitis\s*[abcde]?":        "hepatitis",
    r"jaundice\w*":                                    "jaundice",
    # Urological
    r"burning\s+(urination|micturition)":              "urinary tract infection",
    r"\bdysuria\b":                                    "urinary tract infection",
    r"\buti\b":                                        "urinary tract infection",
    # Neurological / Psychiatric
    r"seizure\w*|convulsion\w*":                       "seizure",
    r"\bfits?\b":                                      "seizure",
    r"depressive\s+disorder|major\s+depression":      "depression",
    r"\bdepression\b":                                 "depression",
    r"anxiety\s+(disorder|neurosis)?":                 "anxiety",
    # Dermatological
    r"skin\s+rash|maculopapular\s+rash":               "skin rash",
    r"\burticaria\b|\bhives\b":                        "allergy",
    r"skin\s+infection|infected\s+wound":              "skin infection",
    # Other NCD
    r"iron\s+deficiency\s+an[ae]mia":                 "anemia",
    r"\ban[ae]mia\b":                                  "anemia",
    r"hypothyroid(ism)?":                              "hypothyroidism",
    r"hyperthyroid(ism)?":                             "hyperthyroidism",
    r"chest\s+pain":                                   "chest pain",
    # Ocular / ENT
    r"conjunctivitis|red\s+eye|eye\s+(infection|redness)": "red eye",
    r"\bearache\b":                                    "ear pain",
    r"tooth\s+pain|dental\s+pain":                    "toothache",
    # Obstetric
    r"antenatal(\s+visit)?|anc\s+visit":              "antenatal",
}

_NORMALIZE_COMPILED = [(re.compile(k, re.IGNORECASE), v) for k, v in _NORMALIZE_MAP.items()]


def normalize_disease(raw: str) -> str:
    """Normalize raw disease string to canonical form."""
    if not raw:
        return raw
    text = raw.lower().strip()
    for pattern, replacement in _NORMALIZE_COMPILED:
        if pattern.search(text):
            text = pattern.sub(replacement, text)
    return text.strip()


class NERExtractor:
    """
    Hybrid NER: Regex for structured fields, optional ClinicalBERT for disease NER.
    The ZeaCares data has a consistent template, so regex covers 95%+ of fields accurately.
    ClinicalBERT NER is used for free-text disease descriptions not covered by the template.
    """

    # Regex patterns
    _P = {
        "gender_age": re.compile(r"(Male|Female)\s+(\d+)\s+years?", re.IGNORECASE),
        "disease":    re.compile(r"presented with\s+(.+?)\.\s*Onset", re.IGNORECASE),
        "onset":      re.compile(r"Onset was\s+(\w+)", re.IGNORECASE),
        "duration":   re.compile(r"duration of\s+(\d+)\s+day", re.IGNORECASE),
        "severity":   re.compile(r"Severity:\s+(\w+)", re.IGNORECASE),
        "temp":       re.compile(r"Temperature\s+([\d.]+)\s*F", re.IGNORECASE),
        "pulse":      re.compile(r"Pulse\s+(\d+)\s*bpm", re.IGNORECASE),
        "bp":         re.compile(r"BP\s+(\d+)/(\d+)\s*mmHg", re.IGNORECASE),
        "spo2":       re.compile(r"SpO2\s+([\d.]+)\s*%", re.IGNORECASE),
        "bmi":        re.compile(r"BMI status:\s+(\w+)", re.IGNORECASE),
        "facility":   re.compile(r"Attended\s+(.+?)\.?\s*$", re.IGNORECASE),
    }

    def __init__(self, use_model: bool = False):
        self.use_model = use_model
        self._ner_model = None
        if use_model:
            self._load_ner_model()

    def _load_ner_model(self) -> None:
        try:
            from transformers import pipeline
            self._ner_model = pipeline(
                "ner",
                model="emilyalsentzer/Bio_ClinicalBERT",
                aggregation_strategy="simple",
                device=-1,
            )
            logger.info("ClinicalBERT NER pipeline loaded")
        except Exception as e:
            logger.warning(f"Could not load ClinicalBERT NER: {e}. Using regex only.")
            self._ner_model = None

    def extract(self, text: str, district: str = "Unknown") -> ExtractedEntities:
        """Extract all entities from a single clinical text record."""
        entities = ExtractedEntities(
            gender=None, age=None, age_band=None,
            disease_raw=None, disease_normalized=None,
            onset=None, duration_days=None, severity=None,
            temp_f=None, pulse=None, bp_sys=None, bp_dia=None,
            spo2=None, bmi_status=None, facility=None, district=district,
        )

        # Demographics
        m = self._P["gender_age"].search(text)
        if m:
            entities.gender = m.group(1)[0].upper()
            entities.age = int(m.group(2))
            entities.age_band = age_to_band(entities.age)

        # Disease (primary extraction target)
        m = self._P["disease"].search(text)
        if m:
            entities.disease_raw = m.group(1).strip().lower()
            entities.disease_normalized = normalize_disease(entities.disease_raw)
        elif self.use_model and self._ner_model:
            entities.disease_raw = self._extract_disease_with_model(text)
            entities.disease_normalized = normalize_disease(entities.disease_raw or "")

        # Clinical parameters
        m = self._P["onset"].search(text)
        if m:
            entities.onset = m.group(1).lower()

        m = self._P["duration"].search(text)
        if m:
            entities.duration_days = int(m.group(1))

        m = self._P["severity"].search(text)
        if m:
            entities.severity = m.group(1).lower()

        m = self._P["temp"].search(text)
        if m:
            entities.temp_f = float(m.group(1))

        m = self._P["pulse"].search(text)
        if m:
            entities.pulse = int(m.group(1))

        m = self._P["bp"].search(text)
        if m:
            entities.bp_sys = int(m.group(1))
            entities.bp_dia = int(m.group(2))

        m = self._P["spo2"].search(text)
        if m:
            entities.spo2 = float(m.group(1))

        m = self._P["bmi"].search(text)
        if m:
            entities.bmi_status = m.group(1).capitalize()

        m = self._P["facility"].search(text)
        if m:
            entities.facility = m.group(1).strip()

        return entities

    def _extract_disease_with_model(self, text: str) -> Optional[str]:
        """Use ClinicalBERT NER to find disease entity when regex fails."""
        if not self._ner_model:
            return None
        try:
            ner_results = self._ner_model(text)
            disease_tokens = [r["word"] for r in ner_results
                              if r.get("entity_group", "").startswith("Disease")]
            return " ".join(disease_tokens).strip() or None
        except Exception:
            return None

    def extract_batch(self, records: list[dict]) -> list[ExtractedEntities]:
        """Process a list of raw records."""
        results = []
        for rec in records:
            text = str(rec.get("clinicalText", ""))
            district = str(rec.get("district", "Unknown"))
            results.append(self.extract(text, district))
        return results

    def get_vitals_summary(self, entity: ExtractedEntities) -> dict:
        """Return a vitals dict with risk flags."""
        summary = {}
        if entity.bp_sys is not None:
            summary["bp"] = f"{entity.bp_sys}/{entity.bp_dia}"
            summary["bp_risk"] = (
                "high" if entity.bp_sys >= 140 or (entity.bp_dia or 0) >= 90
                else "elevated" if entity.bp_sys >= 130
                else "normal"
            )
        if entity.temp_f is not None:
            summary["temp_f"] = entity.temp_f
            summary["fever"] = entity.temp_f >= 99.5
        if entity.spo2 is not None:
            summary["spo2"] = entity.spo2
            summary["low_spo2"] = entity.spo2 < 95.0
        if entity.pulse is not None:
            summary["pulse"] = entity.pulse
            summary["tachycardia"] = entity.pulse > 100
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extractor = NERExtractor(use_model=False)

    test_records = [
        {"clinicalText": "Female 60 years, presented with hypertension stage 2. Onset was gradual with duration of 3 day(s). Severity: mild. Vitals: Temperature 98F, Pulse 72bpm, BP 110/80mmHg, SpO2 98%. BMI status: Healthy. Attended UPHC Lankapatnam.", "district": "Vizianagaram"},
        {"clinicalText": "Male 70 years, presented with gas pain. Onset was gradual with duration of 1 day(s). Severity: mild. Vitals: Temperature 98F, Pulse 72bpm, BP 110/80mmHg, SpO2 98%. BMI status: Healthy. Attended UPHC Lankapatnam.", "district": "Vizianagaram"},
        {"clinicalText": "Female 51 years, presented with diabetic on oral treatment. Onset was gradual with duration of 3 day(s). Severity: mild. Vitals: Temperature 98F, Pulse 72bpm, BP 110/80mmHg, SpO2 98%. BMI status: Healthy. Attended UPHC Lankapatnam.", "district": "Vizianagaram"},
    ]

    print("=== NER Extraction Demo ===\n")
    for rec in test_records:
        e = extractor.extract(rec["clinicalText"], rec["district"])
        print(f"Raw disease:        {e.disease_raw}")
        print(f"Normalized disease: {e.disease_normalized}")
        print(f"Demographics:       {e.gender}, Age Band {e.age_band}")
        print(f"Severity/Duration:  {e.severity}, {e.duration_days} days")
        print(f"BP:                 {e.bp_sys}/{e.bp_dia} mmHg")
        vitals = extractor.get_vitals_summary(e)
        print(f"Vitals flags:       {vitals}")
        print()
