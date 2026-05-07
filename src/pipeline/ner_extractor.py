"""
NER (Named Entity Recognition) Extractor
Hybrid approach: Regex for structured fields + disease normalization.
Handles both PHC template format and free-form structured section format.
"""
import re
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntities:
    # Standard fields
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
    # Extra fields — everything beyond the standard template
    extra_fields: Dict[str, Any] = field(default_factory=dict)


def age_to_band(age: int) -> str:
    bands = [(0, 14), (15, 24), (25, 34), (35, 44), (45, 54),
             (55, 64), (65, 74), (75, 84), (85, 120)]
    for lo, hi in bands:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return "Unknown"


# ── Disease normalisation map ─────────────────────────────────────────────────
_NORMALIZE_MAP = {
    r"hypertension\s*stage\s*[12i]+":                  "hypertension",
    r"stage\s*[12]\s*hypertension":                    "hypertension",
    r"htn[-\s]?[12]?":                                 "hypertension",
    r"high\s*bp":                                      "hypertension",
    r"bp\s+high":                                      "hypertension",
    r"fear\s+of\s+hypertension":                       "hypertension",
    r"diabetic\s+(on\s+oral\s+treatment|monitored|diet|urine|good\s+control)": "diabetes type 2",
    r"diabetes\s+monitored":                           "diabetes type 2",
    r"dm\s*(monitored|type\s*2)?":                     "diabetes type 2",
    r"type\s*2\s+diabetes":                            "diabetes type 2",
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
    r"loose\s+mot?i?on":                               "diarrhea",
    r"loose\s+stools?":                                "diarrhea",
    r"watery\s+diarrhea|profuse\s+diarrhea":           "diarrhea",
    r"rice[- ]water\s+stool":                          "cholera",
    r"vomiting\s+\w+":                                 "vomiting",
    r"nausea\s+(and\s+vomiting|vomiting)?":            "vomiting",
    r"gastric\s+reflux":                               "gastroesophageal reflux",
    r"normal\s+gastric\s+acidity":                     "gastroesophageal reflux",
    r"gas\s+trouble":                                  "gas pain",
    r"acidity\b":                                      "gastroesophageal reflux",
    r"stomach\s+pain":                                 "abdominal pain",
    r"general\s+(illness|weakness|malaise)":           "general illness",
    r"whole\s+body\s+pain":                            "generalized pain",
    r"body\s+(pain|ache)":                             "generalized pain",
    r"back\s+(pain|ache)":                             "backache",
    r"cervical\s+(pain|spondylosis)":                  "neck pain",
    r"dog\s+bite":                                     "dog bite",
    r"dengue\s+(hemorrhagic\s+)?fever":               "dengue fever",
    r"\bdengue\b(?!\s+fever)":                        "dengue fever",
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
    r"burning\s+(urination|micturition)":              "urinary tract infection",
    r"\bdysuria\b":                                    "urinary tract infection",
    r"\buti\b":                                        "urinary tract infection",
    r"seizure\w*|convulsion\w*":                       "seizure",
    r"\bfits?\b":                                      "seizure",
    r"depressive\s+disorder|major\s+depression":      "depression",
    r"\bdepression\b":                                 "depression",
    r"anxiety\s+(disorder|neurosis)?":                 "anxiety",
    r"skin\s+rash|maculopapular\s+rash":               "skin rash",
    r"\burticaria\b|\bhives\b":                        "allergy",
    r"skin\s+infection|infected\s+wound":              "skin infection",
    r"iron\s+deficiency\s+an[ae]{1,2}mia":             "anemia",
    r"\ban[ae]{1,2}mia\b":                            "anemia",
    r"hypothyroid(ism)?":                              "hypothyroidism",
    r"hyperthyroid(ism)?":                             "hyperthyroidism",
    r"chest\s+pain":                                   "chest pain",
    r"conjunctivitis|red\s+eye|eye\s+(infection|redness)": "red eye",
    r"\bearache\b":                                    "ear pain",
    r"tooth\s+pain|dental\s+pain":                    "toothache",
    r"antenatal(\s+visit)?|anc\s+visit":              "antenatal",
    r"cholera\w*":                                     "cholera",
    r"vibri[ao]\s+(cholerae|infection)":               "cholera",
    r"dehydration\s+(severe|moderate)":                "dehydration",
    r"severe\s+dehydration":                           "dehydration",
    r"acute\s+gastroenteritis":                        "gastroenteritis",
    r"gastro[\s-]?enteritis":                          "gastroenteritis",
}

_NORMALIZE_COMPILED = [(re.compile(k, re.IGNORECASE), v) for k, v in _NORMALIZE_MAP.items()]


def normalize_disease(raw: str) -> str:
    if not raw:
        return raw
    text = raw.lower().strip()
    for pattern, replacement in _NORMALIZE_COMPILED:
        if pattern.search(text):
            text = pattern.sub(replacement, text)
    return text.strip()


# ── Detect structured section format ─────────────────────────────────────────
_SECTION_HEADERS = re.compile(
    r"(Patient|Chief\s+Complaints?|History|Examination|Vitals?|Assessment|Lab(?:oratory)?|Diagnosis|Plan|Impression)\s*:",
    re.IGNORECASE
)

def _is_section_format(text: str) -> bool:
    return len(_SECTION_HEADERS.findall(text)) >= 2


def _parse_sections(text: str) -> Dict[str, str]:
    """Split text by section headers; returns dict of section_name → content."""
    sections: Dict[str, str] = {}
    parts = _SECTION_HEADERS.split(text)
    # parts = [pre_text, header1, content1, header2, content2, ...]
    i = 1
    while i < len(parts) - 1:
        raw_header = parts[i].strip().lower()
        # normalise: "chief complaints" → "chief_complaints", "vital" → "vitals"
        key = re.sub(r"\s+", "_", raw_header)
        key = re.sub(r"chief_complaints?", "chief_complaints", key)
        key = re.sub(r"^vital$", "vitals", key)
        content = parts[i + 1].strip()
        sections[key] = content
        i += 2
    return sections


class NERExtractor:
    """
    Flexible NER extractor.
    Handles:
      - Standard PHC template: "Female 60 years, presented with..."
      - Section-based format: "Patient: ... Chief Complaints: ... Vitals: ..."
    Puts standard fields into ExtractedEntities; any extra parsed data goes into extra_fields.
    """

    # Standard template patterns
    _P = {
        "gender_age":  re.compile(r"(Male|Female)\s+(\d+)\s+years?", re.IGNORECASE),
        "disease":     re.compile(r"presented with\s+(.+?)\.\s*Onset", re.IGNORECASE),
        "onset":       re.compile(r"Onset was\s+(\w+)", re.IGNORECASE),
        "duration":    re.compile(r"duration of\s+(\d+)\s+day", re.IGNORECASE),
        "severity":    re.compile(r"Severity:\s*(\w+)", re.IGNORECASE),
        "temp_f":      re.compile(r"Temperature\s+([\d.]+)\s*F", re.IGNORECASE),
        "pulse_bpm":   re.compile(r"Pulse\s+(\d+)\s*bpm", re.IGNORECASE),
        "bp":          re.compile(r"BP\s+(\d+)/(\d+)\s*mmHg", re.IGNORECASE),
        "spo2":        re.compile(r"SpO2\s+([\d.]+)\s*%", re.IGNORECASE),
        "bmi":         re.compile(r"BMI status:\s*(\w+)", re.IGNORECASE),
        "facility":    re.compile(r"Attended\s+(.+?)\.?\s*$", re.IGNORECASE),
    }

    # Extra vitals/lab patterns (used for both formats)
    _EXTRA_P = {
        # Temperature in Celsius  → convert to F
        "temp_c":      re.compile(r"Temp\w*\s+([\d.]+)\s*°?\s*C\b", re.IGNORECASE),
        # Alternate pulse notation: "130/min" or "Pulse 130/min"
        "pulse_min":   re.compile(r"[Pp]ulse\s+(\d+)\s*/\s*min", re.IGNORECASE),
        # SpO2 alternate: "SpO2 97%", "O2 sat 97%"
        "spo2_alt":    re.compile(r"O2\s+sat\w*\s+([\d.]+)\s*%", re.IGNORECASE),
        # Respiratory rate
        "resp_rate":   re.compile(r"RR\s*[:/]?\s*(\d+)\s*/\s*min", re.IGNORECASE),
        # CRT (capillary refill time)
        "crt":         re.compile(r"CRT\s*[><=]?\s*([\d.]+)\s*sec", re.IGNORECASE),
        # Weight
        "weight_kg":   re.compile(r"Weight\s*[:\-]?\s*([\d.]+)\s*kg", re.IGNORECASE),
        # Electrolytes
        "serum_na":    re.compile(r"(?:Serum\s+)?Na\s*[:/]\s*([\d.]+)\s*mEq", re.IGNORECASE),
        "serum_k":     re.compile(r"(?:Serum\s+)?K\s*[:/]\s*([\d.]+)\s*mEq", re.IGNORECASE),
        "serum_cl":    re.compile(r"(?:Serum\s+)?Cl\s*[:/]\s*([\d.]+)\s*mEq", re.IGNORECASE),
        "serum_hco3":  re.compile(r"(?:Serum\s+)?HCO3\s*[:/]\s*([\d.]+)", re.IGNORECASE),
        # Blood glucose
        "blood_glucose":re.compile(r"(?:Blood\s+)?(?:glucose|sugar|RBS|FBS)\s*[:/]?\s*([\d.]+)\s*mg", re.IGNORECASE),
        # Haemoglobin
        "hemoglobin":  re.compile(r"Hb\s*[:/]?\s*([\d.]+)\s*g", re.IGNORECASE),
        # Episode count per day
        "episodes_per_day": re.compile(r"(\d+)[-–](\d+)\s*episodes?/day", re.IGNORECASE),
        # Dehydration plan
        "dehydration_plan": re.compile(r"WHO\s+Plan\s+([ABC])", re.IGNORECASE),
        # Stool character
        "stool_type":  re.compile(r"(rice[- ]water\s+stool|bloody\s+stool|watery\s+stool|mucoid\s+stool)", re.IGNORECASE),
        # Duration in days (alternate formats)
        "duration_days_alt": re.compile(r"for\s+(\d+)\s+days?", re.IGNORECASE),
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

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self, text: str, district: str = "Unknown") -> ExtractedEntities:
        """Extract all entities from a single clinical text record (any format)."""
        entities = ExtractedEntities(
            gender=None, age=None, age_band=None,
            disease_raw=None, disease_normalized=None,
            onset=None, duration_days=None, severity=None,
            temp_f=None, pulse=None, bp_sys=None, bp_dia=None,
            spo2=None, bmi_status=None, facility=None, district=district,
            extra_fields={},
        )

        if _is_section_format(text):
            self._extract_section_format(text, entities)
        else:
            self._extract_template_format(text, entities)

        # Extra vitals/lab from the full text regardless of format
        self._extract_extra_fields(text, entities)

        return entities

    # ── Template format (existing PHC records) ────────────────────────────────

    def _extract_template_format(self, text: str, e: ExtractedEntities) -> None:
        m = self._P["gender_age"].search(text)
        if m:
            e.gender = m.group(1)[0].upper()
            e.age = int(m.group(2))
            e.age_band = age_to_band(e.age)

        m = self._P["disease"].search(text)
        if m:
            e.disease_raw = m.group(1).strip().lower()
            e.disease_normalized = normalize_disease(e.disease_raw)
        elif self.use_model and self._ner_model:
            e.disease_raw = self._extract_disease_with_model(text)
            e.disease_normalized = normalize_disease(e.disease_raw or "")

        for key, pat, attr in [
            ("onset",    self._P["onset"],    "onset"),
            ("severity", self._P["severity"], "severity"),
        ]:
            m = pat.search(text)
            if m:
                setattr(e, attr, m.group(1).lower())

        m = self._P["duration"].search(text)
        if m:
            e.duration_days = int(m.group(1))

        m = self._P["temp_f"].search(text)
        if m:
            e.temp_f = float(m.group(1))

        m = self._P["pulse_bpm"].search(text)
        if m:
            e.pulse = int(m.group(1))

        m = self._P["bp"].search(text)
        if m:
            e.bp_sys = int(m.group(1))
            e.bp_dia = int(m.group(2))

        m = self._P["spo2"].search(text)
        if m:
            e.spo2 = float(m.group(1))

        m = self._P["bmi"].search(text)
        if m:
            e.bmi_status = m.group(1).capitalize()

        m = self._P["facility"].search(text)
        if m:
            e.facility = m.group(1).strip()

    # ── Section format (structured clinical notes) ────────────────────────────

    def _extract_section_format(self, text: str, e: ExtractedEntities) -> None:
        sections = _parse_sections(text)

        # ── Patient / Demographics ────────────────────────────────────────────
        patient_text = sections.get("patient", "")
        if patient_text:
            e.extra_fields["patient_info"] = patient_text.rstrip(".")
            # Age: "6 years", "6-year-old"
            m = re.search(r"(\d+)\s*[- ]?years?", patient_text, re.IGNORECASE)
            if m:
                e.age = int(m.group(1))
                e.age_band = age_to_band(e.age)
            # Gender / age group
            gender_map = {
                r"\bfemale\b|\bgirl\b|\bwoman\b":  "F",
                r"\bmale\b|\bboy\b|\bman\b":        "M",
                r"\bchild\b|\binfant\b|\bneonate\b": "unknown",
            }
            for pat, g in gender_map.items():
                if re.search(pat, patient_text, re.IGNORECASE):
                    e.gender = g if g != "unknown" else None
                    if g == "unknown":
                        e.extra_fields["age_group"] = re.search(
                            r"(child|infant|neonate)", patient_text, re.IGNORECASE
                        ).group(1).lower()
                    break
            # District from "from X district"
            m = re.search(r"from\s+(.+?)\s+district", patient_text, re.IGNORECASE)
            if m and e.district in (None, "Unknown"):
                e.district = m.group(1).strip()

        # ── Chief Complaints → primary disease ───────────────────────────────
        cc = sections.get("chief_complaints", sections.get("chief_complaint", ""))
        if cc:
            e.extra_fields["chief_complaints"] = cc.rstrip(".")
            # Extract primary complaint (text before first comma or parenthesis)
            primary = re.split(r"[,(]", cc)[0].strip().lower()
            # Remove qualifier words
            primary = re.sub(r"(profuse|acute|severe|mild|moderate|recurrent|chronic|watery)\s+", "", primary)
            if not e.disease_raw:
                e.disease_raw = primary
                e.disease_normalized = normalize_disease(primary)
            # Duration: "for 2 days" in CC
            m = re.search(r"for\s+(\d+)\s+days?", cc, re.IGNORECASE)
            if m and not e.duration_days:
                e.duration_days = int(m.group(1))

        # ── History ──────────────────────────────────────────────────────────
        history = sections.get("history", "")
        if history:
            e.extra_fields["history"] = history.rstrip(".")
            # Outbreak signal: "multiple children with similar symptoms"
            if re.search(r"multiple\s+.{0,30}similar\s+symptom", history, re.IGNORECASE):
                e.extra_fields["outbreak_signal"] = True
            # Source of infection
            m = re.search(r"(?:consumed?|drank?)\s+(.+?)(?:\.|,)", history, re.IGNORECASE)
            if m:
                e.extra_fields["exposure_source"] = m.group(1).strip()

        # ── Examination ──────────────────────────────────────────────────────
        exam = sections.get("examination", "")
        if exam:
            e.extra_fields["examination"] = exam.rstrip(".")
            # Dehydration level
            m = re.search(r"(severe|severely|moderate|mild)\s+dehydrat", exam, re.IGNORECASE)
            if m:
                e.extra_fields["dehydration_level"] = m.group(1).lower()
            # Consciousness
            m = re.search(r"(conscious|unconscious|lethargic|obtunded|confused)", exam, re.IGNORECASE)
            if m:
                e.extra_fields["consciousness"] = m.group(1).lower()

        # ── Assessment / Diagnosis ────────────────────────────────────────────
        assessment = sections.get("assessment", sections.get("diagnosis", sections.get("impression", "")))
        if assessment:
            e.extra_fields["assessment"] = assessment.rstrip(".")
            # If disease still not set, pull from assessment
            if not e.disease_raw:
                primary_dx = re.split(r"[.(]", assessment)[0].strip().lower()
                primary_dx = re.sub(r"(severe|moderate|mild|acute)\s+", "", primary_dx)
                e.disease_raw = primary_dx
                e.disease_normalized = normalize_disease(primary_dx)
            # Weight from assessment
            m = re.search(r"Weight\s*[:\-]?\s*([\d.]+)\s*kg", assessment, re.IGNORECASE)
            if m:
                e.extra_fields["weight_kg"] = float(m.group(1))
            # Dehydration plan
            m = re.search(r"WHO\s+Plan\s+([ABC])", assessment, re.IGNORECASE)
            if m:
                e.extra_fields["dehydration_plan"] = f"WHO Plan {m.group(1).upper()}"

        # ── Lab ───────────────────────────────────────────────────────────────
        lab = sections.get("lab", sections.get("laboratory", ""))
        if lab:
            e.extra_fields["lab_findings"] = lab.rstrip(".")
            # Vibrio / cholera signal
            if re.search(r"vibrio|cholera", lab, re.IGNORECASE):
                e.extra_fields["pathogen_identified"] = re.search(
                    r"(motile vibrios?|Vibrio cholerae|cholera)", lab, re.IGNORECASE
                ).group(1)
            # Serum electrolytes
            for key, pat in [
                ("serum_na_meql",  r"Na\s*[:/]\s*([\d.]+)\s*mEq"),
                ("serum_k_meql",   r"K\s*[:/]\s*([\d.]+)\s*mEq"),
                ("serum_cl_meql",  r"Cl\s*[:/]\s*([\d.]+)\s*mEq"),
                ("serum_hco3",     r"HCO3\s*[:/]\s*([\d.]+)"),
            ]:
                m = re.search(pat, lab, re.IGNORECASE)
                if m:
                    e.extra_fields[key] = float(m.group(1))

        # ── Vitals section ────────────────────────────────────────────────────
        vitals_text = sections.get("vitals", sections.get("vital", ""))
        if vitals_text:
            # Temperature °C
            m = re.search(r"Temp\w*\s+([\d.]+)\s*°?\s*C", vitals_text, re.IGNORECASE)
            if m:
                tc = float(m.group(1))
                e.temp_f = round(tc * 9/5 + 32, 1)
                e.extra_fields["temp_celsius"] = tc
            # Temperature F (fallback)
            if not e.temp_f:
                m = re.search(r"Temp\w*\s+([\d.]+)\s*F", vitals_text, re.IGNORECASE)
                if m:
                    e.temp_f = float(m.group(1))

            # BP
            m = self._P["bp"].search(vitals_text)
            if m:
                e.bp_sys = int(m.group(1))
                e.bp_dia = int(m.group(2))

            # Pulse (handles "130/min" and "130 bpm")
            m = re.search(r"[Pp]ulse\s+(\d+)\s*(?:/\s*min|bpm)", vitals_text)
            if m:
                e.pulse = int(m.group(1))

            # SpO2
            m = self._P["spo2"].search(vitals_text)
            if m:
                e.spo2 = float(m.group(1))

            # CRT
            m = re.search(r"CRT\s*[><=]?\s*([\d.]+)\s*sec", vitals_text, re.IGNORECASE)
            if m:
                e.extra_fields["crt_seconds"] = float(m.group(1))
            elif re.search(r"CRT\s*>\s*3", vitals_text, re.IGNORECASE):
                e.extra_fields["crt_seconds"] = ">3"

            # Respiratory rate
            m = re.search(r"RR\s*[:/]?\s*(\d+)", vitals_text, re.IGNORECASE)
            if m:
                e.extra_fields["respiratory_rate"] = int(m.group(1))

    # ── Extra fields (run on full text for both formats) ──────────────────────

    def _extract_extra_fields(self, text: str, e: ExtractedEntities) -> None:
        # Temperature Celsius (if not already set via sections)
        if e.temp_f is None:
            m = self._EXTRA_P["temp_c"].search(text)
            if m:
                tc = float(m.group(1))
                e.temp_f = round(tc * 9/5 + 32, 1)
                e.extra_fields["temp_celsius"] = tc

        # Pulse /min notation
        if e.pulse is None:
            m = self._EXTRA_P["pulse_min"].search(text)
            if m:
                e.pulse = int(m.group(1))

        # SpO2 alternate notation
        if e.spo2 is None:
            m = self._EXTRA_P["spo2_alt"].search(text)
            if m:
                e.spo2 = float(m.group(1))

        # Weight
        if "weight_kg" not in e.extra_fields:
            m = self._EXTRA_P["weight_kg"].search(text)
            if m:
                e.extra_fields["weight_kg"] = float(m.group(1))

        # Serum Na
        if "serum_na_meql" not in e.extra_fields:
            m = self._EXTRA_P["serum_na"].search(text)
            if m:
                e.extra_fields["serum_na_meql"] = float(m.group(1))

        # Serum K
        if "serum_k_meql" not in e.extra_fields:
            m = self._EXTRA_P["serum_k"].search(text)
            if m:
                e.extra_fields["serum_k_meql"] = float(m.group(1))

        # Serum Cl
        if "serum_cl_meql" not in e.extra_fields:
            m = self._EXTRA_P["serum_cl"].search(text)
            if m:
                e.extra_fields["serum_cl_meql"] = float(m.group(1))

        # CRT
        if "crt_seconds" not in e.extra_fields:
            m = self._EXTRA_P["crt"].search(text)
            if m:
                e.extra_fields["crt_seconds"] = float(m.group(1))
            elif re.search(r"CRT\s*>\s*3", text, re.IGNORECASE):
                e.extra_fields["crt_seconds"] = ">3"

        # Episodes per day
        m = self._EXTRA_P["episodes_per_day"].search(text)
        if m:
            e.extra_fields["episodes_per_day_min"] = int(m.group(1))
            e.extra_fields["episodes_per_day_max"] = int(m.group(2))

        # Dehydration plan
        if "dehydration_plan" not in e.extra_fields:
            m = self._EXTRA_P["dehydration_plan"].search(text)
            if m:
                e.extra_fields["dehydration_plan"] = f"WHO Plan {m.group(1).upper()}"

        # Stool type
        m = self._EXTRA_P["stool_type"].search(text)
        if m:
            e.extra_fields["stool_character"] = m.group(1).lower()

        # Blood glucose
        m = self._EXTRA_P["blood_glucose"].search(text)
        if m:
            e.extra_fields["blood_glucose_mgdl"] = float(m.group(1))

        # Haemoglobin
        m = self._EXTRA_P["hemoglobin"].search(text)
        if m:
            e.extra_fields["hemoglobin_gdl"] = float(m.group(1))

        # Respiratory rate
        if "respiratory_rate" not in e.extra_fields:
            m = self._EXTRA_P["resp_rate"].search(text)
            if m:
                e.extra_fields["respiratory_rate"] = int(m.group(1))

        # Alternate duration
        if e.duration_days is None:
            m = self._EXTRA_P["duration_days_alt"].search(text)
            if m:
                e.duration_days = int(m.group(1))

        # Remove empty extra_fields
        e.extra_fields = {k: v for k, v in e.extra_fields.items() if v is not None and v != ""}

    def _extract_disease_with_model(self, text: str) -> Optional[str]:
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
        results = []
        for rec in records:
            text = str(rec.get("clinicalText", ""))
            district = str(rec.get("district", "Unknown"))
            results.append(self.extract(text, district))
        return results

    def get_vitals_summary(self, entity: ExtractedEntities) -> dict:
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
        {
            "clinicalText": (
                "Female 60 years, presented with hypertension stage 2. Onset was gradual with duration of 3 day(s). "
                "Severity: mild. Vitals: Temperature 98F, Pulse 72bpm, BP 110/80mmHg, SpO2 98%. "
                "BMI status: Healthy. Attended UPHC Lankapatnam."
            ),
            "district": "Vizianagaram",
        },
        {
            "clinicalText": (
                "Patient: Child, 6 years, from Krishna district rural area.\n"
                "Chief Complaints: Profuse watery diarrhea (rice-water stools, 15-20 episodes/day) for 2 days, "
                "multiple episodes of vomiting, severe weakness.\n"
                "History: Consumed water from community pond, multiple children in village with similar symptoms, "
                "no blood in stool.\n"
                "Examination: Severely dehydrated, sunken eyes, dry tongue, skin turgor reduced (>2 seconds), "
                "lethargy but conscious.\n"
                "Vitals: Temp 36.5°C, BP 80/50 mmHg, Pulse 130/min (thready), CRT >3 seconds.\n"
                "Assessment: Severe dehydration (WHO Plan C). Weight: 16 kg (expected 20 kg).\n"
                "Lab: Stool microscopy shows motile vibrios. Serum Na: 128 mEq/L, K: 2.8 mEq/L."
            ),
            "district": "Krishna",
        },
    ]

    print("=== NER Extraction Demo ===\n")
    for rec in test_records:
        e = extractor.extract(rec["clinicalText"], rec["district"])
        print(f"Disease raw:        {e.disease_raw}")
        print(f"Disease normalized: {e.disease_normalized}")
        print(f"Demographics:       gender={e.gender}  age={e.age}  age_band={e.age_band}")
        print(f"Vitals:             temp_f={e.temp_f}  pulse={e.pulse}  BP={e.bp_sys}/{e.bp_dia}")
        print(f"SpO2:               {e.spo2}%")
        print(f"Duration:           {e.duration_days} days  severity={e.severity}")
        print(f"Extra fields:       {e.extra_fields}")
        print()
