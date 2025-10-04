import re, math, json, difflib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import json
from langchain_ollama import ChatOllama
import math 
import os 

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
OLLAMA_model = os.getenv("DEFAULT_MODEL", "")

#llm = ChatOllama(model="llama3.2:3b", temperature=0)
llm = ChatOllama(
    model=OLLAMA_model,temperature=0,
    base_url=OLLAMA_BASE_URL
)


# ---------------- Configuration ----------------
CATEGORY_KEYS = ["nutrition_output", "exercise_output", "lifestyle_output", "supplement_output"]
CATEGORY_HEADINGS = ["Nutrition", "Exercise", "Lifestyle", "Supplement"]
FUZZY_CUTOFF = 0.72

# ---------------- Utilities ----------------
def now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"

def normalize_key(s: Optional[str]) -> str:
    """Normalize a string key for fuzzy matching."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[_\-\s]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.replace(" ", "_")

def extract_lab_cond(cond: str) -> Optional[Tuple[str, str]]:
    """Extract lab test and risk from 'Test:Risk' format."""
    if ":" not in cond:
        return None
    left, right = cond.split(":", 1)
    return left.strip(), right.strip()

def split_group(cond_str: Optional[str]) -> Tuple[str, List[str]]:
    """Split condition groups using '&' (AND) or '|' (OR)."""
    if not cond_str or str(cond_str).strip() == "":
        return "AND", []
    if "|" in cond_str and "&" not in cond_str:
        return "OR", [p.strip() for p in cond_str.split("|") if p.strip()]
    return "AND", [p.strip() for p in cond_str.split("&") if p.strip()]

def build_labs_map(patient_rows: List[Dict]) -> Dict[str, str]:
    """Build normalized lab test -> risk mapping from patient rows."""
    labs: Dict[str, str] = {}
    if not patient_rows:
        return labs
    sample = patient_rows[0]
    test_cols = [k for k in sample.keys() if "test" in k.lower() or "name" in k.lower()]
    risk_cols = [k for k in sample.keys() if "risk" in k.lower() or "category" in k.lower()]
    tcol = test_cols[0] if test_cols else None
    rcol = risk_cols[0] if risk_cols else None
    for r in patient_rows:
        if tcol is None:
            continue
        raw_test = r.get(tcol)
        if raw_test is None:
            continue
        key = normalize_key(raw_test)
        val = r.get(rcol) if rcol else r.get("risk_category") or ""
        labs[key] = "" if val in (None, "") else str(val).strip()
    return labs

def match_test_in_labs(query_test: str, labs: Dict[str, str]) -> Optional[str]:
    """Match a lab test string to labs keys using exact, subset, or fuzzy match."""
    if not labs:
        return None
    qk = normalize_key(query_test)
    if qk in labs:
        return qk
    qtokens = set(qk.split("_"))
    for k in labs.keys():
        kt = set(k.split("_"))
        if qtokens and qtokens.issubset(kt):
            return k
    candidate = difflib.get_close_matches(qk, list(labs.keys()), n=1, cutoff=FUZZY_CUTOFF)
    if candidate:
        return candidate[0]
    return None

# ---------------- Condition parser ----------------
def parse_condition_dynamic(cond: str) -> Dict[str, Any]:
    """Parse a single primitive condition into structured dict (lab, numeric, presence)."""
    out = {"raw": cond, "type": "unknown", "parsed": None}
    if not cond or str(cond).strip() == "":
        return out
    cond = cond.strip()
    lab = extract_lab_cond(cond)
    if lab:
        out["type"] = "lab"
        out["parsed"] = {"test": lab[0], "risk": lab[1]}
        return out
    m = re.match(r"^\s*([A-Za-z0-9_\- ]+)\s*(>=|<=|>|<|=)\s*([\d.]+)\s*$", cond)
    if m:
        out["type"] = "numeric"
        out["parsed"] = {"attr": m.group(1).strip(), "op": m.group(2), "value": float(m.group(3))}
        return out
    out["type"] = "presence"
    out["parsed"] = {"text": cond}
    return out

def evaluate_primitive_deterministic(parsed: Dict[str, Any], patient_profile: Dict[str, Any], labs: Dict[str, str]) -> Tuple[Optional[bool], str]:
    """Evaluate a primitive condition deterministically."""
    t = parsed.get("type")
    p = parsed.get("parsed")
    if t == "lab":
        test, required = p["test"], p["risk"]
        matched = match_test_in_labs(test, labs)
        if matched is None:
            return None, f"lab_no_match:{test}"
        val = labs.get(matched)
        if not val:
            return None, f"lab_match_no_value:{matched}"
        return (str(val).strip().lower() == str(required).strip().lower()), f"lab_cmp:{matched}={val} vs required={required}"
    if t == "numeric":
        attr, op, val = p["attr"], p["op"], p["value"]
        keys_to_try = [normalize_key(attr), attr.lower(), attr]
        pv = None
        for k in keys_to_try:
            if k in patient_profile and patient_profile.get(k) not in (None, ""):
                try:
                    pv = float(patient_profile.get(k))
                    break
                except:
                    continue
        if pv is None:
            for k in ("age", "age_years", "years"):
                if k in patient_profile and patient_profile.get(k) not in (None, ""):
                    try:
                        pv = float(patient_profile.get(k))
                        break
                    except:
                        continue
        if pv is None:
            return None, f"numeric_missing:{attr}"
        ops = {">": pv>val, "<": pv<val, ">=": pv>=val, "<=": pv<=val, "=": pv==val}
        return ops.get(op), f"{pv}{op}{val}"
    if t == "presence":
        text = p.get("text").lower()
        found_any = False
        for fld in ("conditions","comorbidities","symptoms","medical_history","history","notes","medications","meds"):
            v = patient_profile.get(fld)
            if not v:
                continue
            found_any = True
            if isinstance(v, (list, tuple, set)):
                for it in v:
                    if it and text in str(it).lower():
                        return True, f"presence_found_in_{fld}"
            else:
                if text in str(v).lower():
                    return True, f"presence_found_in_{fld}"
        return (False if found_any else None), "presence_not_found_or_no_field"
    return None, "unknown_type"

# 1.True → found in any field.
# 2.False → checked fields exist, but no match.
# None → no such fields at all (unknown).

# ---------------- Recommendation deterministic evaluation ----------------
def classify_recommendation_deterministic(rec: Dict, patient_profile: Dict, labs: Dict[str, str]) -> Tuple[str, Dict]:
    """Classify a recommendation: 'keep', 'drop', 'unknown'. Returns audit info."""
    audit = {"timestamp": now_iso(), "rec": rec.get("recommendations") or rec.get("recommendation"),
             "when_not_raw": rec.get("When_not_to_recommend") or "", "when_to_raw": rec.get("When_to_recommend") or "",
             "when_not_eval": [], "when_to_eval": [], "decision": None}

    # ---------------- Evaluate WHEN_NOT conditions ----------------
    op_not, not_conds = split_group(audit["when_not_raw"])
    ambiguous_not = []
    for cond in not_conds:
        parsed = parse_condition_dynamic(cond)
        val, reason = evaluate_primitive_deterministic(parsed, patient_profile, labs)
        audit["when_not_eval"].append({"cond": cond, "deterministic": val, "reason": reason})
        if val is True:
            audit["decision"] = "drop_due_to_when_not"
            return "drop", audit
        # if val is None:
        #     ambiguous_not.append(cond)
 
        # medicasl history, it mapped as None if no such fields at all (unknown). We are considering that recomendations

        if val is None:
            # Special case: presence type → keep but attach note
            if parsed["type"] == "presence":
                audit["decision"] = "presence_unknown_but_included"
                return "presence_unknown_but_included", audit
            ambiguous_not.append(cond)    

    # ---------------- Evaluate WHEN_TO conditions (optional) ----------------
    # op_to, to_conds = split_group(audit["when_to_raw"])
    # ambiguous_to = []
    # any_true, any_false = False, False
    # for cond in to_conds:
    #     parsed = parse_condition_dynamic(cond)
    #     val, reason = evaluate_primitive_deterministic(parsed, patient_profile, labs)
    #     audit["when_to_eval"].append({"cond": cond, "deterministic": val, "reason": reason})
    #     if val is True:
    #         any_true = True
    #     elif val is False:
    #         any_false = True
    #     else:
    #         ambiguous_to.append(cond)

    # if any_false and op_to=="AND":
    #     audit["decision"] = "drop_due_to_when_to_false"
    #     return "drop", audit
    
    # if ambiguous_to or ambiguous_not:
    #     audit["decision"] = "unknown_due_to_ambiguous_conditions"
    #     return "unknown", audit

    # If any ambiguous WHEN_NOT conditions exist, mark as unknown
    if ambiguous_not:
        audit["decision"] = "unknown_due_to_ambiguous_conditions"
        return "unknown", audit

    audit["decision"] = "keep_deterministic"
    return "keep", audit

## Add note for all medical history recomendations so that we know the whom not recoomend
## example "Intermittent Fasting (14:10 or 16:8 protocol) (⚠️ Cannot validate: patient taking insulin medication)"

def attach_presence_notes(rec: Dict) -> str:
    """
    If recommendation has decision 'presence_unknown_but_included',
    append its WHEN_NOT condition(s) as a warning note.
    """
    text = rec.get("recommendations") or rec.get("recommendation") or ""
    audit = rec.get("audit", {})

    if audit.get("decision") == "presence_unknown_but_included":
        when_not_eval = audit.get("when_not_eval", [])
        # collect only conditions not validated
        unvalidated = [
            cond["cond"]
            for cond in when_not_eval
            if  cond.get("deterministic") is None   ##or cond.get("deterministic") is False 
        ]
        if unvalidated:
            return f"{text} (Cannot validate the condition: {', '.join(unvalidated)})"
    
        # raw_cond = audit.get("when_not_raw") or audit.get("when_to_raw") or ""
        # if raw_cond:
        #     return f"{text} (Cannot validate the condition: {raw_cond})"

    return text

######
# ---------------- Stage 1: Deterministic filtering ----------------
def run_deterministic_stage(state: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """
    Apply deterministic rules for all recommendations.
    Returns validated_data: dict(domain -> list of dicts)
    """
    patient_rows = state.get("patient_data") or []
    patients_by_visit: Dict[str, List[Dict]] = {}
    for r in patient_rows:
        vid = str(r.get("visit_id") or "__no_visit__")
        patients_by_visit.setdefault(vid, []).append(r)

    outputs_by_domain: Dict[str, List[Dict]] = {h: state.get(k, []) for k,h in zip(CATEGORY_KEYS, CATEGORY_HEADINGS)}
    validated_data: Dict[str, List[Dict]] = {h: [] for h in CATEGORY_HEADINGS}

    for vid, patient_rows_for_visit in patients_by_visit.items():
        patient_profile = patient_rows_for_visit[0] if patient_rows_for_visit else {}
        labs = build_labs_map(patient_rows_for_visit)
        for domain in CATEGORY_HEADINGS:
            for rec in outputs_by_domain.get(domain, []):
                rec_vid = str(rec.get("visit_id") or "__no_visit__")
                if rec_vid != vid:
                    continue

                # Deterministic evaluation
                status, audit = classify_recommendation_deterministic(rec, patient_profile, labs)
                # print('status',status)
                # print('audit',audit)
                
                if status != "drop":
                    entry = rec.copy()
                    entry["status"] = status
                    entry["audit"] = audit

                    # ---------------- Attach presence notes if needed ----------------
                    entry["final_text"] = attach_presence_notes(entry)

                    validated_data[domain].append(entry)
    return validated_data

# ---------------- Stage 2: LLM cross-validation ----------------

# ---------------- Utility Functions ----------------
def safe_value(val):
    """Convert NaN, None, or empty strings into JSON-safe None."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if val == "" or str(val).lower() == "nan":
        return None
    return val

def format_recommendations(domain_list: List[Dict], patient_data: Dict[str, Any]):
    """
    Format recommendations for LLM input as JSON string.
    Uses patient profile from state for validation context.
    """
    if not domain_list:
        return "None"

    formatted = []
    for rec in domain_list:
        if isinstance(rec, dict):
            formatted.append({
                "recommendation":safe_value(rec.get("final_text") or rec.get("recommendations") or rec.get("recommendation")),
                "When_to_Recommend": safe_value(rec.get("When_to_recommend") or rec.get("when_to_recommend")),
                "When_Not_to_Recommend": safe_value(rec.get("When_not_to_recommend") or rec.get("when_not_to_recommend")),
                "Patient_Age": safe_value(patient_data.get("age")),
                "Patient_Gender": safe_value(patient_data.get("gender")),
                "Comorbidities": safe_value(patient_data.get("comorbidities")),
                "Medications": safe_value(patient_data.get("medications")),
                "Labs": safe_value(patient_data.get("labs")),
            })
        else:
            formatted.append({"recommendation": str(rec)})
    return json.dumps(formatted, indent=2)

# ---------------- LLM Prompt ----------------
CROSS_VALIDATOR_PROMPT = """
You are a healthcare validation system. 

You receive recommendations grouped under four domains: Nutrition, Exercise, Lifestyle, and Supplement.
Each recommendation has metadata fields:
- recommendation (the EXACT text to KEEP if valid)
- When_to_Recommend
- When_Not_to_Recommend
- Patient profile (age, gender, comorbidities, medications, labs)

Your job is to strictly filter recommendations according to these rules:

---------------- STRICT VALIDATION RULES(HARD) ----------------

1. KEEP a recommendation ONLY IF:
   - All conditions in `When_to_Recommend` (if present) are satisfied by the patient profile.
   - NONE of the conditions in `When_Not_to_Recommend` are satisfied.

2. DROP a recommendation if:
   - ANY condition in `When_Not_to_Recommend` is satisfied.
   - ALL conditions in `When_to_Recommend` are not satisfied (if `When_to_Recommend` exists).

3. DO NOT change the text of the recommendation in any way.
4. DO NOT include explanations, reasoning, notes, or metadata.
5. DO NOT add anything extra. Only output valid recommendations.
6. Deduplicate recommendations across domains. Place each recommendation in the most relevant domain.
7. If a recommendation has a note "(Cannot validate the condition: ...)" in its text (i.e., presence-unknown), **always include it** under the correct domain. Do not remove or modify it.

---------------- OUTPUT FORMAT ----------------

- Use **exactly these four headings** in this order:  
  Nutrition:  
  Exercise:  
  Lifestyle:  
  Supplement:  

- Under each domain, list **only valid recommendations** as bullets starting with a single "- ".
- If no valid recommendations exist for a domain, leave it empty (heading only, no bullets).  
- DO NOT use Markdown formatting (**bold, italics, ###, etc.).
- DO NOT add any text other than bullets under the domain headings.

---------------- EXAMPLE ----------------

Nutrition:
- Eat plain Greek yogurt daily

Exercise:
- Practice 3 days per week. Focus on core strength and flexibility.

Lifestyle:
- Continue to sleep 7-9 hours per night

Supplement:
- Start a chromium supplement

IMPORTANT: Output MUST follow this format exactly. Do not add any extra text, reasoning, or commentary. Only valid recommendations as bullets under the correct headings.
"""


# ---------------- Stage 2: Cross-Validator ----------------
# ---------------- Full Pipeline: Deterministic + LLM Cross-Validator ----------------
def cross_validator_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full 2-stage cross-validation pipeline:
    1. Run deterministic rules to get validated_data
    2. Run LLM cross-validator on validated_data using patient profile from state
    Stores final validated text in state['cross_validated_node'].
    """
    if "patient_data" not in state or not state["patient_data"]:
        print("❌ No patient data found in state. Skipping cross-validation.")
        state["cross_validated_node"] = "Nutrition:\n\nExercise:\n\nLifestyle:\n\nSupplement:\n"
        return state

    # ---------------- Stage 1: Deterministic rules ----------------
    validated_data = run_deterministic_stage(state)
    #print("validated_data", validated_data)

    if not validated_data:
        print("⚠️ Deterministic validation returned empty. Skipping LLM stage.")
        state["cross_validated_node"] = "Nutrition:\n\nExercise:\n\nLifestyle:\n\nSupplement:\n"
        return state

    # ---------------- Stage 2: LLM cross-validator ----------------
    patient_rows = state.get("patient_data", [])
    patient_profile = patient_rows[0] if patient_rows else {}

    validation_query = f"""
Patient profile: {json.dumps(patient_profile, indent=2)}

🥗 Nutrition:
{format_recommendations(validated_data.get("Nutrition", []), patient_profile)}

🏃 Exercise:
{format_recommendations(validated_data.get("Exercise", []), patient_profile)}

🛌 Lifestyle:
{format_recommendations(validated_data.get("Lifestyle", []), patient_profile)}

💊 Supplement:
{format_recommendations(validated_data.get("Supplement", []), patient_profile)}
"""
    #print(validation_query)
    try:
        # Use the global llm object already defined
        response = llm.invoke(CROSS_VALIDATOR_PROMPT + "\n\n" + validation_query)
        cross_validated_text = getattr(response, "content", str(response))

        # ---------------- Output Sanitization ----------------
        # Ensure all four headings exist in order and remove any extra text
        domains = ["Nutrition:", "Exercise:", "Lifestyle:", "Supplement:"]

        sanitized_text = []
        for i, domain in enumerate(domains):
            # build safe lookahead: if there are next domains, use them, else assert end-of-string
            next_domains = domains[i+1:]
            if next_domains:
                # escape each domain for regex safety and join with |
                escaped = "|".join(re.escape(d) for d in next_domains)
                lookahead = rf"(?={escaped})"
            else:
                lookahead = r"(?=$)"
            # escape current domain too
            pattern = rf"{re.escape(domain)}(.*?){lookahead}"
            match = re.search(pattern, cross_validated_text, flags=re.DOTALL | re.IGNORECASE)
            content = match.group(1).strip() if match else ""
            # Ensure bullets start with "- ", keep lines that start with a bullet symbol
            bullets = [line.strip() for line in content.splitlines() if line.strip().startswith("-")]
            sanitized_text.append(f"{domain}\n" + "\n".join(bullets))

        state["cross_validated"] = "\n\n".join(sanitized_text)
        #state["cross_validated"] = cross_validated_text.strip()
        print("✅ Cross-Validator completed")
        #print("cross_validated_text\n", cross_validated_text)
    except Exception as e:
        print(f"⚠️ Cross-validation failed: {e}")
        state["cross_validated"] = "Nutrition:\n\nExercise:\n\nLifestyle:\n\nSupplement:\n"

    return state
