# nodes/final_generator_node.py

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import json
import re

import os 

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
OLLAMA_model = os.getenv("DEFAULT_MODEL", "")

#llm = ChatOllama(model="llama3.2:3b", temperature=0)
llm = ChatOllama(
    model=OLLAMA_model,temperature=0,
    base_url=OLLAMA_BASE_URL
)


# Pydantic structured output
class RecommendationOutput(BaseModel):
    Nutrition: List[str] = Field(default_factory=list)
    Exercise: List[str] = Field(default_factory=list)
    Lifestyle: List[str] = Field(default_factory=list)
    Supplement: List[str] = Field(default_factory=list)


import re
from typing import Dict, List

def extract_bullets(text: str) -> Dict[str, List[str]]:
    """
    Robustly extract bullet lists for the four domains from an LLM output string.

    Strategy:
    - Find heading positions (Nutrition/Exercise/Lifestyle/Supplement(s)) using a line-based match.
    - Slice text between headings and extract bullets from each slice.
    - Preserve punctuation and text (do NOT strip '&', parentheses, etc).
    - Accept emoji-prefixed headings and small heading name variants.
    """
    domains = ["Nutrition", "Exercise", "Lifestyle", "Supplement"]
    out: Dict[str, List[str]] = {d: [] for d in domains}
    if not text or not isinstance(text, str):
        return out

    # Normalize some heading synonyms (keep rest of text intact)
    normalized = text.replace("Diet:", "Nutrition:")
    normalized = normalized.replace("Sleep and Lifestyle", "Lifestyle:")

    # Find heading lines (capture heading name and position)
    # Match lines like: [optional emoji/whitespace]Nutrition:  OR  "💊 Supplements:" etc.
    heading_re = re.compile(
        r"(?mi)^[^\S\n\r]*(?:[^\w\n\r]+\s*)*(Nutrition|Exercise|Lifestyle|Supplement)\s*:\s*$",
        flags=re.MULTILINE | re.IGNORECASE,
    )

    matches = list(heading_re.finditer(normalized))

    # If no exact heading lines found, fallback to a looser heading matcher (heading with content on same line)
    if not matches:
        heading_re2 = re.compile(
            r"(?mi)^[^\S\n\r]*(?:[^\w\n\r]+\s*)*(Nutrition|Exercise|Lifestyle|Supplement)\s*:\s*",
            flags=re.MULTILINE | re.IGNORECASE,
        )
        matches = list(heading_re2.finditer(normalized))

    # If still no matches, try to heuristically split by domain words anywhere (last resort)
    if not matches:
        simple = {}
        for d in domains:
            # find the first occurrence of "Nutrition:" etc.
            idx = re.search(r"(?i)\b" + re.escape(d[:-1] if d == "Supplement" else d) + r"\s*:", normalized)
            if idx:
                simple[d] = idx.start()
        if not simple:
            return out
        # turn into pseudo-matches sorted by position
        items = sorted(simple.items(), key=lambda kv: kv[1])
        pseudo = []
        for i, (name, pos) in enumerate(items):
            # end pos is next pos or end
            next_pos = items[i+1][1] if i+1 < len(items) else len(normalized)
            pseudo.append((name, pos, next_pos))
        # extract content slices
        for name, s, e in pseudo:
            content = normalized[s:e]
            bullets = re.findall(r"^\s*[-*•]\s+(.*)", content, flags=re.MULTILINE)
            out[name] = [b.strip() for b in bullets if b.strip()]
        return out

    # Build list of (heading_name, start_idx, end_idx)
    spans = []
    for m in matches:
        name = m.group(1)
        start = m.start()
        end = m.end()
        spans.append((name, start, end))

    # Compute content windows between headings
    for idx, (name, start, end) in enumerate(spans):
        # content begins at end of heading line
        content_start = end
        content_end = spans[idx + 1][1] if idx + 1 < len(spans) else len(normalized)
        content = normalized[content_start:content_end]

        # Normalize heading to our canonical domain names
        domain = "Supplement" if name.lower().startswith("supplement") else name.capitalize()

        # Primary: extract bullets that start with -, *, or • (keep entire line)
        bullets = re.findall(r"^\s*[-*•]\s+(.*(?:\S.*)?)", content, flags=re.MULTILINE)
        bullets = [b.strip() for b in bullets if b and b.strip()]

        # If no bullets, fallback: split by line, keep non-empty non-heading lines
        if not bullets:
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            # Remove lines that look like headers or the "After applying..." preamble
            lines = [ln for ln in lines if not re.match(r"(?i)^(after applying|based on the provided|here is the output)", ln)]
            # If lines look like "A, B, C" on one line, split by comma
            if len(lines) == 1 and "," in lines[0] and len(lines[0]) < 400:
                parts = [p.strip() for p in lines[0].split(",") if p.strip()]
                bullets = parts
            else:
                bullets = lines

        out[domain] = bullets

    # Ensure keys present
    for d in domains:
        out.setdefault(d, [])

    return out


def final_generator_agent_node(state: dict):
    """
    Generate final_output from cross-validated text stored in state.
    Reads state['cross_validated'], extracts bullets,
    normalizes domains, and writes state['final_output'].
    """
    print("📝 Generating Final Output...")

    # Read from the canonical place where cross-validator writes output
    cross_text = state.get("cross_validated") or ""
    if not cross_text:
        print("⚠️ No cross-validated text found in state ('cross_validated').")
        state["final_output"] = {"Nutrition": [], "Exercise": [], "Lifestyle": [], "Supplement": []}
        return state

    # Debug: show raw cross-validated text (comment out in production)
    print("DEBUG: raw cross_validated text preview:\n", cross_text[:1000])

    extracted = extract_bullets(cross_text)
    #print("extracted", extracted)
    print("DEBUG: extracted counts:", {k: len(v) for k, v in extracted.items()})

    # Normalize keys and ensure all four present
    final_output = {
        "Nutrition": extracted.get("Nutrition", []),
        "Exercise": extracted.get("Exercise", []),
        "Lifestyle": extracted.get("Lifestyle", []),
        "Supplement": extracted.get("Supplement", []),
    }

    # If you have RecommendationOutput Pydantic model use it; otherwise set dict
    try:
        ro = RecommendationOutput(**final_output)
        state["final_output"] = ro.dict()
    except Exception:
        state["final_output"] = final_output

    print("✅ Final output generated (counts):", {k: len(v) for k, v in state["final_output"].items()})
    return state
