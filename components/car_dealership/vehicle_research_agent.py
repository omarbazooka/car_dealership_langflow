from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from lfx.custom import Component
from lfx.io import IntInput, MessageTextInput, Output, SecretStrInput
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, get_car, lexical_knowledge_search

logger = logging.getLogger(__name__)

INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|rules|constraints)",
    r"you\s+are\s+now",
    r"system\s+prompt",
    r"jailbreak",
    r"override\s+(?:the\s+)?(?:rules|instructions)",
    r"act\s+as\s+a",
    r"forget\s+(?:everything|rules|instructions)",
]

TIER_1_MANUAL = "official_manual"
TIER_2_CATALOG = "local_catalog_doc"
TIER_3_BRAND = "brand_website"
TIER_4_DEALER = "dealer_portal"
TIER_5_GENERAL = "general_web"


def sanitize_and_check_injection(text: str) -> tuple[str, bool]:
    """Check text for prompt injection and sanitize."""
    lower = text.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, lower):
            return "[Content redacted due to safety guidelines]", True
    # Strip potential HTML/script artifacts
    clean = re.sub(r"<script.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<[^>]+>", " ", clean)
    return clean.strip(), False


def extract_airbag_count(text: str) -> int | None:
    """Extract strict numeric airbag count. Never guess."""
    matches = re.findall(r"(\d{1,2})\s*(?:وسائد|وسادة|وسايد|ايرباج|airbags?|air\s*bags?)", text, re.IGNORECASE)
    if matches:
        for m in matches:
            val = int(m)
            if 1 <= val <= 14:
                return val
    patterns = [
        (r"(?:airbags?|air\s*bags?|وسائد\s*هوائية)\s*[:=]\s*(\d{1,2})", 1),
        (r"(?:عدد\s*الوسائد|الوسائد\s*الهوائية)\s*[:=]?\s*(\d{1,2})", 1),
    ]
    for pat, grp in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(grp))
            if 1 <= val <= 14:
                return val
    arabic_num_map = {
        "وسادتين": 2, "وسادتان": 2, "ايرباجين": 2,
        "اربع وسائد": 4, "أربع وسائد": 4, "4 وسائد": 4,
        "ست وسائد": 6, "6 وسائد": 6,
        "سبع وسائد": 7, "7 وسائد": 7,
        "ثماني وسائد": 8, "8 وسائد": 8,
    }
    for k, v in arabic_num_map.items():
        if k in text:
            return v
    return None


def extract_numeric_fact(attribute: str, text: str) -> str | None:
    """Extract specific automotive technical fact based on requested attribute."""
    norm = text.lower()
    if attribute in ("airbags", "ايرباج", "وسائد هوائية"):
        cnt = extract_airbag_count(text)
        return f"{cnt} وسائد هوائية" if cnt is not None else None

    if attribute in ("engine_cc", "cc", "سعة المحرك"):
        m = re.search(r"(\d{3,4})\s*(?:cc|سي\s*سي|cm3|سم مكعب)", text, re.IGNORECASE)
        if m:
            return f"{m.group(1)} cc"
        m_l = re.search(r"(\d\.\d)\s*(?:l|liter|لتر)", text, re.IGNORECASE)
        if m_l:
            return f"{m_l.group(1)}L"

    if attribute in ("acceleration", "تسارع"):
        m = re.search(r"(?:0[-–]100|من\s*صفر\s*ل\s*100).*?(\d{1,2}(?:\.\d)?)\s*(?:ثانية|ثواني|sec|s\b)", text, re.IGNORECASE)
        if m:
            return f"0-100 في {m.group(1)} ثانية"

    if attribute in ("fuel_consumption", "استهلاك الوقود"):
        m = re.search(r"(\d{1,2}(?:\.\d)?)\s*(?:لتر|liters?)\s*(?:لكل|\/)\s*100\s*(?:كم|km)", text, re.IGNORECASE)
        if m:
            return f"{m.group(1)} لتر / 100 كم"

    if attribute in ("boot_size", "شنطة", "مساحة الشنطة"):
        m = re.search(r"(\d{3,4})\s*(?:لتر|liters?|l\b)", text, re.IGNORECASE)
        if m:
            return f"{m.group(1)} لتر"

    return None


class VehicleResearchAgent(Component):
    display_name = "Vehicle Research Agent"
    description = (
        "Internal subordinate agent for SalesOrchestrator to research out-of-catalog vehicle specifications "
        "and market facts following a strict 5-tier source hierarchy, exact variant matching, and injection defense."
    )
    icon = "search-code"
    name = "VehicleResearchAgent"

    inputs = [
        IntInput(name="car_id", display_name="Car ID", tool_mode=True, required=False),
        MessageTextInput(name="brand", display_name="Brand", tool_mode=True, required=False),
        MessageTextInput(name="model", display_name="Model", tool_mode=True, required=False),
        IntInput(name="year", display_name="Year", tool_mode=True, required=False),
        MessageTextInput(name="trim_variant", display_name="Trim / Variant", tool_mode=True, required=False),
        MessageTextInput(name="attribute", display_name="Attribute to Research", tool_mode=True, required=True, info="e.g. airbags, engine_cc, acceleration, dimensions, fuel_consumption"),
        MessageTextInput(name="query", display_name="Specific Research Query", tool_mode=True, required=False),
        MessageTextInput(name="db_path", display_name="DB Path", value=DEFAULT_DB, advanced=True),
        SecretStrInput(name="google_api_key", display_name="Google API Key", required=False, advanced=True),
    ]
    outputs = [Output(display_name="Research Result", name="research_result", method="run_research")]

    def run_research(self) -> Data:
        db = self.db_path or DEFAULT_DB
        car_id = getattr(self, "car_id", None)
        attribute = str(getattr(self, "attribute", "") or "general_spec").strip().lower()
        user_query = str(getattr(self, "query", "") or "").strip()
        user_trim = str(getattr(self, "trim_variant", "") or "").strip().lower()

        car_data = get_car(db, int(car_id)) if car_id else None
        brand = str(getattr(self, "brand", "") or (car_data.get("brand") if car_data else "")).strip()
        model = str(getattr(self, "model", "") or (car_data.get("model") if car_data else "")).strip()
        year = getattr(self, "year", None) or (car_data.get("year") if car_data else None)

        matched_obj = None
        if car_data:
            matched_obj = {
                "id": car_data.get("id"),
                "brand": car_data.get("brand"),
                "model": car_data.get("model"),
                "year": car_data.get("year"),
                "condition": car_data.get("condition"),
                "source_url": car_data.get("source_url"),
            }

        # ---------------------------------------------------------------------
        # 1. Exact Variant Matching Rule
        # ---------------------------------------------------------------------
        # If user requests e.g. 1.6L when car in DB is 2.0L or vice-versa
        combined_text = f"{user_query} {user_trim}".lower()
        if car_data:
            car_desc = f"{car_data.get('model', '')} {car_data.get('variant', '')} {car_data.get('description', '')}".lower()
            # Check displacement mismatch
            req_16 = bool(re.search(r"\b1\.6\b|\b1600\b", combined_text))
            req_20 = bool(re.search(r"\b2\.0\b|\b2000\b", combined_text))
            car_16 = bool(re.search(r"\b1\.6\b|\b1600\b", car_desc))
            car_20 = bool(re.search(r"\b2\.0\b|\b2000\b", car_desc))

            if req_16 and car_20 and not car_16:
                res_payload = {
                    "status": "ambiguous",
                    "car_matched": matched_obj,
                    "attribute": attribute,
                    "fact_value": None,
                    "confidence": 0.0,
                    "source_tier": TIER_4_DEALER,
                    "source_url": car_data.get("source_url"),
                    "notes": "حسب البيانات المتاحة عندي، هناك تعارض في سعة المحرك (المطلوب محرك 1.6L بينما السيارة المسجلة بمحرك 2.0L). يرجى توضيح الفئة المطلوبة.",
                }
                self.status = res_payload
                return Data(data=res_payload)

            if req_20 and car_16 and not car_20:
                res_payload = {
                    "status": "ambiguous",
                    "car_matched": matched_obj,
                    "attribute": attribute,
                    "fact_value": None,
                    "confidence": 0.0,
                    "source_tier": TIER_4_DEALER,
                    "source_url": car_data.get("source_url"),
                    "notes": "حسب البيانات المتاحة عندي، هناك تعارض في سعة المحرك (المطلوب محرك 2.0L بينما السيارة المسجلة بمحرك 1.6L). يرجى توضيح الفئة المطلوبة.",
                }
                self.status = res_payload
                return Data(data=res_payload)

        # ---------------------------------------------------------------------
        # 2. Tier 2: Check Local Catalog & Knowledge Documents
        # ---------------------------------------------------------------------
        search_kw = f"{brand} {model} {attribute}".strip()
        local_docs = lexical_knowledge_search(db, search_kw, limit=2) if db else []
        for doc in local_docs:
            content = doc.get("content", "")
            clean_content, injected = sanitize_and_check_injection(content)
            if injected:
                continue
            fact = extract_numeric_fact(attribute, clean_content)
            if fact:
                res_payload = {
                    "status": "found",
                    "car_matched": matched_obj,
                    "attribute": attribute,
                    "fact_value": fact,
                    "confidence": 0.95,
                    "source_tier": TIER_2_CATALOG,
                    "source_url": f"knowledge://{doc.get('title', 'catalog_doc')}",
                    "notes": f"حسب البيانات المتاحة عندي ضمن الوثائق المستوردة، {attribute} هي {fact}.",
                }
                self.status = res_payload
                return Data(data=res_payload)

        # ---------------------------------------------------------------------
        # 3. Web Research: Bounded to 3 Queries, 3 Sources
        # ---------------------------------------------------------------------
        fact_found: str | None = None
        source_found_tier = TIER_5_GENERAL
        source_found_url: str | None = car_data.get("source_url") if car_data else None
        confidence = 0.0

        # Construct 3 targeted queries across Tier 3 (Official/Brand), Tier 4 (ContactCars/Hatla2ee), Tier 5 (General)
        queries = [
            # Tier 4 Dealer Portal
            (TIER_4_DEALER, f"{brand} {model} {year or ''} {attribute} site:contactcars.com"),
            # Tier 3 Brand Egypt
            (TIER_3_BRAND, f"{brand} Egypt official {model} {attribute} مواصفات"),
            # Tier 5 General
            (TIER_5_GENERAL, f"{brand} {model} {year or ''} مواصفات {attribute}"),
        ]

        query_count = 0
        max_queries = 3

        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                for tier, q_str in queries:
                    if query_count >= max_queries or fact_found:
                        break
                    query_count += 1
                    try:
                        results = list(ddgs.text(q_str, max_results=3))
                        for r in results:
                            snippet = r.get("body", "") + " " + r.get("title", "")
                            url = r.get("href", "")
                            clean_snippet, injected = sanitize_and_check_injection(snippet)
                            if injected:
                                logger.warning("[RESEARCH] Prompt injection pattern detected in external snippet, skipping.")
                                continue

                            extracted = extract_numeric_fact(attribute, clean_snippet)
                            if extracted:
                                fact_found = extracted
                                source_found_tier = tier
                                source_found_url = url
                                confidence = 0.88 if tier in (TIER_3_BRAND, TIER_4_DEALER) else 0.75
                                break
                    except Exception as e:
                        logger.warning("[RESEARCH] Web query '%s' error: %s", q_str, e)
        except Exception as e:
            logger.warning("[RESEARCH] DDGS initialization failed: %s", e)

        # ---------------------------------------------------------------------
        # 4. Result Formatting & Honest Phrasing
        # ---------------------------------------------------------------------
        if fact_found:
            res_payload = {
                "status": "found",
                "car_matched": matched_obj,
                "attribute": attribute,
                "fact_value": fact_found,
                "confidence": confidence,
                "source_tier": source_found_tier,
                "source_url": source_found_url,
                "notes": f"حسب البيانات المتاحة عندي ضمن المصادر المسجلة ({source_found_tier})، مواصفة {attribute} لطراز {brand} {model} هي {fact_found}.",
            }
        else:
            # Explicit not-found behavior, particularly for airbags
            res_payload = {
                "status": "not_found",
                "car_matched": matched_obj,
                "attribute": attribute,
                "fact_value": None,
                "confidence": 0.0,
                "source_tier": source_found_tier,
                "source_url": source_found_url,
                "notes": f"حسب البيانات المتاحة عندي، لم يُذكر عدد أو تفاصيل {attribute} بشكل دقيق ومؤكد لهذا الطراز في المصادر المعتمدة.",
            }

        self.status = res_payload
        return Data(data=res_payload)
