#!/usr/bin/env python3
"""Build and validate an Egyptian NEW + USED car catalog from licensed datasets.

The builder starts with the user-selected Kaggle sources, inspects their real
headers, normalizes rows, removes genuine duplicate NEW records, and records a
shortfall instead of fabricating data. ContactCars is never crawled by this
script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import kagglehub

USED_SOURCE = {
    "name": "Kaggle - Egyptian Used Car Price Dataset",
    "url": "https://www.kaggle.com/datasets/alyahmedts13/egyptian-used-car-price-dataset/data",
    "license": "Apache 2.0",
    "slug": "alyahmedts13-egyptian-used",
}

NEW_SOURCES = [
    {
        "dataset": "noortariq20/egyptian-car-market-dataset",
        "download_path": None,
        "preferred_files": {"cleaned_cars_data.csv"},
        "name": "Kaggle - Egyptian Car Market Dataset",
        "url": "https://www.kaggle.com/datasets/noortariq20/egyptian-car-market-dataset/data",
        "license": "MIT",
        "slug": "noortariq20-egypt-market",
        "all_rows_new": False,
    },
    {
        "dataset": "shamsfathalla/egyptian-automotive-market-new-and-used-cars",
        "download_path": "new_cars.csv",
        "preferred_files": {"new_cars.csv"},
        "name": "Kaggle - Egyptian Automotive Market: New & Used Cars",
        "url": "https://www.kaggle.com/datasets/shamsfathalla/egyptian-automotive-market-new-and-used-cars/data",
        "license": "CC BY-NC 4.0",
        "slug": "shamsfathalla-egypt-automotive",
        "all_rows_new": True,
    },
]

BASE_COLUMNS = [
    "Brand",
    "Model",
    "Kilometers",
    "Year",
    "Fuel Type",
    "Transmission Type",
    "Engine Capacity (CC)",
    "Body Type",
    "Price_EGP",
]
EXTRA_COLUMNS = [
    "Condition",
    "Trim",
    "Location",
    "Origin",
    "Horsepower",
    "Color",
    "Source",
    "Source_URL",
    "Source_ID",
    "Collected_At",
]
OUTPUT_COLUMNS = BASE_COLUMNS + EXTRA_COLUMNS

ALIASES = {
    "Brand": ["brand", "make", "manufacturer", "company", "manufacturer_brand", "ماركة", "الماركة", "الصانع"],
    # car_name is preferred over model_name because the Noor source uses
    # model_name for a raw title that also contains brand and year.
    "Model": ["model", "car_name", "car model", "car_model", "model_name", "name", "الموديل", "موديل", "الطراز"],
    "Kilometers": ["kilometers", "kilometres", "km", "mileage", "distance", "kms_driven", "عداد", "المسافة", "الكيلومترات"],
    "Year": ["year", "year_model", "model_year", "manufacturing_year", "manufacture_year", "production_year", "السنة", "سنة"],
    "Fuel Type": ["fuel type", "fuel_type", "fuel", "نوع الوقود", "الوقود"],
    "Transmission Type": ["transmission type", "transmission_type", "transmission", "gearbox", "gear_box", "type", "ناقل الحركة", "الفتيس"],
    "Engine Capacity (CC)": ["engine capacity (cc)", "engine_capacity", "engine capacity", "engine_cc", "cc", "engine displacement", "engine_displacement", "سعة المحرك", "المحرك"],
    "Body Type": ["body type", "body_type", "body", "category", "car_type", "نوع الهيكل", "الهيكل"],
    "Price_EGP": ["price_egp", "price egp", "price", "selling_price", "market_price", "official_price", "السعر"],
    "Condition": ["condition", "car_condition", "vehicle_condition", "status", "الحالة", "حالة السيارة"],
    "Trim": ["trim", "variant", "version", "grade", "الفئة", "الفئه"],
    "Location": ["location", "city", "governorate", "area", "الموقع", "المدينة", "المحافظة"],
    "Origin": ["origin", "country", "country_of_origin", "بلد المنشأ", "المنشأ"],
    "Horsepower": ["horsepower", "engine_power", "hp", "power", "القوة", "حصان"],
    "Color": ["color", "colour", "exterior_color", "اللون"],
}

NEW_MARKERS = {"new", "brand new", "new car", "zero", "0 km", "0km", "جديدة", "جديد", "زيرو", "صفر"}
USED_MARKERS = {"used", "pre owned", "preowned", "second hand", "مستعملة", "مستعمل"}
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_key(value: Any) -> str:
    text = str(value or "").replace("\ufeff", "").strip().casefold().translate(ARABIC_DIGITS)
    text = re.sub(r"[_\-]+", " ", text)
    return re.sub(r"\s+", " ", text)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").strip()
    if text.casefold() in {"nan", "none", "null", "n/a", "na", "-"}:
        return ""
    return re.sub(r"\s+", " ", text)


def numeric_text(value: Any) -> str:
    text = clean_text(value).translate(ARABIC_DIGITS)
    if not text:
        return ""
    match = re.search(r"-?\d+(?:[,.]\d+)*", text.replace(" ", ""))
    if not match:
        return ""
    raw = match.group(0)
    if "," in raw and "." not in raw:
        parts = raw.split(",")
        raw = "".join(parts) if all(len(part) == 3 for part in parts[1:]) else raw.replace(",", ".")
    else:
        raw = raw.replace(",", "")
    try:
        number = float(raw)
    except ValueError:
        return ""
    return str(int(number)) if number.is_integer() else str(number)


def normalize_condition(value: Any) -> str:
    key = clean_key(value)
    if key in {clean_key(marker) for marker in NEW_MARKERS}:
        return "new"
    if key in {clean_key(marker) for marker in USED_MARKERS}:
        return "used"
    return ""


def normalize_fuel(value: Any) -> str:
    key = clean_key(value)
    if not key:
        return ""
    if "natural gas" in key or "cng" in key or "غاز طبيعي" in key:
        return "Natural Gas"
    if key in {"gas", "gasoline", "petrol", "benzine", "benzin", "بنزين"}:
        return "Gasoline"
    if "diesel" in key or "ديزل" in key or "سولار" in key:
        return "Diesel"
    if "hybrid" in key or "هجين" in key:
        return "Hybrid"
    if "electric" in key or "كهرب" in key:
        return "Electric"
    return clean_text(value)


def normalize_transmission(value: Any) -> str:
    key = clean_key(value)
    if not key:
        return ""
    if any(marker in key for marker in ("automatic", "auto", "cvt", "dct", "أوتوماتيك", "اوتوماتيك")):
        return "Automatic"
    if "manual" in key or "مانيوال" in key or "يدوي" in key:
        return "Manual"
    return clean_text(value)


def header_map(headers: list[str]) -> dict[str, str]:
    normalized = {clean_key(header): header for header in headers}
    mapping: dict[str, str] = {}
    for target, aliases in ALIASES.items():
        for alias in [target, *aliases]:
            match = normalized.get(clean_key(alias))
            if match:
                mapping[target] = match
                break
    return mapping


def read_csv_flexible(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1256", "latin1"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                sample = handle.read(8192)
                handle.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                except csv.Error:
                    dialect = csv.excel
                reader = csv.DictReader(handle, dialect=dialect)
                if not reader.fieldnames:
                    continue
                return list(reader.fieldnames), list(reader), encoding
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path}: {last_error}")


def source_id(slug: str, source_file: str, index: int, raw: dict[str, Any]) -> str:
    payload = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{slug}|{source_file}|{index}|{payload}".encode("utf-8")).hexdigest()[:24]
    return f"{slug}:{digest}"


def normalize_row(
    raw: dict[str, Any],
    mapping: dict[str, str],
    *,
    condition: str,
    source: dict[str, Any],
    source_file: str,
    index: int,
    collected_at: str,
) -> dict[str, str]:
    row = {column: "" for column in OUTPUT_COLUMNS}
    for column in BASE_COLUMNS + ["Trim", "Location", "Origin", "Horsepower", "Color"]:
        source_column = mapping.get(column)
        if source_column:
            row[column] = clean_text(raw.get(source_column))

    for column in ("Kilometers", "Year", "Engine Capacity (CC)", "Price_EGP", "Horsepower"):
        row[column] = numeric_text(row[column])
    row["Fuel Type"] = normalize_fuel(row["Fuel Type"])
    row["Transmission Type"] = normalize_transmission(row["Transmission Type"])
    row["Condition"] = condition
    row["Source"] = f"{source['name']} ({source['license']})"
    row["Source_URL"] = source["url"]
    row["Source_ID"] = source_id(source["slug"], source_file, index, raw)
    row["Collected_At"] = collected_at
    return row


def positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def usable_new(row: dict[str, str]) -> bool:
    if not (row["Brand"] and row["Model"] and positive_number(row["Year"]) and positive_number(row["Price_EGP"])):
        return False
    year = int(float(row["Year"]))
    return 1886 <= year <= datetime.now(timezone.utc).year + 2


def completeness(row: dict[str, str]) -> int:
    return sum(bool(clean_text(row.get(column))) for column in OUTPUT_COLUMNS)


def dedupe_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        clean_key(row.get(column, ""))
        for column in (
            "Brand",
            "Model",
            "Trim",
            "Year",
            "Price_EGP",
            "Location",
            "Transmission Type",
            "Engine Capacity (CC)",
            "Body Type",
            "Color",
        )
    )


def deduplicate_new(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    best: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = dedupe_key(row)
        previous = best.get(key)
        if previous is None or completeness(row) > completeness(previous):
            best[key] = row
    unique = list(best.values())
    unique.sort(
        key=lambda row: (
            int(float(row["Year"] or 0)),
            completeness(row),
            float(row["Price_EGP"] or 0),
            row["Source_ID"],
        ),
        reverse=True,
    )
    return unique, len(rows) - len(unique)


def inspect_new_source(source: dict[str, Any], collected_at: str) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    downloaded = Path(
        kagglehub.dataset_download(source["dataset"], path=source["download_path"])
        if source["download_path"]
        else kagglehub.dataset_download(source["dataset"])
    )
    csv_files = [downloaded] if downloaded.is_file() else sorted(downloaded.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {downloaded}")

    candidates: list[dict[str, str]] = []
    diagnostics: list[dict[str, Any]] = []
    for path in csv_files:
        headers, raw_rows, encoding = read_csv_flexible(path)
        mapping = header_map(headers)
        selected = not source["preferred_files"] or path.name in source["preferred_files"]
        condition_column = mapping.get("Condition")
        condition_counts = Counter(
            "new" if source["all_rows_new"] else (normalize_condition(raw.get(condition_column, "")) if condition_column else "")
            for raw in raw_rows
        )
        usable_count = 0
        rejected_count = 0
        if selected:
            for index, raw in enumerate(raw_rows, 1):
                condition = "new" if source["all_rows_new"] else normalize_condition(raw.get(condition_column, "") if condition_column else "")
                if condition != "new":
                    continue
                row = normalize_row(
                    raw,
                    mapping,
                    condition="new",
                    source=source,
                    source_file=path.name,
                    index=index,
                    collected_at=collected_at,
                )
                if usable_new(row):
                    candidates.append(row)
                    usable_count += 1
                else:
                    rejected_count += 1
        diagnostics.append(
            {
                "dataset": source["dataset"],
                "dataset_url": source["url"],
                "license": source["license"],
                "file": str(path),
                "selected_for_normalization": selected,
                "encoding": encoding,
                "rows": len(raw_rows),
                "columns": headers,
                "mapped_columns": mapping,
                "condition_counts": dict(condition_counts),
                "usable_new_rows": usable_count,
                "rejected_new_rows": rejected_count,
            }
        )
    return candidates, diagnostics


def signature_duplicate_count(rows: list[dict[str, str]]) -> int:
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for row in rows:
        key = (clean_key(row.get("Condition")), *dedupe_key(row))
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def numeric_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            if clean_text(row.get(column)):
                values.append(float(row[column]))
        except (TypeError, ValueError):
            continue
    return values


def validate_output(path: Path, target_used: int, target_new: int) -> dict[str, Any]:
    headers, rows, encoding = read_csv_flexible(path)
    conditions = [clean_key(row.get("Condition")) for row in rows]
    used_count = conditions.count("used")
    new_count = conditions.count("new")
    invalid_conditions = sum(condition not in {"used", "new"} for condition in conditions)
    source_ids = [clean_text(row.get("Source_ID")) for row in rows]
    years = numeric_values(rows, "Year")
    prices = numeric_values(rows, "Price_EGP")
    mileages = numeric_values(rows, "Kilometers")
    duplicate_rows = signature_duplicate_count(rows)
    replacement_characters = sum("\ufffd" in clean_text(value) for row in rows for value in row.values())
    missing = {
        column: sum(not clean_text(row.get(column)) for row in rows)
        for column in ("Brand", "Model", "Year", "Price_EGP", "Source", "Source_URL", "Source_ID", "Collected_At")
    }
    return {
        "csv_readable": True,
        "encoding": encoding,
        "columns_match": headers == OUTPUT_COLUMNS,
        "total_rows": len(rows),
        "used_rows": used_count,
        "new_rows": new_count,
        "target_used_rows": target_used,
        "target_new_rows": target_new,
        "unique_source_id_count": len(set(source_ids)),
        "missing_source_id_count": source_ids.count(""),
        "duplicate_signature_rows": duplicate_rows,
        "duplicate_rate": round(duplicate_rows / len(rows), 8) if rows else 0.0,
        "missing": missing,
        "invalid_condition_rows": invalid_conditions,
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "price_egp_min": min(prices) if prices else None,
        "price_egp_max": max(prices) if prices else None,
        "negative_price_rows": sum(value < 0 for value in prices),
        "negative_mileage_rows": sum(value < 0 for value in mileages),
        "replacement_character_cells": replacement_characters,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--used", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-used", type=int, default=8374)
    parser.add_argument("--target-new", type=int, default=8374)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    used_path = Path(args.used)
    output_path = Path(args.out)
    report_path = Path(args.report) if args.report else output_path.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    collected_at = now_iso()

    used_headers, used_raw, used_encoding = read_csv_flexible(used_path)
    used_mapping = header_map(used_headers)
    used_rows = [
        normalize_row(
            raw,
            used_mapping,
            condition="used",
            source=USED_SOURCE,
            source_file=used_path.name,
            index=index,
            collected_at=collected_at,
        )
        for index, raw in enumerate(used_raw, 1)
    ]
    print(f"USED rows preserved: {len(used_rows):,}", flush=True)

    raw_new: list[dict[str, str]] = []
    diagnostics: list[dict[str, Any]] = []
    source_errors: list[dict[str, str]] = []
    for source in NEW_SOURCES:
        print(f"Inspecting NEW source: {source['dataset']}", flush=True)
        try:
            rows, source_diagnostics = inspect_new_source(source, collected_at)
            raw_new.extend(rows)
            diagnostics.extend(source_diagnostics)
            print(f"  usable NEW rows from selected files: {len(rows):,}", flush=True)
        except Exception as exc:
            source_errors.append({"dataset": source["dataset"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    unique_new, removed_new_duplicates = deduplicate_new(raw_new)
    selected_new = unique_new[: args.target_new]
    all_rows = used_rows + selected_new
    write_csv(output_path, all_rows)
    validation = validate_output(output_path, args.target_used, args.target_new)

    fatal_validation = any(
        (
            not validation["csv_readable"],
            not validation["columns_match"],
            validation["used_rows"] != args.target_used,
            validation["invalid_condition_rows"] > 0,
            validation["negative_price_rows"] > 0,
            validation["negative_mileage_rows"] > 0,
            validation["replacement_character_cells"] > 0,
        )
    )
    if fatal_validation:
        status = "validation_failed"
    elif validation["new_rows"] < args.target_new:
        status = "insufficient_new_rows"
    elif validation["total_rows"] != args.target_used + args.target_new:
        status = "validation_failed"
    else:
        status = "ok"

    report = {
        "status": status,
        "built_at": collected_at,
        "used_source": USED_SOURCE,
        "used_input": str(used_path),
        "used_input_rows": len(used_rows),
        "used_encoding": used_encoding,
        "used_mapping": used_mapping,
        "new_sources": [
            {key: value for key, value in source.items() if key not in {"preferred_files", "all_rows_new"}}
            for source in NEW_SOURCES
        ],
        "source_files": diagnostics,
        "source_errors": source_errors,
        "new_rows_before_deduplication": len(raw_new),
        "new_duplicate_rows_removed": removed_new_duplicates,
        "usable_unique_new_rows": len(unique_new),
        "selected_new_rows": len(selected_new),
        "missing_new_rows": max(0, args.target_new - len(selected_new)),
        "output": str(output_path),
        "output_columns": OUTPUT_COLUMNS,
        "validation": validation,
        "license_notice": "The Shams Fathalla source is CC BY-NC 4.0 and must not be used commercially without separate permission.",
        "contactcars_bulk_crawling": "NOT PERFORMED",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Unique genuine NEW rows available: {len(unique_new):,}", flush=True)
    print(f"Combined output rows: {len(all_rows):,}", flush=True)
    print(f"Status: {status}", flush=True)
    print(f"CSV: {output_path}", flush=True)
    print(f"Report: {report_path}", flush=True)
    if status == "insufficient_new_rows":
        print(
            f"Shortfall: {report['missing_new_rows']:,} NEW rows. No rows were fabricated or multiplied.",
            file=sys.stderr,
        )
        return 3
    return 0 if status == "ok" else 4


if __name__ == "__main__":
    raise SystemExit(main())
