#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "components/car_dealership/input_guardrails.py",
    "components/car_dealership/gemini_sales_agent.py",
    "components/car_dealership/search_used_cars.py",
    "components/car_dealership/get_car_details.py",
    "components/car_dealership/compare_cars.py",
    "components/car_dealership/knowledge_rag.py",
    "components/car_dealership/create_test_drive.py",
    "components/car_dealership/create_sales_lead.py",
    "components/car_dealership/output_guardrails.py",
    "lib/car_dealership_core.py",
    "scripts/build_and_install_flow.py",
    "scripts/prepare_db.py",
    "scripts/smoke_test.py",
    "data/sample_used_cars.csv",
    "README.md",
]


def main():
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    syntax = {}
    py_files = []
    for folder in ("components", "lib", "scripts", "tests"):
        d = ROOT / folder
        if d.is_dir():
            py_files.extend(d.rglob("*.py"))
    for path in sorted(py_files):
        if "__pycache__" in path.parts:
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            syntax[str(path.relative_to(ROOT))] = "PASS"
        except SyntaxError as exc:
            syntax[str(path.relative_to(ROOT))] = f"FAIL: {exc}"
    hashes = {}
    for rel in REQUIRED:
        p = ROOT / rel
        if p.exists() and p.is_file():
            hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    runtime_dir = Path(os.getenv("CAR_DEALERSHIP_DB", str(ROOT / "runtime" / "car_dealership.db"))).parent
    runtime_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = runtime_dir / "smoke_test_report.json"
    if not smoke_path.exists():
        smoke_path = ROOT / "runtime" / "smoke_test_report.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.exists() else None
    report = {
        "status": "PASS" if not missing and all(v == "PASS" for v in syntax.values()) and smoke and smoke.get("status") == "PASS" else "FAIL",
        "missing": missing,
        "python_syntax": syntax,
        "smoke_test": smoke,
        "sha256": hashes,
    }
    out = runtime_dir / "package_validation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        if runtime_dir != ROOT / "runtime":
            (ROOT / "runtime").mkdir(parents=True, exist_ok=True)
            (ROOT / "runtime" / "package_validation_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    except Exception:
        pass
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
