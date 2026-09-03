#!/usr/bin/env python3
"""Download the exact user-selected Kaggle dataset and locate its CSV.
Requires kagglehub and network access. Public dataset; authentication may be needed depending on Kaggle environment.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parents[1]
cache_dir = Path(kagglehub.dataset_download("volkanastasia/dataset-of-used-cars"))
csvs = sorted(cache_dir.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
if not csvs:
    raise SystemExit(f"No CSV found in Kaggle download: {cache_dir}")
out = ROOT / "data" / "kaggle_used_cars.csv"
shutil.copy2(csvs[0], out)
print(f"Downloaded dataset to: {cache_dir}")
print(f"Selected CSV: {csvs[0]}")
print(f"Copied to: {out}")
print("Next: python scripts/prepare_db.py data/kaggle_used_cars.csv --db runtime/car_dealership.db")
