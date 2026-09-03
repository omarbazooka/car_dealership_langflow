#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from car_dealership_core import import_cars_csv, init_db  # noqa: E402

parser = argparse.ArgumentParser(description="Import the normalized Egypt NEW+USED car CSV into the prototype SQLite database.")
parser.add_argument("csv", nargs="?", default=str(ROOT / "data" / "sample_used_cars.csv"))
parser.add_argument("--db", default=str(ROOT / "runtime" / "car_dealership.db"))
parser.add_argument("--append", action="store_true")
args = parser.parse_args()
init_db(args.db)
result = import_cars_csv(args.csv, args.db, replace=not args.append)
print(result)
