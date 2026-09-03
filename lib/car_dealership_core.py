from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DB = os.getenv("CAR_DEALERSHIP_DB", "/data/car_dealership.db")

ALIASES = {
    "brand": ["brand", "make", "manufacturer", "car_brand", "company"],
    "model": ["model", "car_model", "name"],
    "price": ["price", "price_egp", "egp_price", "usd_price", "price_usd", "cost", "official_price", "market_price"],
    "city": ["city", "location", "sale_city", "city_of_sale", "governorate", "area"],
    "fuel": ["fuel", "fuel_type", "fueltype"],
    "transmission": ["transmission", "transmission_type", "gearbox", "gear_box"],
    "drive": ["drive", "drive_type", "drivetrain", "drive_train"],
    "mileage": ["mileage", "miles", "km", "kilometers", "odometer"],
    "origin": ["origin", "country", "country_of_origin", "made_in"],
    "engine_capacity": ["engine_capacity", "engine_capacity_cc", "enginecapacity", "engine_size", "engine_volume", "capacity"],
    "engine_power": ["engine_power", "enginepower", "horsepower", "hp", "power"],
    "age": ["age", "car_age", "vehicle_age"],
    "year": ["year", "model_year", "production_year", "manufacturing_year", "manufacture_year", "year_model"],
    "condition": ["condition", "car_status", "vehicle_condition", "new_or_used", "car_condition", "status"],
    "body_type": ["body_type", "body_shape", "body", "vehicle_type"],
    "trim": ["trim", "variant", "grade", "class"],
    "color": ["color", "colour", "exterior_color"],
    "source": ["source", "data_source"],
    "source_url": ["source_url", "listing_url", "detail_link", "url"],
    "source_id": ["source_id", "listing_id", "ad_id"],
    "collected_at": ["collected_at", "collected", "scraped_at", "downloaded_at"],
}

CAR_EXTRA_COLUMNS = {
    "condition": "TEXT",
    "body_type": "TEXT",
    "trim": "TEXT",
    "color": "TEXT",
    "source": "TEXT",
    "source_url": "TEXT",
    "source_id": "TEXT",
    "collected_at": "TEXT",
    "catalog_active": "INTEGER NOT NULL DEFAULT 1",
}

SESSION_EXTRA_COLUMNS = {
    "selected_snapshot_id": "INTEGER",
    "selected_position": "INTEGER",
}

TEST_DRIVE_EXTRA_COLUMNS = {
    "cancelled_at": "TEXT",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect(db_path: str = DEFAULT_DB) -> Iterable[sqlite3.Connection]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 60000")
    try:
        yield conn
        for attempt in range(5):
            try:
                conn.commit()
                break
            except sqlite3.OperationalError:
                if attempt == 4:
                    raise
                import time
                time.sleep(0.08 * (attempt + 1))
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


def _ensure_extra_columns(conn: sqlite3.Connection) -> None:
    # cars
    existing_cars = {row[1] for row in conn.execute("PRAGMA table_info(cars)").fetchall()}
    for name, sql_type in CAR_EXTRA_COLUMNS.items():
        if name not in existing_cars:
            conn.execute(f"ALTER TABLE cars ADD COLUMN {name} {sql_type}")

    # conversation_sessions
    existing_sessions = {row[1] for row in conn.execute("PRAGMA table_info(conversation_sessions)").fetchall()}
    for name, sql_type in SESSION_EXTRA_COLUMNS.items():
        if name not in existing_sessions:
            conn.execute(f"ALTER TABLE conversation_sessions ADD COLUMN {name} {sql_type}")

    # test_drive_requests
    existing_td = {row[1] for row in conn.execute("PRAGMA table_info(test_drive_requests)").fetchall()}
    for name, sql_type in TEST_DRIVE_EXTRA_COLUMNS.items():
        if name not in existing_td:
            conn.execute(f"ALTER TABLE test_drive_requests ADD COLUMN {name} {sql_type}")


def init_db(db_path: str = DEFAULT_DB) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand TEXT,
                model TEXT,
                price REAL,
                city TEXT,
                fuel TEXT,
                transmission TEXT,
                drive TEXT,
                mileage REAL,
                origin TEXT,
                engine_capacity REAL,
                engine_power REAL,
                age REAL,
                year INTEGER,
                condition TEXT,
                body_type TEXT,
                trim TEXT,
                color TEXT,
                source TEXT,
                source_url TEXT,
                source_id TEXT,
                collected_at TEXT,
                catalog_active INTEGER NOT NULL DEFAULT 1,
                source_row TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS test_drive_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                car_id INTEGER NOT NULL,
                preferred_date TEXT NOT NULL,
                preferred_time TEXT NOT NULL,
                notes TEXT,
                status TEXT NOT NULL DEFAULT 'NEW',
                cancelled_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(car_id) REFERENCES cars(id)
            );

            CREATE TABLE IF NOT EXISTS sales_leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                email TEXT,
                car_id INTEGER,
                notes TEXT,
                status TEXT NOT NULL DEFAULT 'NEW',
                created_at TEXT NOT NULL,
                FOREIGN KEY(car_id) REFERENCES cars(id)
            );

            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                summary TEXT NOT NULL DEFAULT '',
                summarized_until_message_id INTEGER,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_sessions (
                session_id TEXT PRIMARY KEY,
                condition TEXT,
                min_price REAL,
                max_price REAL,
                brand TEXT,
                model TEXT,
                body_type TEXT,
                fuel_type TEXT,
                transmission TEXT,
                min_year INTEGER,
                max_year INTEGER,
                location TEXT,
                max_mileage REAL,
                last_recommended_car_ids TEXT,
                selected_car_id INTEGER,
                selected_snapshot_id INTEGER,
                selected_position INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pending_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                entity_id INTEGER,
                payload_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS recommendation_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sequence_no INTEGER NOT NULL,
                criteria_json TEXT NOT NULL DEFAULT '{}',
                items_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        _ensure_extra_columns(conn)
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_cars_brand ON cars(brand);
            CREATE INDEX IF NOT EXISTS idx_cars_model ON cars(model);
            CREATE INDEX IF NOT EXISTS idx_cars_price ON cars(price);
            CREATE INDEX IF NOT EXISTS idx_cars_fuel ON cars(fuel);
            CREATE INDEX IF NOT EXISTS idx_cars_transmission ON cars(transmission);
            CREATE INDEX IF NOT EXISTS idx_cars_condition ON cars(condition);
            CREATE INDEX IF NOT EXISTS idx_cars_year ON cars(year);
            CREATE INDEX IF NOT EXISTS idx_cars_body_type ON cars(body_type);
            CREATE INDEX IF NOT EXISTS idx_cars_city ON cars(city);
            CREATE INDEX IF NOT EXISTS idx_cars_mileage ON cars(mileage);
            CREATE INDEX IF NOT EXISTS idx_cars_source_id ON cars(source_id);
            CREATE INDEX IF NOT EXISTS idx_cars_catalog_active ON cars(catalog_active);
            CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation_messages(session_id, id);
            CREATE INDEX IF NOT EXISTS idx_conversation_session_created ON conversation_messages(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_summaries_session ON conversation_summaries(session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_session ON conversation_sessions(session_id);
            CREATE INDEX IF NOT EXISTS idx_pending_actions_session ON pending_actions(session_id);
            CREATE INDEX IF NOT EXISTS idx_pending_actions_session_status ON pending_actions(session_id, status);
            CREATE INDEX IF NOT EXISTS idx_rec_snapshots_session_seq ON recommendation_snapshots(session_id, sequence_no);
            CREATE INDEX IF NOT EXISTS idx_rec_snapshots_session_status ON recommendation_snapshots(session_id, status);
            """
        )


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "n/a", "na", "-", "negotiable"}:
        return None
    m = re.search(r"-?\d+(?:[.,]\d+)*", s.replace(" ", ""))
    if not m:
        return None
    raw = m.group(0)
    if raw.count(",") > 0 and raw.count(".") == 0:
        parts = raw.split(",")
        raw = "".join(parts) if all(len(p) == 3 for p in parts[1:]) else raw.replace(",", ".")
    else:
        raw = raw.replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "n/a", "na"}:
        return None
    return s


def _find_columns(fieldnames: Iterable[str]) -> dict[str, str | None]:
    normalized = {_norm(c): c for c in fieldnames}
    mapping: dict[str, str | None] = {}
    for target, aliases in ALIASES.items():
        match = None
        for alias in aliases:
            if _norm(alias) in normalized:
                match = normalized[_norm(alias)]
                break
        mapping[target] = match
    return mapping


def import_cars_csv(csv_path: str, db_path: str = DEFAULT_DB, replace: bool = True) -> dict[str, Any]:
    init_db(db_path)
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(csv_path)

    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    sample = raw[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    reader = csv.DictReader(raw.splitlines(), dialect=dialect)
    if not reader.fieldnames:
        raise ValueError("CSV has no header row")
    mapping = _find_columns(reader.fieldnames)
    if not mapping["brand"] or not mapping["model"]:
        raise ValueError(f"Could not identify brand/model columns. Found: {reader.fieldnames}")

    rows = []
    current_year = datetime.now(timezone.utc).year
    for r in reader:
        year_num = _num(r.get(mapping["year"])) if mapping["year"] else None
        age_num = _num(r.get(mapping["age"])) if mapping["age"] else None
        year = int(year_num) if year_num is not None else None
        condition = _text(r.get(mapping["condition"])) if mapping["condition"] else None
        condition = condition.lower() if condition else None
        if condition in {"zero", "brand new", "brand-new"}:
            condition = "new"
        elif condition in {"preowned", "pre-owned", "second hand", "second-hand"}:
            condition = "used"

        source_row = json.dumps(r, ensure_ascii=False, sort_keys=True)
        source_id = _text(r.get(mapping["source_id"])) if mapping["source_id"] else None
        if not source_id:
            source_id = "csv:" + hashlib.sha256(source_row.encode("utf-8")).hexdigest()[:24]

        item = {
            "brand": _text(r.get(mapping["brand"])) if mapping["brand"] else None,
            "model": _text(r.get(mapping["model"])) if mapping["model"] else None,
            "price": _num(r.get(mapping["price"])) if mapping["price"] else None,
            "city": _text(r.get(mapping["city"])) if mapping["city"] else None,
            "fuel": _text(r.get(mapping["fuel"])) if mapping["fuel"] else None,
            "transmission": _text(r.get(mapping["transmission"])) if mapping["transmission"] else None,
            "drive": _text(r.get(mapping["drive"])) if mapping["drive"] else None,
            "mileage": _num(r.get(mapping["mileage"])) if mapping["mileage"] else None,
            "origin": _text(r.get(mapping["origin"])) if mapping["origin"] else None,
            "engine_capacity": _num(r.get(mapping["engine_capacity"])) if mapping["engine_capacity"] else None,
            "engine_power": _num(r.get(mapping["engine_power"])) if mapping["engine_power"] else None,
            "age": age_num if age_num is not None else (max(0, current_year - year) if year else None),
            "year": year,
            "condition": condition,
            "body_type": _text(r.get(mapping["body_type"])) if mapping["body_type"] else None,
            "trim": _text(r.get(mapping["trim"])) if mapping["trim"] else None,
            "color": _text(r.get(mapping["color"])) if mapping["color"] else None,
            "source": _text(r.get(mapping["source"])) if mapping["source"] else None,
            "source_url": _text(r.get(mapping["source_url"])) if mapping["source_url"] else None,
            "source_id": source_id,
            "collected_at": _text(r.get(mapping["collected_at"])) if mapping["collected_at"] else None,
            "catalog_active": 1,
            "source_row": source_row,
        }
        if item["condition"] == "new" and item["mileage"] is None:
            item["mileage"] = 0.0
        # Preserve source truth, including the small number of authoritative
        # rows with a missing brand or model. Search filters naturally exclude
        # unusable rows when a brand/model is requested.
        if any(_text(value) for value in r.values()):
            rows.append(item)

    with _connect(db_path) as conn:
        if replace:
            # Never delete business actions during a catalog refresh. Keep any
            # cars referenced by an existing request/lead as inactive history;
            # remove only unreferenced catalog rows before loading the new set.
            conn.execute("UPDATE cars SET catalog_active=0")
            conn.execute(
                """DELETE FROM cars
                   WHERE id NOT IN (
                       SELECT car_id FROM test_drive_requests
                       UNION
                       SELECT car_id FROM sales_leads WHERE car_id IS NOT NULL
                   )"""
            )
        conn.executemany(
            """
            INSERT INTO cars
            (brand, model, price, city, fuel, transmission, drive, mileage, origin,
             engine_capacity, engine_power, age, year, condition, body_type, trim, color,
             source, source_url, source_id, collected_at, catalog_active, source_row)
            VALUES
            (:brand, :model, :price, :city, :fuel, :transmission, :drive, :mileage, :origin,
             :engine_capacity, :engine_power, :age, :year, :condition, :body_type, :trim, :color,
             :source, :source_url, :source_id, :collected_at, :catalog_active, :source_row)
            """,
            rows,
        )
        active_rows = conn.execute("SELECT COUNT(*) FROM cars WHERE catalog_active=1").fetchone()[0]
        retained_history = conn.execute("SELECT COUNT(*) FROM cars WHERE catalog_active=0").fetchone()[0]
    return {
        "rows_imported": len(rows),
        "active_catalog_rows": active_rows,
        "retained_referenced_cars": retained_history,
        "columns": reader.fieldnames,
        "mapping": mapping,
        "db_path": db_path,
    }


def _rowdict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d.pop("source_row", None)
    return d


def search_cars(
    db_path: str = DEFAULT_DB,
    *,
    brand: str | None = None,
    model: str | None = None,
    condition: str | None = None,
    body_type: str | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    city: str | None = None,
    fuel: str | None = None,
    transmission: str | None = None,
    drive: str | None = None,
    max_mileage: float | None = None,
    max_age: float | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    init_db(db_path)
    clauses: list[str] = ["catalog_active = 1"]
    params: list[Any] = []

    for col, value in (
        ("brand", brand), ("model", model), ("city", city), ("fuel", fuel),
        ("transmission", transmission), ("drive", drive), ("body_type", body_type),
    ):
        if value:
            clauses.append(f"LOWER({col}) LIKE LOWER(?)")
            params.append(f"%{value.strip()}%")
    if condition:
        clauses.append("LOWER(condition) = LOWER(?)")
        params.append(condition.strip())

    for col, op, value in (
        ("price", ">=", min_price),
        ("price", "<=", max_price),
        ("mileage", "<=", max_mileage),
        ("age", "<=", max_age),
        ("year", ">=", min_year),
        ("year", "<=", max_year),
    ):
        if value is not None:
            clauses.append(f"{col} {op} ?")
            params.append(float(value))

    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    # With a stated budget, rank toward the budget ceiling instead of surfacing
    # anomalously tiny source prices first. Without a budget, prefer newer cars.
    order_by = (
        "(price IS NULL), price DESC, (year IS NULL), year DESC, (mileage IS NULL), mileage ASC"
        if max_price is not None
        else "(year IS NULL), year DESC, (price IS NULL), price ASC, (mileage IS NULL), mileage ASC"
    )
    sql = f"SELECT * FROM cars{where} ORDER BY {order_by} LIMIT ?"
    params.append(max(1, min(int(limit), 20)))
    with _connect(db_path) as conn:
        return [_rowdict(r) for r in conn.execute(sql, params).fetchall()]


def get_car(db_path: str, car_id: int) -> dict[str, Any] | None:
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM cars WHERE id=?", (int(car_id),)).fetchone()
    return _rowdict(row) if row else None


def compare_cars(db_path: str, car_ids: list[int]) -> list[dict[str, Any]]:
    return [c for cid in car_ids if (c := get_car(db_path, cid)) is not None]


def create_test_drive(
    db_path: str,
    customer_name: str,
    phone: str,
    car_id: int,
    preferred_date: str,
    preferred_time: str,
    notes: str | None = None,
) -> dict[str, Any]:
    init_db(db_path)
    car = get_car(db_path, int(car_id))
    if not car:
        raise ValueError(f"Car id {car_id} does not exist")
    if len(re.sub(r"\D", "", phone)) < 7:
        raise ValueError("Phone number is too short")
    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO test_drive_requests
               (customer_name, phone, car_id, preferred_date, preferred_time, notes, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'NEW', ?)""",
            (customer_name.strip(), phone.strip(), int(car_id), preferred_date.strip(), preferred_time.strip(), notes, utc_now()),
        )
        request_id = cur.lastrowid
    return {"request_id": request_id, "status": "NEW", "car": car, "preferred_date": preferred_date, "preferred_time": preferred_time}


def cancel_test_drive(
    db_path: str,
    request_id: int,
    notes: str | None = None,
) -> dict[str, Any]:
    """Cancel a test drive request in the database and record timestamp."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM test_drive_requests WHERE id=?", (int(request_id),)).fetchone()
        if not row:
            raise ValueError(f"Test drive request #{request_id} does not exist")
        now = utc_now()
        existing_notes = row["notes"] or ""
        new_notes = f"{existing_notes} | Cancellation notes: {notes}".strip(" |") if notes else existing_notes
        conn.execute(
            "UPDATE test_drive_requests SET status='CANCELLED', cancelled_at=?, notes=? WHERE id=?",
            (now, new_notes, int(request_id)),
        )
    return {
        "request_id": int(request_id),
        "status": "CANCELLED",
        "cancelled_at": now,
        "car_id": row["car_id"],
        "customer_name": row["customer_name"],
    }


def get_test_drive_requests(
    db_path: str,
    *,
    phone: str | None = None,
    car_id: int | None = None,
    status: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    init_db(db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if phone:
        digits = re.sub(r"\D", "", phone)
        clauses.append("phone LIKE ?")
        params.append(f"%{digits[-8:]}%")
    if car_id is not None:
        clauses.append("car_id = ?")
        params.append(int(car_id))
    if status:
        clauses.append("LOWER(status) = LOWER(?)")
        params.append(status.strip())
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM test_drive_requests{where} ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def build_recommendation_set(cars: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Group duplicate vehicle listings/variants so hidden rows do not consume visible positions."""
    grouped: list[dict[str, Any]] = []
    seen_keys: dict[tuple, int] = {}

    for c in cars:
        # Grouping key: brand, model, year, condition, body_type, transmission, price
        key = (
            (c.get("brand") or "").strip().lower(),
            (c.get("model") or "").strip().lower(),
            c.get("year"),
            (c.get("condition") or "").strip().lower(),
            (c.get("body_type") or "").strip().lower(),
            (c.get("transmission") or "").strip().lower(),
            c.get("price"),
        )
        color = c.get("color")
        car_id = int(c["id"])

        if key in seen_keys:
            idx = seen_keys[key]
            if car_id not in grouped[idx]["variant_ids"]:
                grouped[idx]["variant_ids"].append(car_id)
            if color and color not in grouped[idx]["display_metadata"]["colors"]:
                grouped[idx]["display_metadata"]["colors"].append(color)
        else:
            if len(grouped) >= limit:
                continue
            idx = len(grouped)
            seen_keys[key] = idx
            pos = idx + 1
            brand_str = c.get("brand") or ""
            model_str = c.get("model") or ""
            display_name = f"{brand_str} {model_str}".strip() or f"Car #{car_id}"
            grouped.append({
                "position": pos,
                "primary_car_id": car_id,
                "variant_ids": [car_id],
                "display_name": display_name,
                "brand": brand_str,
                "model": model_str,
                "year": c.get("year"),
                "condition": c.get("condition"),
                "body_type": c.get("body_type"),
                "transmission": c.get("transmission"),
                "price": c.get("price"),
                "display_metadata": {
                    "colors": [color] if color else []
                },
            })

    return grouped



def create_sales_lead(
    db_path: str,
    customer_name: str,
    phone: str,
    car_id: int | None = None,
    email: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    init_db(db_path)
    car = None
    if car_id is not None:
        car = get_car(db_path, int(car_id))
        if not car:
            raise ValueError(f"Car id {car_id} does not exist")
    if len(re.sub(r"\D", "", phone)) < 7:
        raise ValueError("Phone number is too short")
    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO sales_leads
               (customer_name, phone, email, car_id, notes, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'NEW', ?)""",
            (customer_name.strip(), phone.strip(), email.strip() if email else None, car_id, notes, utc_now()),
        )
        lead_id = cur.lastrowid
    return {"lead_id": lead_id, "status": "NEW", "car": car}


def save_message(db_path: str, session_id: str, role: str, content: str) -> None:
    init_db(db_path)
    if role not in {"user", "assistant"}:
        raise ValueError("role must be user or assistant")
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO conversation_messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
            (session_id, role, content, utc_now()),
        )


def load_messages(db_path: str, session_id: str, limit: int = 20) -> list[dict[str, str]]:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT role, content FROM (
                   SELECT id, role, content FROM conversation_messages
                   WHERE session_id=? ORDER BY id DESC LIMIT ?
               ) ORDER BY id ASC""",
            (session_id, max(1, int(limit))),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def sync_knowledge_folder(db_path: str, knowledge_dir: str) -> int:
    init_db(db_path)
    root = Path(knowledge_dir)
    files = sorted([*root.glob("*.md"), *root.glob("*.txt")]) if root.exists() else []
    with _connect(db_path) as conn:
        for p in files:
            content = p.read_text(encoding="utf-8")
            title = next((line.lstrip("# ").strip() for line in content.splitlines() if line.strip()), p.stem)
            conn.execute(
                """INSERT INTO knowledge_documents(source,title,content,updated_at)
                   VALUES(?,?,?,?)
                   ON CONFLICT(source) DO UPDATE SET title=excluded.title, content=excluded.content, updated_at=excluded.updated_at""",
                (str(p.resolve()), title, content, utc_now()),
            )
    return len(files)


def lexical_knowledge_search(db_path: str, query: str, limit: int = 4) -> list[dict[str, Any]]:
    init_db(db_path)
    tokens = [t for t in re.findall(r"[\w\u0600-\u06ff]+", query.lower()) if len(t) > 2]
    with _connect(db_path) as conn:
        docs = conn.execute("SELECT source,title,content FROM knowledge_documents").fetchall()
    scored = []
    for d in docs:
        hay = (d["title"] + " " + d["content"]).lower()
        score = sum(hay.count(t) for t in tokens)
        if score:
            scored.append((score, dict(d)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{**d, "score": score} for score, d in scored[:limit]]


def list_business_actions(db_path: str = DEFAULT_DB) -> dict[str, list[dict[str, Any]]]:
    init_db(db_path)
    with _connect(db_path) as conn:
        td = [dict(r) for r in conn.execute("SELECT * FROM test_drive_requests ORDER BY id").fetchall()]
        leads = [dict(r) for r in conn.execute("SELECT * FROM sales_leads ORDER BY id").fetchall()]
    return {"test_drives": td, "sales_leads": leads}


INJECTION_PATTERNS = [
    r"ignore\s+(all|any|the)?\s*(previous|prior|system)\s+instructions",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(me\s+)?(your\s+)?system\s+(prompt|instructions)",
    r"developer\s+message",
    r"bypass\s+(the\s+)?guardrails",
    r"jailbreak",
]


def guard_input(text: str) -> tuple[bool, str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return False, "Please enter a message."
    if len(cleaned) > 8000:
        return False, "The message is too long for this sales assistant."
    low = cleaned.lower()
    if any(re.search(p, low, re.I) for p in INJECTION_PATTERNS):
        return False, "I can help with cars, dealership information, test drives, and sales inquiries, but I can't follow requests to override or reveal system instructions."
    return True, cleaned


def guard_output(text: str) -> str:
    out = (text or "").strip()
    out = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED_API_KEY]", out)
    out = re.sub(r"sk-[A-Za-z0-9_-]{16,}", "[REDACTED_API_KEY]", out)
    if "Traceback (most recent call last)" in out:
        return "I couldn't complete that action safely. Please try again or choose another dealership action."
    return out
