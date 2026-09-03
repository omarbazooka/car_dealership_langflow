from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sqlite3
from typing import Any, Callable

from car_dealership_core import (
    DEFAULT_DB,
    _connect,
    build_recommendation_set,
    cancel_test_drive,
    create_sales_lead,
    create_test_drive,
    get_car,
    get_test_drive_requests,
    init_db,
    utc_now,
)

logger = logging.getLogger(__name__)

# Centralized memory configuration constants
MEMORY_RECENT_MESSAGE_LIMIT = 12
MEMORY_SUMMARY_TRIGGER = 12
MEMORY_MAX_SUMMARY_CHARS = 4000

TEST_DRIVE_REQUIRED_FIELDS = ("customer_name", "phone", "car_id", "preferred_date", "preferred_time")
SALES_LEAD_REQUIRED_FIELDS = ("customer_name", "phone")

# ContextVar to track the active session across components in the current request
CURRENT_SESSION_ID: contextvars.ContextVar[str] = contextvars.ContextVar("CURRENT_SESSION_ID", default="default-session")


def get_current_session_id() -> str:
    return CURRENT_SESSION_ID.get()


def set_current_session_id(session_id: str) -> None:
    CURRENT_SESSION_ID.set(session_id)


class MemoryManager:
    """Central Memory Manager implementing the Intermediate Hybrid Memory Architecture.
    
    Layers:
    1. Conversation Memory (raw messages, recent window retrieval)
    2. Conversation Summary Memory (incremental compression past threshold)
    3. Structured Session / Working Memory (preferences, recommended IDs, selected car)
    4. Task / Action Memory (multi-turn workflows, missing fields, verified completion)
    """

    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path or DEFAULT_DB
        init_db(self.db_path)

    # -------------------------------------------------------------------------
    # Layer 1: Conversation Memory
    # -------------------------------------------------------------------------

    def save_message(self, session_id: str, role: str, content: str) -> int:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Invalid role: {role}. Must be 'user' or 'assistant'.")
        with _connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO conversation_messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
                (session_id, role, content, utc_now()),
            )
            return int(cur.lastrowid)

    def get_recent_messages(self, session_id: str, limit: int = MEMORY_RECENT_MESSAGE_LIMIT) -> list[dict[str, Any]]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT id, role, content, created_at FROM (
                       SELECT id, role, content, created_at FROM conversation_messages
                       WHERE session_id=? ORDER BY id DESC LIMIT ?
                   ) ORDER BY id ASC""",
                (session_id, max(1, int(limit))),
            ).fetchall()
        return [{"id": r["id"], "role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in rows]

    def get_message_count(self, session_id: str) -> int:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM conversation_messages WHERE session_id=?",
                (session_id,),
            ).fetchone()
            return int(row[0]) if row else 0

    def get_unsummarized_messages(self, session_id: str, up_to_id: int | None = None) -> list[dict[str, Any]]:
        summary_info = self.get_summary(session_id)
        last_summarized_id = summary_info.get("summarized_until_message_id") or 0
        with _connect(self.db_path) as conn:
            if up_to_id is not None:
                rows = conn.execute(
                    """SELECT id, role, content, created_at FROM conversation_messages
                       WHERE session_id=? AND id > ? AND id <= ? ORDER BY id ASC""",
                    (session_id, last_summarized_id, int(up_to_id)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, role, content, created_at FROM conversation_messages
                       WHERE session_id=? AND id > ? ORDER BY id ASC""",
                    (session_id, last_summarized_id),
                ).fetchall()
        return [{"id": r["id"], "role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in rows]

    # -------------------------------------------------------------------------
    # Layer 2: Conversation Summary Memory
    # -------------------------------------------------------------------------

    def get_summary(self, session_id: str) -> dict[str, Any]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT summary, summarized_until_message_id, updated_at FROM conversation_summaries WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if row:
                return {
                    "summary": row["summary"] or "",
                    "summarized_until_message_id": row["summarized_until_message_id"],
                    "updated_at": row["updated_at"],
                }
        return {"summary": "", "summarized_until_message_id": 0, "updated_at": None}

    def update_summary(self, session_id: str, summary: str, summarized_until_message_id: int) -> None:
        trimmed_summary = (summary or "")[:MEMORY_MAX_SUMMARY_CHARS].strip()
        with _connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO conversation_summaries (session_id, summary, summarized_until_message_id, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       summary=excluded.summary,
                       summarized_until_message_id=excluded.summarized_until_message_id,
                       updated_at=excluded.updated_at""",
                (session_id, trimmed_summary, int(summarized_until_message_id), utc_now()),
            )

    def _default_summarize_messages(self, existing_summary: str, new_messages: list[dict[str, Any]]) -> str:
        """Concise deterministic summarizer fallback when no LLM summarizer is provided."""
        facts: list[str] = []
        for msg in new_messages:
            role = msg["role"]
            content = msg["content"].strip()
            if not content:
                continue
            prefs = extract_preferences(content)
            extracted_bits = []
            if prefs.get("condition"):
                extracted_bits.append(f"condition: {prefs['condition']}")
            if prefs.get("max_price"):
                extracted_bits.append(f"budget max: {prefs['max_price']:,.0f} EGP")
            if prefs.get("body_type"):
                extracted_bits.append(f"body: {prefs['body_type']}")
            if prefs.get("transmission"):
                extracted_bits.append(f"transmission: {prefs['transmission']}")
            if prefs.get("brand"):
                extracted_bits.append(f"brand: {prefs['brand']}")
            if prefs.get("model"):
                extracted_bits.append(f"model: {prefs['model']}")

            car_ids = re.findall(r"(?:رقم|id|ID|عربية)\s*(\d+)", content)
            if car_ids:
                extracted_bits.append(f"car IDs referenced: {', '.join(car_ids[:3])}")

            if extracted_bits:
                facts.append(f"{role.capitalize()}: {'; '.join(extracted_bits)}")
            elif len(content) < 80 and not any(w in content for w in ("سلام", "شكرا", "أهلا", "ازيك", "تمام")):
                facts.append(f"{role.capitalize()}: {content}")

        new_segment = "\n".join(facts)
        if existing_summary and new_segment:
            combined = f"{existing_summary}\n---\n{new_segment}"
        elif new_segment:
            combined = new_segment
        else:
            combined = existing_summary
        return combined[:MEMORY_MAX_SUMMARY_CHARS]

    def maybe_update_summary(
        self,
        session_id: str,
        summarizer_fn: Callable[[str, list[dict[str, Any]]], str] | None = None,
        force: bool = False,
    ) -> bool:
        """Incrementally update conversation summary if unsummarized messages meet MEMORY_SUMMARY_TRIGGER.
        
        Zero-failure guarantee: If summarizer fails or throws, logs the error, preserves existing summary,
        and returns False without crashing the chat flow.
        """
        try:
            total_messages = self.get_message_count(session_id)
            if total_messages <= MEMORY_RECENT_MESSAGE_LIMIT and not force:
                return False

            summary_info = self.get_summary(session_id)
            last_summarized_id = summary_info.get("summarized_until_message_id") or 0

            # Eligible messages are those older than the latest MEMORY_RECENT_MESSAGE_LIMIT raw messages
            with _connect(self.db_path) as conn:
                recent_ids = [
                    r[0] for r in conn.execute(
                        "SELECT id FROM conversation_messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                        (session_id, MEMORY_RECENT_MESSAGE_LIMIT),
                    ).fetchall()
                ]

            if not recent_ids:
                return False

            cutoff_id = min(recent_ids) - 1
            if cutoff_id <= last_summarized_id and not force:
                return False

            target_cutoff = cutoff_id if not force else max(recent_ids)
            unsummarized = self.get_unsummarized_messages(session_id, up_to_id=target_cutoff)

            if len(unsummarized) < MEMORY_SUMMARY_TRIGGER and not force:
                return False

            existing_summary = summary_info.get("summary", "")
            if summarizer_fn:
                new_summary = summarizer_fn(existing_summary, unsummarized)
            else:
                new_summary = self._default_summarize_messages(existing_summary, unsummarized)

            new_cutoff_id = unsummarized[-1]["id"]
            self.update_summary(session_id, new_summary, new_cutoff_id)
            logger.info("Updated summary for session %s up to message id %s", session_id, new_cutoff_id)
            return True
        except Exception as exc:
            # Memory summary failure must never crash the main conversation
            logger.warning("Summary update failed for session %s (continuing chat safely): %s", session_id, exc)
            return False

    # -------------------------------------------------------------------------
    # Layer 3: Structured Session / Working Memory
    # -------------------------------------------------------------------------

    def get_session_state(self, session_id: str) -> dict[str, Any]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM conversation_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if row:
                data = dict(row)
                raw_ids = data.get("last_recommended_car_ids")
                if raw_ids:
                    try:
                        data["last_recommended_car_ids"] = json.loads(raw_ids)
                    except Exception:
                        data["last_recommended_car_ids"] = []
                else:
                    data["last_recommended_car_ids"] = []
                return data

        now = utc_now()
        return {
            "session_id": session_id,
            "condition": None,
            "min_price": None,
            "max_price": None,
            "brand": None,
            "model": None,
            "body_type": None,
            "fuel_type": None,
            "transmission": None,
            "min_year": None,
            "max_year": None,
            "location": None,
            "max_mileage": None,
            "last_recommended_car_ids": [],
            "selected_car_id": None,
            "selected_snapshot_id": None,
            "selected_position": None,
            "created_at": now,
            "updated_at": now,
        }

    def update_session_state(self, session_id: str, **updates: Any) -> dict[str, Any]:
        """Update structured session state with smart non-destructive merging.
        
        Only non-None, non-empty fields in updates are modified; existing fields are preserved.
        """
        valid_columns = {
            "condition", "min_price", "max_price", "brand", "model", "body_type",
            "fuel_type", "transmission", "min_year", "max_year", "location",
            "max_mileage", "last_recommended_car_ids", "selected_car_id",
            "selected_snapshot_id", "selected_position",
        }
        current = self.get_session_state(session_id)
        now = utc_now()

        filtered_updates: dict[str, Any] = {}
        for k, v in updates.items():
            if k not in valid_columns:
                continue
            if v is not None and v != "":
                if k == "last_recommended_car_ids":
                    if isinstance(v, (list, tuple)):
                        filtered_updates[k] = json.dumps([int(x) for x in v])
                    elif isinstance(v, str):
                        filtered_updates[k] = v
                elif k in {"min_price", "max_price", "max_mileage"}:
                    try:
                        filtered_updates[k] = float(v)
                    except (ValueError, TypeError):
                        pass
                elif k in {"min_year", "max_year", "selected_car_id", "selected_snapshot_id", "selected_position"}:
                    try:
                        filtered_updates[k] = int(v)
                    except (ValueError, TypeError):
                        pass
                else:
                    filtered_updates[k] = str(v).strip()

        if not filtered_updates:
            return current

        # Check if record exists
        with _connect(self.db_path) as conn:
            exists = conn.execute("SELECT 1 FROM conversation_sessions WHERE session_id=?", (session_id,)).fetchone()
            if exists:
                set_clauses = [f"{col} = ?" for col in filtered_updates.keys()]
                set_clauses.append("updated_at = ?")
                params = list(filtered_updates.values()) + [now, session_id]
                conn.execute(f"UPDATE conversation_sessions SET {', '.join(set_clauses)} WHERE session_id=?", params)
            else:
                cols = ["session_id", "created_at", "updated_at"] + list(filtered_updates.keys())
                placeholders = ["?"] * len(cols)
                params = [session_id, now, now] + list(filtered_updates.values())
                conn.execute(f"INSERT INTO conversation_sessions ({', '.join(cols)}) VALUES ({', '.join(placeholders)})", params)

        return self.get_session_state(session_id)

    def clear_session_state_field(self, session_id: str, field_name: str) -> None:
        valid_columns = {
            "condition", "min_price", "max_price", "brand", "model", "body_type",
            "fuel_type", "transmission", "min_year", "max_year", "location",
            "max_mileage", "last_recommended_car_ids", "selected_car_id",
            "selected_snapshot_id", "selected_position",
        }
        if field_name not in valid_columns:
            return
        now = utc_now()
        with _connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE conversation_sessions SET {field_name}=NULL, updated_at=? WHERE session_id=?",
                (now, session_id),
            )

    def set_recommended_cars(self, session_id: str, car_ids: list[int]) -> None:
        clean_ids = [int(x) for x in car_ids if isinstance(x, (int, str)) and str(x).isdigit()]
        self.update_session_state(session_id, last_recommended_car_ids=clean_ids)

    def get_recommended_cars(self, session_id: str) -> list[int]:
        state = self.get_session_state(session_id)
        ids = state.get("last_recommended_car_ids")
        return [int(x) for x in ids] if isinstance(ids, list) else []

    def set_selected_car(
        self,
        session_id: str,
        car_id: int | None,
        snapshot_id: int | None = None,
        position: int | None = None,
    ) -> None:
        if car_id is None:
            self.clear_session_state_field(session_id, "selected_car_id")
            self.clear_session_state_field(session_id, "selected_snapshot_id")
            self.clear_session_state_field(session_id, "selected_position")
        else:
            updates: dict[str, Any] = {"selected_car_id": int(car_id)}
            if snapshot_id is not None:
                updates["selected_snapshot_id"] = int(snapshot_id)
            if position is not None:
                updates["selected_position"] = int(position)
            self.update_session_state(session_id, **updates)

    def get_selected_car(self, session_id: str) -> int | None:
        state = self.get_session_state(session_id)
        cid = state.get("selected_car_id")
        return int(cid) if cid is not None else None

    # -------------------------------------------------------------------------
    # Recommendation Snapshots Management
    # -------------------------------------------------------------------------

    def create_recommendation_snapshot(
        self,
        session_id: str,
        criteria: dict[str, Any],
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create an active recommendation snapshot with visible items and update session recommendations."""
        now = utc_now()
        with _connect(self.db_path) as conn:
            conn.execute(
                "UPDATE recommendation_snapshots SET status='inactive' WHERE session_id=? AND status='active'",
                (session_id,),
            )
            cur_seq = conn.execute(
                "SELECT COALESCE(MAX(sequence_no), 0) + 1 FROM recommendation_snapshots WHERE session_id=?",
                (session_id,),
            ).fetchone()[0]
            cur = conn.execute(
                """INSERT INTO recommendation_snapshots
                   (session_id, sequence_no, criteria_json, items_json, status, created_at)
                   VALUES (?, ?, ?, ?, 'active', ?)""",
                (
                    session_id,
                    cur_seq,
                    json.dumps(criteria, ensure_ascii=False),
                    json.dumps(items, ensure_ascii=False),
                    now,
                ),
            )
            snapshot_id = cur.lastrowid

        primary_ids = [int(it["primary_car_id"]) for it in items if "primary_car_id" in it]
        self.set_recommended_cars(session_id, primary_ids)
        logger.info("[MEMORY] snapshot created id=%s seq=%s items=%s session=%s", snapshot_id, cur_seq, len(items), session_id)
        return {
            "id": snapshot_id,
            "session_id": session_id,
            "sequence_no": cur_seq,
            "criteria": criteria,
            "items": items,
            "status": "active",
            "created_at": now,
        }

    def get_active_snapshot(self, session_id: str) -> dict[str, Any] | None:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT * FROM recommendation_snapshots
                   WHERE session_id=? AND status='active'
                   ORDER BY sequence_no DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
            if row:
                d = dict(row)
                try:
                    d["criteria"] = json.loads(d.get("criteria_json", "{}"))
                    d["items"] = json.loads(d.get("items_json", "[]"))
                except Exception:
                    d["criteria"] = {}
                    d["items"] = []
                return d
        return None

    def get_snapshot_by_id(self, snapshot_id: int) -> dict[str, Any] | None:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM recommendation_snapshots WHERE id=?",
                (int(snapshot_id),),
            ).fetchone()
            if row:
                d = dict(row)
                try:
                    d["criteria"] = json.loads(d.get("criteria_json", "{}"))
                    d["items"] = json.loads(d.get("items_json", "[]"))
                except Exception:
                    d["criteria"] = {}
                    d["items"] = []
                return d
        return None

    def get_historical_snapshots(self, session_id: str) -> list[dict[str, Any]]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT * FROM recommendation_snapshots
                   WHERE session_id=?
                   ORDER BY sequence_no ASC""",
                (session_id,),
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                try:
                    d["criteria"] = json.loads(d.get("criteria_json", "{}"))
                    d["items"] = json.loads(d.get("items_json", "[]"))
                except Exception:
                    d["criteria"] = {}
                    d["items"] = []
                results.append(d)
            return results

    def invalidate_active_snapshot(self, session_id: str, reason: str = "") -> bool:
        with _connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE recommendation_snapshots SET status='invalidated' WHERE session_id=? AND status='active'",
                (session_id,),
            )
            count = cur.rowcount
        if count > 0:
            self.set_recommended_cars(session_id, [])
            logger.info("[MEMORY] snapshot invalidated reason=%s session=%s", reason, session_id)
            return True
        return False

    def apply_preference_updates_and_invalidate(self, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Apply preference updates and invalidate incompatible active snapshots or selected vehicles."""
        current_state = self.get_session_state(session_id)
        active_snapshot = self.get_active_snapshot(session_id)
        selected_car_id = current_state.get("selected_car_id")
        selected_car = get_car(self.db_path, selected_car_id) if selected_car_id else None

        # 1. Condition change: new <-> used
        new_cond = updates.get("condition")
        old_cond = current_state.get("condition")
        if not old_cond and active_snapshot:
            old_cond = active_snapshot.get("criteria", {}).get("condition")

        if new_cond:
            if old_cond and new_cond.lower() != old_cond.lower():
                self.invalidate_active_snapshot(session_id, f"condition {old_cond}->{new_cond}")
                if selected_car and (selected_car.get("condition") or "").lower() != new_cond.lower():
                    self.set_selected_car(session_id, None)
                    logger.info("[MEMORY] selected car cleared incompatible (condition %s != %s)", selected_car.get("condition"), new_cond)
            elif active_snapshot:
                items = active_snapshot.get("items", [])
                if items and any((it.get("condition") or "").lower() != new_cond.lower() for it in items):
                    self.invalidate_active_snapshot(session_id, f"items condition mismatch with {new_cond}")
                    if selected_car and (selected_car.get("condition") or "").lower() != new_cond.lower():
                        self.set_selected_car(session_id, None)

        # 2. Body type change
        new_body = updates.get("body_type")
        old_body = current_state.get("body_type")
        if new_body and old_body and new_body.lower() != old_body.lower():
            if active_snapshot:
                items = active_snapshot.get("items", [])
                if items and all((it.get("body_type") or "").lower() != new_body.lower() for it in items):
                    self.invalidate_active_snapshot(session_id, f"body_type {old_body}->{new_body}")
            if selected_car and (selected_car.get("body_type") or "").lower() != new_body.lower():
                self.set_selected_car(session_id, None)
                logger.info("[MEMORY] selected car cleared incompatible (body_type %s != %s)", selected_car.get("body_type"), new_body)

        # 3. Max price ceiling decrease
        new_max_price = updates.get("max_price")
        if new_max_price is not None:
            if selected_car and selected_car.get("price") and selected_car["price"] > new_max_price:
                self.set_selected_car(session_id, None)
                logger.info("[MEMORY] selected car cleared incompatible (price %s > %s)", selected_car.get("price"), new_max_price)
            if active_snapshot:
                items = active_snapshot.get("items", [])
                if items and all((it.get("price") or 0) > new_max_price for it in items):
                    self.invalidate_active_snapshot(session_id, f"max_price {new_max_price}")

        return self.update_session_state(session_id, **updates)

    # -------------------------------------------------------------------------
    # Layer 4: Task / Action Memory
    # -------------------------------------------------------------------------

    def create_pending_action(
        self,
        session_id: str,
        action_type: str,
        entity_id: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new pending action. Cancels any existing active pending action for this session."""
        now = utc_now()
        self.cancel_pending_action(session_id)

        default_payloads = {
            "test_drive": {
                "customer_name": None,
                "phone": None,
                "preferred_date": None,
                "preferred_time": None,
                "notes": None,
            },
            "sales_lead": {
                "customer_name": None,
                "phone": None,
                "email": None,
                "notes": None,
            },
        }
        merged_payload = dict(default_payloads.get(action_type, {}))
        if payload:
            merged_payload.update(payload)

        with _connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO pending_actions (session_id, action_type, entity_id, payload_json, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
                (session_id, action_type, entity_id, json.dumps(merged_payload, ensure_ascii=False), now, now),
            )
            action_id = cur.lastrowid

        return {
            "id": action_id,
            "session_id": session_id,
            "action_type": action_type,
            "entity_id": entity_id,
            "payload": merged_payload,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        }

    def get_pending_action(self, session_id: str) -> dict[str, Any] | None:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT * FROM pending_actions
                   WHERE session_id=? AND status='pending'
                   ORDER BY id DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
            if row:
                data = dict(row)
                try:
                    data["payload"] = json.loads(data.get("payload_json", "{}"))
                except Exception:
                    data["payload"] = {}
                return data
        return None

    def update_pending_action(
        self,
        session_id: str,
        payload_updates: dict[str, Any] | None = None,
        entity_id: int | None = None,
        status: str | None = None,
    ) -> dict[str, Any] | None:
        action = self.get_pending_action(session_id)
        if not action:
            return None

        now = utc_now()
        payload = action.get("payload", {})
        if payload_updates:
            for k, v in payload_updates.items():
                payload[k] = v

        new_entity_id = entity_id if entity_id is not None else action.get("entity_id")
        new_status = status if status is not None else action.get("status", "pending")

        with _connect(self.db_path) as conn:
            conn.execute(
                """UPDATE pending_actions
                   SET payload_json=?, entity_id=?, status=?, updated_at=?
                   WHERE id=?""",
                (json.dumps(payload, ensure_ascii=False), new_entity_id, new_status, now, action["id"]),
            )

        action["payload"] = payload
        action["entity_id"] = new_entity_id
        action["status"] = new_status
        action["updated_at"] = now
        return action

    def cancel_pending_action(self, session_id: str) -> bool:
        with _connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE pending_actions SET status='cancelled', updated_at=? WHERE session_id=? AND status='pending'",
                (utc_now(), session_id),
            )
            return cur.rowcount > 0

    def complete_pending_action(self, session_id: str, result_metadata: dict[str, Any] | None = None) -> bool:
        action = self.get_pending_action(session_id)
        if not action:
            return False
        payload = action.get("payload", {})
        if result_metadata:
            payload["_result"] = result_metadata

        with _connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE pending_actions SET status='completed', payload_json=?, updated_at=? WHERE id=?",
                (json.dumps(payload, ensure_ascii=False), utc_now(), action["id"]),
            )
            return cur.rowcount > 0

    def clear_pending_action(self, session_id: str) -> None:
        self.cancel_pending_action(session_id)

    def get_last_completed_action(self, session_id: str, action_type: str | None = None) -> dict[str, Any] | None:
        with _connect(self.db_path) as conn:
            clauses = ["session_id=?", "status='completed'"]
            params: list[Any] = [session_id]
            if action_type:
                clauses.append("action_type=?")
                params.append(action_type)
            sql = f"SELECT * FROM pending_actions WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT 1"
            row = conn.execute(sql, params).fetchone()
            if row:
                d = dict(row)
                try:
                    d["payload"] = json.loads(d.get("payload_json", "{}"))
                except Exception:
                    d["payload"] = {}
                return d
        return None

    # -------------------------------------------------------------------------
    # Context Builder
    # -------------------------------------------------------------------------

    def build_memory_context(self, session_id: str, user_text: str = "") -> dict[str, Any]:
        """Build compact structured memory context for Gemini prompt injection."""
        session_state = self.get_session_state(session_id)
        summary_info = self.get_summary(session_id)
        recent_messages = self.get_recent_messages(session_id, limit=MEMORY_RECENT_MESSAGE_LIMIT)
        pending_action = self.get_pending_action(session_id)
        active_snapshot = self.get_active_snapshot(session_id)
        resolved_ref_info = resolve_recommendation_reference(session_id, user_text, self.db_path) if user_text else None

        lines: list[str] = []

        # SECTION 1: CURRENT STRUCTURED STATE (Authoritative)
        lines.append("CURRENT STRUCTURED STATE (Authoritative — Always Outranks Summary)")
        lines.append("----------------------------------------------------------------")
        state_parts = []
        if session_state.get("condition"):
            state_parts.append(f"Condition: {session_state['condition']}")
        if session_state.get("max_price"):
            state_parts.append(f"Max Price: {session_state['max_price']:,.0f} EGP")
        if session_state.get("min_price"):
            state_parts.append(f"Min Price: {session_state['min_price']:,.0f} EGP")
        if session_state.get("body_type"):
            state_parts.append(f"Body Type: {session_state['body_type']}")
        if session_state.get("transmission"):
            state_parts.append(f"Transmission: {session_state['transmission']}")
        if session_state.get("fuel_type"):
            state_parts.append(f"Fuel: {session_state['fuel_type']}")
        if session_state.get("brand"):
            state_parts.append(f"Brand: {session_state['brand']}")
        if session_state.get("model"):
            state_parts.append(f"Model: {session_state['model']}")
        if session_state.get("min_year") or session_state.get("max_year"):
            state_parts.append(f"Year: {session_state.get('min_year')} - {session_state.get('max_year')}")
        if session_state.get("location"):
            state_parts.append(f"Location: {session_state['location']}")
        if not state_parts:
            state_parts.append("No explicit constraints set yet.")
        lines.append("\n".join(state_parts))
        lines.append("")

        # SECTION 2: ACTIVE RECOMMENDATION SNAPSHOT
        lines.append("ACTIVE RECOMMENDATION SNAPSHOT (Single Source of Truth for Visible Positions)")
        lines.append("-----------------------------------------------------------------------------")
        if active_snapshot and active_snapshot.get("items"):
            items = active_snapshot["items"]
            lines.append(f"Snapshot ID: #{active_snapshot['id']} (Sequence: {active_snapshot['sequence_no']})")
            for it in items:
                pos = it.get("position")
                cid = it.get("primary_car_id")
                name = it.get("display_name")
                price = it.get("price")
                price_str = f"{price:,.0f} EGP" if price else "Price on inquiry"
                colors = it.get("display_metadata", {}).get("colors", [])
                color_str = f" [Colors: {', '.join(colors)}]" if colors else ""
                lines.append(f"  Position {pos}: {name} ({it.get('year')}) — {price_str} (Primary ID: {cid}){color_str}")
        else:
            lines.append("None (Search has not been run or previous active list was invalidated due to preference change)")
        lines.append("")

        # SECTION 3: SELECTED VEHICLE
        lines.append("SELECTED VEHICLE")
        lines.append("----------------")
        sel_id = session_state.get("selected_car_id")
        if sel_id:
            car_info = get_car(self.db_path, sel_id)
            if car_info:
                car_desc = f"{car_info.get('brand')} {car_info.get('model')} ({car_info.get('year')}) [{car_info.get('condition')}]"
                lines.append(f"Selected Car ID: {sel_id} [{car_desc}]")
            else:
                lines.append(f"Selected Car ID: {sel_id}")
            if session_state.get("selected_position"):
                lines.append(f"Selected from Snapshot #{session_state.get('selected_snapshot_id')} Position {session_state.get('selected_position')}")
        else:
            lines.append("None")
        lines.append("")

        # SECTION 4: PENDING ACTION
        lines.append("PENDING ACTION (In-Progress Business Workflow)")
        lines.append("---------------------------------------------")
        if pending_action:
            lines.append(f"Action Type: {pending_action['action_type']}")
            lines.append(f"Status: {pending_action.get('status', 'pending')}")
            lines.append(f"Target Car ID: {pending_action.get('entity_id') or 'Not selected'}")
            payload = pending_action.get("payload", {})
            lines.append(f"Collected Info: {json.dumps(payload, ensure_ascii=False)}")
            missing = [k for k, v in payload.items() if not v and not k.startswith("_")]
            lines.append(f"Missing Required Info: {missing}")
            lines.append("CRITICAL: Do not execute database insert until all required fields are provided and verified.")
        else:
            lines.append("None")
        lines.append("")

        # SECTION 5: RESOLVED REFERENCE HINT (If applicable)
        if resolved_ref_info:
            status = resolved_ref_info.get("status")
            if status == "resolved":
                c_ids = [it["primary_car_id"] for it in resolved_ref_info.get("items", [])]
                lines.append("DETERMINISTIC ORDINAL RESOLUTION")
                lines.append("--------------------------------")
                lines.append(f"Status: Resolved to snapshot #{resolved_ref_info.get('snapshot_id')} positions {resolved_ref_info.get('positions')} -> Inventory ID(s): {c_ids}")
                lines.append("")
            elif status == "no_active_snapshot":
                lines.append("DETERMINISTIC ORDINAL RESOLUTION")
                lines.append("--------------------------------")
                lines.append("Status: Unqualified ordinal used but active recommendation list is invalidated. Search current criteria or ask customer.")
                lines.append("")
            elif status == "ambiguous":
                lines.append("DETERMINISTIC ORDINAL RESOLUTION")
                lines.append("--------------------------------")
                lines.append(f"Status: Ambiguous reference ({resolved_ref_info.get('reason')}). Ask customer for clarification.")
                lines.append("")

        # SECTION 6: HISTORICAL SUMMARY
        lines.append("HISTORICAL SUMMARY (Earlier Discussion — Do NOT override current state)")
        lines.append("----------------------------------------------------------------------")
        if summary_info.get("summary"):
            lines.append(summary_info["summary"])
        else:
            lines.append("None")
        lines.append("")

        # SECTION 7: RECENT CONVERSATION
        lines.append("RECENT CONVERSATION")
        lines.append("-------------------")
        if recent_messages:
            for m in recent_messages:
                lines.append(f"{m['role'].capitalize()}: {m['content']}")
        else:
            lines.append("None")
        lines.append("")

        # SECTION 8: CURRENT USER MESSAGE
        lines.append("CURRENT USER MESSAGE")
        lines.append("--------------------")
        lines.append(user_text)

        formatted_context = "\n".join(lines).strip()

        # Deterministic primary car ID(s) for compatibility
        legacy_resolved = None
        if resolved_ref_info and resolved_ref_info.get("status") == "resolved":
            its = resolved_ref_info.get("items", [])
            if len(its) == 1:
                legacy_resolved = its[0]["primary_car_id"]
            elif len(its) > 1:
                legacy_resolved = [it["primary_car_id"] for it in its]

        return {
            "summary": summary_info.get("summary", ""),
            "recent_messages": recent_messages,
            "session_state": session_state,
            "pending_action": pending_action,
            "active_snapshot": active_snapshot,
            "resolved_reference": legacy_resolved,
            "resolved_reference_details": resolved_ref_info,
            "formatted_context": formatted_context,
        }


# -----------------------------------------------------------------------------
# Deterministic Arabic & Business Logic Helpers
# -----------------------------------------------------------------------------

def _normalize_arabic(text: str) -> str:
    """Normalize common Arabic spelling variants and punctuation."""
    t = str(text or "").lower()
    # Normalize alef variants
    t = re.sub(r"[إأآا]", "ا", t)
    # Normalize taa marbuta / haa
    t = re.sub(r"[ة]", "ه", t)
    # Normalize yaa / alef maksura
    t = re.sub(r"[يى]", "ي", t)
    # Remove diacritics / tashkeel
    t = re.sub(r"[\u064B-\u065F\u0670]", "", t)
    # Normalize multiple spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


def resolve_recommendation_reference(
    session_id: str,
    user_text: str,
    db_path: str = DEFAULT_DB,
) -> dict[str, Any]:
    """Deterministically resolve Arabic ordinal and historical vehicle references against snapshots."""
    manager = MemoryManager(db_path)
    norm = _normalize_arabic(user_text)

    # 1. Check for historical intent (e.g. "في الأول", "اللي قولتهم في الأول", "قبل كده")
    is_historical = bool(
        re.search(r"(?:في|ف|من)\s+(?:الاول|الأول|البدايه|البداية)", norm)
        or re.search(r"اللي\s+(?:فات|فاتوا|قولتهم|قلتهم|عرضتهم|رشحتهم)", norm)
        or re.search(r"قبل\s*كده", norm)
        or re.search(r"عربيه\s+من\s+العربيات", norm)
        or re.search(r"كنت\s+(?:طالب|قايل|مختار)", norm)
    )

    target_snapshot: dict[str, Any] | None = None
    if is_historical:
        historical = manager.get_historical_snapshots(session_id)
        if historical:
            for snap in historical:
                if len(snap.get("items", [])) >= 2:
                    target_snapshot = snap
                    break
            if not target_snapshot:
                target_snapshot = historical[0]
    else:
        target_snapshot = manager.get_active_snapshot(session_id)

    # Fallback for legacy set_recommended_cars calls without explicit snapshot
    if not target_snapshot or not target_snapshot.get("items"):
        legacy_rec = manager.get_recommended_cars(session_id)
        if legacy_rec:
            target_snapshot = {
                "id": 0,
                "session_id": session_id,
                "sequence_no": 0,
                "items": [
                    {
                        "position": i + 1,
                        "primary_car_id": cid,
                        "display_name": f"Car #{cid}",
                    }
                    for i, cid in enumerate(legacy_rec)
                ],
            }

    # If no snapshot found
    if not target_snapshot or not target_snapshot.get("items"):
        selected = manager.get_selected_car(session_id)
        if selected and any(w in norm for w in ("فيها", "عنها", "دي", "العربيه دي", "السياره دي")):
            car_row = get_car(db_path, selected)
            return {
                "status": "resolved",
                "snapshot_id": None,
                "positions": [],
                "items": [{"primary_car_id": selected, "display_name": f"{car_row.get('brand', '')} {car_row.get('model', '')}" if car_row else ""}],
                "reason": "Resolved to currently selected vehicle",
            }
        ordinal_words = ["اول", "الاول", "الاولى", "الاولي", "تاني", "التاني", "التانيه", "الثاني", "الثانيه", "تالت", "التالت", "التالته", "الثالث", "الثالثه", "رابع", "خامس", "اخر"]
        if any(re.search(rf"(?:^|[^\w\u0600-\u06ff]){w}(?:[^\w\u0600-\u06ff]|$)", norm) for w in ordinal_words):
            return {
                "status": "no_active_snapshot",
                "snapshot_id": None,
                "positions": [],
                "items": [],
                "reason": "Active recommendation snapshot is invalidated or does not exist",
            }
        return {
            "status": "not_found",
            "snapshot_id": None,
            "positions": [],
            "items": [],
            "reason": "No vehicle reference found",
        }

    items = target_snapshot["items"]

    # 2. Check for negative constraint: e.g. "مش الساوليت", "مش ساوايست"
    excluded_brand = None
    neg_match = re.search(r"(?:مش|غير|ماعدا|بدون)\s+([^\s]+)", norm)
    if neg_match:
        excluded_term = neg_match.group(1).lower()
        if any(b in excluded_term for b in ("ساوليت", "ساوايست", "soueast")):
            excluded_brand = "soueast"
        elif any(b in excluded_term for b in ("ميتسوبيشي", "mitsubishi")):
            excluded_brand = "mitsubishi"
        elif any(b in excluded_term for b in ("سوزوكي", "suzuki")):
            excluded_brand = "suzuki"
        elif any(b in excluded_term for b in ("تويوتا", "toyota")):
            excluded_brand = "toyota"
        elif any(b in excluded_term for b in ("هيونداي", "hyundai")):
            excluded_brand = "hyundai"
        elif any(b in excluded_term for b in ("شيري", "chery")):
            excluded_brand = "chery"
        else:
            excluded_brand = excluded_term

    # 3. Check multi-car ordinals: e.g. "أول اتنين", "قارن أول اتنين"
    multi_patterns = [
        r"(?:الاول|الاولى|الاولي|اول)\s*(?:و\s*)?(?:اتنين|اثنين|التاني|التانيه|الثاني|الثانيه|2)",
        r"قارن\s*(?:بين\s*)?(?:اول|الاول|الاولي)",
        r"اول\s*سيارتين|اول\s*عربيتين",
    ]
    for pat in multi_patterns:
        if re.search(pat, norm):
            matched_items = items[:2]
            return {
                "status": "resolved",
                "snapshot_id": target_snapshot["id"],
                "positions": [it["position"] for it in matched_items],
                "items": matched_items,
                "reason": "First two items from snapshot",
            }

    # 4. Check last item
    last_patterns = [
        r"اخر\s*(?:واحده|عربيه|سياره|اختيار)?",
        r"(?:^|[^\w\u0600-\u06ff])(?:الاخيره|الاخير)(?:[^\w\u0600-\u06ff]|$)",
    ]
    for pat in last_patterns:
        if re.search(pat, norm) and not re.search(r"اخر\s*(?:حاجه|سعر|كلام)", norm):
            it = items[-1]
            return {
                "status": "resolved",
                "snapshot_id": target_snapshot["id"],
                "positions": [it["position"]],
                "items": [it],
                "reason": "Last item from snapshot",
            }

    # 5. Check single position
    eval_text = norm
    if is_historical:
        eval_text = re.sub(
            r"(?:في|ف|من)?\s*(?:الاول|الأول|البدايه|البداية|اللي فات|اللي قولتهم|اللي قلتهم|اللي قولتهم في الاول|اللي قلتهم في الاول|التلاته اللي|الثلاثة اللي)",
            " ",
            eval_text,
        )
        eval_text = re.sub(r"\s+", " ", eval_text).strip()

    target_pos = None
    if re.search(r"اول\s*(?:واحده|عربيه|سياره|اختيار)|(?:^|[^\w\u0600-\u06ff])(?:الاولى|الاولي|الاول)(?:[^\w\u0600-\u06ff]|$)|رقم\s*(?:1|واحد|١)", eval_text):
        target_pos = 1
    elif re.search(r"(?:تاني|ثاني)\s*(?:واحده|عربيه|سياره|اختيار)?|(?:^|[^\w\u0600-\u06ff])(?:التانيه|الثانيه|التاني|الثاني|تاني|ثاني)(?:[^\w\u0600-\u06ff]|$)|رقم\s*(?:2|اتنين|اثنين|٢)", eval_text):
        target_pos = 2
    elif re.search(r"(?:تالت|ثالث)\s*(?:واحده|عربيه|سياره|اختيار)?|(?:^|[^\w\u0600-\u06ff])(?:التالته|الثالثه|التالت|الثالث|تالت|ثالث)(?:[^\w\u0600-\u06ff]|$)|رقم\s*(?:3|تلاته|ثلاثه|٣)", eval_text):
        target_pos = 3
    elif re.search(r"رابع\s*(?:واحده|عربيه|سياره)?|(?:^|[^\w\u0600-\u06ff])(?:الرابعه|الرابع|رابع)(?:[^\w\u0600-\u06ff]|$)|رقم\s*(?:4|اربعه|٤)", eval_text):
        target_pos = 4
    elif re.search(r"خامس\s*(?:واحده|عربيه|سياره)?|(?:^|[^\w\u0600-\u06ff])(?:الخامسه|الخامس|خامس)(?:[^\w\u0600-\u06ff]|$)|رقم\s*(?:5|خمسه|٥)", eval_text):
        target_pos = 5

    if target_pos is not None:
        matched = [it for it in items if it.get("position") == target_pos]
        if matched:
            candidate = matched[0]
            if excluded_brand:
                cand_text = f"{candidate.get('display_name', '')} {candidate.get('brand', '')}".lower()
                is_excluded = (
                    excluded_brand in cand_text
                    or (excluded_brand == "soueast" and any(b in cand_text for b in ("ساوليت", "ساوايست", "soueast")))
                    or (excluded_brand == "mitsubishi" and any(b in cand_text for b in ("ميتسوبيشي", "mitsubishi")))
                    or (excluded_brand == "suzuki" and any(b in cand_text for b in ("سوزوكي", "suzuki")))
                    or (excluded_brand == "toyota" and any(b in cand_text for b in ("تويوتا", "toyota")))
                    or (excluded_brand == "hyundai" and any(b in cand_text for b in ("هيونداي", "hyundai")))
                    or (excluded_brand == "chery" and any(b in cand_text for b in ("شيري", "chery")))
                )
                if is_excluded:
                    return {
                        "status": "ambiguous",
                        "snapshot_id": target_snapshot["id"],
                        "positions": [target_pos],
                        "items": [candidate],
                        "reason": f"Customer requested position {target_pos} but explicitly excluded brand '{excluded_brand}'",
                    }
            return {
                "status": "resolved",
                "snapshot_id": target_snapshot["id"],
                "positions": [target_pos],
                "items": [candidate],
                "reason": f"Position {target_pos} from snapshot",
            }

    # 6. Check explicit primary car ID mentioned directly in text
    for it in items:
        cid = it.get("primary_car_id")
        if cid and re.search(rf"\b{cid}\b", norm):
            return {
                "status": "resolved",
                "snapshot_id": target_snapshot["id"],
                "positions": [it.get("position", 1)],
                "items": [it],
                "reason": f"Explicit primary ID {cid} mentioned",
            }

    return {
        "status": "not_found",
        "snapshot_id": target_snapshot["id"],
        "positions": [],
        "items": [],
        "reason": "No matching position or car found in snapshot",
    }


def resolve_car_reference(session_id: str, user_text: str, db_path: str = DEFAULT_DB) -> list[int] | int | None:
    """Backward-compatible wrapper returning int or list[int] from snapshot resolver."""
    res = resolve_recommendation_reference(session_id, user_text, db_path)
    if res.get("status") == "resolved":
        items = res.get("items", [])
        if len(items) == 1:
            return items[0].get("primary_car_id")
        elif len(items) > 1:
            return [it.get("primary_car_id") for it in items]
    return None


def extract_preferences(user_text: str, current_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Deterministically extract structured car search preferences from Arabic / English input.
    
    Returns a dict with non-null values for explicitly stated constraints.
    """
    norm = _normalize_arabic(user_text)
    updates: dict[str, Any] = {}

    # 1. Condition: new vs used
    if re.search(r"(?:^|[^\w\u0600-\u06ff]|و)(?:زيرو|جديده|جديد|zero|brand[- ]?new|new)(?:[^\w\u0600-\u06ff]|$)", norm):
        updates["condition"] = "new"
    elif re.search(r"(?:^|[^\w\u0600-\u06ff]|و)(?:مستعمل|مستعمله|used|pre[- ]?owned)(?:[^\w\u0600-\u06ff]|$)", norm):
        updates["condition"] = "used"

    # 2. Budget / Price
    m_plus_k = re.search(r"(?:(\d+)\s*)?مليون\s*(?:و\s*)?(\d+)\s*(?:الف|k)", norm)
    if m_plus_k:
        m_val = float(m_plus_k.group(1) or 1) * 1_000_000.0
        k_val = float(m_plus_k.group(2)) * 1_000.0
        updates["max_price"] = m_val + k_val
    elif re.search(r"مليون\s*(?:و\s*)?نص", norm):
        updates["max_price"] = 1_500_000.0
    elif re.search(r"مليون\s*(?:و\s*)?ربع", norm):
        updates["max_price"] = 1_250_000.0
    elif re.search(r"مليون\s*(?:و\s*)?تلت", norm):
        updates["max_price"] = 1_330_000.0
    elif re.search(r"(?:2|اتنين|اثنين)\s*مليون|مليونين", norm):
        updates["max_price"] = 2_000_000.0
    elif re.search(r"\bمليون\b", norm) and not re.search(r"\d+\s*مليون", norm):
        updates["max_price"] = 1_000_000.0

    num_million = re.search(r"(\d+(?:\.\d+)?)\s*مليون", norm)
    if num_million:
        updates["max_price"] = float(num_million.group(1)) * 1_000_000.0

    num_k = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:الف|k)\b", norm)
    if num_k and "max_price" not in updates:
        val = float(num_k.group(1).replace(",", ".")) * 1000.0
        if re.search(r"(?:ميزاني|حد|اخر|سعر|تحت|اقل|في حدود|معايا)\b", norm) or val >= 50000:
            updates["max_price"] = val

    plain_large_num = re.search(r"\b([1-9]\d{5,7})\b", norm)
    if plain_large_num and "max_price" not in updates:
        updates["max_price"] = float(plain_large_num.group(1))

    range_match = re.search(r"من\s*(\d+)\s*(?:الف)?\s*(?:ل|الي|إلى|حتى)\s*(\d+)\s*(?:الف)?", norm)
    if range_match:
        n1 = float(range_match.group(1))
        n2 = float(range_match.group(2))
        if n1 < 1000:
            n1 *= 1000.0
        if n2 < 1000:
            n2 *= 1000.0
        updates["min_price"] = min(n1, n2)
        updates["max_price"] = max(n1, n2)

    # 3. Body Type
    if re.search(r"(?:^|[^\w\u0600-\u06ff]|و)(?:suv|اس\s*يو\s*في|فور\s*باي\s*فور|جيب|كروس\s*اوفر|crossover)(?:[^\w\u0600-\u06ff]|$)", norm):
        updates["body_type"] = "SUV"
    elif re.search(r"(?:^|[^\w\u0600-\u06ff]|و)(?:سيدان|sedan)(?:[^\w\u0600-\u06ff]|$)", norm):
        updates["body_type"] = "Sedan"
    elif re.search(r"(?:^|[^\w\u0600-\u06ff]|و)(?:هاتشباك|hatchback)(?:[^\w\u0600-\u06ff]|$)", norm):
        updates["body_type"] = "Hatchback"
    elif re.search(r"(?:^|[^\w\u0600-\u06ff]|و)(?:كوبيه|coupe)(?:[^\w\u0600-\u06ff]|$)", norm):
        updates["body_type"] = "Coupe"

    # 4. Transmission
    if re.search(r"(?:^|[^\w\u0600-\u06ff]|و)(?:اوتوماتيك|اتوماتيك|automatic|auto)(?:[^\w\u0600-\u06ff]|$)", norm):
        updates["transmission"] = "Automatic"
    elif re.search(r"(?:^|[^\w\u0600-\u06ff]|و)(?:مانيوال|عادي|يدوي|manual)(?:[^\w\u0600-\u06ff]|$)", norm):
        updates["transmission"] = "Manual"

    # 5. Fuel Type
    if re.search(r"(?:^|[^\w\u0600-\u06ff]|و)(?:بنزين|gasoline|petrol)(?:[^\w\u0600-\u06ff]|$)", norm):
        updates["fuel_type"] = "Gasoline"
    elif re.search(r"(?:^|[^\w\u0600-\u06ff]|و)(?:سولار|ديزل|diesel)(?:[^\w\u0600-\u06ff]|$)", norm):
        updates["fuel_type"] = "Diesel"
    elif re.search(r"(?:^|[^\w\u0600-\u06ff]|و)(?:كهرباء|كهربائيه|electric|ev)(?:[^\w\u0600-\u06ff]|$)", norm):
        updates["fuel_type"] = "Electric"
    elif re.search(r"(?:^|[^\w\u0600-\u06ff]|و)(?:هايبرد|hybrid)(?:[^\w\u0600-\u06ff]|$)", norm):
        updates["fuel_type"] = "Hybrid"

    # 6. Year
    year_match = re.search(r"\b(?:موديل|سنه|عام)?\s*(20[12]\d)\b", norm)
    if year_match:
        yr = int(year_match.group(1))
        if re.search(r"(?:فوق|من|بعد)\s*" + str(yr), norm):
            updates["min_year"] = yr
        elif re.search(r"(?:تحت|قبل)\s*" + str(yr), norm):
            updates["max_year"] = yr
        elif re.search(r"(?:موديل|سنه)\s*" + str(yr), norm):
            updates["min_year"] = yr
            updates["max_year"] = yr

    # 7. Common Brands
    brands = {
        "Toyota": [r"\bتويوتا\b", r"\btoyota\b"],
        "Hyundai": [r"\bهيونداي\b", r"\bhyundai\b"],
        "Renault": [r"\bرينو\b", r"\brenault\b"],
        "Kia": [r"\bكيا\b", r"\bkia\b"],
        "Chery": [r"\bشيري\b", r"\bchery\b"],
        "Nissan": [r"\bنيسان\b", r"\bnissan\b"],
        "MG": [r"\bام جي\b", r"\bmg\b"],
        "Soueast": [r"\bساوايست\b", r"\bsoueast\b", r"\bساوليت\b"],
        "Mercedes": [r"\bمرسيدس\b", r"\bmercedes\b"],
        "BMW": [r"\bبي ام\b", r"\bbmw\b"],
    }
    for bname, pats in brands.items():
        if any(re.search(p, norm) for p in pats):
            updates["brand"] = bname
            break

    # 8. Location
    locations = {
        "Cairo": [r"\bالقاهره\b", r"\bcairo\b"],
        "Giza": [r"\bالجيزه\b", r"\bgiza\b"],
        "Alexandria": [r"\bالاسكندريه\b", r"\bاسكندريه\b", r"\balexandria\b"],
    }
    for loc_name, pats in locations.items():
        if any(re.search(p, norm) for p in pats):
            updates["location"] = loc_name
            break

    return updates


def detect_cancellation(user_text: str) -> bool:
    """Detect common Arabic cancellation and abort phrases."""
    norm = _normalize_arabic(user_text)
    patterns = [
        r"خلاص\s*بلاش",
        r"بلاش\s*(?:الحجز|الميعاد|التجربه)?",
        r"\bالغ(?:ي|اء|ي الحجز)\b",
        r"\bإلغاء\b",
        r"\bكنسل\b",
        r"مش\s*عايز\s*(?:حد\s*يكلمني|احجز|test drive|تجربه)",
        r"\bcancel\b",
        r"بلاش\s*تكمل",
    ]
    return any(re.search(p, norm) for p in patterns)


def detect_reschedule(user_text: str) -> bool:
    """Detect phrases asking to reschedule or book with a different schedule."""
    norm = _normalize_arabic(user_text)
    return bool(re.search(r"(?:بمواعيد|بميعاد|بموعد|معاد|ميعاد|وقت)\s*(?:تاني|تانيه|اخر|مختلف|جديد)", norm))


def detect_new_details(user_text: str) -> bool:
    """Detect phrases asking to book with completely new contact details."""
    norm = _normalize_arabic(user_text)
    return bool(re.search(r"بيانات\s*(?:جديده|تانيه|مختلفه|كليه)", norm))


def detect_action_trigger(user_text: str) -> tuple[str | None, str | None]:
    """Detect if user is asking for a test drive or a sales lead."""
    norm = _normalize_arabic(user_text)
    if any(k in norm for k in ("test drive", "تيست درايف", "تجربه قياده", "احجزلي", "احجز ميعاد", "معاينه", "اجرب العربيه")):
        return "test_drive", "احجزلي test drive"

    if any(k in norm for k in ("حد يكلمني", "المبيعات تكلمني", "مندوب مبيعات", "تواصل معايا", "سجل بياناتي", "حد يتصل بيا")):
        return "sales_lead", "خلي حد من المبيعات يكلمني"

    return None, None


def extract_action_inputs(action_type: str, user_text: str, current_payload: dict[str, Any]) -> dict[str, Any]:
    """Extract missing fields for pending test drive or sales lead from user reply."""
    raw = str(user_text or "").strip()
    norm = _normalize_arabic(raw)
    updates: dict[str, Any] = {}

    days = {
        "السبت": "السبت", "الاحد": "الأحد", "الأحد": "الأحد",
        "الاثنين": "الاثنين", "الاتنين": "الاثنين", "الاتنن": "الاثنين", "الإثنين": "الاثنين",
        "الثلاثاء": "الثلاثاء", "التلات": "الثلاثاء", "الثلاثا": "الثلاثاء",
        "الاربعاء": "الأربعاء", "الاربع": "الأربعاء", "الأربعاء": "الأربعاء",
        "الخميس": "الخميس",
        "الجمعه": "الجمعة", "الجمعة": "الجمعة",
        "بكره": "غداً", "غدا": "غداً", "بعد بكره": "بعد غد",
    }

    # Combined single-message input: e.g. "أحمد محمد01123456789الاتنن 2الظهر"
    phone_combined = re.search(r"(?:(?:\+?20)|0)?(1[0125]\d{8})", raw)
    if phone_combined:
        extracted_phone = "0" + phone_combined.group(1)
        before_text = raw[:phone_combined.start()].strip()
        after_text = raw[phone_combined.end():].strip()

        if before_text and not current_payload.get("customer_name"):
            words = before_text.split()
            if 1 <= len(words) <= 5 and not any(w in _normalize_arabic(before_text) for w in ("احجز", "عايز", "كلمني", "تيست")):
                updates["customer_name"] = before_text

        if not current_payload.get("phone"):
            updates["phone"] = extracted_phone

        if action_type == "test_drive" and after_text:
            after_norm = _normalize_arabic(after_text)
            for d_key, d_val in days.items():
                if d_key in after_norm:
                    updates["preferred_date"] = d_val
                    remainder = re.sub(d_key, "", after_norm).strip()
                    time_m = re.search(r"(\d{1,2}(?::\d{2})?\s*(?:مساء|صباحا|م|ص|am|pm|الظهر|العصر|المغرب)?|الساعه\s*\d{1,2}|الظهر|العصر|المغرب)", remainder)
                    if time_m and not current_payload.get("preferred_time"):
                        updates["preferred_time"] = time_m.group(1).strip()
                    break

    # Standard fallback extractions if not combined or fields still missing:
    # 1. Phone number
    if "phone" not in updates and not current_payload.get("phone"):
        phone_match = re.search(r"(?:(?:\+?20)|0)?(1[0125]\d{8})\b", raw.replace(" ", "").replace("-", ""))
        if phone_match:
            updates["phone"] = "0" + phone_match.group(1)

    # 2. Date extraction for test drive
    if action_type == "test_drive" and "preferred_date" not in updates and not current_payload.get("preferred_date"):
        for d_key, d_val in days.items():
            if d_key in norm:
                updates["preferred_date"] = d_val
                break
        if "preferred_date" not in updates:
            date_iso = re.search(r"\b(202\d[-/]\d{1,2}[-/]\d{1,2})\b", raw)
            if date_iso:
                updates["preferred_date"] = date_iso.group(1)

    # 3. Time extraction for test drive
    if action_type == "test_drive" and "preferred_time" not in updates and not current_payload.get("preferred_time"):
        time_match = re.search(r"(\d{1,2}(?::\d{2})?\s*(?:مساء|صباحا|م|ص|am|pm|الظهر|العصر|المغرب)?|الساعه\s*\d{1,2}|الظهر|العصر|المغرب)", norm)
        if time_match and not any(time_match.group(1).startswith(p) for p in ("01", "20")):
            candidate = time_match.group(1)
            if len(re.sub(r"\D", "", candidate)) <= 4:
                updates["preferred_time"] = candidate

    # 4. Customer Name extraction
    if "customer_name" not in updates and not current_payload.get("customer_name"):
        words = [w for w in raw.split() if w]
        is_phone = bool(re.search(r"\d{7,}", raw))
        is_date = any(w in norm for w in days.keys())
        is_time = any(w in norm for w in ("مساء", "صباحا", "الساعه", "العصر", "الظهر"))
        is_command = any(w in norm for w in ("عايز", "احجز", "كلمني", "عربيه", "سياره", "كام", "بكام", "فين"))
        if not is_phone and not is_date and not is_time and not is_command and 1 <= len(words) <= 4:
            updates["customer_name"] = raw.strip()

    return updates
