from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from lfx.custom import Component
from lfx.io import HandleInput, IntInput, MessageInput, MessageTextInput, MultilineInput, Output, SecretStrInput
from lfx.schema import Message

from car_dealership_core import (
    DEFAULT_DB,
    build_recommendation_set,
    cancel_test_drive,
    create_sales_lead,
    create_test_drive,
    get_car,
    search_cars,
)
from memory_manager import (
    MEMORY_RECENT_MESSAGE_LIMIT,
    MemoryManager,
    detect_action_trigger,
    detect_cancellation,
    detect_new_details,
    detect_reschedule,
    extract_action_inputs,
    extract_preferences,
    resolve_recommendation_reference,
    set_current_session_id,
)
from request_context import set_current_user_text

logger = logging.getLogger(__name__)
BLOCK_PREFIX = "__GUARDRAIL_BLOCKED__:"

SYSTEM_PROMPT = r"""
You are AutoDrive Egypt AI, a concise Egyptian car-sales and customer-service orchestrator.
Mirror the customer's language. If they write Egyptian Arabic, answer naturally in Egyptian Arabic.

GROUNDING
- Never invent inventory, price, mileage, availability, vehicle specs, booking IDs, cancellation status, or lead IDs.
- Imported catalog rows are NOT guaranteed live stock. Say "حسب البيانات المتاحة عندي" / "ضمن البيانات المسجلة".
- Current structured state is authoritative for what the customer wants NOW.
- Historical summary is only historical context and must never override current structured state.
- The ACTIVE RECOMMENDATION SNAPSHOT is the only source of truth for visible positions: الأولى/التانية/التالتة.
- If an ordinal is ambiguous or there is no active displayed snapshot, ask for clarification or show a fresh list. Never guess.
- Dealership policies use Dealership Knowledge RAG.
- Missing external technical facts use VehicleResearchAgent only. Do not call generic web tools directly.
- Respect VehicleResearchAgent status exactly: verified / partial / not_found / conflict / error.
- Never call a third-party marketplace/database "official".
- Never claim a test-drive booking unless the deterministic workflow has all required fields and the DB returns request_id.
- Never claim cancellation unless the actual DB row was updated.
""".strip()

SUMMARIZER_INSTRUCTION = r"""
Summarize the older conversation for an Egyptian car dealership agent.
Use only explicit facts. Preserve preferences, vehicles discussed, decisions and business-action progress.
Do not invent facts. Keep historical facts historical; do not rewrite them as the customer's current state.
Keep it concise.
""".strip()


class GeminiCarSalesAgent(Component):
    display_name = "Gemini Car Sales Agent + Memory"
    description = "Stateful Egypt new+used car sales orchestrator with deterministic workflow gates."
    icon = "bot"
    name = "GeminiCarSalesAgent"

    inputs = [
        MessageInput(name="input_value", display_name="Guarded Customer Message", required=True),
        HandleInput(name="tools", display_name="Tools", input_types=["Tool"], required=False, is_list=True),
        SecretStrInput(name="google_api_key", display_name="Google API Key", required=False),
        MessageTextInput(name="model", display_name="Model", value="gemini-3.5-flash-lite", advanced=True),
        MultilineInput(name="system_prompt", display_name="Agent Instructions", value=SYSTEM_PROMPT, advanced=True),
        IntInput(name="memory_messages", display_name="Memory Messages", value=MEMORY_RECENT_MESSAGE_LIMIT, advanced=True),
        IntInput(name="max_iterations", display_name="Max Agent Iterations", value=10, advanced=True),
        MessageTextInput(name="db_path", display_name="Memory / Business DB", value=DEFAULT_DB, advanced=True),
    ]
    outputs = [Output(display_name="Response", name="response", method="run_agent")]

    def _key(self) -> str:
        raw = getattr(self, "google_api_key", None)
        if raw:
            try:
                return raw.get_secret_value()
            except AttributeError:
                return str(raw)
        return os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")

    def _session_id(self, msg: Message) -> str:
        return getattr(msg, "session_id", "") or getattr(self.graph, "session_id", "") or "langflow-playground"

    @staticmethod
    def _missing_test_drive_fields(action: dict[str, Any]) -> list[str]:
        payload = action.get("payload", {})
        missing: list[str] = []
        if not action.get("entity_id") and not payload.get("car_id"):
            missing.append("car_id")
        for field in ("customer_name", "phone", "preferred_date", "preferred_time"):
            if not payload.get(field):
                missing.append(field)
        return missing

    @staticmethod
    def _missing_sales_lead_fields(action: dict[str, Any]) -> list[str]:
        payload = action.get("payload", {})
        return [field for field in ("customer_name", "phone") if not payload.get(field)]

    @staticmethod
    def _ask_for_missing(missing: list[str], action_type: str) -> str:
        if not missing:
            return ""
        if "car_id" in missing:
            return "حددلي أنهي عربية تقصد من آخر قائمة ظاهرة قدامك الأول."
        if action_type == "test_drive":
            labels = {
                "customer_name": "اسم حضرتك",
                "phone": "رقم الموبايل",
                "preferred_date": "اليوم/التاريخ المناسب",
                "preferred_time": "الساعة المناسبة",
            }
        else:
            labels = {"customer_name": "اسم حضرتك", "phone": "رقم الموبايل"}
        wanted = [labels[m] for m in missing if m in labels]
        if len(wanted) == 1:
            return f"تمام. ناقص بس {wanted[0]}."
        return "تمام. ناقص بس: " + "، ".join(wanted) + "."

    @staticmethod
    def _is_browse_again(text: str) -> bool:
        t = str(text or "").lower()
        return bool(re.search(r"(?:عايز|عاوز|وريني|هات|شوف)\s+(?:اشوف|أشوف|عربيه|عربية|سياره|سيارة).*?(?:تاني|تانية|اخرى|أخرى)", t))

    @staticmethod
    def _has_ordinal(text: str) -> bool:
        return bool(re.search(
            r"(?:الأولى|الاولى|الأول|الاول|التانية|الثانية|التالتة|الثالثة|الرابعة|الخامسة|آخر واحدة|اخر واحده|أول اتنين|اول اتنين)",
            str(text or ""),
        ))

    @staticmethod
    def _is_selection_phrase(text: str) -> bool:
        t = str(text or "").lower()
        return any(x in t for x in ("عجبتني", "خلينا في", "اختار", "اخترت", "مهتم بـ", "مهتم ب"))

    @staticmethod
    def _format_recommendations(rec_set: list[dict[str, Any]], criteria: dict[str, Any]) -> str:
        if not rec_set:
            return "حسب البيانات المتاحة عندي، ملقتش نتائج مطابقة للمواصفات الحالية. ممكن نوسع شرط من الشروط."
        lines = ["حسب البيانات المتاحة عندي، دي أقرب الخيارات المطابقة:"]
        for item in rec_set:
            pos = item.get("position")
            name = item.get("display_name") or f"Car #{item.get('primary_car_id')}"
            year = item.get("year") or "-"
            price = item.get("price")
            price_text = f"{price:,.0f} جنيه" if isinstance(price, (int, float)) else "السعر غير مسجل"
            bits = [str(item.get("body_type") or ""), str(item.get("condition") or ""), str(item.get("transmission") or "")]
            bits = [b for b in bits if b]
            extra = "، ".join(bits)
            lines.append(f"{pos}. **{name} ({year})** — {price_text}" + (f" — {extra}" if extra else ""))
        lines.append("قولّي رقم/ترتيب العربية لو عايز تفاصيل أو مقارنة.")
        return "\n".join(lines)

    def _search_current_state(self, mm: MemoryManager, session_id: str, db_path: str) -> tuple[str, dict[str, Any] | None]:
        state = mm.get_session_state(session_id)
        kwargs: dict[str, Any] = {}
        mapping = {
            "brand": "brand",
            "model": "model",
            "condition": "condition",
            "body_type": "body_type",
            "min_year": "min_year",
            "max_year": "max_year",
            "min_price": "min_price",
            "max_price": "max_price",
            "location": "city",
            "fuel_type": "fuel",
            "transmission": "transmission",
            "max_mileage": "max_mileage",
        }
        for state_key, search_key in mapping.items():
            value = state.get(state_key)
            if value not in (None, "", []):
                kwargs[search_key] = value
        if not kwargs:
            return "", None

        kwargs["limit"] = 15
        raw_rows = search_cars(db_path, **kwargs)
        rec_set = build_recommendation_set(raw_rows, limit=5)
        criteria = dict(kwargs)
        criteria["limit"] = 5
        snapshot = mm.create_recommendation_snapshot(session_id, criteria, rec_set) if rec_set else None
        return self._format_recommendations(rec_set, criteria), snapshot

    def _persist_direct(self, mm: MemoryManager, session_id: str, user_text: str, response: str) -> Message:
        mm.save_message(session_id, "user", user_text)
        mm.save_message(session_id, "assistant", response)
        mm.maybe_update_summary(session_id)
        self.status = response
        return Message(text=response, session_id=session_id)

    def run_agent(self) -> Message:
        msg = self.input_value if isinstance(self.input_value, Message) else Message(text=str(self.input_value or ""))
        session_id = self._session_id(msg)
        if msg.text.startswith(BLOCK_PREFIX):
            return Message(text=msg.text[len(BLOCK_PREFIX):], session_id=session_id)

        key = self._key()
        if not key:
            return Message(
                text="Configuration error: GOOGLE_API_KEY is not set. Add your Gemini API key to the Agent or environment.",
                session_id=session_id,
            )

        db_path = self.db_path or DEFAULT_DB
        set_current_session_id(session_id)
        user_text = msg.text.strip()
        set_current_user_text(user_text)
        mm = MemoryManager(db_path)

        # 1) Current explicit preferences are applied BEFORE selection/action handling.
        new_prefs = extract_preferences(user_text)
        if new_prefs:
            mm.apply_preference_updates_and_invalidate(session_id, new_prefs)

        # 2) Cancellation is session-scoped only. Never cancel a global "latest" booking.
        if detect_cancellation(user_text):
            pending = mm.get_pending_action(session_id)
            if pending:
                mm.cancel_pending_action(session_id)
                return self._persist_direct(mm, session_id, user_text, "تم إلغاء الطلب اللي كان لسه قيد التجهيز، ومفيش حجز جديد اتسجل منه.")

            last_booking = mm.get_last_completed_action(session_id, action_type="test_drive")
            req_id = None
            if last_booking:
                req_id = (last_booking.get("payload", {}).get("_result") or {}).get("request_id")
            if req_id and cancel_test_drive(db_path, int(req_id), notes="Customer cancelled via chat"):
                return self._persist_direct(
                    mm,
                    session_id,
                    user_text,
                    f"تم إلغاء حجز تجربة القيادة رقم #{req_id} بنجاح في النظام.",
                )
            return self._persist_direct(
                mm,
                session_id,
                user_text,
                "مش لاقي حجز مكتمل مرتبط بالمحادثة دي أقدر ألغيه تلقائياً.",
            )

        # 3) Reschedule/new-details only manipulate pending state; they never insert.
        if detect_reschedule(user_text):
            selected = mm.get_selected_car(session_id)
            ref_res = resolve_recommendation_reference(session_id, user_text, db_path)
            target = (
                ref_res.get("items", [{}])[0].get("primary_car_id")
                if ref_res.get("status") == "resolved"
                else selected
            )
            last_completed = mm.get_last_completed_action(session_id, action_type="test_drive")
            old_payload = (last_completed or {}).get("payload", {})
            mm.create_pending_action(
                session_id,
                "test_drive",
                entity_id=target,
                payload={
                    "customer_name": old_payload.get("customer_name"),
                    "phone": old_payload.get("phone"),
                    "preferred_date": None,
                    "preferred_time": None,
                    "notes": None,
                },
            )
            if target:
                mm.set_selected_car(session_id, int(target))
            return self._persist_direct(
                mm,
                session_id,
                user_text,
                "تمام. الحجز الجديد لسه **ما اتسجلش**. ابعتلي اليوم/التاريخ والساعة الجديدة.",
            )

        if detect_new_details(user_text):
            current = mm.get_pending_action(session_id)
            target = current.get("entity_id") if current else mm.get_selected_car(session_id)
            mm.create_pending_action(
                session_id,
                "test_drive",
                entity_id=target,
                payload={
                    "customer_name": None,
                    "phone": None,
                    "preferred_date": None,
                    "preferred_time": None,
                    "notes": None,
                },
            )
            return self._persist_direct(
                mm,
                session_id,
                user_text,
                "تمام، هنستخدم بيانات جديدة بالكامل. ابعت الاسم ورقم الموبايل واليوم والساعة.",
            )

        # 4) Existing pending action consumes the turn deterministically.
        active_action = mm.get_pending_action(session_id)
        if active_action:
            action_type = active_action.get("action_type")
            updates = extract_action_inputs(action_type, user_text, active_action.get("payload", {}))
            if updates:
                active_action = mm.update_pending_action(session_id, payload_updates=updates) or active_action

            if action_type == "test_drive":
                missing = self._missing_test_drive_fields(active_action)
                if missing:
                    return self._persist_direct(mm, session_id, user_text, self._ask_for_missing(missing, "test_drive"))
                payload = active_action.get("payload", {})
                car_id = active_action.get("entity_id") or payload.get("car_id")
                try:
                    result = create_test_drive(
                        db_path,
                        str(payload["customer_name"]),
                        str(payload["phone"]),
                        int(car_id),
                        str(payload["preferred_date"]),
                        str(payload["preferred_time"]),
                        payload.get("notes"),
                    )
                    mm.complete_pending_action(session_id, result)
                    car = get_car(db_path, int(car_id))
                    car_name = f"{car.get('brand')} {car.get('model')}" if car else f"ID {car_id}"
                    response = (
                        f"تم حجز تجربة القيادة بنجاح. رقم الحجز **#{result['request_id']}** "
                        f"لـ **{car_name}** يوم {payload['preferred_date']} الساعة {payload['preferred_time']}."
                    )
                    return self._persist_direct(mm, session_id, user_text, response)
                except Exception as exc:
                    logger.warning("Deterministic test-drive insert failed: %s", exc)
                    return self._persist_direct(
                        mm,
                        session_id,
                        user_text,
                        "حصلت مشكلة أثناء تسجيل الحجز في قاعدة البيانات، فمش هأكد الحجز دلوقتي.",
                    )

            if action_type == "sales_lead":
                missing = self._missing_sales_lead_fields(active_action)
                if missing:
                    return self._persist_direct(mm, session_id, user_text, self._ask_for_missing(missing, "sales_lead"))
                payload = active_action.get("payload", {})
                car_id = active_action.get("entity_id") or payload.get("car_id")
                try:
                    result = create_sales_lead(
                        db_path,
                        str(payload["customer_name"]),
                        str(payload["phone"]),
                        int(car_id) if car_id else None,
                        payload.get("email"),
                        payload.get("notes"),
                    )
                    mm.complete_pending_action(session_id, result)
                    return self._persist_direct(
                        mm,
                        session_id,
                        user_text,
                        f"تم تسجيل طلب التواصل مع المبيعات بنجاح. رقم الطلب **#{result['lead_id']}**.",
                    )
                except Exception as exc:
                    logger.warning("Deterministic lead insert failed: %s", exc)
                    return self._persist_direct(
                        mm,
                        session_id,
                        user_text,
                        "حصلت مشكلة أثناء تسجيل طلب التواصل، فمش هأكد إنه اتسجل.",
                    )

        # 5) If preference changed, immediately show a fresh list from CURRENT state.
        if new_prefs:
            response, _ = self._search_current_state(mm, session_id, db_path)
            if response:
                return self._persist_direct(mm, session_id, user_text, response)

        # 6) "Show me another car" should reuse known current state, never re-ask it.
        if self._is_browse_again(user_text):
            response, _ = self._search_current_state(mm, session_id, db_path)
            if response:
                return self._persist_direct(mm, session_id, user_text, response)

        # 7) Ordinal selection is deterministic and only against an actually active snapshot.
        ref_res = resolve_recommendation_reference(session_id, user_text, db_path)
        if self._has_ordinal(user_text):
            if ref_res.get("status") == "no_active_snapshot":
                response, _ = self._search_current_state(mm, session_id, db_path)
                if response:
                    return self._persist_direct(
                        mm,
                        session_id,
                        user_text,
                        "القائمة القديمة مبقتش مناسبة لتفضيلاتك الحالية، فدي قائمة جديدة:\n\n" + response,
                    )
                return self._persist_direct(
                    mm,
                    session_id,
                    user_text,
                    "مفيش قائمة حالية ظاهرة أقدر أفسر منها «الأولى/التانية». خلينا نعمل بحث جديد الأول.",
                )
            if ref_res.get("status") == "ambiguous":
                return self._persist_direct(
                    mm,
                    session_id,
                    user_text,
                    "المرجع مش واضح بالنسبة للقوائم اللي اتعرضت. قولّي اسم العربية أو تقصد أنهي قائمة.",
                )
            if ref_res.get("status") == "resolved" and len(ref_res.get("items", [])) == 1 and self._is_selection_phrase(user_text):
                item = ref_res["items"][0]
                mm.set_selected_car(
                    session_id,
                    int(item["primary_car_id"]),
                    snapshot_id=ref_res.get("snapshot_id"),
                    position=item.get("position"),
                )
                return self._persist_direct(
                    mm,
                    session_id,
                    user_text,
                    f"تمام، ثبتّ اختيارك على **{item.get('display_name')}** (رقم {item.get('position')} في القائمة الحالية).",
                )

        # 8) New business action requires a deterministic target; ambiguity is never guessed.
        new_action_type, _ = detect_action_trigger(user_text)
        if new_action_type == "test_drive":
            ref_res = resolve_recommendation_reference(session_id, user_text, db_path)
            target = None
            if ref_res.get("status") == "resolved" and len(ref_res.get("items", [])) == 1:
                target = ref_res["items"][0].get("primary_car_id")
            if not target:
                target = mm.get_selected_car(session_id)
            if not target:
                return self._persist_direct(
                    mm,
                    session_id,
                    user_text,
                    "تمام، بس لازم تحددلي العربية الأول من آخر قائمة ظاهرة قبل ما أبدأ طلب تجربة القيادة.",
                )
            mm.create_pending_action(session_id, "test_drive", entity_id=int(target))
            mm.set_selected_car(session_id, int(target))
            action = mm.get_pending_action(session_id)
            missing = self._missing_test_drive_fields(action or {})
            return self._persist_direct(mm, session_id, user_text, self._ask_for_missing(missing, "test_drive"))

        if new_action_type == "sales_lead":
            mm.create_pending_action(session_id, "sales_lead", entity_id=mm.get_selected_car(session_id))
            action = mm.get_pending_action(session_id)
            missing = self._missing_sales_lead_fields(action or {})
            return self._persist_direct(mm, session_id, user_text, self._ask_for_missing(missing, "sales_lead"))

        # 9) Remaining technical/policy/conversational queries go to Gemini with gated tools.
        context_data = mm.build_memory_context(session_id, user_text)
        recent_raw = context_data["recent_messages"]
        lc_messages = [{"role": item["role"], "content": item["content"]} for item in recent_raw]
        lc_messages.append({"role": "user", "content": user_text})

        augmented_system_prompt = (
            f"{self.system_prompt or SYSTEM_PROMPT}\n\n"
            "=== CURRENT INTERMEDIATE MEMORY CONTEXT ===\n"
            f"{context_data['formatted_context']}\n"
            "==========================================="
        )

        model = ChatGoogleGenerativeAI(model=self.model or "gemini-3.5-flash-lite", google_api_key=key)
        tools = list(self.tools or [])
        agent = create_agent(model=model, tools=tools, system_prompt=augmented_system_prompt)
        result = agent.invoke(
            {"messages": lc_messages},
            config={"recursion_limit": max(4, int(self.max_iterations or 10) * 2)},
        )

        final = result["messages"][-1]
        content = getattr(final, "content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        content = str(content)

        mm.save_message(session_id, "user", user_text)
        mm.save_message(session_id, "assistant", content)

        def _llm_summarizer(existing_sum: str, new_msgs: list[dict[str, Any]]) -> str:
            try:
                transcript = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in new_msgs)
                prompt = (
                    f"{SUMMARIZER_INSTRUCTION}\n\n"
                    f"EXISTING SUMMARY:\n{existing_sum if existing_sum else 'None'}\n\n"
                    f"NEW CONVERSATION TURNS:\n{transcript}\n\n"
                    "UPDATED CONCISE SUMMARY:"
                )
                response = model.invoke(prompt)
                return str(getattr(response, "content", "") or existing_sum).strip()
            except Exception as exc:
                logger.warning("LLM summarizer failed; using deterministic fallback: %s", exc)
                return mm._default_summarize_messages(existing_sum, new_msgs)

        mm.maybe_update_summary(session_id, summarizer_fn=_llm_summarizer)
        self.status = content
        return Message(text=content, session_id=session_id)
