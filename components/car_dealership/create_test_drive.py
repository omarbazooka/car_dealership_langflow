from lfx.custom import Component
from lfx.io import IntInput, MessageTextInput, Output
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, create_test_drive
from memory_manager import MemoryManager, get_current_session_id


_REQUIRED = ("customer_name", "phone", "preferred_date", "preferred_time")


class CreateTestDrive(Component):
    display_name = "Create Test Drive"
    description = (
        "Creates a REAL test-drive request in SQLite only when the deterministic pending-action "
        "state confirms the customer name, phone, selected car, date and time."
    )
    icon = "calendar-check"
    name = "CreateTestDrive"

    inputs = [
        MessageTextInput(name="customer_name", display_name="Customer Name", tool_mode=True, required=True),
        MessageTextInput(name="phone", display_name="Phone", tool_mode=True, required=True),
        IntInput(name="car_id", display_name="Car ID", tool_mode=True, required=True),
        MessageTextInput(name="preferred_date", display_name="Preferred Date", tool_mode=True, required=True),
        MessageTextInput(name="preferred_time", display_name="Preferred Time", tool_mode=True, required=True),
        MessageTextInput(name="notes", display_name="Notes", tool_mode=True, required=False),
        MessageTextInput(name="db_path", display_name="DB Path", value=DEFAULT_DB, advanced=True),
    ]
    outputs = [Output(display_name="Created Request", name="request", method="create_request")]

    def create_request(self) -> Data:
        db = self.db_path or DEFAULT_DB
        sid = get_current_session_id()
        mm = MemoryManager(db)

        # Live-path safety gate: the LLM is not allowed to invent or complete missing
        # booking fields. The pending action is the authoritative workflow state.
        action = mm.get_pending_action(sid) if sid else None
        if not action or action.get("action_type") != "test_drive":
            result = {
                "status": "blocked",
                "reason": "no_pending_test_drive",
                "missing_fields": ["pending_action"],
                "message": "لازم يبدأ طلب تجربة قيادة واضح الأول قبل إنشاء الحجز.",
            }
            self.status = result
            return Data(data=result)

        payload = action.get("payload", {})
        car_id = action.get("entity_id") or payload.get("car_id")
        missing = [field for field in _REQUIRED if not payload.get(field)]
        if not car_id:
            missing.insert(0, "car_id")

        if missing:
            result = {
                "status": "blocked",
                "reason": "pending_action_incomplete",
                "action_id": action.get("id"),
                "missing_fields": missing,
                "message": "الحجز لم يتم لأن بيانات تجربة القيادة ما زالت غير مكتملة.",
            }
            self.status = result
            return Data(data=result)

        # Idempotency: a pending action can be completed only once. Once completed,
        # get_pending_action() no longer returns it, so retries cannot create duplicates.
        result = create_test_drive(
            db,
            str(payload["customer_name"]),
            str(payload["phone"]),
            int(car_id),
            str(payload["preferred_date"]),
            str(payload["preferred_time"]),
            payload.get("notes") or self.notes or None,
        )
        if result.get("request_id"):
            mm.complete_pending_action(sid, result)

        data = Data(data=result)
        self.status = result
        return data
