from lfx.custom import Component
from lfx.io import IntInput, MessageTextInput, Output
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, create_sales_lead
from memory_manager import MemoryManager, get_current_session_id


class CreateSalesLead(Component):
    display_name = "Create Sales Lead"
    description = (
        "Creates a REAL sales lead in SQLite only from the deterministic pending-action "
        "state after the customer name and phone are confirmed."
    )
    icon = "user-plus"
    name = "CreateSalesLead"

    inputs = [
        MessageTextInput(name="customer_name", display_name="Customer Name", tool_mode=True, required=True),
        MessageTextInput(name="phone", display_name="Phone", tool_mode=True, required=True),
        MessageTextInput(name="email", display_name="Email", tool_mode=True, required=False),
        IntInput(name="car_id", display_name="Car ID", tool_mode=True, required=False),
        MessageTextInput(name="notes", display_name="Notes", tool_mode=True, required=False),
        MessageTextInput(name="db_path", display_name="DB Path", value=DEFAULT_DB, advanced=True),
    ]
    outputs = [Output(display_name="Created Lead", name="lead", method="create_lead")]

    def create_lead(self) -> Data:
        db = self.db_path or DEFAULT_DB
        sid = get_current_session_id()
        mm = MemoryManager(db)

        action = mm.get_pending_action(sid) if sid else None
        if not action or action.get("action_type") != "sales_lead":
            result = {
                "status": "blocked",
                "reason": "no_pending_sales_lead",
                "missing_fields": ["pending_action"],
                "message": "لازم يبدأ طلب تواصل مع المبيعات الأول قبل إنشاء الـLead.",
            }
            self.status = result
            return Data(data=result)

        payload = action.get("payload", {})
        missing = [field for field in ("customer_name", "phone") if not payload.get(field)]
        if missing:
            result = {
                "status": "blocked",
                "reason": "pending_action_incomplete",
                "action_id": action.get("id"),
                "missing_fields": missing,
                "message": "طلب المبيعات لم يتم لأن بيانات التواصل غير مكتملة.",
            }
            self.status = result
            return Data(data=result)

        car_id = action.get("entity_id") or payload.get("car_id")
        result = create_sales_lead(
            db,
            str(payload["customer_name"]),
            str(payload["phone"]),
            int(car_id) if car_id else None,
            payload.get("email") or self.email or None,
            payload.get("notes") or self.notes or None,
        )
        if result.get("lead_id"):
            mm.complete_pending_action(sid, result)

        data = Data(data=result)
        self.status = result
        return data
