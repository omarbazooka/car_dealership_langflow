from lfx.custom import Component
from lfx.io import IntInput, MessageTextInput, Output
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, cancel_test_drive
from memory_manager import MemoryManager, get_current_session_id


class CancelTestDrive(Component):
    display_name = "Cancel Test Drive"
    description = (
        "Formally cancel a test-drive booking linked to the current conversation session. "
        "Never falls back to a global latest booking."
    )
    icon = "calendar-x"
    name = "CancelTestDrive"

    inputs = [
        IntInput(name="request_id", display_name="Request ID", tool_mode=True, required=False),
        MessageTextInput(name="notes", display_name="Cancellation Reason / Notes", tool_mode=True, required=False),
        MessageTextInput(name="session_id", display_name="Session ID", tool_mode=True, required=False),
        MessageTextInput(name="db_path", display_name="DB Path", value=DEFAULT_DB, advanced=True),
    ]
    outputs = [Output(display_name="Result", name="result", method="run_cancel")]

    def run_cancel(self) -> Data:
        db = self.db_path or DEFAULT_DB
        sid = getattr(self, "session_id", None) or get_current_session_id()
        mm = MemoryManager(db)

        if not sid:
            result = {
                "success": False,
                "status": "blocked",
                "message": "تعذر تحديد جلسة المحادثة، لذلك لن يتم إلغاء أي حجز تلقائياً.",
            }
            self.status = result
            return Data(data=result)

        last_action = mm.get_last_completed_action(sid, action_type="test_drive")
        session_request_id = None
        if last_action:
            session_request_id = (last_action.get("payload", {}).get("_result") or {}).get("request_id")

        requested_id = getattr(self, "request_id", None)
        if requested_id not in (None, "", 0):
            requested_id = int(requested_id)
            # Never let an LLM-supplied request ID escape the current session lineage.
            if not session_request_id or requested_id != int(session_request_id):
                result = {
                    "success": False,
                    "status": "blocked",
                    "request_id": requested_id,
                    "message": "رقم الحجز المطلوب مش مرتبط بآخر حجز مؤكد في المحادثة دي، لذلك لم يتم إلغاؤه.",
                }
                self.status = result
                return Data(data=result)
            request_id = requested_id
        else:
            request_id = int(session_request_id) if session_request_id else None

        if not request_id:
            pending_cancelled = mm.cancel_pending_action(sid)
            result = {
                "success": bool(pending_cancelled),
                "status": "cancelled_pending" if pending_cancelled else "not_found",
                "message": (
                    "تم إلغاء الطلب اللي كان لسه قيد التجهيز، ومفيش حجز مكتمل اتأثر."
                    if pending_cancelled
                    else "مش لاقي حجز مكتمل مرتبط بالمحادثة دي لإلغائه."
                ),
            }
            self.status = result
            return Data(data=result)

        cancelled = cancel_test_drive(db, request_id, notes=self.notes or "Customer requested cancellation")
        result = {
            "success": cancelled.get("status") == "CANCELLED",
            "status": "cancelled" if cancelled.get("status") == "CANCELLED" else "failed",
            "request_id": request_id,
            "cancelled_at": cancelled.get("cancelled_at"),
            "message": f"تم إلغاء حجز تجربة القيادة رقم #{request_id} بنجاح في النظام.",
        }
        self.status = result
        return Data(data=result)
