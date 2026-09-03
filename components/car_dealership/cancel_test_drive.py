import json
from lfx.custom import Component
from lfx.io import IntInput, MessageTextInput, Output
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, cancel_test_drive, get_test_drive_requests
from memory_manager import MemoryManager, get_current_session_id


class CancelTestDrive(Component):
    display_name = "Cancel Test Drive"
    description = (
        "Formally cancel an existing test drive booking in the database. "
        "Use this tool whenever a customer asks to cancel a previously confirmed test drive booking."
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
        req_id = getattr(self, "request_id", None)
        notes = getattr(self, "notes", None)

        mm = MemoryManager(db)
        if not req_id and sid:
            # Look up last completed test drive action for this session
            last_action = mm.get_last_completed_action(sid, action_type="test_drive")
            if last_action:
                result_meta = last_action.get("payload", {}).get("_result", {})
                req_id = result_meta.get("request_id")
            if not req_id:
                # Look up recent test drive requests for customer
                recent_reqs = get_test_drive_requests(db, limit=5)
                if recent_reqs:
                    req_id = recent_reqs[0]["id"]

        if not req_id:
            res_data = {
                "success": False,
                "status": "not_found",
                "message": "لم يتم العثور على حجز تجربة قيادة سابق لإلغائه.",
            }
            self.status = res_data
            return Data(data=res_data)

        req_id = int(req_id)
        success = cancel_test_drive(db, req_id, notes=notes or "Customer requested cancellation")

        if success:
            if sid:
                mm.cancel_pending_action(sid)
            res_data = {
                "success": True,
                "status": "cancelled",
                "request_id": req_id,
                "message": f"تم إلغاء حجز تجربة القيادة رقم #{req_id} بنجاح في النظام.",
            }
        else:
            res_data = {
                "success": False,
                "status": "failed",
                "request_id": req_id,
                "message": f"تعذر إلغاء الحجز رقم #{req_id} أو أنه ملغي بالفعل.",
            }

        self.status = res_data
        return Data(data=res_data)
