import re

from lfx.custom import Component
from lfx.io import IntInput, MessageTextInput, Output
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, get_car
from memory_manager import MemoryManager, get_current_session_id, resolve_recommendation_reference
from request_context import get_current_user_text


class GetCarDetails(Component):
    display_name = "Get Car Details"
    description = "Get factual details for a specific visible recommendation or inventory id."
    icon = "car"
    name = "GetCarDetails"

    inputs = [
        IntInput(name="car_id", display_name="Car ID", tool_mode=True, required=True),
        MessageTextInput(name="session_id", display_name="Session ID", tool_mode=True, required=False),
        MessageTextInput(name="db_path", display_name="DB Path", value=DEFAULT_DB, advanced=True),
    ]
    outputs = [Output(display_name="Car", name="car", method="get_details")]

    def get_details(self) -> Data:
        db = self.db_path or DEFAULT_DB
        sid = getattr(self, "session_id", None) or get_current_session_id()
        cid = int(self.car_id)
        current_user_text = get_current_user_text()

        # Visible ordinal references are authoritative over LLM-supplied numeric IDs.
        if sid and current_user_text and re.search(
            r"(?:الاول|الأول|الاولى|الأولى|التاني|الثاني|التانيه|الثانية|التالت|الثالث|التالته|الثالثة|الرابع|الخامس|اخر|آخر)",
            current_user_text,
        ):
            ref_res = resolve_recommendation_reference(sid, current_user_text, db)
            if ref_res.get("status") == "resolved" and len(ref_res.get("items", [])) == 1:
                cid = int(ref_res["items"][0]["primary_car_id"])
            elif ref_res.get("status") in {"ambiguous", "no_active_snapshot"}:
                result = Data(
                    data={
                        "found": False,
                        "status": "needs_clarification",
                        "reason": ref_res.get("reason"),
                        "car": None,
                    }
                )
                self.status = result.data
                return result

        car = get_car(db, cid)
        if car and sid:
            mm = MemoryManager(db)
            active_snap = mm.get_active_snapshot(sid)
            snap_id = None
            pos = None
            if active_snap and active_snap.get("items"):
                for it in active_snap["items"]:
                    if it.get("primary_car_id") == cid or cid in it.get("variant_ids", []):
                        snap_id = active_snap["id"]
                        pos = it.get("position")
                        break
            mm.set_selected_car(sid, cid, snapshot_id=snap_id, position=pos)

        result = Data(data={"found": bool(car), "status": "ok" if car else "not_found", "car": car})
        self.status = result.data
        return result
