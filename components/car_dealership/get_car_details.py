from lfx.custom import Component
from lfx.io import IntInput, MessageTextInput, Output
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, get_car
from memory_manager import MemoryManager, get_current_session_id


class GetCarDetails(Component):
    display_name = "Get Car Details"
    description = "Get factual details for a specific car listing by inventory id."
    icon = "car"
    name = "GetCarDetails"

    inputs = [
        IntInput(name="car_id", display_name="Car ID", tool_mode=True, required=True),
        MessageTextInput(name="session_id", display_name="Session ID", tool_mode=True, required=False),
        MessageTextInput(name="db_path", display_name="DB Path", value=DEFAULT_DB, advanced=True),
    ]
    outputs = [Output(display_name="Car", name="car", method="get_details")]

    def get_details(self) -> Data:
        cid = int(self.car_id)
        db = self.db_path or DEFAULT_DB
        car = get_car(db, cid)
        if car:
            sid = getattr(self, "session_id", None) or get_current_session_id()
            if sid:
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
        result = Data(data={"found": bool(car), "car": car})
        self.status = result.data
        return result
