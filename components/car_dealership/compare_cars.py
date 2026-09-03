import re

from lfx.custom import Component
from lfx.io import MessageTextInput, Output
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, compare_cars
from memory_manager import get_current_session_id, resolve_recommendation_reference


class CompareCars(Component):
    display_name = "Compare Cars"
    description = "Return structured factual fields for two or more car IDs so the agent can compare them without inventing specs."
    icon = "columns-2"
    name = "CompareCars"

    inputs = [
        MessageTextInput(name="car_ids", display_name="Car IDs", info="Comma-separated IDs or ordinals e.g. 2,5 or 'أول اتنين'", tool_mode=True, required=True),
        MessageTextInput(name="session_id", display_name="Session ID", tool_mode=True, required=False),
        MessageTextInput(name="db_path", display_name="DB Path", value=DEFAULT_DB, advanced=True),
    ]
    outputs = [Output(display_name="Comparison Data", name="comparison", method="run_compare")]

    def run_compare(self) -> Data:
        raw_ids_str = str(self.car_ids or "")
        ids = [int(x) for x in re.findall(r"\d+", raw_ids_str)]
        sid = getattr(self, "session_id", None) or get_current_session_id()
        
        # If user passed text/ordinals like "أول اتنين" instead of numeric IDs
        if not ids and sid:
            ref_res = resolve_recommendation_reference(sid, raw_ids_str, self.db_path or DEFAULT_DB)
            if ref_res.get("status") == "resolved":
                ids = [it["primary_car_id"] for it in ref_res.get("items", [])]

        cars = compare_cars(self.db_path or DEFAULT_DB, ids[:5])
        result = Data(data={"requested_ids": ids[:5], "cars": cars})
        self.status = result.data
        return result
