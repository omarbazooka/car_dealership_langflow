import json

from lfx.custom import Component
from lfx.io import FloatInput, IntInput, MessageTextInput, Output
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, build_recommendation_set, search_cars
from memory_manager import MemoryManager, get_current_session_id


class SearchUsedCars(Component):
    display_name = "Search Cars (New + Used)"
    description = (
        "Search the structured Egyptian car inventory containing both NEW and USED cars. "
        "Use condition='new' or condition='used' when the customer specifies it."
    )
    icon = "search"
    name = "SearchUsedCars"  # Keep stable so existing flow builder can resolve it.

    inputs = [
        MessageTextInput(name="brand", display_name="Brand", tool_mode=True, required=False),
        MessageTextInput(name="model", display_name="Model", tool_mode=True, required=False),
        MessageTextInput(name="condition", display_name="Condition (new/used)", tool_mode=True, required=False),
        MessageTextInput(name="body_type", display_name="Body Type", tool_mode=True, required=False),
        IntInput(name="min_year", display_name="Min Year", tool_mode=True, required=False),
        IntInput(name="max_year", display_name="Max Year", tool_mode=True, required=False),
        FloatInput(name="min_price", display_name="Min Price EGP", tool_mode=True, required=False),
        FloatInput(name="max_price", display_name="Max Price EGP", tool_mode=True, required=False),
        MessageTextInput(name="city", display_name="Location / City", tool_mode=True, required=False),
        MessageTextInput(name="fuel", display_name="Fuel", tool_mode=True, required=False),
        MessageTextInput(name="transmission", display_name="Transmission", tool_mode=True, required=False),
        MessageTextInput(name="drive", display_name="Drive", tool_mode=True, required=False),
        FloatInput(name="max_mileage", display_name="Max Mileage", tool_mode=True, required=False),
        FloatInput(name="max_age", display_name="Max Vehicle Age", tool_mode=True, required=False),
        IntInput(name="limit", display_name="Result Limit", value=5, tool_mode=True, required=False),
        MessageTextInput(name="session_id", display_name="Session ID", tool_mode=True, required=False),
        MessageTextInput(name="db_path", display_name="DB Path", value=DEFAULT_DB, advanced=True),
    ]
    outputs = [Output(display_name="Results", name="results", method="run_search")]

    def run_search(self) -> Data:
        kwargs = {}
        for k in ("brand", "model", "condition", "body_type", "city", "fuel", "transmission", "drive"):
            v = getattr(self, k, None)
            if v:
                kwargs[k] = v
        for k in ("min_price", "max_price", "max_mileage", "max_age"):
            v = getattr(self, k, None)
            if v not in (None, ""):
                kwargs[k] = float(v)
        for k in ("min_year", "max_year"):
            v = getattr(self, k, None)
            if v not in (None, "", 0):
                kwargs[k] = int(v)
        limit = int(getattr(self, "limit", 5) or 5)
        # Query slightly more rows from database so variant grouping doesn't truncate distinct models
        kwargs["limit"] = max(limit * 3, 15)
        db = self.db_path or DEFAULT_DB
        raw_rows = search_cars(db, **kwargs)

        # Group duplicate variants (e.g. colors/packages) into unified customer-visible positions
        rec_set = build_recommendation_set(raw_rows, limit=limit)

        sid = getattr(self, "session_id", None) or get_current_session_id()
        snapshot_id = None
        if sid and rec_set:
            mm = MemoryManager(db)
            criteria = {k: v for k, v in kwargs.items() if k != "limit"}
            criteria["limit"] = limit
            snap = mm.create_recommendation_snapshot(sid, criteria, rec_set)
            snapshot_id = snap["id"]

        result = Data(data={
            "count": len(rec_set),
            "snapshot_id": snapshot_id,
            "recommendations": rec_set,
            "cars": rec_set,
        })
        self.status = json.dumps(result.data, ensure_ascii=False)
        return result
