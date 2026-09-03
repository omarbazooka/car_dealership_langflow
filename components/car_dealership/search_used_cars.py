import json

from lfx.custom import Component
from lfx.io import FloatInput, IntInput, MessageTextInput, Output
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, search_cars


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
        kwargs["limit"] = int(getattr(self, "limit", 5) or 5)
        rows = search_cars(self.db_path or DEFAULT_DB, **kwargs)
        result = Data(data={"count": len(rows), "cars": rows})
        self.status = json.dumps(result.data, ensure_ascii=False)
        return result
