import re

from lfx.custom import Component
from lfx.io import MessageTextInput, Output
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, compare_cars


class CompareCars(Component):
    display_name = "Compare Cars"
    description = "Return structured factual fields for two or more car IDs so the agent can compare them without inventing specs."
    icon = "columns-2"
    name = "CompareCars"

    inputs = [
        MessageTextInput(name="car_ids", display_name="Car IDs", info="Comma-separated IDs, e.g. 2,5", tool_mode=True, required=True),
        MessageTextInput(name="db_path", display_name="DB Path", value=DEFAULT_DB, advanced=True),
    ]
    outputs = [Output(display_name="Comparison Data", name="comparison", method="run_compare")]

    def run_compare(self) -> Data:
        ids = [int(x) for x in re.findall(r"\d+", str(self.car_ids))]
        cars = compare_cars(self.db_path or DEFAULT_DB, ids[:5])
        result = Data(data={"requested_ids": ids[:5], "cars": cars})
        self.status = result.data
        return result
