from lfx.custom import Component
from lfx.io import IntInput, MessageTextInput, Output
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, get_car


class GetCarDetails(Component):
    display_name = "Get Car Details"
    description = "Get factual details for a specific car listing by inventory id."
    icon = "car"
    name = "GetCarDetails"

    inputs = [
        IntInput(name="car_id", display_name="Car ID", tool_mode=True, required=True),
        MessageTextInput(name="db_path", display_name="DB Path", value=DEFAULT_DB, advanced=True),
    ]
    outputs = [Output(display_name="Car", name="car", method="get_details")]

    def get_details(self) -> Data:
        car = get_car(self.db_path or DEFAULT_DB, int(self.car_id))
        result = Data(data={"found": bool(car), "car": car})
        self.status = result.data
        return result
