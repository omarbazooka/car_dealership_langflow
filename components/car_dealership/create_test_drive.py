from lfx.custom import Component
from lfx.io import IntInput, MessageTextInput, Output
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, create_test_drive


class CreateTestDrive(Component):
    display_name = "Create Test Drive"
    description = "Creates a REAL test-drive request in SQLite. Call only after customer name, phone, car, date and time are known."
    icon = "calendar-check"
    name = "CreateTestDrive"

    inputs = [
        MessageTextInput(name="customer_name", display_name="Customer Name", tool_mode=True, required=True),
        MessageTextInput(name="phone", display_name="Phone", tool_mode=True, required=True),
        IntInput(name="car_id", display_name="Car ID", tool_mode=True, required=True),
        MessageTextInput(name="preferred_date", display_name="Preferred Date", tool_mode=True, required=True),
        MessageTextInput(name="preferred_time", display_name="Preferred Time", tool_mode=True, required=True),
        MessageTextInput(name="notes", display_name="Notes", tool_mode=True, required=False),
        MessageTextInput(name="db_path", display_name="DB Path", value=DEFAULT_DB, advanced=True),
    ]
    outputs = [Output(display_name="Created Request", name="request", method="create_request")]

    def create_request(self) -> Data:
        result = create_test_drive(
            self.db_path or DEFAULT_DB,
            self.customer_name,
            self.phone,
            int(self.car_id),
            self.preferred_date,
            self.preferred_time,
            self.notes or None,
        )
        data = Data(data=result)
        self.status = result
        return data
