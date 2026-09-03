from lfx.custom import Component
from lfx.io import IntInput, MessageTextInput, Output
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, create_sales_lead


class CreateSalesLead(Component):
    display_name = "Create Sales Lead"
    description = "Creates a REAL sales lead in SQLite when the customer wants a salesperson to contact them."
    icon = "user-plus"
    name = "CreateSalesLead"

    inputs = [
        MessageTextInput(name="customer_name", display_name="Customer Name", tool_mode=True, required=True),
        MessageTextInput(name="phone", display_name="Phone", tool_mode=True, required=True),
        MessageTextInput(name="email", display_name="Email", tool_mode=True, required=False),
        IntInput(name="car_id", display_name="Car ID", tool_mode=True, required=False),
        MessageTextInput(name="notes", display_name="Notes", tool_mode=True, required=False),
        MessageTextInput(name="db_path", display_name="DB Path", value=DEFAULT_DB, advanced=True),
    ]
    outputs = [Output(display_name="Created Lead", name="lead", method="create_lead")]

    def create_lead(self) -> Data:
        car_id = int(self.car_id) if getattr(self, "car_id", None) not in (None, "", 0) else None
        result = create_sales_lead(
            self.db_path or DEFAULT_DB,
            self.customer_name,
            self.phone,
            car_id,
            self.email or None,
            self.notes or None,
        )
        data = Data(data=result)
        self.status = result
        return data
