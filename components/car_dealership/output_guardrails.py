from lfx.custom import Component
from lfx.io import MessageInput, Output
from lfx.schema import Message

from car_dealership_core import guard_output


class OutputGuardrails(Component):
    display_name = "Output Guardrails"
    description = "Redacts common secret patterns and prevents raw tool tracebacks from reaching the customer."
    icon = "shield-check"
    name = "OutputGuardrails"

    inputs = [MessageInput(name="input_value", display_name="Agent Message", required=True)]
    outputs = [Output(display_name="Safe Message", name="message", method="build_message")]

    def build_message(self) -> Message:
        msg = self.input_value if isinstance(self.input_value, Message) else Message(text=str(self.input_value or ""))
        msg.text = guard_output(msg.text)
        self.status = msg.text
        return msg
