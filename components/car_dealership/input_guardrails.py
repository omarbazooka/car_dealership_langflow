from lfx.custom import Component
from lfx.io import MessageInput, Output
from lfx.schema import Message

from car_dealership_core import guard_input

BLOCK_PREFIX = "__GUARDRAIL_BLOCKED__:"


class InputGuardrails(Component):
    display_name = "Input Guardrails"
    description = "Checks empty/oversized input and common prompt-injection attempts before the Agent."
    icon = "shield"
    name = "InputGuardrails"

    inputs = [MessageInput(name="input_value", display_name="Customer Message", required=True)]
    outputs = [Output(display_name="Guarded Message", name="message", method="guard_message")]

    def guard_message(self) -> Message:
        msg = self.input_value if isinstance(self.input_value, Message) else Message(text=str(self.input_value or ""))
        ok, cleaned = guard_input(msg.text)
        msg.text = cleaned if ok else BLOCK_PREFIX + cleaned
        self.status = "PASS" if ok else "BLOCKED"
        return msg
