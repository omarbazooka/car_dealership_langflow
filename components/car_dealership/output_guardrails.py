import re

from lfx.custom import Component
from lfx.io import MessageInput, Output
from lfx.schema import Message

from car_dealership_core import guard_output


_THIRD_PARTY_MARKERS = (
    "egycars",
    "egy cars",
    "إيجي كار",
    "ايجي كار",
    "contactcars",
    "كونتكت كارز",
    "hatla2ee",
    "هتلاقي",
    "auto-data",
    "autodata",
    "zigwheels",
)


def enforce_honest_catalog_wording(text: str) -> str:
    """Deterministically remove claims that imported catalog data is live showroom stock."""
    out = str(text or "")
    replacements = {
        "أفضل العربيات المتاحة حالياً عندنا": "من العربيات المطابقة ضمن البيانات المتاحة عندي",
        "أفضل العربيات المتاحة حاليا عندنا": "من العربيات المطابقة ضمن البيانات المتاحة عندي",
        "متاحة عندنا حالياً": "مسجلة ضمن البيانات المتاحة عندي",
        "متاحة عندنا حاليا": "مسجلة ضمن البيانات المتاحة عندي",
        "المتاحة حالياً عندنا": "المسجلة ضمن البيانات المتاحة عندي",
        "المتاحة حاليا عندنا": "المسجلة ضمن البيانات المتاحة عندي",
        "المخزون الحالي": "البيانات المستوردة الحالية",
        "موجودة الآن في المعرض": "مسجلة ضمن البيانات المتاحة",
        "موجوده الآن في المعرض": "مسجلة ضمن البيانات المتاحة",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)

    # Third-party sites/databases are evidence sources, never manufacturer-official sources.
    lower = out.lower()
    if any(marker.lower() in lower for marker in _THIRD_PARTY_MARKERS):
        out = re.sub(r"المواصفات\s+الرسمية", "المواصفات المنشورة", out)
        out = re.sub(r"المصدر\s+الرسمي", "المصدر المنشور", out)
        out = re.sub(r"حسب\s+المصدر\s+الرسمي", "بحسب المصدر المنشور", out)
    return out


class OutputGuardrails(Component):
    display_name = "Output Guardrails"
    description = (
        "Redacts secrets/tool tracebacks and enforces honest imported-catalog/source wording "
        "before messages reach the customer."
    )
    icon = "shield-check"
    name = "OutputGuardrails"

    inputs = [MessageInput(name="input_value", display_name="Agent Message", required=True)]
    outputs = [Output(display_name="Safe Message", name="message", method="build_message")]

    def build_message(self) -> Message:
        msg = self.input_value if isinstance(self.input_value, Message) else Message(text=str(self.input_value or ""))
        msg.text = enforce_honest_catalog_wording(guard_output(msg.text))
        self.status = msg.text
        return msg
