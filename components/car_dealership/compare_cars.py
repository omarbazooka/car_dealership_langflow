import re

from lfx.custom import Component
from lfx.io import MessageTextInput, Output
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, compare_cars
from memory_manager import get_current_session_id, resolve_recommendation_reference
from request_context import get_current_user_text


class CompareCars(Component):
    display_name = "Compare Cars"
    description = (
        "Return structured factual fields for visible recommendation positions or explicit car IDs. "
        "When the customer used an ordinal such as 'أول اتنين', the visible recommendation snapshot is authoritative."
    )
    icon = "columns-2"
    name = "CompareCars"

    inputs = [
        MessageTextInput(
            name="car_ids",
            display_name="Car IDs",
            info="Comma-separated IDs or ordinals e.g. 2,5 or 'أول اتنين'",
            tool_mode=True,
            required=True,
        ),
        MessageTextInput(name="session_id", display_name="Session ID", tool_mode=True, required=False),
        MessageTextInput(name="db_path", display_name="DB Path", value=DEFAULT_DB, advanced=True),
    ]
    outputs = [Output(display_name="Comparison Data", name="comparison", method="run_compare")]

    @staticmethod
    def _contains_ordinal_or_compare(text: str) -> bool:
        t = str(text or "").lower()
        return bool(
            re.search(
                r"(?:قارن|اول\s*اتنين|أول\s*اتنين|الاول|الأول|الاولى|الأولى|التاني|الثاني|التانيه|الثانية|التالت|الثالث)",
                t,
            )
        )

    def run_compare(self) -> Data:
        db = self.db_path or DEFAULT_DB
        sid = getattr(self, "session_id", None) or get_current_session_id()
        raw_ids_str = str(self.car_ids or "")
        current_user_text = get_current_user_text()

        # P0 hardening: if the customer used visible-list ordinals, NEVER trust numeric IDs
        # synthesized by the LLM. Resolve the original utterance against the active snapshot.
        if sid and current_user_text and self._contains_ordinal_or_compare(current_user_text):
            ref_res = resolve_recommendation_reference(sid, current_user_text, db)
            if ref_res.get("status") == "resolved":
                ids = [int(it["primary_car_id"]) for it in ref_res.get("items", [])]
                if len(ids) >= 2:
                    cars = compare_cars(db, ids[:5])
                    result = Data(
                        data={
                            "status": "ok",
                            "source": "visible_recommendation_snapshot",
                            "snapshot_id": ref_res.get("snapshot_id"),
                            "positions": ref_res.get("positions", []),
                            "requested_ids": ids[:5],
                            "cars": cars,
                        }
                    )
                    self.status = result.data
                    return result
            elif ref_res.get("status") in {"ambiguous", "no_active_snapshot"}:
                result = Data(
                    data={
                        "status": "needs_clarification",
                        "reason": ref_res.get("reason"),
                        "requested_ids": [],
                        "cars": [],
                    }
                )
                self.status = result.data
                return result

        # Explicit IDs remain supported when the customer actually supplied IDs.
        ids = [int(x) for x in re.findall(r"\d+", raw_ids_str)]
        if not ids and sid:
            ref_res = resolve_recommendation_reference(sid, raw_ids_str, db)
            if ref_res.get("status") == "resolved":
                ids = [int(it["primary_car_id"]) for it in ref_res.get("items", [])]

        cars = compare_cars(db, ids[:5])
        result = Data(data={"status": "ok", "source": "explicit_ids", "requested_ids": ids[:5], "cars": cars})
        self.status = result.data
        return result
