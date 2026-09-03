import os
import sys
import unittest
import uuid

sys.path.extend([
    "/app/lib",
    "/app/components",
    "/app/components/car_dealership",
])

from lfx.schema import Message

from car_dealership_core import DEFAULT_DB, create_test_drive, get_test_drive_requests, init_db
from cancel_test_drive import CancelTestDrive
from compare_cars import CompareCars
from create_test_drive import CreateTestDrive
from gemini_sales_agent import GeminiCarSalesAgent
from memory_manager import MemoryManager, set_current_session_id
from output_guardrails import enforce_honest_catalog_wording
from request_context import set_current_user_text
from vehicle_research_agent import classify_source_url


class TestLivePathGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = "/data/car_dealership.db" if os.path.exists("/data/car_dealership.db") else DEFAULT_DB
        init_db(cls.db_path)

    def _session(self, suffix: str) -> str:
        return f"live-guard-{suffix}-{uuid.uuid4().hex[:8]}"

    def test_compare_uses_visible_snapshot_not_llm_hidden_ids(self):
        sid = self._session("compare")
        mm = MemoryManager(self.db_path)
        mm.create_recommendation_snapshot(
            sid,
            {"condition": "new", "body_type": "SUV"},
            [
                {"position": 1, "primary_car_id": 9067, "variant_ids": [9067, 9069], "display_name": "Soueast S09"},
                {"position": 2, "primary_car_id": 9638, "variant_ids": [9638, 10053], "display_name": "Mitsubishi Eclipse Cross"},
                {"position": 3, "primary_car_id": 10157, "variant_ids": [10157], "display_name": "Suzuki Fronx"},
            ],
        )
        set_current_session_id(sid)
        set_current_user_text("قارن أول اتنين")

        tool = CompareCars()
        tool.db_path = self.db_path
        tool.session_id = sid
        # Simulate the exact bad live LLM call from the regression: hidden variants.
        tool.car_ids = "9067,9069"
        result = tool.run_compare().data

        self.assertEqual(result["source"], "visible_recommendation_snapshot")
        self.assertEqual(result["positions"], [1, 2])
        self.assertEqual(result["requested_ids"], [9067, 9638])

    def test_create_test_drive_rejects_llm_filled_missing_time_and_is_idempotent(self):
        sid = self._session("booking-tool")
        mm = MemoryManager(self.db_path)
        set_current_session_id(sid)
        mm.create_pending_action(
            sid,
            "test_drive",
            entity_id=9067,
            payload={
                "customer_name": "عمر أحمد",
                "phone": "01012345678",
                "preferred_date": "السبت",
                "preferred_time": None,
            },
        )

        before = len(get_test_drive_requests(self.db_path, phone="01012345678", limit=500))

        tool = CreateTestDrive()
        tool.db_path = self.db_path
        tool.customer_name = "عمر أحمد"
        tool.phone = "01012345678"
        tool.car_id = 9067
        tool.preferred_date = "السبت"
        # Malicious/incorrect LLM guess: the memory still does not have a time.
        tool.preferred_time = "5 مساء"
        tool.notes = ""
        blocked = tool.create_request().data
        self.assertEqual(blocked["status"], "blocked")
        self.assertIn("preferred_time", blocked["missing_fields"])
        self.assertEqual(len(get_test_drive_requests(self.db_path, phone="01012345678", limit=500)), before)

        mm.update_pending_action(sid, payload_updates={"preferred_time": "5 مساء"})
        created = tool.create_request().data
        self.assertIn("request_id", created)
        after_one = len(get_test_drive_requests(self.db_path, phone="01012345678", limit=500))
        self.assertEqual(after_one, before + 1)

        # Retrying the same tool after action completion cannot insert again.
        retry = tool.create_request().data
        self.assertEqual(retry["status"], "blocked")
        self.assertEqual(len(get_test_drive_requests(self.db_path, phone="01012345678", limit=500)), after_one)

    def test_sales_orchestrator_does_not_insert_after_date_only(self):
        sid = self._session("booking-agent")
        mm = MemoryManager(self.db_path)
        mm.create_pending_action(
            sid,
            "test_drive",
            entity_id=9067,
            payload={"customer_name": "عمر أحمد", "phone": "01012345678"},
        )
        before = len(get_test_drive_requests(self.db_path, phone="01012345678", limit=500))

        agent = GeminiCarSalesAgent()
        agent.google_api_key = "dummy-ci-key"
        agent.db_path = self.db_path
        agent.input_value = Message(text="السبت", session_id=sid)
        response = agent.run_agent()

        self.assertIn("الساعة", response.text)
        self.assertEqual(len(get_test_drive_requests(self.db_path, phone="01012345678", limit=500)), before)
        pending = mm.get_pending_action(sid)
        self.assertEqual(pending["payload"]["preferred_date"], "السبت")
        self.assertFalse(pending["payload"].get("preferred_time"))

        agent.input_value = Message(text="الساعة 5 مساء", session_id=sid)
        response2 = agent.run_agent()
        self.assertIn("رقم الحجز", response2.text)
        self.assertEqual(len(get_test_drive_requests(self.db_path, phone="01012345678", limit=500)), before + 1)

    def test_cancel_tool_is_scoped_to_current_session(self):
        sid_a = self._session("cancel-a")
        sid_b = self._session("cancel-b")
        mm = MemoryManager(self.db_path)

        mm.create_pending_action(
            sid_a, "test_drive", entity_id=9067,
            payload={
                "customer_name": "عميل أ",
                "phone": "01011111111",
                "preferred_date": "السبت",
                "preferred_time": "5 مساء",
            },
        )
        booking_a = create_test_drive(
            self.db_path, "عميل أ", "01011111111", 9067, "السبت", "5 مساء"
        )
        mm.complete_pending_action(sid_a, booking_a)

        mm.create_pending_action(
            sid_b, "test_drive", entity_id=9067,
            payload={
                "customer_name": "عميل ب",
                "phone": "01022222222",
                "preferred_date": "الأحد",
                "preferred_time": "4 مساء",
            },
        )
        booking_b = create_test_drive(
            self.db_path, "عميل ب", "01022222222", 9067, "الأحد", "4 مساء"
        )
        mm.complete_pending_action(sid_b, booking_b)

        set_current_session_id(sid_a)
        tool = CancelTestDrive()
        tool.db_path = self.db_path
        tool.session_id = sid_a
        tool.request_id = booking_b["request_id"]
        tool.notes = "wrong session attempt"
        blocked = tool.run_cancel().data
        self.assertEqual(blocked["status"], "blocked")

        tool.request_id = None
        cancelled = tool.run_cancel().data
        self.assertTrue(cancelled["success"])
        self.assertEqual(cancelled["request_id"], booking_a["request_id"])

        a_row = get_test_drive_requests(self.db_path, phone="01011111111", limit=20)[0]
        b_row = get_test_drive_requests(self.db_path, phone="01022222222", limit=20)[0]
        self.assertEqual(a_row["status"], "CANCELLED")
        self.assertNotEqual(b_row["status"], "CANCELLED")

    def test_output_guard_removes_live_stock_and_false_official_third_party_wording(self):
        text = "دي أفضل العربيات المتاحة حالياً عندنا. حسب المواصفات الرسمية في EgyCars العربية فيها 6 Airbags."
        cleaned = enforce_honest_catalog_wording(text)
        self.assertNotIn("المتاحة حالياً عندنا", cleaned)
        self.assertNotIn("المواصفات الرسمية", cleaned)
        self.assertIn("المواصفات المنشورة", cleaned)

    def test_third_party_domains_are_never_official(self):
        for url in (
            "https://www.contactcars.com/example",
            "https://www.hatla2ee.com/example",
            "https://www.auto-data.net/en/example",
            "https://www.zigwheels.com/example",
            "https://www.egycars.example/specs",
        ):
            source_type, official = classify_source_url(url, "official_manufacturer")
            self.assertFalse(official, url)
            self.assertNotEqual(source_type, "official_manufacturer")


if __name__ == "__main__":
    unittest.main()
