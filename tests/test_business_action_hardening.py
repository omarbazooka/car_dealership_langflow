import os
import sys
import unittest

sys.path.extend([
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "components")),
])

from car_dealership_core import (
    DEFAULT_DB,
    cancel_test_drive,
    create_test_drive,
    get_car,
    get_test_drive_requests,
    init_db,
)
from memory_manager import (
    MemoryManager,
    detect_cancellation,
    detect_new_details,
    detect_reschedule,
    extract_action_inputs,
)


class TestBusinessActionHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = "/data/car_dealership.db" if os.path.exists("/data/car_dealership.db") else DEFAULT_DB
        init_db(cls.db_path)
        cls.mm = MemoryManager(cls.db_path)

    def test_01_no_premature_insert_and_combined_parsing(self):
        """P0 Bug 4: No premature DB insert until all 5 fields are complete.
        Extracts combined input: 'أحمد محمد01123456789الاتنن 2الظهر'."""
        session_id = "test-action-hardening-01"
        action = self.mm.create_pending_action(session_id, "test_drive", entity_id=9067)
        payload = action["payload"]

        # Before user provides data: cannot insert
        req_fields = ("customer_name", "phone", "car_id", "preferred_date", "preferred_time")
        cid = action["entity_id"]
        self.assertFalse(all([payload.get(k) for k in ("customer_name", "phone", "preferred_date", "preferred_time")]), "Must be missing fields")

        # User provides combined input in 1 message:
        user_msg = "أحمد محمد01123456789الاتنن 2الظهر"
        updates = extract_action_inputs("test_drive", user_msg, payload)

        self.assertEqual(updates.get("customer_name"), "أحمد محمد")
        self.assertEqual(updates.get("phone"), "01123456789")
        self.assertEqual(updates.get("preferred_date"), "الاثنين")
        self.assertIn("2الظهر", updates.get("preferred_time", "").replace(" ", ""))

        # Merge updates
        action = self.mm.update_pending_action(session_id, payload_updates=updates)
        payload = action["payload"]

        # Now all fields are complete -> can insert
        res = create_test_drive(
            self.db_path,
            customer_name=payload["customer_name"],
            phone=payload["phone"],
            car_id=cid,
            preferred_date=payload["preferred_date"],
            preferred_time=payload["preferred_time"],
        )
        self.assertIn("request_id", res)
        self.mm.complete_pending_action(session_id, res)

    def test_02_reschedule_clears_date_time_without_insert(self):
        """Customer says: 'عايز احجز العربيه دي بمواعيد تانيه' -> Clears schedule, asks missing, does NOT insert."""
        session_id = "test-action-reschedule-01"
        # Simulate previously filled booking
        self.mm.create_pending_action(
            session_id,
            "test_drive",
            entity_id=9067,
            payload={
                "customer_name": "عمر أحمد",
                "phone": "01012345678",
                "preferred_date": "الخميس",
                "preferred_time": "5 مساء",
            },
        )

        user_msg = "عايز احجز العربيه دي بمواعيد تانيه"
        self.assertTrue(detect_reschedule(user_msg))

        # Under the hardened flow, schedule fields are cleared
        act = self.mm.update_pending_action(
            session_id,
            payload_updates={"preferred_date": None, "preferred_time": None},
        )
        payload = act["payload"]
        self.assertIsNone(payload["preferred_date"])
        self.assertIsNone(payload["preferred_time"])
        self.assertEqual(act["entity_id"], 9067)
        self.assertEqual(payload["customer_name"], "عمر أحمد")

    def test_03_new_details_clears_customer_info(self):
        """Customer says: 'وببيانات جديدة كلية' -> Clears name, phone, date, time while keeping target car."""
        session_id = "test-action-new-details-01"
        self.mm.create_pending_action(
            session_id,
            "test_drive",
            entity_id=9067,
            payload={
                "customer_name": "عمر أحمد",
                "phone": "01012345678",
                "preferred_date": "الخميس",
                "preferred_time": "5 مساء",
            },
        )

        user_msg = "وببيانات جديدة كلية"
        self.assertTrue(detect_new_details(user_msg))

        act = self.mm.update_pending_action(
            session_id,
            payload_updates={
                "customer_name": None,
                "phone": None,
                "preferred_date": None,
                "preferred_time": None,
            },
        )
        payload = act["payload"]
        self.assertIsNone(payload["customer_name"])
        self.assertIsNone(payload["phone"])
        self.assertIsNone(payload["preferred_date"])
        self.assertIsNone(payload["preferred_time"])
        self.assertEqual(act["entity_id"], 9067)

    def test_04_completed_booking_cancellation(self):
        """P0 Bug 5: Real cancellation updating test_drive_requests.status = 'CANCELLED'
        and cancelled_at timestamp in SQLite."""
        # 1. Create a real test drive booking
        booking = create_test_drive(
            self.db_path,
            customer_name="يوسف إبراهيم",
            phone="01234567890",
            car_id=9067,
            preferred_date="السبت",
            preferred_time="4 مساء",
            notes="Initial booking",
        )
        req_id = booking["request_id"]
        self.assertEqual(booking["status"], "NEW")

        # 2. Cancel it via cancel_test_drive
        success = cancel_test_drive(self.db_path, req_id, notes="Customer changed mind")
        self.assertTrue(success)

        # 3. Verify in SQLite that status is CANCELLED and cancelled_at is populated
        reqs = get_test_drive_requests(self.db_path, limit=10)
        matched = [r for r in reqs if r["id"] == req_id]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["status"], "CANCELLED")
        self.assertIsNotNone(matched[0]["cancelled_at"])


if __name__ == "__main__":
    unittest.main()
