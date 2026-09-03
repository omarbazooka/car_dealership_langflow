from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from car_dealership_core import init_db
from memory_manager import (
    MEMORY_RECENT_MESSAGE_LIMIT,
    MEMORY_SUMMARY_TRIGGER,
    MemoryManager,
    detect_cancellation,
    extract_action_inputs,
    extract_preferences,
    resolve_car_reference,
)


class TestIntermediateMemory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_memory.db")
        init_db(self.db_path)
        self.mm = MemoryManager(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # TEST A: Conversation Memory
    # -------------------------------------------------------------------------
    def test_a_conversation_memory(self):
        session_id = "test-session-a"
        # Save 15 messages
        for i in range(15):
            role = "user" if i % 2 == 0 else "assistant"
            self.mm.save_message(session_id, role, f"Message {i+1}")

        self.assertEqual(self.mm.get_message_count(session_id), 15)

        # Verify recent message limit defaults to 12
        recent = self.mm.get_recent_messages(session_id)
        self.assertEqual(len(recent), MEMORY_RECENT_MESSAGE_LIMIT)
        self.assertEqual(recent[-1]["content"], "Message 15")
        self.assertEqual(recent[0]["content"], "Message 4")  # 15 - 12 + 1 = 4

    # -------------------------------------------------------------------------
    # TEST B: Structured State Merge
    # -------------------------------------------------------------------------
    def test_b_structured_state_merge(self):
        session_id = "test-session-b"
        # Initial state
        self.mm.update_session_state(session_id, condition="new", max_price=1_500_000.0)
        s1 = self.mm.get_session_state(session_id)
        self.assertEqual(s1["condition"], "new")
        self.assertEqual(s1["max_price"], 1_500_000.0)
        self.assertIsNone(s1["body_type"])

        # Update body_type only
        self.mm.update_session_state(session_id, body_type="SUV")
        s2 = self.mm.get_session_state(session_id)
        # All three must remain intact
        self.assertEqual(s2["condition"], "new")
        self.assertEqual(s2["max_price"], 1_500_000.0)
        self.assertEqual(s2["body_type"], "SUV")

        # Update transmission only
        self.mm.update_session_state(session_id, transmission="Automatic")
        s3 = self.mm.get_session_state(session_id)
        self.assertEqual(s3["condition"], "new")
        self.assertEqual(s3["max_price"], 1_500_000.0)
        self.assertEqual(s3["body_type"], "SUV")
        self.assertEqual(s3["transmission"], "Automatic")

    # -------------------------------------------------------------------------
    # TEST C: Preference Override
    # -------------------------------------------------------------------------
    def test_c_preference_override(self):
        session_id = "test-session-c"
        # User initially wants new
        self.mm.update_session_state(session_id, condition="new")
        self.assertEqual(self.mm.get_session_state(session_id)["condition"], "new")

        # Later explicitly changes mind to used
        self.mm.update_session_state(session_id, condition="used")
        self.assertEqual(self.mm.get_session_state(session_id)["condition"], "used")

    # -------------------------------------------------------------------------
    # TEST D: Recommendation IDs
    # -------------------------------------------------------------------------
    def test_d_recommendation_ids(self):
        session_id = "test-session-d"
        ordered_ids = [144, 281, 633]
        self.mm.set_recommended_cars(session_id, ordered_ids)

        loaded = self.mm.get_recommended_cars(session_id)
        self.assertEqual(loaded, ordered_ids)
        self.assertEqual(loaded[0], 144)
        self.assertEqual(loaded[1], 281)
        self.assertEqual(loaded[2], 633)

    # -------------------------------------------------------------------------
    # TEST E: Ordinal Resolver
    # -------------------------------------------------------------------------
    def test_e_ordinal_resolver(self):
        session_id = "test-session-e"
        self.mm.set_recommended_cars(session_id, [144, 281, 633])

        # First car variants
        for phrase in ["الأولى", "الاولى", "أول واحدة", "اول واحدة", "أول عربية", "اول عربية", "الاول"]:
            res = resolve_car_reference(session_id, phrase, self.db_path)
            self.assertEqual(res, 144, f"Failed for phrase: {phrase}")

        # Second car variants
        for phrase in ["التانية", "الثانية", "تاني واحدة", "تانى واحده", "تاني عربية", "التاني", "الثاني"]:
            res = resolve_car_reference(session_id, phrase, self.db_path)
            self.assertEqual(res, 281, f"Failed for phrase: {phrase}")

        # Third car variants
        for phrase in ["التالتة", "الثالثة", "تالت واحدة", "ثالث واحده", "التالت"]:
            res = resolve_car_reference(session_id, phrase, self.db_path)
            self.assertEqual(res, 633, f"Failed for phrase: {phrase}")

        # Last car variants
        for phrase in ["آخر واحدة", "اخر واحدة", "الأخيرة", "الاخيره", "اخر عربيه"]:
            res = resolve_car_reference(session_id, phrase, self.db_path)
            self.assertEqual(res, 633, f"Failed for phrase: {phrase}")

        # Multiple cars: first two
        for phrase in ["أول اتنين", "اول اتنين", "الاول والتاني", "الأول والثاني", "اول واثنين", "أول اثنين"]:
            res = resolve_car_reference(session_id, phrase, self.db_path)
            self.assertEqual(res, [144, 281], f"Failed for phrase: {phrase}")

    # -------------------------------------------------------------------------
    # TEST F: Selected Car
    # -------------------------------------------------------------------------
    def test_f_selected_car(self):
        session_id = "test-session-f"
        self.assertIsNone(self.mm.get_selected_car(session_id))

        self.mm.set_selected_car(session_id, 281)
        self.assertEqual(self.mm.get_selected_car(session_id), 281)

        # Clear selected car
        self.mm.set_selected_car(session_id, None)
        self.assertIsNone(self.mm.get_selected_car(session_id))

    # -------------------------------------------------------------------------
    # TEST G: Pending Test Drive
    # -------------------------------------------------------------------------
    def test_g_pending_test_drive(self):
        session_id = "test-session-g"
        action = self.mm.create_pending_action(session_id, "test_drive", entity_id=144)
        self.assertEqual(action["action_type"], "test_drive")
        self.assertEqual(action["entity_id"], 144)
        self.assertEqual(action["status"], "pending")
        self.assertIsNone(action["payload"]["customer_name"])

        # Sequential updates
        self.mm.update_pending_action(session_id, payload_updates={"customer_name": "عمر أحمد"})
        self.mm.update_pending_action(session_id, payload_updates={"phone": "01012345678"})
        self.mm.update_pending_action(session_id, payload_updates={"preferred_date": "2026-09-06"})
        updated = self.mm.update_pending_action(session_id, payload_updates={"preferred_time": "17:00"})

        self.assertEqual(updated["payload"]["customer_name"], "عمر أحمد")
        self.assertEqual(updated["payload"]["phone"], "01012345678")
        self.assertEqual(updated["payload"]["preferred_date"], "2026-09-06")
        self.assertEqual(updated["payload"]["preferred_time"], "17:00")
        self.assertEqual(updated["entity_id"], 144)

    # -------------------------------------------------------------------------
    # TEST H: Cancellation
    # -------------------------------------------------------------------------
    def test_h_cancellation(self):
        session_id = "test-session-h"
        self.mm.create_pending_action(session_id, "test_drive", entity_id=144)
        self.assertIsNotNone(self.mm.get_pending_action(session_id))

        # Cancellation phrases test
        self.assertTrue(detect_cancellation("خلاص بلاش"))
        self.assertTrue(detect_cancellation("خلاص بلاش الحجز"))
        self.assertTrue(detect_cancellation("إلغاء"))
        self.assertTrue(detect_cancellation("الغي"))
        self.assertTrue(detect_cancellation("مش عايز حد يكلمني"))

        # Perform cancellation
        cancelled = self.mm.cancel_pending_action(session_id)
        self.assertTrue(cancelled)
        self.assertIsNone(self.mm.get_pending_action(session_id))

    # -------------------------------------------------------------------------
    # TEST I: Summary Threshold
    # -------------------------------------------------------------------------
    def test_i_summary_threshold(self):
        session_id = "test-session-i"

        # Under threshold: 10 messages total (less than MEMORY_RECENT_MESSAGE_LIMIT=12)
        for i in range(10):
            self.mm.save_message(session_id, "user" if i % 2 == 0 else "assistant", f"Chat turn {i+1}")

        updated = self.mm.maybe_update_summary(session_id)
        self.assertFalse(updated, "Summary should not update when under message threshold")
        self.assertEqual(self.mm.get_summary(session_id)["summary"], "")

        # Add messages to exceed recent window + trigger (12 recent + 12 old = 24 messages)
        for i in range(10, 25):
            self.mm.save_message(session_id, "user" if i % 2 == 0 else "assistant", f"Turn {i+1}: عايز عربية SUV زيرو")

        updated = self.mm.maybe_update_summary(session_id)
        self.assertTrue(updated, "Summary should update when trigger threshold of unsummarized messages is met")
        summary_info = self.mm.get_summary(session_id)
        self.assertTrue(len(summary_info["summary"]) > 0)
        self.assertGreater(summary_info["summarized_until_message_id"], 0)

    # -------------------------------------------------------------------------
    # TEST J: Summary Failure Does Not Crash
    # -------------------------------------------------------------------------
    def test_j_summary_failure(self):
        session_id = "test-session-j"
        for i in range(25):
            self.mm.save_message(session_id, "user" if i % 2 == 0 else "assistant", f"Turn {i+1}")

        # Defective summarizer that deliberately throws an exception
        def broken_summarizer(existing, msgs):
            raise RuntimeError("Simulated external API network outage / timeout")

        # Must not raise an exception; must catch, log, and return False safely
        res = self.mm.maybe_update_summary(session_id, summarizer_fn=broken_summarizer)
        self.assertFalse(res)

        # Verify main conversation messages and session state remain completely intact
        self.assertEqual(self.mm.get_message_count(session_id), 25)
        recent = self.mm.get_recent_messages(session_id)
        self.assertEqual(len(recent), MEMORY_RECENT_MESSAGE_LIMIT)

    # -------------------------------------------------------------------------
    # TEST K: Session Isolation
    # -------------------------------------------------------------------------
    def test_k_session_isolation(self):
        session_a = "user-alice"
        session_b = "user-bob"

        self.mm.update_session_state(session_a, condition="new", max_price=1_500_000.0, body_type="SUV")
        self.mm.set_recommended_cars(session_a, [144, 281])
        self.mm.set_selected_car(session_a, 144)
        self.mm.create_pending_action(session_a, "test_drive", entity_id=144)

        # Bob's session must be completely clean and isolated
        state_b = self.mm.get_session_state(session_b)
        self.assertIsNone(state_b["condition"])
        self.assertIsNone(state_b["max_price"])
        self.assertEqual(state_b["last_recommended_car_ids"], [])
        self.assertIsNone(state_b["selected_car_id"])
        self.assertIsNone(self.mm.get_pending_action(session_b))
        self.assertEqual(self.mm.get_recent_messages(session_b), [])


if __name__ == "__main__":
    unittest.main()
