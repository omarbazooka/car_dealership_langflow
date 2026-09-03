import json
import os
import sys
import unittest

sys.path.extend([
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "components")),
])

from car_dealership_core import (
    DEFAULT_DB,
    build_recommendation_set,
    get_car,
    init_db,
    search_cars,
)
from memory_manager import (
    MemoryManager,
    resolve_car_reference,
    resolve_recommendation_reference,
)


class TestRecommendationSnapshots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = "/data/car_dealership.db" if os.path.exists("/data/car_dealership.db") else DEFAULT_DB
        init_db(cls.db_path)
        cls.mm = MemoryManager(cls.db_path)

    def test_01_variant_grouping(self):
        """P0 Bug 1: Visible recommendation order must match memory IDs.
        Duplicate variants (e.g. colors) must group into 1 visible position."""
        raw_cars = search_cars(self.db_path, condition="new", max_price=1750000.0, limit=10)
        self.assertTrue(len(raw_cars) >= 2, "Need at least 2 cars in search")

        rec_set = build_recommendation_set(raw_cars, limit=5)
        self.assertTrue(len(rec_set) <= 5)

        # Positions must be strictly 1-indexed sequential
        positions = [c["position"] for c in rec_set]
        self.assertEqual(positions, list(range(1, len(rec_set) + 1)))

        # Position 1 should have primary_car_id and variant_ids
        pos1 = rec_set[0]
        self.assertIn("primary_car_id", pos1)
        self.assertIn("variant_ids", pos1)
        self.assertIn("display_name", pos1)

    def test_02_snapshot_creation_and_ordinal_resolution(self):
        """Visible recommendation order matches memory positions deterministically."""
        import uuid
        session_id = f"test-snap-01-{uuid.uuid4().hex[:6]}"
        raw_cars = search_cars(self.db_path, condition="new", max_price=1750000.0, limit=10)
        rec_set = build_recommendation_set(raw_cars, limit=5)

        snap = self.mm.create_recommendation_snapshot(session_id, {"condition": "new"}, rec_set)
        self.assertEqual(snap["status"], "active")
        self.assertEqual(snap["sequence_no"], 1)

        # Position 1: "الأولى"
        res1 = resolve_recommendation_reference(session_id, "عايز تفاصيل العربية الأولى", self.db_path)
        self.assertEqual(res1["status"], "resolved")
        self.assertEqual(res1["positions"], [1])
        self.assertEqual(res1["items"][0]["primary_car_id"], rec_set[0]["primary_car_id"])

        # Position 2: "التانية"
        if len(rec_set) >= 2:
            res2 = resolve_recommendation_reference(session_id, "احجزلي العربية التانية", self.db_path)
            self.assertEqual(res2["status"], "resolved")
            self.assertEqual(res2["positions"], [2])
            self.assertEqual(res2["items"][0]["primary_car_id"], rec_set[1]["primary_car_id"])

        # Compare first two: "أول اتنين"
        if len(rec_set) >= 2:
            res_comp = resolve_recommendation_reference(session_id, "قارن أول اتنين", self.db_path)
            self.assertEqual(res_comp["status"], "resolved")
            self.assertEqual(res_comp["positions"], [1, 2])
            self.assertEqual(len(res_comp["items"]), 2)

        # Last one: "آخر واحدة"
        res_last = resolve_recommendation_reference(session_id, "كام سعر آخر واحدة؟", self.db_path)
        self.assertEqual(res_last["status"], "resolved")
        self.assertEqual(res_last["positions"], [rec_set[-1]["position"]])
        self.assertEqual(res_last["items"][0]["primary_car_id"], rec_set[-1]["primary_car_id"])

    def test_03_negative_filter_ambiguity(self):
        """User asks for car #2 but excludes the brand at position #2 -> returns ambiguous."""
        import uuid
        session_id = f"test-snap-neg-{uuid.uuid4().hex[:6]}"
        mock_items = [
            {"position": 1, "primary_car_id": 101, "display_name": "Toyota Corolla", "brand": "Toyota"},
            {"position": 2, "primary_car_id": 102, "display_name": "Soueast S09", "brand": "Soueast"},
            {"position": 3, "primary_car_id": 103, "display_name": "Mitsubishi Xpander", "brand": "Mitsubishi"},
        ]
        self.mm.create_recommendation_snapshot(session_id, {"condition": "new"}, mock_items)

        res = resolve_recommendation_reference(session_id, "احجزلي العربية التانية مش الساوليت", self.db_path)
        self.assertEqual(res["status"], "ambiguous")
        self.assertIn("Customer requested position 2 but explicitly excluded brand", res["reason"])

    def test_04_preference_change_invalidates_snapshot(self):
        """P0 Bug 2: Stale incompatible lists on material preference change.
        When customer switches from 'new' to 'used', active snapshot is invalidated."""
        import uuid
        session_id = f"test-snap-inval-{uuid.uuid4().hex[:6]}"
        mock_items = [
            {"position": 1, "primary_car_id": 201, "display_name": "Brand New S09", "condition": "new", "price": 1500000.0},
        ]
        self.mm.create_recommendation_snapshot(session_id, {"condition": "new"}, mock_items)
        self.mm.set_selected_car(session_id, 201)

        active_before = self.mm.get_active_snapshot(session_id)
        self.assertIsNotNone(active_before)

        # User changes condition to used
        self.mm.apply_preference_updates_and_invalidate(session_id, {"condition": "used"})

        # Active snapshot must be invalidated
        active_after = self.mm.get_active_snapshot(session_id)
        self.assertIsNone(active_after)

        # Unqualified ordinal against invalidated list returns no_active_snapshot
        res = resolve_recommendation_reference(session_id, "عايز العربية الأولى", self.db_path)
        self.assertEqual(res["status"], "no_active_snapshot")

    def test_05_historical_snapshot_resolution(self):
        """Customer explicitly references historical list:
        'عربيه من العربيات اللي انت قولتهم في الاول التلاته عايز العربيه التانيه منهم'"""
        import uuid
        session_id = f"test-snap-hist-{uuid.uuid4().hex[:6]}"
        # Snapshot 1 (Initial new search)
        snap1_items = [
            {"position": 1, "primary_car_id": 301, "display_name": "Soueast S09 (New)", "year": 2025},
            {"position": 2, "primary_car_id": 302, "display_name": "Mitsubishi Xpander (New)", "year": 2024},
            {"position": 3, "primary_car_id": 303, "display_name": "Suzuki Ertiga (New)", "year": 2024},
        ]
        self.mm.create_recommendation_snapshot(session_id, {"condition": "new"}, snap1_items)

        # Snapshot 2 (Later used search)
        snap2_items = [
            {"position": 1, "primary_car_id": 401, "display_name": "Hyundai Elantra (Used)", "year": 2020},
            {"position": 2, "primary_car_id": 402, "display_name": "Toyota Corolla (Used)", "year": 2019},
        ]
        self.mm.create_recommendation_snapshot(session_id, {"condition": "used"}, snap2_items)

        # Current active snapshot has Hyundai and Toyota
        active = self.mm.get_active_snapshot(session_id)
        self.assertEqual(active["sequence_no"], 2)

        # User asks for car #2 from the FIRST list:
        q = "عربيه من العربيات اللي انت قولتهم في الاول التلاته عايز العربيه التانيه منهم"
        res = resolve_recommendation_reference(session_id, q, self.db_path)
        self.assertEqual(res["status"], "resolved")
        self.assertEqual(res["items"][0]["primary_car_id"], 302)
        self.assertEqual(res["items"][0]["display_name"], "Mitsubishi Xpander (New)")


if __name__ == "__main__":
    unittest.main()
