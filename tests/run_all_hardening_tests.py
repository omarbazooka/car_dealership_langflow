import os
import sys
import unittest

sys.path.extend([
    "/app",
    "/app/lib",
    "/app/components",
    "/app/components/car_dealership",
    "/app/tests",
])

from test_memory_manager import TestIntermediateMemory
from test_recommendation_snapshots import TestRecommendationSnapshots
from test_business_action_hardening import TestBusinessActionHardening
from test_vehicle_research_agent import TestVehicleResearchAgent
from test_live_path_guards import TestLivePathGuards
from test_32_turn_hardened_scenario import Test32TurnHardenedScenario

if __name__ == "__main__":
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for case in (
        TestIntermediateMemory,
        TestRecommendationSnapshots,
        TestBusinessActionHardening,
        TestVehicleResearchAgent,
        TestLivePathGuards,
        Test32TurnHardenedScenario,
    ):
        suite.addTests(loader.loadTestsFromTestCase(case))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n=======================================================")
    print(f"ALL TESTS SUMMARY: Ran {result.testsRun} tests.")
    print(f"Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    print("STATUS: ALL TESTS PASSED!" if result.wasSuccessful() else "STATUS: SOME TESTS FAILED!")
    print("=======================================================\n")
    sys.exit(0 if result.wasSuccessful() else 1)
