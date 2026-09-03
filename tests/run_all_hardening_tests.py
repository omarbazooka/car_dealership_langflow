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
from test_32_turn_hardened_scenario import Test32TurnHardenedScenario

if __name__ == "__main__":
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestIntermediateMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestRecommendationSnapshots))
    suite.addTests(loader.loadTestsFromTestCase(TestBusinessActionHardening))
    suite.addTests(loader.loadTestsFromTestCase(TestVehicleResearchAgent))
    suite.addTests(loader.loadTestsFromTestCase(Test32TurnHardenedScenario))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n=======================================================")
    print(f"ALL TESTS SUMMARY: Ran {result.testsRun} tests.")
    print(f"Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    if result.wasSuccessful():
        print("STATUS: ALL TESTS PASSED!")
    else:
        print("STATUS: SOME TESTS FAILED!")
    print("=======================================================\n")
    sys.exit(0 if result.wasSuccessful() else 1)
