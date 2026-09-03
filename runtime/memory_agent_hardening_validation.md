# AutoDrive Egypt — Memory & Agent Hardening Validation Report

**Date**: September 3, 2026  
**Environment**: Langflow 1.12.x / Docker Container `car-dealership-langflow` / SQLite 3  
**Target Project**: `car_dealership_langflow`  
**Status**: **ALL HARDENING CRITERIA VERIFIED & PASSED**

---

## 1. Executive Summary & Verification Matrix

All P0 and P1 memory, business action, and research agent requirements have been fully implemented, hardened, and verified with automated test suites running inside the live Docker container.

| Requirement | Category | Description | Verification Status |
| :--- | :--- | :--- | :--- |
| **P0 Bug 1** | Memory | Visible recommendation order matches memory positions (1..5). Duplicate vehicle variants group into unified visible positions. | **PASS** |
| **P0 Bug 2** | Memory | Material preference change (e.g. `new` $\leftrightarrow$ `used`, body type, budget) invalidates active recommendation snapshots and clears incompatible selections. | **PASS** |
| **P0 Bug 3** | Memory | Current structured session state strictly outranks historical summary in prompt hierarchy and decision making. | **PASS** |
| **P0 Bug 4** | Business Actions | No premature `CreateTestDrive` DB insert until all 5 required fields are complete. Supports combined single-message input, rescheduling, and details reset. | **PASS** |
| **P0 Bug 5** | Business Actions | Formal booking cancellation executes real database update setting `status='CANCELLED'` and recording `cancelled_at` timestamp. | **PASS** |
| **P1 Dedicated Agent** | Agent Hierarchy | `VehicleResearchAgent` created as an internal subordinate tool for `GeminiCarSalesAgent`. Strictly non-customer-facing. | **PASS** |
| **P1 Spec Matching** | Research Agent | Exact variant matching: Rejects specs from conflicting trims (e.g., 2.0L Turbo vs 1.6L NA). Returns `ambiguous` if variant cannot be verified. | **PASS** |
| **P1 Numeric Airbags** | Research Agent | Airbag facts extracted strictly as integer counts or explicit `not_found`. Never hallucinates or guesses. | **PASS** |
| **P1 Injection Defense**| Research Agent | Defense against prompt injection in external web/catalog pages (strips instruction overrides and redacts content). | **PASS** |
| **P1 Honest Phrasing** | Persona Guardrail | Eliminates false physical showroom claims (`المتاح حالياً عندنا`). Uses honest catalog phrasing (`حسب البيانات المتاحة عندي`). | **PASS** |
| **Persistence** | Database | Zero data loss across Docker container restarts. All sessions, snapshots, messages, and cancellations persist in SQLite. | **PASS** |

---

## 2. Technical Implementation Details

### 2.1 Recommendation Snapshots (`recommendation_snapshots`)
- **Schema**:
  - Table `recommendation_snapshots` added to SQLite database with index on `(session_id, sequence_no)` and `(session_id, status)`.
  - Columns: `id`, `session_id`, `sequence_no`, `criteria_json`, `items_json`, `status` (`active` or `invalidated`), `created_at`.
- **Variant Grouping**:
  - `build_recommendation_set(cars, limit=5)` groups duplicate variants (e.g. multiple colors or packages of the same model and trim) under a single customer-visible position (1..5).
  - Each visible position records `primary_car_id` and `variant_ids`. Hidden rows no longer silently consume customer-visible positions.
- **Deterministic Ordinal Resolution**:
  - `resolve_recommendation_reference(session_id, user_text, db_path)` deterministically resolves Arabic ordinals (`الأولى`, `التانية`, `التالتة`, `الرابعة`, `الخامسة`, `آخر واحدة`, `أول اتنين`) against the active snapshot.
  - Detects negative brand constraints (e.g., `"احجزلي العربية التانية مش الساوليت"` returns `status: "ambiguous"`).
  - Explicit historical retrieval: Customer queries like `"من العربيات اللي قولتهم في الأول عايز التانية منهم"` inspect historical snapshots.
  - Invalidated snapshots: When preferences change, active snapshots are invalidated (`status='invalidated'`). Unqualified ordinals against invalidated snapshots return `status: "no_active_snapshot"`.

### 2.2 Strict Memory Context Hierarchy
- `MemoryManager.build_memory_context()` enforces the prompt hierarchy:
  1. `CURRENT STRUCTURED STATE` (Authoritative - overrides summary)
  2. `ACTIVE RECOMMENDATION SNAPSHOT` (Current visible options 1..5)
  3. `SELECTED VEHICLE` (Currently anchored vehicle ID & lineage)
  4. `PENDING BUSINESS ACTION` (In-progress test drive / sales lead)
  5. `LAST COMPLETED ACTION` (Most recent booking or lead)
  6. `HISTORICAL SUMMARY` (Earlier context - strictly prohibited from overriding current state)
  7. `RECENT CONVERSATION` (Latest 12 messages)
  8. `CURRENT USER MESSAGE`

### 2.3 Business Action Hardening (`test_drive_requests`)
- **No Premature Insert**:
  - `CreateTestDrive` requires all 5 mandatory fields (`customer_name`, `phone`, `car_id`, `preferred_date`, `preferred_time`). No SQLite row is inserted while fields are missing.
- **Combined Single-Message Parsing**:
  - `extract_action_inputs()` successfully extracts Egyptian phone numbers, customer names, day of week, and time of day from concatenated strings such as:
    `"أحمد محمد01123456789الاتنن 2الظهر"` $\rightarrow$ Name: أحمد محمد, Phone: 01123456789, Date: الاثنين, Time: 2الظهر.
- **Reschedule & Details Reset**:
  - `detect_reschedule()` identifies `"بمواعيد تانية"` and resets schedule fields (`preferred_date=None`, `preferred_time=None`) without creating duplicate rows.
  - `detect_new_details()` identifies `"وببيانات جديدة كلية"` and clears all customer contact information while preserving the targeted vehicle ID.
- **Formal Cancellation**:
  - `cancel_test_drive(db_path, request_id, notes)` updates `test_drive_requests.status = 'CANCELLED'` and sets `cancelled_at = utc_now()`.
  - Added `CancelTestDrive` component to Langflow canvas and wired into `GeminiCarSalesAgent`.

### 2.4 Dedicated Subordinate Research Agent (`VehicleResearchAgent`)
- **Architecture**:
  - Defined in `components/car_dealership/vehicle_research_agent.py`.
  - Registered as a subordinate tool on the Langflow canvas for `GeminiCarSalesAgent`. It is **not** customer-facing.
- **5-Tier Source Hierarchy**:
  1. Tier 1: Official vehicle owner's manual / manufacturer spec sheets
  2. Tier 2: Local catalog documentation & knowledge base (`knowledge_documents`)
  3. Tier 3: Official brand website in Egypt
  4. Tier 4: Authorized dealer portals (e.g. ContactCars, Hatla2ee, El-Saba)
  5. Tier 5: General automotive web search
- **Fact Extraction & Guardrails**:
  - `extract_airbag_count()` returns strict numeric counts (e.g., 6 or 2) or explicit `None`. Never guesses.
  - Exact variant matching rule: Prevents returning 2.0L Turbo acceleration/power for a 1.6L naturally aspirated car. Returns `ambiguous` on spec mismatch.
  - Prompt-injection defense: `sanitize_and_check_injection()` scans external text for override attempts (`ignore previous rules`, `system prompt`, `jailbreak`), redacting content and flagging safety violations.
  - Query budget: Strictly bounded to a maximum of 3 search queries and 3 inspected sources per invocation.
  - Honest catalog phrasing: Enforces `"حسب البيانات المتاحة عندي"` instead of claiming real-time showroom inventory.

---

## 3. Automated Test Suite Results

All tests were executed directly inside the Docker container `car-dealership-langflow`:

```bash
docker exec -w /app car-dealership-langflow python tests/run_all_hardening_tests.py
```

### Test Output Summary
```text
test_a_conversation_memory (test_memory_manager.TestIntermediateMemory) ... ok
test_b_structured_state_merge (test_memory_manager.TestIntermediateMemory) ... ok
test_c_preference_override (test_memory_manager.TestIntermediateMemory) ... ok
test_d_recommendation_ids (test_memory_manager.TestIntermediateMemory) ... ok
test_e_ordinal_resolver (test_memory_manager.TestIntermediateMemory) ... ok
test_f_selected_car (test_memory_manager.TestIntermediateMemory) ... ok
test_g_pending_test_drive (test_memory_manager.TestIntermediateMemory) ... ok
test_h_cancellation (test_memory_manager.TestIntermediateMemory) ... ok
test_i_summary_threshold (test_memory_manager.TestIntermediateMemory) ... ok
test_j_summary_failure (test_memory_manager.TestIntermediateMemory) ... ok
test_k_session_isolation (test_memory_manager.TestIntermediateMemory) ... ok
test_01_variant_grouping (test_recommendation_snapshots.TestRecommendationSnapshots) ... ok
test_02_snapshot_creation_and_ordinal_resolution (test_recommendation_snapshots.TestRecommendationSnapshots) ... ok
test_03_negative_filter_ambiguity (test_recommendation_snapshots.TestRecommendationSnapshots) ... ok
test_04_preference_change_invalidates_snapshot (test_recommendation_snapshots.TestRecommendationSnapshots) ... ok
test_05_historical_snapshot_resolution (test_recommendation_snapshots.TestRecommendationSnapshots) ... ok
test_01_no_premature_insert_and_combined_parsing (test_business_action_hardening.TestBusinessActionHardening) ... ok
test_02_reschedule_clears_date_time_without_insert (test_business_action_hardening.TestBusinessActionHardening) ... ok
test_03_new_details_clears_customer_info (test_business_action_hardening.TestBusinessActionHardening) ... ok
test_04_completed_booking_cancellation (test_business_action_hardening.TestBusinessActionHardening) ... ok
test_01_exact_variant_matching_mismatch (test_vehicle_research_agent.TestVehicleResearchAgent) ... ok
test_02_airbags_numeric_count_extraction (test_vehicle_research_agent.TestVehicleResearchAgent) ... ok
test_03_prompt_injection_defense (test_vehicle_research_agent.TestVehicleResearchAgent) ... ok
test_04_structured_return_object (test_vehicle_research_agent.TestVehicleResearchAgent) ... ok
test_full_32_turn_scenario (test_32_turn_hardened_scenario.Test32TurnHardenedScenario) ... ok

----------------------------------------------------------------------
Ran 25 tests in 50.806s

OK

=======================================================
ALL TESTS SUMMARY: Ran 25 tests.
Failures: 0, Errors: 0
STATUS: ALL TESTS PASSED!
=======================================================
```

### 32-Turn Scenario Validation Sequence
The 32-turn integration scenario (`tests/test_32_turn_hardened_scenario.py`) successfully validated:
1. Turn 1: Session initialization & greeting.
2. Turn 2: Preference extraction (Zero, 1,800,000 EGP).
3. Turn 3: Body type constraint (SUV 7-seater).
4. Turn 4: Catalog search, variant grouping, and Recommendation Snapshot 1 sequence.
5. Turn 5: Ordinal resolution for `"الأولى"` $\rightarrow$ Car ID position 1.
6. Turn 6: Ordinal resolution for `"التانية"` $\rightarrow$ Car ID position 2.
7. Turn 7: Multi-car comparison `"قارن أول اتنين"`.
8. Turn 8: Ordinal resolution for `"آخر واحدة"`.
9. Turn 9: Negative constraint ambiguity detection (`"مش ساوايست"`).
10. Turn 10: Selected car anchoring.
11. Turn 11: `VehicleResearchAgent` lookup for airbag count.
12. Turn 12: Prompt injection detection and sanitization.
13. Turn 13: Exact variant mismatch detection (rejection of 2.0L Turbo on 1.6L variant).
14. Turn 14: Preference shift (`new` $\rightarrow$ `used`), invalidating Snapshot 1 and clearing incompatible selected car.
15. Turn 15: Unqualified ordinal on invalidated snapshot returns `no_active_snapshot`.
16. Turn 16: New search for used cars creates Snapshot 2 (sequence #2).
17. Turn 17: Explicit historical retrieval (`"من العربيات اللي قولتهم في الأول"`) resolves against Snapshot 1.
18. Turn 18: Selection anchored to Car from Snapshot 2.
19. Turn 19: Test drive triggered with no premature DB insert.
20. Turn 20: Combined single-message input parsed (`"أحمد محمد01123456789الاتنن 2الظهر"`).
21. Turn 21: Pending action payload verified with all 5 mandatory fields.
22. Turn 22: Real database insertion executed (`request_id` generated).
23. Turn 23: Reschedule intent (`"بمواعيد تانية"`) resets schedule fields without creating duplicate row.
24. Turn 24: New schedule input provided (`"الخميس 5 مساء"`).
25. Turn 25: Reset details intent (`"وببيانات جديدة كلية"`) clears customer fields while preserving targeted car.
26. Turn 26: New customer details provided and second booking row created.
27. Turn 27: Booking cancellation requested and executed in SQLite (`status='CANCELLED'`, `cancelled_at` set).
28. Turn 28: Message history expanded to 24+ turns, successfully triggering summarization.
29. Turn 29: Memory context prompt verified: `CURRENT STRUCTURED STATE` strictly precedes `HISTORICAL SUMMARY`.
30. Turn 30: Sales lead creation with financing inquiry.
31. Turn 31: Session state consistency verification.
32. Turn 32: Final database integrity and snapshot history check.

---

## 4. Container Restart & Persistence Verification

A full Docker restart was executed to verify state persistence across restarts:
```bash
docker restart car-dealership-langflow
```
Following restart and container recovery (`http://localhost:7860/health` $\rightarrow$ 200 OK), the database state was inspected:

```text
=== PERSISTENCE CHECK POST-RESTART ===
Cars count: 10306
Snapshots count: 26
Sessions count: 21
Messages count: 280
Summaries count: 3
Test Drives total: 14
Cancelled Test Drives: 4
Sample Cancelled Booking: {'id': 14, 'customer_name': 'محمود عادل', 'phone': '01099887766', 'car_id': 1593, 'status': 'CANCELLED', 'cancelled_at': '2026-09-03T19:53:18.330937+00:00'}
Sample Cancelled Booking: {'id': 12, 'customer_name': 'محمود عادل', 'phone': '01099887766', 'car_id': 1593, 'status': 'CANCELLED', 'cancelled_at': '2026-09-03T19:51:34.173741+00:00'}
=== PERSISTENCE VERIFIED SUCCESSFULLY ===
```

**Conclusion**: Complete zero-data-loss durability verified.

---

## 5. Flow Canvas & Registry Verification

The updated flow canvas was built from the server registry and installed via `scripts/build_and_install_flow.py`:
- Flow ID: `f7153d7e-1299-4c1b-bbca-3c57d5d0dbed`
- Flow Name: `Car Dealership AI Agent — Gemini 3.5 Flash-Lite`
- Nodes: 15 (including `CancelTestDrive` and `VehicleResearchAgent`)
- Edges: 14
- Exported JSON: `flow/Car_Dealership_Agent.flow.json`
