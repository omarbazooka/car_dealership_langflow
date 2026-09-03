# AutoDrive Egypt end-to-end validation

Overall status: **FAIL**

- FAIL: balanced_dataset_target
- PASS: live_database_integrity
- PASS: structured_search_new_suv_automatic_budget
- PASS: structured_search_used_gasoline_year_budget
- PASS: comparison_by_prior_ids
- PASS: persistent_conversation_memory
- PASS: real_business_action_inserts
- PASS: rag_policy_retrieval_fallback
- PASS: input_output_guardrails
- PASS: langflow_health
- PASS: deployed_flow_graph
- PASS: web_fallback_wiring
- NOT TESTED: live_llm_conversation_scenarios

The validation database is an isolated copy; automated booking/lead test rows were not added to the live database.
