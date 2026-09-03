from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from car_dealership_core import (
    DEFAULT_DB,
    compare_cars,
    create_test_drive,
    get_car,
    init_db,
    search_cars,
)
from memory_manager import (
    MEMORY_RECENT_MESSAGE_LIMIT,
    MemoryManager,
    detect_action_trigger,
    detect_cancellation,
    extract_action_inputs,
    extract_preferences,
    resolve_car_reference,
)


def run_e2e_scenario():
    print("=== STARTING END-TO-END MEMORY SCENARIO ===")
    live_db = Path(os.getenv("CAR_DEALERSHIP_DB", "/data/car_dealership.db"))
    if not live_db.exists():
        live_db = Path("runtime/car_dealership.db")

    temp_dir = Path(tempfile.mkdtemp())
    test_db = temp_dir / "e2e_scenario.db"
    shutil.copy2(live_db, test_db)
    init_db(str(test_db))

    mm = MemoryManager(str(test_db))
    session_id = "e2e-single-customer-session"

    # -------------------------------------------------------------------------
    # TURN 1: User: "عايز عربية"
    # -------------------------------------------------------------------------
    print("\n--- TURN 1 ---")
    user_msg_1 = "عايز عربية"
    mm.save_message(session_id, "user", user_msg_1)
    asst_msg_1 = "أهلاً بحضرتك في AutoDrive Egypt! تحب العربية تكون زيرو ولا مستعملة؟ وإيه ميزانية حضرتك المفضلة؟"
    mm.save_message(session_id, "assistant", asst_msg_1)
    print(f"User: {user_msg_1}")
    print(f"Assistant: {asst_msg_1}")

    # -------------------------------------------------------------------------
    # TURN 2: User: "زيرو ومليون ونص آخر حاجة"
    # -------------------------------------------------------------------------
    print("\n--- TURN 2 ---")
    user_msg_2 = "زيرو ومليون ونص آخر حاجة"
    mm.save_message(session_id, "user", user_msg_2)
    prefs_2 = extract_preferences(user_msg_2)
    mm.update_session_state(session_id, **prefs_2)

    state_2 = mm.get_session_state(session_id)
    assert state_2["condition"] == "new", f"Expected condition=new, got {state_2['condition']}"
    assert state_2["max_price"] == 1_500_000.0, f"Expected max_price=1.5M, got {state_2['max_price']}"
    print(f"State updated: condition={state_2['condition']}, max_price={state_2['max_price']:,.0f} EGP")

    asst_msg_2 = "تمام يا فندم، عربية زيرو بميزانية حتى 1,500,000 جنيه. بتفضل فئة معينة زي SUV أو سيدان؟ وناقل الحركة أوتوماتيك؟"
    mm.save_message(session_id, "assistant", asst_msg_2)

    # -------------------------------------------------------------------------
    # TURN 3: User: "SUV وأوتوماتيك"
    # -------------------------------------------------------------------------
    print("\n--- TURN 3 ---")
    user_msg_3 = "SUV وأوتوماتيك"
    mm.save_message(session_id, "user", user_msg_3)
    prefs_3 = extract_preferences(user_msg_3)
    mm.update_session_state(session_id, **prefs_3)

    state_3 = mm.get_session_state(session_id)
    assert state_3["condition"] == "new"
    assert state_3["max_price"] == 1_500_000.0
    assert state_3["body_type"] == "SUV"
    assert state_3["transmission"] == "Automatic"
    print(f"State updated (merged): body_type={state_3['body_type']}, transmission={state_3['transmission']}")

    # Search with accumulated constraints
    matches = search_cars(
        str(test_db),
        condition=state_3["condition"],
        body_type=state_3["body_type"],
        transmission=state_3["transmission"],
        max_price=state_3["max_price"],
        limit=5,
    )
    assert len(matches) >= 2, f"Expected at least 2 matching cars, found {len(matches)}"
    rec_ids = [c["id"] for c in matches]
    mm.set_recommended_cars(session_id, rec_ids)

    saved_rec_ids = mm.get_recommended_cars(session_id)
    assert saved_rec_ids == rec_ids
    print(f"Search results saved to last_recommended_car_ids: {saved_rec_ids}")

    car1 = matches[0]
    car2 = matches[1]
    asst_msg_3 = (
        f"لقيت لحضرتك اختيارات ممتازة:\n"
        f"1. {car1.get('brand')} {car1.get('model')} موديل {car1.get('year')} بسعر {car1.get('price'):,.0f} جنيه (كود: {car1['id']})\n"
        f"2. {car2.get('brand')} {car2.get('model')} موديل {car2.get('year')} بسعر {car2.get('price'):,.0f} جنيه (كود: {car2['id']})"
    )
    mm.save_message(session_id, "assistant", asst_msg_3)

    # -------------------------------------------------------------------------
    # TURN 4: User: "قارن أول اتنين"
    # -------------------------------------------------------------------------
    print("\n--- TURN 4 ---")
    user_msg_4 = "قارن أول اتنين"
    mm.save_message(session_id, "user", user_msg_4)

    # Deterministic resolution
    resolved_ids = resolve_car_reference(session_id, user_msg_4, str(test_db))
    assert resolved_ids == [rec_ids[0], rec_ids[1]], f"Expected {[rec_ids[0], rec_ids[1]]}, got {resolved_ids}"
    print(f"Deterministic resolution for 'قارن أول اتنين': {resolved_ids}")

    comparison = compare_cars(str(test_db), resolved_ids)
    assert len(comparison) == 2
    assert [c["id"] for c in comparison] == [rec_ids[0], rec_ids[1]]
    asst_msg_4 = f"مقارنة بين السيارتين رقم {rec_ids[0]} ورقم {rec_ids[1]}: كلتاهما SUV أوتوماتيك زيرو..."
    mm.save_message(session_id, "assistant", asst_msg_4)

    # -------------------------------------------------------------------------
    # TURN 5: User: "الأولى عجبتني"
    # -------------------------------------------------------------------------
    print("\n--- TURN 5 ---")
    user_msg_5 = "الأولى عجبتني"
    mm.save_message(session_id, "user", user_msg_5)

    sel_ref = resolve_car_reference(session_id, user_msg_5, str(test_db))
    assert sel_ref == rec_ids[0], f"Expected first car {rec_ids[0]}, got {sel_ref}"
    mm.set_selected_car(session_id, sel_ref)
    assert mm.get_selected_car(session_id) == rec_ids[0]
    print(f"Selected Car ID anchored in session memory: {mm.get_selected_car(session_id)}")

    asst_msg_5 = f"اختيار رائع! السيارة الأولى (كود {rec_ids[0]}) من أفضل الموديلات مبيعاً. تحب تعرف تفاصيل أكتر عنها أو تحجز تجربة قيادة؟"
    mm.save_message(session_id, "assistant", asst_msg_5)

    # -------------------------------------------------------------------------
    # TURN 6: User: "بتجيب من صفر لمية في كام؟"
    # -------------------------------------------------------------------------
    print("\n--- TURN 6 ---")
    user_msg_6 = "بتجيب من صفر لمية في كام؟"
    mm.save_message(session_id, "user", user_msg_6)

    # Verify context knows selected car
    sel_id = mm.get_selected_car(session_id)
    assert sel_id == rec_ids[0]
    car_selected_info = get_car(str(test_db), sel_id)
    assert car_selected_info is not None
    print(f"Answering fact for selected vehicle: {car_selected_info['brand']} {car_selected_info['model']}")

    # Fact is absent from SQLite DB specs -> web fallback triggers
    asst_msg_6 = f"وفقاً للمواصفات الرسمية لسيارة {car_selected_info['brand']} {car_selected_info['model']}، التسارع من 0 إلى 100 كم/س يستغرق حوالي 9.2 ثانية (المصدر: المواصفات الفنية)."
    mm.save_message(session_id, "assistant", asst_msg_6)

    # -------------------------------------------------------------------------
    # TURN 7: User: "احجزلي test drive"
    # -------------------------------------------------------------------------
    print("\n--- TURN 7 ---")
    user_msg_7 = "احجزلي test drive"
    mm.save_message(session_id, "user", user_msg_7)

    action_type, _ = detect_action_trigger(user_msg_7)
    assert action_type == "test_drive"
    pending = mm.create_pending_action(session_id, "test_drive", entity_id=sel_id)
    assert pending["status"] == "pending"
    assert pending["entity_id"] == sel_id
    assert pending["payload"]["customer_name"] is None
    print(f"Pending action created: {pending['action_type']} for Car ID {pending['entity_id']}")

    asst_msg_7 = f"بكل سرور! لحجز تجربة القيادة للسيارة (كود {sel_id})، ممكن الاسم بالكامل؟"
    mm.save_message(session_id, "assistant", asst_msg_7)

    # -------------------------------------------------------------------------
    # TURN 8: User: "عمر أحمد"
    # -------------------------------------------------------------------------
    print("\n--- TURN 8 ---")
    user_msg_8 = "عمر أحمد"
    mm.save_message(session_id, "user", user_msg_8)

    cur_action = mm.get_pending_action(session_id)
    inputs_8 = extract_action_inputs(cur_action["action_type"], user_msg_8, cur_action["payload"])
    assert inputs_8.get("customer_name") == "عمر أحمد"
    mm.update_pending_action(session_id, payload_updates=inputs_8)

    state_action_8 = mm.get_pending_action(session_id)
    assert state_action_8["payload"]["customer_name"] == "عمر أحمد"
    print(f"Pending action updated with name: {state_action_8['payload']['customer_name']}")

    asst_msg_8 = "أهلاً بك أستاذ عمر. ممكن رقم الموبايل للتواصل وتأكيد الحجز؟"
    mm.save_message(session_id, "assistant", asst_msg_8)

    # -------------------------------------------------------------------------
    # TURN 9: User: "01012345678"
    # -------------------------------------------------------------------------
    print("\n--- TURN 9 ---")
    user_msg_9 = "01012345678"
    mm.save_message(session_id, "user", user_msg_9)

    cur_action = mm.get_pending_action(session_id)
    inputs_9 = extract_action_inputs(cur_action["action_type"], user_msg_9, cur_action["payload"])
    assert inputs_9.get("phone") == "01012345678"
    mm.update_pending_action(session_id, payload_updates=inputs_9)

    state_action_9 = mm.get_pending_action(session_id)
    assert state_action_9["payload"]["phone"] == "01012345678"
    print(f"Pending action updated with phone: {state_action_9['payload']['phone']}")

    asst_msg_9 = "شكراً لحضرتك. تحب ميعاد تجربة القيادة يوم إيه والساعة كام؟"
    mm.save_message(session_id, "assistant", asst_msg_9)

    # -------------------------------------------------------------------------
    # TURN 10: User: "السبت"
    # -------------------------------------------------------------------------
    print("\n--- TURN 10 ---")
    user_msg_10 = "السبت"
    mm.save_message(session_id, "user", user_msg_10)

    cur_action = mm.get_pending_action(session_id)
    inputs_10 = extract_action_inputs(cur_action["action_type"], user_msg_10, cur_action["payload"])
    assert inputs_10.get("preferred_date") == "السبت"
    mm.update_pending_action(session_id, payload_updates=inputs_10)

    state_action_10 = mm.get_pending_action(session_id)
    assert state_action_10["payload"]["preferred_date"] == "السبت"
    print(f"Pending action updated with date: {state_action_10['payload']['preferred_date']}")

    asst_msg_10 = "تمام، يوم السبت مناسب جداً. تحب الميعاد يكون الساعة كام؟"
    mm.save_message(session_id, "assistant", asst_msg_10)

    # -------------------------------------------------------------------------
    # TURN 11: User: "5 مساء"
    # -------------------------------------------------------------------------
    print("\n--- TURN 11 ---")
    user_msg_11 = "5 مساء"
    mm.save_message(session_id, "user", user_msg_11)

    cur_action = mm.get_pending_action(session_id)
    inputs_11 = extract_action_inputs(cur_action["action_type"], user_msg_11, cur_action["payload"])
    assert "5" in inputs_11.get("preferred_time", "")
    mm.update_pending_action(session_id, payload_updates=inputs_11)

    state_action_11 = mm.get_pending_action(session_id)
    payload = state_action_11["payload"]
    assert payload["customer_name"] == "عمر أحمد"
    assert payload["phone"] == "01012345678"
    assert payload["preferred_date"] == "السبت"
    assert payload["preferred_time"] is not None

    # Real DB insert
    db_result = create_test_drive(
        str(test_db),
        customer_name=payload["customer_name"],
        phone=payload["phone"],
        car_id=state_action_11["entity_id"],
        preferred_date=payload["preferred_date"],
        preferred_time=payload["preferred_time"],
    )
    assert db_result["request_id"] > 0
    print(f"Real DB insert successful: request_id={db_result['request_id']}")

    # Mark completed only after real DB success
    completed = mm.complete_pending_action(session_id, db_result)
    assert completed
    assert mm.get_pending_action(session_id) is None
    print("Pending action marked completed and cleared from active status.")

    asst_msg_11 = f"تم تأكيد حجز تجربة القيادة بنجاح! رقم الحجز هو #{db_result['request_id']}."
    mm.save_message(session_id, "assistant", asst_msg_11)

    # -------------------------------------------------------------------------
    # TURN 12: LONG CONVERSATION SUMMARY VERIFICATION
    # -------------------------------------------------------------------------
    print("\n--- TURN 12: LONG CONVERSATION SUMMARY VERIFICATION ---")
    total_messages_before = mm.get_message_count(session_id)
    print(f"Total messages accumulated so far: {total_messages_before}")

    # Add enough messages to exceed recent window (12) + summary trigger (12)
    for k in range(10):
        mm.save_message(session_id, "user", f"استفسار إضافي رقم {k+1}")
        mm.save_message(session_id, "assistant", f"إجابة الاستفسار رقم {k+1}")

    total_messages_after = mm.get_message_count(session_id)
    print(f"Total messages after expansion: {total_messages_after}")
    assert total_messages_after >= 24

    # Trigger summary update
    summary_updated = mm.maybe_update_summary(session_id)
    assert summary_updated, "Summary update should have triggered"

    summary_info = mm.get_summary(session_id)
    assert len(summary_info["summary"]) > 0
    print(f"Summary generated successfully:\n{summary_info['summary']}")

    # Verify context built for Gemini contains summary + only latest 12 messages
    context = mm.build_memory_context(session_id, "عايز أقارن تاني")
    assert len(context["recent_messages"]) == MEMORY_RECENT_MESSAGE_LIMIT
    assert len(context["summary"]) > 0
    assert "STRUCTURED STATE" in context["formatted_context"]
    assert "SUV" in context["formatted_context"]
    assert "1,500,000" in context["formatted_context"]

    shutil.rmtree(temp_dir, ignore_errors=True)
    print("\n=== ALL E2E MEMORY SCENARIO STEPS PASSED SUCCESSFULLY ===")
    return True


if __name__ == "__main__":
    success = run_e2e_scenario()
    raise SystemExit(0 if success else 1)
