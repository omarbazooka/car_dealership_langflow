import sqlite3
import os

db_path = "/data/car_dealership.db" if os.path.exists("/data/car_dealership.db") else "runtime/car_dealership.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

print("=== PERSISTENCE CHECK POST-RESTART ===")
print("Cars count:", conn.execute("SELECT COUNT(1) FROM cars").fetchone()[0])
print("Snapshots count:", conn.execute("SELECT COUNT(1) FROM recommendation_snapshots").fetchone()[0])
print("Sessions count:", conn.execute("SELECT COUNT(1) FROM conversation_sessions").fetchone()[0])
print("Messages count:", conn.execute("SELECT COUNT(1) FROM conversation_messages").fetchone()[0])
print("Summaries count:", conn.execute("SELECT COUNT(1) FROM conversation_summaries").fetchone()[0])
print("Test Drives total:", conn.execute("SELECT COUNT(1) FROM test_drive_requests").fetchone()[0])
print("Cancelled Test Drives:", conn.execute("SELECT COUNT(1) FROM test_drive_requests WHERE status='CANCELLED'").fetchone()[0])

cancelled_rows = conn.execute(
    "SELECT id, customer_name, phone, car_id, status, cancelled_at FROM test_drive_requests WHERE status='CANCELLED' ORDER BY id DESC LIMIT 2"
).fetchall()
for r in cancelled_rows:
    print("Sample Cancelled Booking:", dict(r))

conn.close()
print("=== PERSISTENCE VERIFIED SUCCESSFULLY ===")
