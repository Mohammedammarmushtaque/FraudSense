import sqlite3
import os

db_path = r"c:\Users\Ammar A16\OneDrive\Desktop\HackupTrial-final\HackupTrial\backend\db\fraudsense.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE transactions ADD COLUMN created_at TIMESTAMP;")
    print("Column created_at added.")
except sqlite3.OperationalError as e:
    print(f"Error adding column: {e}")

conn.commit()
conn.close()
