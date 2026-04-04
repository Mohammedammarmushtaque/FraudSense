import sqlite3
import os

db_path = r"c:\Users\Ammar A16\OneDrive\Desktop\HackupTrial-final\HackupTrial\backend\db\fraudsense.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(transactions);")
cols = cursor.fetchall()
print("Columns in transactions:")
for col in cols:
    print(col)
conn.close()
