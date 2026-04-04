import json

class ProfileService:
    def __init__(self, db):
        self.db = db

    def get_profile(self, user_id):
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        
        if row:
            # We assume user_profiles has these standard fields.
            return {
                "avg_amount":         row["avg_amount"],
                "std_amount":         row["std_amount"],
                "tx_count":           row["tx_count"],
                "typical_hour":       row["typical_hour"] if "typical_hour" in row.keys() else 12,
                "common_merchants":   json.loads(row["common_merchants"]) if row["common_merchants"] else [],
                "frequent_locations": json.loads(row["common_merchants"]) if row["common_merchants"] else [],
                "trusted_devices":    self._get_trusted_devices(user_id),
            }
        
        # Default empty profile
        return {
            "avg_amount":       0.0,
            "std_amount":       0.0,
            "tx_count":         0,
            "typical_hour":     12,
            "common_merchants": [],
            "frequent_locations": [],
            "trusted_devices":  [],
        }

    def _get_trusted_devices(self, user_id):
        cursor = self.db.cursor()
        cursor.execute("SELECT device_id FROM user_devices WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        return [r["device_id"] for r in rows]

    def update_profile(self, user_id, transaction_data):
        """Update running averages and history after a transaction."""
        cursor = self.db.cursor()
        
        # 1. Ensure profile exists
        cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        
        amt = transaction_data.get("amount", 0)
        merch = transaction_data.get("merchant_id", "")
        city = transaction_data.get("city", "")
        
        if not row:
            cursor.execute(
                "INSERT INTO user_profiles (user_id, avg_amount, std_amount, tx_count, common_merchants, frequent_locations, typical_hour) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, amt, 0.0, 1, json.dumps([merch]) if merch else "[]", json.dumps([city]) if city else "[]", 12)
            )
        else:
            old_count = row["tx_count"]
            new_count = old_count + 1
            old_avg = row["avg_amount"]
            
            # Running average
            new_avg = old_avg + (amt - old_avg) / new_count
            
            # Update merchants and locations lists
            merchants = json.loads(row["common_merchants"]) if row["common_merchants"] else []
            if merch and merch not in merchants: merchants.append(merch)
            
            locations = json.loads(row["frequent_locations"]) if row["frequent_locations"] else []
            if city and city not in locations: locations.append(city)

            cursor.execute(
                "UPDATE user_profiles SET avg_amount = ?, tx_count = ?, common_merchants = ?, frequent_locations = ? WHERE user_id = ?",
                (new_avg, new_count, json.dumps(merchants), json.dumps(locations), user_id)
            )
            
        self.db.commit()
