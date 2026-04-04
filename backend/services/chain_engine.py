import json
from datetime import datetime

class ChainEngine:
    def __init__(self, db):
        self.db = db

    def _get_record(self, user_id):
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM chain_states WHERE user_id = ?", (user_id,))
        return cursor.fetchone()

    def process_event(self, user_id, event):
        rec = self._get_record(user_id)
        if not rec:
            self.db.execute("INSERT INTO chain_states (user_id, state, event_log, suspicion_score, last_event_time) VALUES (?, ?, ?, ?, ?)",
                            (user_id, "CLEAN", "[]", 0, datetime.now().isoformat()))
            self.db.commit()
            rec = self._get_record(user_id)
        
        event_log = json.loads(rec["event_log"] or "[]")
        event_log.append({"event": event, "time": datetime.now().isoformat()})
        
        # simple state machine
        new_state = rec["state"]
        if event == "LOGIN_NEW_DEVICE":
            new_state = "WATCH"
        elif event == "ANALYST_CONFIRM_FRAUD":
            new_state = "BLOCKED"
        elif event == "TRANSACTION_ATTEMPT" and new_state != "BLOCKED":
            new_state = "MFA_REQUIRED"
        elif event == "FAILED_MFA":
            new_state = "BLOCKED"
        elif event == "MFA_SUCCESS" and new_state != "BLOCKED":
            new_state = "CLEAN"
            
        self.db.execute("UPDATE chain_states SET state = ?, event_log = ?, last_event_time = ? WHERE user_id = ?",
                        (new_state, json.dumps(event_log), datetime.now().isoformat(), user_id))
        self.db.commit()
        return new_state

    def get_chain_risk_boost(self, user_id):
        rec = self._get_record(user_id)
        if not rec:
            return 0.0
        if rec["state"] == "WATCH":
            return 15.0
        elif rec["state"] == "MFA_REQUIRED":
            return 25.0
        elif rec["state"] == "BLOCKED":
            return 100.0
        return 0.0

    def get_current_state(self, user_id):
        rec = self._get_record(user_id)
        if not rec:
            return "CLEAN"
        return rec["state"]

    def reset_chain(self, user_id):
        self.db.execute("UPDATE chain_states SET state = 'CLEAN', event_log = '[]', suspicion_score = 0 WHERE user_id = ?", (user_id,))
        self.db.commit()
