import json
from datetime import datetime
import networkx as nx

class GraphService:
    def __init__(self, db):
        self.db = db
        # Pre-initialize directed graph
        self.G = nx.DiGraph()

    # =========================================================
    # Chain Engine Logic (Merged)
    # =========================================================
    def _get_chain_record(self, user_id):
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM chain_states WHERE user_id = ?", (user_id,))
        return cursor.fetchone()

    def process_chain_event(self, user_id, event):
        rec = self._get_chain_record(user_id)
        if not rec:
            self.db.execute("INSERT INTO chain_states (user_id, state, event_log, suspicion_score, last_event_time) VALUES (?, ?, ?, ?, ?)",
                            (user_id, "CLEAN", "[]", 0, datetime.now().isoformat()))
            self.db.commit()
            rec = self._get_chain_record(user_id)
        
        event_log = json.loads(rec["event_log"] or "[]")
        event_log.append({"event": event, "time": datetime.now().isoformat()})
        
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
        rec = self._get_chain_record(user_id)
        if not rec:
            return 0.0
        if rec["state"] == "WATCH":
            return 15.0
        elif rec["state"] == "MFA_REQUIRED":
            return 25.0
        elif rec["state"] == "BLOCKED":
            return 100.0
        return 0.0

    def get_current_chain_state(self, user_id):
        rec = self._get_chain_record(user_id)
        if not rec:
            return "CLEAN"
        return rec["state"]

    def reset_chain(self, user_id):
        self.db.execute("UPDATE chain_states SET state = 'CLEAN', event_log = '[]', suspicion_score = 0 WHERE user_id = ?", (user_id,))
        self.db.commit()

    # =========================================================
    # NetworkX Graph Logic (Fraud Ring Detection)
    # =========================================================
    def build_fraud_graph(self, window_hours=24):
        """
        Builds a multi-layer graph of:
          - Users (nodes)
          - Devices (nodes)
          - Transactions (edges between users or user->merchant)
          - Device Usage (edges between users and devices)
        """
        cursor = self.db.cursor()
        
        # 1. Device Usage (Shared Device Detection)
        cursor.execute("""
            SELECT user_id, device_id FROM user_devices 
            WHERE last_seen > datetime('now', ?)
        """, (f'-{window_hours} hours',))
        device_rows = cursor.fetchall()
        
        self.G.clear()
        for row in device_rows:
            uid = f"USER_{row['user_id']}"
            did = f"DEV_{row['device_id']}"
            self.G.add_node(uid, type='USER')
            self.G.add_node(did, type='DEVICE')
            self.G.add_edge(uid, did, relationship='usage')
            self.G.add_edge(did, uid, relationship='usage') # Undirected device sharing

        # 2. Transaction Flow (Circular Money Detection)
        cursor.execute("""
            SELECT user_id, merchant_id, amount, tx_id, timestamp 
            FROM transactions 
            WHERE timestamp > datetime('now', ?)
        """, (f'-{window_hours} hours',))
        tx_rows = cursor.fetchall()
        
        for row in tx_rows:
            uid = f"USER_{row['user_id']}"
            # For circular flow, check if merchant is actually another user
            # (In some systems, merchant_id might be a user_id for peer-to-peer transfers)
            is_p2p = row['merchant_id'].startswith('user_')
            target_node = f"USER_{row['merchant_id']}" if is_p2p else f"MERCH_{row['merchant_id']}"
            
            self.G.add_node(uid, type='USER')
            self.G.add_node(target_node, type='USER' if is_p2p else 'MERCHANT')
            self.G.add_edge(uid, target_node, 
                           relationship='transfer', 
                           weight=row['amount'], 
                           tx_id=row['tx_id'])

    def detect_suspicious_clusters(self, current_user_id, current_merchant_id):
        """Detects if the current transaction is part of a fraud ring or circular flow."""
        self.build_fraud_graph(window_hours=24) # Wider window for ring detection
        
        uid = f"USER_{current_user_id}"
        mid = f"MERCH_{current_merchant_id}"
        
        graph_risk = 0
        reasons = []

        # ── 1. Shared Device Detection (User -> Device <- User) ──────────
        user_devices = [n for n in self.G.neighbors(uid) if self.G.nodes[n].get('type') == 'DEVICE']
        for dev in user_devices:
            other_users = [n for n in self.G.neighbors(dev) if n != uid and self.G.nodes[n].get('type') == 'USER']
            if len(other_users) >= 2: # 3+ users on one device
                graph_risk += 40
                reasons.append(f"Fraud Ring: Device {dev} shared with {len(other_users)} other users")

        # ── 2. Circular Flow Detection (nx.simple_cycles) ────────────────
        # We only care about cycles in 'transfer' relationships
        try:
            # Create a subgraph of only user-to-user transfers for cycle detection
            transfer_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d['relationship'] == 'transfer']
            T = nx.DiGraph(transfer_edges)
            cycles = list(nx.simple_cycles(T))
            for cycle in cycles:
                if uid in cycle:
                    graph_risk += 60
                    reasons.append(f"Circular Flow: User detected in money recycling ring (length {len(cycle)})")
                    break
        except Exception as e:
            # simple_cycles can be computationally expensive on large graphs
            pass

        # ── 3. High In-Degree (Coordinated Merchants) ───────────────────
        if mid in self.G:
            in_degree = self.G.in_degree(mid)
            if in_degree >= 5:
                graph_risk += 25
                reasons.append(f"Coordinated Activity: {in_degree} users targeting same merchant")

        return {
            "graph_risk": min(100, graph_risk),
            "reasons": list(dict.fromkeys(reasons)) # uniq
        }
