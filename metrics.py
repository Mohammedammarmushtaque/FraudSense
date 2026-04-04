import time
import threading
from typing import Dict, Any

class SystemMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_transactions = 0
        self.flagged_transactions = 0  # MFA_HOLD or BLOCK
        self.total_latency_ms = 0.0
        self.start_time = time.time()
        
    def record_transaction(self, latency_ms: float, decision: str):
        with self.lock:
            self.total_transactions += 1
            self.total_latency_ms += latency_ms
            if decision in ("MFA_HOLD", "BLOCK"):
                self.flagged_transactions += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            avg_latency = (self.total_latency_ms / self.total_transactions) if self.total_transactions > 0 else 0
            fraud_rate = (self.flagged_transactions / self.total_transactions * 100) if self.total_transactions > 0 else 0
            uptime_seconds = time.time() - self.start_time
            throughput = (self.total_transactions / uptime_seconds) if uptime_seconds > 0 else 0
            
            return {
                "avg_latency_ms": round(avg_latency, 2),
                "total_transactions": self.total_transactions,
                "fraud_detection_rate": round(fraud_rate, 2),
                "throughput_tps": round(throughput, 2)
            }

# Global singleton
metrics_tracker = SystemMetrics()
