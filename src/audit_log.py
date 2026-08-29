"""
Append-only audit trail (locked doc §9). Every alert gets a permanent,
timestamped record: what fired, why (from SHAP), what threshold was in
effect, and what model version produced it.
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone


def get_model_version(model_path):
    """SHA-256 of the model file, used as a stable version identifier."""
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


class AuditLog:
    def __init__(self, log_path, model_path, threshold_txn):
        self.log_path = Path(log_path)
        self.model_version = get_model_version(model_path)
        self.threshold_txn = threshold_txn
        if not self.log_path.exists():
            self.log_path.touch()

    def log_transaction_alert(self, transaction_id, merchant_id, fraud_score, explanations):
        record = dict(
            record_type="transaction_alert",
            logged_at=datetime.now(timezone.utc).isoformat(),
            transaction_id=transaction_id,
            merchant_id=merchant_id,
            fraud_score=float(fraud_score),
            threshold_used=self.threshold_txn,
            model_version=self.model_version,
            explanations=explanations,
        )
        self._append(record)
        return record

    def log_burst_alert(self, merchant_id, fire_time, burst_explanation):
        record = dict(
            record_type="burst_alert",
            logged_at=datetime.now(timezone.utc).isoformat(),
            merchant_id=merchant_id,
            fire_time=str(fire_time),
            model_version=self.model_version,
            explanation=burst_explanation,
        )
        self._append(record)
        return record

    def _append(self, record):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read_all(self):
        if not self.log_path.exists():
            return []
        with open(self.log_path) as f:
            return [json.loads(line) for line in f if line.strip()]
