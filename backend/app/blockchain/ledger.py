"""
Local SHA-256 hash-linked audit ledger.
Provides tamper-evident cryptographic hash chaining stored in SQLite
(not a decentralized blockchain with distributed consensus).
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

GENESIS_PREV_HASH = "0" * 64


def sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def compute_record_hash(record: Dict[str, Any]) -> str:
    canonical = json.dumps(
        {
            "event_type": record["event_type"],
            "hospital_id": record["hospital_id"],
            "payload_hash": record["payload_hash"],
            "payload_summary": record["payload_summary"],
            "previous_hash": record["previous_hash"],
            "record_id": int(record["record_id"]),
            "timestamp": record["timestamp"],
            "training_round": int(record["training_round"]),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_hex(canonical)


def create_audit_record(
    record_id: int,
    previous_hash: str,
    hospital_id: str,
    training_round: int,
    event_type: str,
    payload_summary: str,
    payload_data: Any,
) -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    payload_hash = sha256_hex(json.dumps(payload_data, sort_keys=True, separators=(",", ":")))
    rec = {
        "record_id": int(record_id),
        "timestamp": timestamp,
        "hospital_id": hospital_id,
        "training_round": int(training_round),
        "event_type": event_type,
        "payload_summary": payload_summary,
        "payload_hash": payload_hash,
        "previous_hash": previous_hash,
    }
    rec["current_hash"] = compute_record_hash(rec)
    return rec


def verify_ledger_chain(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    verified_at = datetime.now(timezone.utc).isoformat()
    if not records:
        return {
            "valid": True,
            "total_records": 0,
            "checked_records": 0,
            "broken_record_id": None,
            "failure_reason": None,
            "verified_at": verified_at,
        }

    for idx, rec in enumerate(records):
        expected_prev = GENESIS_PREV_HASH if idx == 0 else records[idx - 1]["current_hash"]
        if rec["previous_hash"] != expected_prev:
            return {
                "valid": False,
                "total_records": len(records),
                "checked_records": idx + 1,
                "broken_record_id": int(rec["record_id"]),
                "failure_reason": (
                    f"Chain linkage mismatch at Record #{rec['record_id']}: "
                    "previous_hash does not match prior record's current_hash."
                ),
                "verified_at": verified_at,
            }

        recomputed = compute_record_hash(rec)
        if recomputed != rec["current_hash"]:
            return {
                "valid": False,
                "total_records": len(records),
                "checked_records": idx + 1,
                "broken_record_id": int(rec["record_id"]),
                "failure_reason": (
                    f"Hash mismatch at Record #{rec['record_id']}: "
                    "stored SHA-256 digest does not match recomputed record hash."
                ),
                "verified_at": verified_at,
            }

    return {
        "valid": True,
        "total_records": len(records),
        "checked_records": len(records),
        "broken_record_id": None,
        "failure_reason": None,
        "verified_at": verified_at,
    }


def run_controlled_tamper_test(
    live_records: List[Dict[str, Any]],
    target_record_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs a non-destructive tampering simulation on an in-memory clone of the ledger.
    Never mutates live SQLite audit records.
    """
    if not live_records:
        raise ValueError("Audit ledger is empty. Initialize an experiment first.")

    cloned = copy.deepcopy(live_records)
    target_idx = len(cloned) - 1
    if target_record_id is not None:
        for i, rec in enumerate(cloned):
            if int(rec["record_id"]) == int(target_record_id):
                target_idx = i
                break
    elif len(cloned) >= 2:
        target_idx = len(cloned) - 2

    target = cloned[target_idx]
    original_summary = target["payload_summary"]
    tampered_summary = f"{original_summary} [ALTERED_WEIGHT_OVERRIDE=+99.4]"
    target["payload_summary"] = tampered_summary

    recomputed_hash = compute_record_hash(target)
    verification_after_tamper = verify_ledger_chain(cloned)
    live_check = verify_ledger_chain(live_records)

    return {
        "test_executed_at": datetime.now(timezone.utc).isoformat(),
        "live_ledger_untouched": True,
        "live_ledger_valid": live_check["valid"],
        "target_record_id": int(target["record_id"]),
        "original_summary": original_summary,
        "tampered_summary": tampered_summary,
        "stored_hash": target["current_hash"],
        "recomputed_hash": recomputed_hash,
        "verification_after_tamper": verification_after_tamper,
    }
