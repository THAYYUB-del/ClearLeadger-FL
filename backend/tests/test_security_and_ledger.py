"""Stdlib unit tests for Python HMAC authentication, SHA-256 ledger, and SQLite storage."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.blockchain.ledger import (
    GENESIS_PREV_HASH,
    create_audit_record,
    run_controlled_tamper_test,
    verify_ledger_chain,
)
from app.security.auth import (
    inspect_and_validate_update,
    sign_model_update,
    verify_hmac_signature,
)
from app.services.database import ClearLedgerDatabase


class TestPythonSecurityAndLedger(unittest.TestCase):
    def test_hmac_signing_and_rejection(self):
        weights = [0.05] * 30
        valid = sign_model_update("hospital_a", 1, 150, weights, 0.02)
        self.assertTrue(verify_hmac_signature(valid))

        tampered = {**valid, "hmac_signature": "0" * 64}
        self.assertFalse(verify_hmac_signature(tampered))

        forged = sign_model_update(
            "hospital_a", 1, 150, weights, 0.02, override_secret="wrong-secret-key"
        )
        self.assertFalse(verify_hmac_signature(forged))

    def test_distinguish_hmac_from_exaggerated_update(self):
        zero_w = [0.0] * 30
        huge_w = [3.0] * 30
        signed_huge = sign_model_update("hospital_b", 1, 150, huge_w, 1.0)
        res = inspect_and_validate_update(signed_huge, zero_w, 0.0)
        self.assertTrue(res["hmac_valid"])
        self.assertFalse(res["validation_passed"])
        self.assertEqual(res["decision"], "REJECTED")
        self.assertEqual(res["layer_failed"], "MALICIOUS_VALIDATION")

    def test_hash_linked_ledger_and_tamper_detection(self):
        r1 = create_audit_record(
            1, GENESIS_PREV_HASH, "aggregator", 0, "EXPERIMENT_INITIALIZED", "Init", {"seed": 42}
        )
        r2 = create_audit_record(
            2, r1["current_hash"], "hospital_a", 1, "AUTHENTICATED_UPDATE_ACCEPTED", "Update A", {"n": 150}
        )
        ver = verify_ledger_chain([r1, r2])
        self.assertTrue(ver["valid"])
        self.assertEqual(ver["checked_records"], 2)

        tamper = run_controlled_tamper_test([r1, r2], target_record_id=1)
        self.assertFalse(tamper["verification_after_tamper"]["valid"])
        self.assertEqual(tamper["verification_after_tamper"]["broken_record_id"], 1)
        self.assertTrue(tamper["live_ledger_valid"])

    def test_sqlite_database_roundtrip(self):
        db = ClearLedgerDatabase(":memory:")
        rec = create_audit_record(
            1, GENESIS_PREV_HASH, "aggregator", 0, "EXPERIMENT_INITIALIZED", "Init", {"seed": 42}
        )
        db.append_audit_record(rec)
        ledger = db.get_audit_ledger()
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["current_hash"], rec["current_hash"])
        db.reset_all()
        self.assertEqual(len(db.get_audit_ledger()), 0)
        db.close()


if __name__ == "__main__":
    unittest.main()
