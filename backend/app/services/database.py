"""Parameterized SQLite storage service for ClearLedger-FL."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class ClearLedgerDatabase:
    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            env_path = os.getenv("CLEARLEDGER_DB_PATH", "./results/clearledger.db")
            resolved = Path(env_path).resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(resolved)
        elif db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            resolved = Path(db_path).resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(resolved)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiment_meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                initialized INTEGER NOT NULL DEFAULT 0,
                distribution TEXT NOT NULL DEFAULT 'balanced',
                seed INTEGER NOT NULL DEFAULT 42,
                local_epochs INTEGER NOT NULL DEFAULT 12,
                learning_rate REAL NOT NULL DEFAULT 0.12,
                current_round INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'not_initialized',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS training_rounds (
                round_number INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                distribution TEXT NOT NULL,
                accepted_updates INTEGER NOT NULL,
                rejected_updates INTEGER NOT NULL,
                global_accuracy REAL NOT NULL,
                global_precision REAL NOT NULL,
                global_recall REAL NOT NULL,
                global_f1 REAL NOT NULL,
                global_roc_auc REAL NOT NULL,
                global_loss REAL NOT NULL,
                confusion_matrix_json TEXT NOT NULL,
                hospital_contributions_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                hospital_id TEXT NOT NULL,
                claimed_sender TEXT NOT NULL,
                training_round INTEGER NOT NULL,
                scenario TEXT NOT NULL,
                hmac_valid INTEGER NOT NULL,
                validation_passed INTEGER NOT NULL,
                decision TEXT NOT NULL,
                layer_failed TEXT NOT NULL,
                reason TEXT NOT NULL,
                delta_norm REAL NOT NULL,
                cosine_similarity REAL,
                signature_preview TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_ledger (
                record_id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                hospital_id TEXT NOT NULL,
                training_round INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_summary TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                current_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS system_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                detail TEXT NOT NULL,
                severity TEXT NOT NULL
            );
            """
        )
        row = cur.execute("SELECT id FROM experiment_meta WHERE id = 1").fetchone()
        if row is None:
            cur.execute(
                """
                INSERT INTO experiment_meta (
                    id, initialized, distribution, seed, local_epochs, learning_rate, current_round, status, updated_at
                ) VALUES (1, 0, 'balanced', 42, 12, 0.12, 0, 'not_initialized', ?)
                """,
                (datetime.now(timezone.utc).isoformat(),),
            )
        self.conn.commit()

    def save_experiment_meta(self, meta: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            UPDATE experiment_meta
            SET initialized = ?, distribution = ?, seed = ?, local_epochs = ?, learning_rate = ?,
                current_round = ?, status = ?, updated_at = ?
            WHERE id = 1
            """,
            (
                1 if meta["initialized"] else 0,
                meta["distribution"],
                int(meta["seed"]),
                int(meta["local_epochs"]),
                float(meta["learning_rate"]),
                int(meta["current_round"]),
                meta["status"],
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def get_experiment_meta(self) -> Dict[str, Any]:
        row = self.conn.execute("SELECT * FROM experiment_meta WHERE id = 1").fetchone()
        return {
            "initialized": bool(row["initialized"]),
            "distribution": row["distribution"],
            "seed": int(row["seed"]),
            "local_epochs": int(row["local_epochs"]),
            "learning_rate": float(row["learning_rate"]),
            "current_round": int(row["current_round"]),
            "status": row["status"],
            "updated_at": row["updated_at"],
        }

    def insert_round(self, round_data: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO training_rounds (
                round_number, timestamp, distribution, accepted_updates, rejected_updates,
                global_accuracy, global_precision, global_recall, global_f1, global_roc_auc, global_loss,
                confusion_matrix_json, hospital_contributions_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(round_data["round_number"]),
                round_data["timestamp"],
                round_data["distribution"],
                int(round_data["accepted_updates"]),
                int(round_data["rejected_updates"]),
                float(round_data["global_accuracy"]),
                float(round_data["global_precision"]),
                float(round_data["global_recall"]),
                float(round_data["global_f1"]),
                float(round_data["global_roc_auc"]),
                float(round_data["global_loss"]),
                json.dumps(round_data["confusion_matrix"]),
                json.dumps(round_data["hospital_contributions"]),
            ),
        )
        self.conn.commit()

    def get_rounds(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM training_rounds ORDER BY round_number ASC"
        ).fetchall()
        return [
            {
                "round_number": int(r["round_number"]),
                "timestamp": r["timestamp"],
                "distribution": r["distribution"],
                "accepted_updates": int(r["accepted_updates"]),
                "rejected_updates": int(r["rejected_updates"]),
                "global_accuracy": float(r["global_accuracy"]),
                "global_precision": float(r["global_precision"]),
                "global_recall": float(r["global_recall"]),
                "global_f1": float(r["global_f1"]),
                "global_roc_auc": float(r["global_roc_auc"]),
                "global_loss": float(r["global_loss"]),
                "confusion_matrix": json.loads(r["confusion_matrix_json"]),
                "hospital_contributions": json.loads(r["hospital_contributions_json"]),
            }
            for r in rows
        ]

    def insert_security_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        cur = self.conn.execute(
            """
            INSERT INTO security_events (
                timestamp, hospital_id, claimed_sender, training_round, scenario,
                hmac_valid, validation_passed, decision, layer_failed, reason,
                delta_norm, cosine_similarity, signature_preview
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["timestamp"],
                event["hospital_id"],
                event["claimed_sender"],
                int(event["training_round"]),
                event["scenario"],
                1 if event["hmac_valid"] else 0,
                1 if event["validation_passed"] else 0,
                event["decision"],
                event["layer_failed"],
                event["reason"],
                float(event["delta_norm"]),
                event.get("cosine_similarity"),
                event["signature_preview"],
            ),
        )
        self.conn.commit()
        return {**event, "id": int(cur.lastrowid)}

    def get_security_events(self, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM security_events ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
        return [
            {
                "id": int(r["id"]),
                "timestamp": r["timestamp"],
                "hospital_id": r["hospital_id"],
                "claimed_sender": r["claimed_sender"],
                "training_round": int(r["training_round"]),
                "scenario": r["scenario"],
                "hmac_valid": bool(r["hmac_valid"]),
                "validation_passed": bool(r["validation_passed"]),
                "decision": r["decision"],
                "layer_failed": r["layer_failed"],
                "reason": r["reason"],
                "delta_norm": float(r["delta_norm"]),
                "cosine_similarity": r["cosine_similarity"],
                "signature_preview": r["signature_preview"],
            }
            for r in rows
        ]

    def append_audit_record(self, rec: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO audit_ledger (
                record_id, timestamp, hospital_id, training_round, event_type,
                payload_summary, payload_hash, previous_hash, current_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(rec["record_id"]),
                rec["timestamp"],
                rec["hospital_id"],
                int(rec["training_round"]),
                rec["event_type"],
                rec["payload_summary"],
                rec["payload_hash"],
                rec["previous_hash"],
                rec["current_hash"],
            ),
        )
        self.conn.commit()

    def get_audit_ledger(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM audit_ledger ORDER BY record_id ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def insert_activity(self, entry: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO system_activity (timestamp, category, title, detail, severity)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                entry["timestamp"],
                entry["category"],
                entry["title"],
                entry["detail"],
                entry["severity"],
            ),
        )
        self.conn.commit()

    def get_recent_activity(self, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM system_activity ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
        return [dict(r) for r in rows]

    def reset_all(self) -> None:
        self.conn.executescript(
            """
            DELETE FROM training_rounds;
            DELETE FROM security_events;
            DELETE FROM audit_ledger;
            DELETE FROM system_activity;
            """
        )
        self.save_experiment_meta(
            {
                "initialized": False,
                "distribution": "balanced",
                "seed": 42,
                "local_epochs": 12,
                "learning_rate": 0.12,
                "current_round": 0,
                "status": "not_initialized",
            }
        )

    def close(self) -> None:
        self.conn.close()
