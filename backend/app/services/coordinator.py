"""Federated Learning Coordinator for the Python FastAPI backend."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from app.blockchain.ledger import (
    GENESIS_PREV_HASH,
    create_audit_record,
    run_controlled_tamper_test,
    verify_ledger_chain,
)
from app.explainability.shap_explainer import (
    compute_global_shap,
    explain_individual_prediction,
)
from app.federated.dataset import (
    HOSPITAL_METADATA,
    DatasetSplit,
    HospitalPartition,
    partition_hospital_data,
    prepare_dataset_split,
)
from app.federated.model import (
    ModelParameters,
    evaluate_model,
    federated_average,
    initialize_zero_model,
    train_local_logistic_regression,
)
from app.security.auth import inspect_and_validate_update, sign_model_update
from app.services.database import ClearLedgerDatabase

HOSPITAL_IDS = ("hospital_a", "hospital_b", "hospital_c")


def _safe_csv_value(val: Any) -> Any:
    if isinstance(val, str) and val and val[0] in ("=", "+", "-", "@", "\t", "\r"):
        return f"'{val}"
    return val


class FederatedCoordinator:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db = ClearLedgerDatabase(db_path)
        self.initialized = False
        self.distribution = "balanced"
        self.seed = 42
        self.local_epochs = 12
        self.learning_rate = 0.12
        self.current_round = 0

        self.split: Optional[DatasetSplit] = None
        self.partitions: Optional[Dict[str, HospitalPartition]] = None
        self.global_params: Optional[ModelParameters] = None
        self.global_metrics: Optional[Dict[str, Any]] = None
        self.centralized_metrics: Optional[Dict[str, Any]] = None

        self.hospital_models: Dict[str, Optional[ModelParameters]] = {h: None for h in HOSPITAL_IDS}
        self.hospital_local_metrics: Dict[str, Optional[Dict[str, Any]]] = {h: None for h in HOSPITAL_IDS}
        self.hospital_standalone_metrics: Dict[str, Optional[Dict[str, Any]]] = {h: None for h in HOSPITAL_IDS}
        self.hospital_last_round: Dict[str, int] = {h: 0 for h in HOSPITAL_IDS}
        self.hospital_status: Dict[str, str] = {h: "Uninitialized" for h in HOSPITAL_IDS}
        self.hospital_progress: Dict[str, int] = {h: 0 for h in HOSPITAL_IDS}
        self.hospital_delta_norm: Dict[str, Optional[float]] = {h: None for h in HOSPITAL_IDS}
        self.hospital_updated_at: Dict[str, Optional[str]] = {h: None for h in HOSPITAL_IDS}

    def _compute_baselines(self) -> None:
        assert self.split is not None and self.partitions is not None and self.global_params is not None
        epochs = max(60, self.local_epochs * 5)
        cent_res = train_local_logistic_regression(
            self.split.train_x,
            self.split.train_y,
            initialize_zero_model(len(self.split.feature_names)),
            learning_rate=self.learning_rate,
            epochs=epochs,
        )
        self.centralized_metrics = evaluate_model(
            cent_res["parameters"], self.split.test_x, self.split.test_y
        )
        for hid in HOSPITAL_IDS:
            part = self.partitions[hid]
            solo = train_local_logistic_regression(
                part.train_x,
                part.train_y,
                initialize_zero_model(len(self.split.feature_names)),
                learning_rate=self.learning_rate,
                epochs=epochs,
            )
            self.hospital_standalone_metrics[hid] = evaluate_model(
                solo["parameters"], self.split.test_x, self.split.test_y
            )
            self.hospital_models[hid] = self.global_params.copy()
            self.hospital_status[hid] = "Ready"
            self.hospital_progress[hid] = 0

    def _append_ledger(
        self,
        hospital_id: str,
        training_round: int,
        event_type: str,
        payload_summary: str,
        payload_data: Any,
    ) -> Dict[str, Any]:
        existing = self.db.get_audit_ledger()
        next_id = existing[-1]["record_id"] + 1 if existing else 1
        prev_hash = existing[-1]["current_hash"] if existing else GENESIS_PREV_HASH
        rec = create_audit_record(
            record_id=next_id,
            previous_hash=prev_hash,
            hospital_id=hospital_id,
            training_round=training_round,
            event_type=event_type,
            payload_summary=payload_summary,
            payload_data=payload_data,
        )
        self.db.append_audit_record(rec)
        return rec

    def initialize_experiment(
        self,
        distribution: str = "balanced",
        seed: int = 42,
        local_epochs: int = 12,
        learning_rate: float = 0.12,
    ) -> Dict[str, Any]:
        self.db.reset_all()
        self.distribution = "non_iid" if distribution == "non_iid" else "balanced"
        self.seed = seed
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.current_round = 0
        self.initialized = True

        self.split = prepare_dataset_split(seed=seed)
        self.partitions = partition_hospital_data(
            self.split.train_x, self.split.train_y, self.distribution, seed=seed
        )
        self.global_params = initialize_zero_model(len(self.split.feature_names))
        self.global_metrics = None

        now = datetime.now(timezone.utc).isoformat()
        for hid in HOSPITAL_IDS:
            self.hospital_local_metrics[hid] = None
            self.hospital_last_round[hid] = 0
            self.hospital_delta_norm[hid] = None
            self.hospital_updated_at[hid] = now

        self._compute_baselines()
        self.db.save_experiment_meta(
            {
                "initialized": True,
                "distribution": self.distribution,
                "seed": self.seed,
                "local_epochs": self.local_epochs,
                "learning_rate": self.learning_rate,
                "current_round": 0,
                "status": "initialized",
            }
        )

        self._append_ledger(
            hospital_id="aggregator",
            training_round=0,
            event_type="EXPERIMENT_INITIALIZED",
            payload_summary=(
                f"Initialized global logistic model (30 features, "
                f"{'Balanced IID' if self.distribution == 'balanced' else 'Non-IID'} split, seed={seed})"
            ),
            payload_data={
                "distribution": self.distribution,
                "seed": seed,
                "train_samples": len(self.split.train_x),
                "test_samples": len(self.split.test_x),
            },
        )
        self.db.insert_activity(
            {
                "timestamp": now,
                "category": "federated",
                "title": "Experiment Initialized",
                "detail": f"Partitioned {len(self.split.train_x)} training records across 3 hospitals.",
                "severity": "info",
            }
        )
        return self.get_federated_status()

    def reset_experiment(self) -> Dict[str, Any]:
        self.db.reset_all()
        self.initialized = False
        self.current_round = 0
        self.split = None
        self.partitions = None
        self.global_params = None
        self.global_metrics = None
        self.centralized_metrics = None
        for hid in HOSPITAL_IDS:
            self.hospital_models[hid] = None
            self.hospital_local_metrics[hid] = None
            self.hospital_standalone_metrics[hid] = None
            self.hospital_last_round[hid] = 0
            self.hospital_status[hid] = "Uninitialized"
            self.hospital_progress[hid] = 0
            self.hospital_delta_norm[hid] = None
            self.hospital_updated_at[hid] = None
        return {"status": "reset", "message": "Experiment reset cleanly."}

    def train_single_hospital(self, hospital_id: str, local_epochs: Optional[int] = None) -> Dict[str, Any]:
        if not self.initialized or not self.split or not self.partitions or not self.global_params:
            raise ValueError("Experiment not initialized.")
        part = self.partitions[hospital_id]
        epochs = local_epochs or self.local_epochs
        start = self.hospital_models[hospital_id].copy() if self.hospital_models[hospital_id] else self.global_params.copy()
        res = train_local_logistic_regression(
            part.train_x, part.train_y, start, learning_rate=self.learning_rate, epochs=epochs
        )
        metrics = evaluate_model(res["parameters"], self.split.test_x, self.split.test_y)
        self.hospital_models[hospital_id] = res["parameters"]
        self.hospital_local_metrics[hospital_id] = metrics
        self.hospital_status[hospital_id] = "Trained Locally"
        self.hospital_progress[hospital_id] = 100
        self.hospital_delta_norm[hospital_id] = round(res["delta_norm"], 4)
        self.hospital_updated_at[hospital_id] = datetime.now(timezone.utc).isoformat()
        return {
            "hospital": self.get_hospital_detail(hospital_id),
            "train_loss": res["train_loss"],
            "train_accuracy": res["train_accuracy"],
            "delta_norm": round(res["delta_norm"], 4),
            "epochs_completed": epochs,
        }

    def run_federated_training(self, num_rounds: int = 1) -> Dict[str, Any]:
        if not self.initialized or not self.split or not self.partitions or not self.global_params:
            raise ValueError("Experiment not initialized.")

        executed: List[Dict[str, Any]] = []
        for _ in range(num_rounds):
            next_round = self.current_round + 1
            global_before = self.global_params.copy()

            local_items = []
            for hid in HOSPITAL_IDS:
                part = self.partitions[hid]
                res = train_local_logistic_regression(
                    part.train_x,
                    part.train_y,
                    global_before,
                    learning_rate=self.learning_rate,
                    epochs=self.local_epochs,
                )
                metrics = evaluate_model(res["parameters"], self.split.test_x, self.split.test_y)
                local_items.append((hid, part, res, metrics))

            mean_delta = np.mean(
                [np.append(item[2]["delta_weights"], item[2]["delta_bias"]) for item in local_items],
                axis=0,
            ).tolist()

            accepted_updates = []
            contributions = []
            accepted_count = 0
            rejected_count = 0
            total_samples = sum(item[1].sample_count for item in local_items)
            now = datetime.now(timezone.utc).isoformat()

            for hid, part, res, metrics in local_items:
                signed = sign_model_update(
                    hospital_id=hid,
                    training_round=next_round,
                    num_samples=part.sample_count,
                    weights=res["parameters"].weights.tolist(),
                    bias=res["parameters"].bias,
                )
                ver = inspect_and_validate_update(
                    signed,
                    global_before.weights.tolist(),
                    global_before.bias,
                    reference_delta=mean_delta,
                )
                self.db.insert_security_event(
                    {
                        "timestamp": now,
                        "hospital_id": hid,
                        "claimed_sender": signed["claimed_sender"],
                        "training_round": next_round,
                        "scenario": "valid_update",
                        "hmac_valid": ver["hmac_valid"],
                        "validation_passed": ver["validation_passed"],
                        "decision": ver["decision"],
                        "layer_failed": ver["layer_failed"],
                        "reason": ver["reason"],
                        "delta_norm": ver["delta_norm"],
                        "cosine_similarity": ver["cosine_similarity"],
                        "signature_preview": ver["signature_preview"],
                    }
                )
                self.hospital_models[hid] = res["parameters"]
                self.hospital_local_metrics[hid] = metrics
                self.hospital_last_round[hid] = next_round
                self.hospital_status[hid] = "Synced with Global" if ver["accepted"] else "Trained Locally"
                self.hospital_progress[hid] = 100
                self.hospital_delta_norm[hid] = ver["delta_norm"]
                self.hospital_updated_at[hid] = now

                if ver["accepted"]:
                    accepted_count += 1
                    accepted_updates.append((hid, part.sample_count, res["parameters"]))
                    self._append_ledger(
                        hospital_id=hid,
                        training_round=next_round,
                        event_type="AUTHENTICATED_UPDATE_ACCEPTED",
                        payload_summary=(
                            f"{part.name} update verified (HMAC-SHA256 valid, "
                            f"n={part.sample_count}, ||Δw||={ver['delta_norm']:.3f})"
                        ),
                        payload_data={
                            "hospital_id": hid,
                            "round": next_round,
                            "num_samples": part.sample_count,
                            "delta_norm": ver["delta_norm"],
                        },
                    )
                else:
                    rejected_count += 1

                contributions.append(
                    {
                        "hospital_id": hid,
                        "hospital_name": part.name,
                        "num_samples": part.sample_count,
                        "weight_share": round(part.sample_count / total_samples, 4),
                        "local_accuracy": metrics["accuracy"],
                        "local_loss": metrics["loss"],
                        "delta_norm": ver["delta_norm"],
                        "accepted": ver["accepted"],
                    }
                )

            self.global_params = federated_average(accepted_updates)
            self.global_metrics = evaluate_model(
                self.global_params, self.split.test_x, self.split.test_y
            )
            self.current_round = next_round
            for hid in HOSPITAL_IDS:
                self.hospital_models[hid] = self.global_params.copy()

            round_rec = {
                "round_number": next_round,
                "timestamp": now,
                "distribution": self.distribution,
                "accepted_updates": accepted_count,
                "rejected_updates": rejected_count,
                "global_accuracy": self.global_metrics["accuracy"],
                "global_precision": self.global_metrics["precision"],
                "global_recall": self.global_metrics["recall"],
                "global_f1": self.global_metrics["f1_score"],
                "global_roc_auc": self.global_metrics["roc_auc"],
                "global_loss": self.global_metrics["loss"],
                "confusion_matrix": self.global_metrics["confusion_matrix"],
                "hospital_contributions": contributions,
            }
            self.db.insert_round(round_rec)
            self.db.save_experiment_meta(
                {
                    "initialized": True,
                    "distribution": self.distribution,
                    "seed": self.seed,
                    "local_epochs": self.local_epochs,
                    "learning_rate": self.learning_rate,
                    "current_round": self.current_round,
                    "status": "trained",
                }
            )
            self._append_ledger(
                hospital_id="aggregator",
                training_round=next_round,
                event_type="FEDAVG_AGGREGATION",
                payload_summary=(
                    f"Round {next_round} FedAvg aggregated {accepted_count}/3 updates -> "
                    f"Global Accuracy {self.global_metrics['accuracy'] * 100:.2f}%"
                ),
                payload_data=self.global_metrics,
            )
            executed.append(round_rec)

        return {
            "rounds_executed": len(executed),
            "current_round": self.current_round,
            "latest_round": executed[-1],
            "global_metrics": self.global_metrics,
        }

    def simulate_security_experiment(self, scenario: str, target_hospital: str = "hospital_b") -> Dict[str, Any]:
        if not self.initialized or not self.split or not self.partitions or not self.global_params:
            raise ValueError("Initialize an experiment first.")
        part = self.partitions[target_hospital]
        base = self.global_params.copy()
        honest = train_local_logistic_regression(
            part.train_x, part.train_y, base, learning_rate=self.learning_rate, epochs=self.local_epochs
        )

        peer_deltas = []
        for pid in HOSPITAL_IDS:
            if pid == target_hospital:
                continue
            ppart = self.partitions[pid]
            pres = train_local_logistic_regression(
                ppart.train_x, ppart.train_y, base, learning_rate=self.learning_rate, epochs=self.local_epochs
            )
            peer_deltas.append(np.append(pres["delta_weights"], pres["delta_bias"]))
        ref_delta = np.mean(peer_deltas, axis=0).tolist()

        round_label = self.current_round + 1
        signed = sign_model_update(
            hospital_id=target_hospital,
            training_round=round_label,
            num_samples=part.sample_count,
            weights=honest["parameters"].weights.tolist(),
            bias=honest["parameters"].bias,
        )
        explanation = ""

        if scenario == "valid_update":
            explanation = f"{part.name} signed an honest local update with its genuine HMAC-SHA256 secret key."
        elif scenario == "invalid_hmac":
            signed["hmac_signature"] = "deadbeef" + signed["hmac_signature"][8:]
            explanation = f"Corrupted HMAC-SHA256 digest claiming to be {part.name}; Layer 1 authentication rejected it."
        elif scenario == "forged_sender":
            signed = sign_model_update(
                hospital_id=target_hospital,
                training_round=round_label,
                num_samples=part.sample_count,
                weights=honest["parameters"].weights.tolist(),
                bias=honest["parameters"].bias,
                override_secret="unauthorized-attacker-key-0000",
                claimed_sender=target_hospital,
            )
            explanation = f"Unauthorized actor impersonating {part.name} with wrong key; Layer 1 rejected it."
        elif scenario == "exaggerated_update":
            scaled_w = base.weights + honest["delta_weights"] * 18.0
            scaled_b = base.bias + honest["delta_bias"] * 18.0
            signed = sign_model_update(
                hospital_id=target_hospital,
                training_round=round_label,
                num_samples=part.sample_count,
                weights=scaled_w.tolist(),
                bias=scaled_b,
            )
            explanation = (
                f"{part.name} signed an 18x amplified update using its valid key. "
                "Layer 1 HMAC passed, but Layer 2 L2-norm validation quarantined the update."
            )
        elif scenario == "label_flipping":
            flipped_y = 1 - part.train_y
            poisoned = train_local_logistic_regression(
                part.train_x,
                flipped_y,
                base,
                learning_rate=self.learning_rate * 1.4,
                epochs=self.local_epochs + 6,
            )
            signed = sign_model_update(
                hospital_id=target_hospital,
                training_round=round_label,
                num_samples=part.sample_count,
                weights=poisoned["parameters"].weights.tolist(),
                bias=poisoned["parameters"].bias,
            )
            explanation = (
                f"{part.name} trained on inverted labels and signed with its valid key. "
                "Layer 1 HMAC passed, but Layer 2 cosine-similarity screening quarantined the poisoned update."
            )

        ver = inspect_and_validate_update(
            signed, base.weights.tolist(), base.bias, reference_delta=ref_delta
        )
        stored = self.db.insert_security_event(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hospital_id": target_hospital,
                "claimed_sender": signed["claimed_sender"],
                "training_round": round_label,
                "scenario": scenario,
                "hmac_valid": ver["hmac_valid"],
                "validation_passed": ver["validation_passed"],
                "decision": ver["decision"],
                "layer_failed": ver["layer_failed"],
                "reason": ver["reason"],
                "delta_norm": ver["delta_norm"],
                "cosine_similarity": ver["cosine_similarity"],
                "signature_preview": ver["signature_preview"],
            }
        )
        return {
            "event": stored,
            "explanation": explanation,
            "honest_reference_norm": round(honest["delta_norm"], 4),
        }

    def get_hospital_detail(self, hospital_id: str) -> Dict[str, Any]:
        meta = HOSPITAL_METADATA[hospital_id]
        part = self.partitions[hospital_id] if self.partitions else None
        return {
            "hospital_id": hospital_id,
            "name": meta["name"],
            "institution_type": meta["institution_type"],
            "region": meta["region"],
            "connection_status": "Connected" if self.initialized else "Standby",
            "model_status": self.hospital_status[hospital_id],
            "training_progress": self.hospital_progress[hospital_id],
            "last_training_round": self.hospital_last_round[hospital_id],
            "num_samples": part.sample_count if part else 0,
            "benign_count": part.benign_count if part else 0,
            "malignant_count": part.malignant_count if part else 0,
            "local_metrics": self.hospital_local_metrics[hospital_id],
            "standalone_metrics": self.hospital_standalone_metrics[hospital_id],
            "last_delta_norm": self.hospital_delta_norm[hospital_id],
            "last_updated_at": self.hospital_updated_at[hospital_id],
            "privacy_guarantee": (
                "Raw patient feature vectors remain isolated inside this hospital node. "
                "Only signed parameter weights are transmitted to the aggregator."
            ),
        }

    def get_hospitals(self) -> List[Dict[str, Any]]:
        return [self.get_hospital_detail(h) for h in HOSPITAL_IDS]

    def get_federated_status(self) -> Dict[str, Any]:
        return {
            "initialized": self.initialized,
            "status": (
                "Not Initialized"
                if not self.initialized
                else ("Initialized — Ready to Train" if self.current_round == 0 else "Trained")
            ),
            "distribution": self.distribution,
            "seed": self.seed,
            "local_epochs": self.local_epochs,
            "learning_rate": self.learning_rate,
            "current_round": self.current_round,
            "total_train_samples": len(self.split.train_x) if self.split else 0,
            "total_test_samples": len(self.split.test_x) if self.split else 0,
            "global_metrics": self.global_metrics,
            "hospitals": self.get_hospitals(),
            "rounds_count": len(self.db.get_rounds()),
        }

    def get_dashboard_summary(self) -> Dict[str, Any]:
        rounds = self.db.get_rounds()
        events = self.db.get_security_events(200)
        ledger = self.db.get_audit_ledger()
        ver = verify_ledger_chain(ledger)
        return {
            "experiment_initialized": self.initialized,
            "federated_status": (
                "Not Initialized"
                if not self.initialized
                else (
                    "Ready for Round 1"
                    if self.current_round == 0
                    else f"Active ({self.current_round} rounds completed)"
                )
            ),
            "connected_hospitals": 3 if self.initialized else 0,
            "total_hospitals": 3,
            "current_round": self.current_round,
            "global_accuracy": self.global_metrics["accuracy"] if self.global_metrics else None,
            "global_roc_auc": self.global_metrics["roc_auc"] if self.global_metrics else None,
            "global_loss": self.global_metrics["loss"] if self.global_metrics else None,
            "total_authenticated_updates": sum(1 for e in events if e["decision"] == "ACCEPTED"),
            "security_alerts": sum(1 for e in events if e["decision"] == "REJECTED"),
            "ledger_records_count": len(ledger),
            "ledger_integrity": {
                "valid": ver["valid"],
                "status": (
                    "Empty (Awaiting Initialization)"
                    if not ledger
                    else ("Integrity Verified" if ver["valid"] else "Tampering Detected")
                ),
                "verified_at": ver["verified_at"],
            },
            "distribution": self.distribution,
            "performance_series": [
                {
                    "round": r["round_number"],
                    "accuracy": round(r["global_accuracy"] * 100, 2),
                    "f1_score": round(r["global_f1"] * 100, 2),
                    "roc_auc": round(r["global_roc_auc"] * 100, 2),
                    "loss": r["global_loss"],
                }
                for r in rounds
            ],
            "recent_activity": self.db.get_recent_activity(12),
        }

    def get_ledger(self) -> Dict[str, Any]:
        records = self.db.get_audit_ledger()
        return {
            "architecture_note": (
                "Local SHA-256 hash-linked audit ledger persisted in SQLite. "
                "Provides tamper-evident cryptographic chaining without decentralized consensus."
            ),
            "records": records,
            "verification": verify_ledger_chain(records),
        }

    def verify_ledger(self) -> Dict[str, Any]:
        return verify_ledger_chain(self.db.get_audit_ledger())

    def run_tamper_test(self, target_record_id: Optional[int] = None) -> Dict[str, Any]:
        return run_controlled_tamper_test(self.db.get_audit_ledger(), target_record_id)

    def get_global_explainability(self) -> Dict[str, Any]:
        if not self.initialized or not self.split or not self.global_params or self.current_round == 0:
            return {
                "available": False,
                "method": "shap.LinearExplainer (interventional)",
                "current_round": self.current_round,
                "base_value_logit": None,
                "test_sample_count": 0,
                "features": [],
                "sample_preview": [],
            }
        res = compute_global_shap(
            self.global_params, self.split.train_x, self.split.test_x, self.split.feature_names
        )
        preview = []
        for idx in range(min(20, len(self.split.test_x))):
            exp = explain_individual_prediction(
                self.global_params,
                self.split.train_x,
                idx,
                self.split.test_x[idx],
                self.split.test_raw_x[idx],
                int(self.split.test_y[idx]),
                self.split.feature_names,
            )
            preview.append(
                {
                    "sample_index": idx,
                    "actual_class": exp["actual_class"],
                    "predicted_class": exp["predicted_class"],
                    "probability_benign": exp["probability_benign"],
                    "mean_radius": round(float(self.split.test_raw_x[idx][0]), 2),
                    "worst_perimeter": round(float(self.split.test_raw_x[idx][22]), 2),
                    "worst_concave_points": round(float(self.split.test_raw_x[idx][27]), 4),
                }
            )
        return {
            "available": True,
            "method": "shap.LinearExplainer (interventional)",
            "current_round": self.current_round,
            "base_value_logit": res["base_value_logit"],
            "test_sample_count": len(self.split.test_x),
            "features": res["features"],
            "sample_preview": preview,
        }

    def explain_prediction(self, sample_index: int) -> Dict[str, Any]:
        if not self.initialized or not self.split or not self.global_params or self.current_round == 0:
            raise ValueError("Train at least one federated round before generating SHAP explanations.")
        if sample_index < 0 or sample_index >= len(self.split.test_x):
            raise ValueError(f"sample_index must be between 0 and {len(self.split.test_x) - 1}.")
        return explain_individual_prediction(
            self.global_params,
            self.split.train_x,
            sample_index,
            self.split.test_x[sample_index],
            self.split.test_raw_x[sample_index],
            int(self.split.test_y[sample_index]),
            self.split.feature_names,
        )

    def get_analytics(self) -> Dict[str, Any]:
        rounds = self.db.get_rounds()
        evaluated = self.initialized and self.current_round > 0 and self.global_metrics is not None
        comparison = []
        if evaluated:
            comparison.append(
                {
                    "model_name": "Centralized Baseline (Pooled)",
                    "scope": "All training data pooled (privacy-breaking upper bound)",
                    "metrics": self.centralized_metrics,
                }
            )
            comparison.append(
                {
                    "model_name": "Federated Global Model (FedAvg)",
                    "scope": f"Aggregated across 3 hospitals over {self.current_round} rounds",
                    "metrics": self.global_metrics,
                }
            )
            for hid in HOSPITAL_IDS:
                comparison.append(
                    {
                        "model_name": f"{HOSPITAL_METADATA[hid]['name']} (Standalone Local)",
                        "scope": f"Trained only on {self.partitions[hid].sample_count} local records",
                        "metrics": self.hospital_standalone_metrics[hid],
                    }
                )
        return {
            "evaluated": evaluated,
            "distribution": self.distribution,
            "current_round": self.current_round,
            "dataset_info": {
                "name": "Wisconsin Breast Cancer Diagnostic (WDBC)",
                "total_samples": 569,
                "feature_count": 30,
                "train_samples": len(self.split.train_x) if self.split else 0,
                "held_out_test_samples": len(self.split.test_x) if self.split else 0,
                "educational_notice": "Educational ML experiment only — not a clinical diagnostic system.",
            },
            "global_metrics": self.global_metrics,
            "centralized_metrics": self.centralized_metrics if evaluated else None,
            "comparison": comparison,
            "rounds": rounds,
        }

    def export_analytics_csv(self) -> str:
        analytics = self.get_analytics()
        if not analytics["evaluated"]:
            raise ValueError("Run at least one federated round before exporting CSV.")
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "section",
                "model_or_round",
                "distribution",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "loss",
                "tp",
                "tn",
                "fp",
                "fn",
            ]
        )
        for item in analytics["comparison"]:
            m = item["metrics"]
            cm = m["confusion_matrix"]
            writer.writerow(
                [
                    "model_comparison",
                    _safe_csv_value(item["model_name"]),
                    analytics["distribution"],
                    m["accuracy"],
                    m["precision"],
                    m["recall"],
                    m["f1_score"],
                    m["roc_auc"],
                    m["loss"],
                    cm["tp"],
                    cm["tn"],
                    cm["fp"],
                    cm["fn"],
                ]
            )
        for r in analytics["rounds"]:
            cm = r["confusion_matrix"]
            writer.writerow(
                [
                    "federated_round",
                    f"Round {r['round_number']}",
                    r["distribution"],
                    r["global_accuracy"],
                    r["global_precision"],
                    r["global_recall"],
                    r["global_f1"],
                    r["global_roc_auc"],
                    r["global_loss"],
                    cm["tp"],
                    cm["tn"],
                    cm["fp"],
                    cm["fn"],
                ]
            )
        return buf.getvalue()
