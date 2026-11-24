"""
Cryptographic HMAC-SHA256 update authentication and independent
malicious-update validation (L2 norm and directional cosine screening).

Important distinction:
- A valid HMAC proves possession of the hospital secret key and payload
  integrity in transit; it does NOT prove the update is honest.
- Layer 2 validation screens parameter updates independently of HMAC.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

VALID_HOSPITALS = ("hospital_a", "hospital_b", "hospital_c")

_ENV_KEY_MAP = {
    "hospital_a": "HOSPITAL_A_SECRET",
    "hospital_b": "HOSPITAL_B_SECRET",
    "hospital_c": "HOSPITAL_C_SECRET",
}

_EPHEMERAL_KEYS = {
    hid: secrets.token_hex(32) for hid in VALID_HOSPITALS
}


def get_hospital_secret(hospital_id: str) -> str:
    """Loads hospital HMAC secret from environment or ephemeral process-local fallback."""
    env_var = _ENV_KEY_MAP.get(hospital_id)
    if env_var:
        val = os.getenv(env_var, "").strip()
        if val and not val.startswith("replace-with-") and len(val) >= 16:
            return val
    if hospital_id not in _EPHEMERAL_KEYS:
        raise ValueError(f"Unknown hospital_id: {hospital_id}")
    return _EPHEMERAL_KEYS[hospital_id]


def canonicalize_update_payload(
    hospital_id: str,
    training_round: int,
    num_samples: int,
    weights: Sequence[float],
    bias: float,
    timestamp: str,
    nonce: str,
) -> str:
    payload = {
        "bias": round(float(bias), 8),
        "hospital_id": hospital_id,
        "nonce": nonce,
        "num_samples": int(num_samples),
        "timestamp": timestamp,
        "training_round": int(training_round),
        "weights": [round(float(w), 8) for w in weights],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sign_model_update(
    hospital_id: str,
    training_round: int,
    num_samples: int,
    weights: Sequence[float],
    bias: float,
    override_secret: Optional[str] = None,
    claimed_sender: Optional[str] = None,
) -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    nonce = secrets.token_hex(12)
    sender = claimed_sender or hospital_id
    secret = override_secret if override_secret is not None else get_hospital_secret(hospital_id)

    canonical = canonicalize_update_payload(
        hospital_id=sender,
        training_round=training_round,
        num_samples=num_samples,
        weights=weights,
        bias=bias,
        timestamp=timestamp,
        nonce=nonce,
    )
    signature = hmac.new(
        secret.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return {
        "hospital_id": hospital_id,
        "claimed_sender": sender,
        "training_round": training_round,
        "num_samples": num_samples,
        "weights": [float(w) for w in weights],
        "bias": float(bias),
        "timestamp": timestamp,
        "nonce": nonce,
        "hmac_signature": signature,
    }


def verify_hmac_signature(update: Dict[str, Any]) -> bool:
    claimed = str(update.get("claimed_sender", ""))
    if claimed not in VALID_HOSPITALS:
        return False

    secret = get_hospital_secret(claimed)
    canonical = canonicalize_update_payload(
        hospital_id=claimed,
        training_round=int(update["training_round"]),
        num_samples=int(update["num_samples"]),
        weights=update["weights"],
        bias=float(update["bias"]),
        timestamp=str(update["timestamp"]),
        nonce=str(update["nonce"]),
    )
    expected = hmac.new(
        secret.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    provided = str(update.get("hmac_signature", ""))
    return hmac.compare_digest(expected, provided)


def _l2_norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = _l2_norm(a)
    nb = _l2_norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def inspect_and_validate_update(
    update: Dict[str, Any],
    global_weights: Sequence[float],
    global_bias: float,
    reference_delta: Optional[Sequence[float]] = None,
    max_delta_norm: float = 4.5,
    min_cosine_similarity: float = -0.15,
) -> Dict[str, Any]:
    delta_vec: List[float] = [
        float(w) - float(gw) for w, gw in zip(update["weights"], global_weights)
    ] + [float(update["bias"]) - float(global_bias)]

    delta_norm = round(_l2_norm(delta_vec), 4)
    ref_norm = _l2_norm(reference_delta) if reference_delta is not None else 0.0
    effective_max_norm = (
        min(max_delta_norm, max(0.85, ref_norm * 3.5))
        if reference_delta is not None and ref_norm > 1e-8
        else max_delta_norm
    )
    cos_sim: Optional[float] = None
    if reference_delta is not None and ref_norm > 1e-8 and delta_norm > 1e-8:
        cos_sim = round(_cosine_similarity(delta_vec, reference_delta), 4)

    sig_preview = f"{str(update.get('hmac_signature', ''))[:16]}..."

    # Layer 1: Cryptographic Authentication
    if not verify_hmac_signature(update):
        return {
            "hmac_valid": False,
            "validation_passed": False,
            "accepted": False,
            "decision": "REJECTED",
            "layer_failed": "AUTHENTICATION",
            "reason": (
                "HMAC-SHA256 verification failed. Signature does not match the "
                "claimed hospital secret key or payload was altered in transit."
            ),
            "delta_norm": delta_norm,
            "cosine_similarity": cos_sim,
            "signature_preview": sig_preview,
        }

    # Layer 2: Malicious Update Screening (independent of valid HMAC)
    if any(not math.isfinite(float(v)) for v in list(update["weights"]) + [update["bias"]]):
        return {
            "hmac_valid": True,
            "validation_passed": False,
            "accepted": False,
            "decision": "REJECTED",
            "layer_failed": "MALICIOUS_VALIDATION",
            "reason": "Valid HMAC-SHA256 sender, but non-finite parameter values detected.",
            "delta_norm": delta_norm,
            "cosine_similarity": cos_sim,
            "signature_preview": sig_preview,
        }

    if delta_norm > effective_max_norm:
        return {
            "hmac_valid": True,
            "validation_passed": False,
            "accepted": False,
            "decision": "REJECTED",
            "layer_failed": "MALICIOUS_VALIDATION",
            "reason": (
                f"Valid HMAC-SHA256 authenticated sender, but L2 parameter delta norm "
                f"({delta_norm:.3f}) exceeds adaptive peer bound ({effective_max_norm:.2f}) — exaggerated update quarantined."
            ),
            "delta_norm": delta_norm,
            "cosine_similarity": cos_sim,
            "signature_preview": sig_preview,
        }

    if cos_sim is not None and cos_sim < min_cosine_similarity:
        return {
            "hmac_valid": True,
            "validation_passed": False,
            "accepted": False,
            "decision": "REJECTED",
            "layer_failed": "MALICIOUS_VALIDATION",
            "reason": (
                f"Valid HMAC-SHA256 authenticated sender, but update gradient opposes peer consensus "
                f"(cosine similarity {cos_sim:.3f} < {min_cosine_similarity}) — label-flipping quarantined."
            ),
            "delta_norm": delta_norm,
            "cosine_similarity": cos_sim,
            "signature_preview": sig_preview,
        }

    return {
        "hmac_valid": True,
        "validation_passed": True,
        "accepted": True,
        "decision": "ACCEPTED",
        "layer_failed": "NONE",
        "reason": "HMAC-SHA256 signature verified and update passed norm and directional integrity checks.",
        "delta_norm": delta_norm,
        "cosine_similarity": cos_sim,
        "signature_preview": sig_preview,
    }
