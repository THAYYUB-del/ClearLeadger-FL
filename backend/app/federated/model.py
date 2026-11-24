"""
NumPy Logistic Regression model with L2 regularization, sample-weighted FedAvg,
and scikit-learn evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ModelParameters:
    weights: np.ndarray
    bias: float

    def copy(self) -> "ModelParameters":
        return ModelParameters(weights=self.weights.copy(), bias=float(self.bias))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def initialize_zero_model(num_features: int = 30) -> ModelParameters:
    return ModelParameters(weights=np.zeros(num_features, dtype=np.float64), bias=0.0)


def train_local_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    initial_params: ModelParameters,
    learning_rate: float = 0.12,
    epochs: int = 12,
    l2_lambda: float = 0.015,
) -> Dict[str, Any]:
    n = len(x)
    w = initial_params.weights.copy()
    b = float(initial_params.bias)

    for _ in range(epochs):
        logits = x @ w + b
        preds = sigmoid(logits)
        errors = preds - y
        grad_w = (x.T @ errors) / n + l2_lambda * w
        grad_b = float(np.mean(errors))
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    delta_w = w - initial_params.weights
    delta_b = b - initial_params.bias
    delta_norm = float(np.linalg.norm(np.append(delta_w, delta_b)))

    trained = ModelParameters(weights=w, bias=b)
    train_metrics = evaluate_model(trained, x, y)

    return {
        "parameters": trained,
        "delta_weights": delta_w,
        "delta_bias": delta_b,
        "delta_norm": delta_norm,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "epochs_completed": epochs,
    }


def federated_average(
    updates: List[Tuple[str, int, ModelParameters]],
) -> ModelParameters:
    if not updates:
        raise ValueError("Cannot aggregate zero updates.")
    total_samples = sum(n for _, n, _ in updates)
    if total_samples <= 0:
        raise ValueError("Total sample count must be positive.")

    num_features = len(updates[0][2].weights)
    agg_w = np.zeros(num_features, dtype=np.float64)
    agg_b = 0.0

    for _, n_k, params in updates:
        factor = n_k / total_samples
        agg_w += factor * params.weights
        agg_b += factor * params.bias

    return ModelParameters(weights=agg_w, bias=float(agg_b))


def evaluate_model(
    params: ModelParameters,
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    probs = sigmoid(x @ params.weights + params.bias)
    preds = (probs >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

    try:
        auc = float(roc_auc_score(y, probs))
    except ValueError:
        auc = 0.5

    return {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y, preds, zero_division=0)), 4),
        "roc_auc": round(auc, 4),
        "loss": round(float(log_loss(y, np.clip(probs, 1e-12, 1 - 1e-12), labels=[0, 1])), 4),
        "confusion_matrix": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        },
        "sample_count": int(len(y)),
    }
