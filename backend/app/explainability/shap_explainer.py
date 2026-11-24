"""
Real SHAP explainability using shap.LinearExplainer on the trained
federated logistic regression model and Wisconsin Breast Cancer test set.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import shap

from app.federated.model import ModelParameters, sigmoid


def _build_linear_explainer(
    params: ModelParameters,
    background_x: np.ndarray,
) -> shap.LinearExplainer:
    return shap.LinearExplainer(
        (params.weights.reshape(1, -1), np.array([params.bias], dtype=np.float64)),
        background_x,
        feature_perturbation="interventional",
    )


def compute_global_shap(
    params: ModelParameters,
    background_x: np.ndarray,
    test_x: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    explainer = _build_linear_explainer(params, background_x)
    shap_values = np.asarray(explainer.shap_values(test_x))
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    elif shap_values.ndim == 2 and shap_values.shape[0] != len(test_x):
        shap_values = shap_values.T

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    features = []
    for idx, name in enumerate(feature_names):
        w_j = float(params.weights[idx])
        features.append(
            {
                "feature_index": idx,
                "feature_name": name,
                "mean_abs_shap": round(float(mean_abs[idx]), 5),
                "weight": round(w_j, 5),
                "direction": "benign_leaning" if w_j >= 0 else "malignant_leaning",
            }
        )

    features.sort(key=lambda item: item["mean_abs_shap"], reverse=True)
    expected = explainer.expected_value
    base_val = float(expected[0] if isinstance(expected, (list, np.ndarray)) else expected)

    return {
        "base_value_logit": round(base_val, 5),
        "features": features,
    }


def explain_individual_prediction(
    params: ModelParameters,
    background_x: np.ndarray,
    sample_index: int,
    standardized_x: np.ndarray,
    raw_x: np.ndarray,
    actual_label: int,
    feature_names: List[str],
) -> Dict[str, Any]:
    explainer = _build_linear_explainer(params, background_x)
    row = standardized_x.reshape(1, -1)
    shap_vals = np.asarray(explainer.shap_values(row)).reshape(-1)
    expected = explainer.expected_value
    base_val = float(expected[0] if isinstance(expected, (list, np.ndarray)) else expected)

    logit = float(standardized_x @ params.weights + params.bias)
    prob_benign = float(sigmoid(np.array([logit]))[0])
    predicted_label = 1 if prob_benign >= 0.5 else 0

    contributions = []
    for idx, name in enumerate(feature_names):
        phi = float(shap_vals[idx])
        contributions.append(
            {
                "feature_index": idx,
                "feature_name": name,
                "raw_value": round(float(raw_x[idx]), 4),
                "standardized_value": round(float(standardized_x[idx]), 4),
                "shap_value": round(phi, 5),
                "direction": "positive" if phi >= 0 else "negative",
            }
        )

    sorted_abs = sorted(contributions, key=lambda c: abs(c["shap_value"]), reverse=True)
    top_positive = sorted(
        [c for c in contributions if c["shap_value"] > 0],
        key=lambda c: c["shap_value"],
        reverse=True,
    )[:8]
    top_negative = sorted(
        [c for c in contributions if c["shap_value"] < 0],
        key=lambda c: c["shap_value"],
    )[:8]

    return {
        "sample_index": int(sample_index),
        "actual_label": int(actual_label),
        "actual_class": "Benign" if actual_label == 1 else "Malignant",
        "predicted_label": predicted_label,
        "predicted_class": "Benign" if predicted_label == 1 else "Malignant",
        "probability_benign": round(prob_benign, 4),
        "probability_malignant": round(1.0 - prob_benign, 4),
        "base_value_logit": round(base_val, 5),
        "predicted_logit": round(logit, 5),
        "shap_sum_check": round(base_val + float(np.sum(shap_vals)), 5),
        "contributions": sorted_abs,
        "top_positive": top_positive,
        "top_negative": top_negative,
    }
