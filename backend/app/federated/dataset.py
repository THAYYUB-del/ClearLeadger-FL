"""
Dataset loading, Z-score preprocessing, and 3-hospital partitioning
using scikit-learn's built-in Wisconsin Breast Cancer Diagnostic dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HOSPITAL_METADATA = {
    "hospital_a": {
        "name": "Hospital A",
        "institution_type": "Metropolitan Oncology Center",
        "region": "North Clinical Network",
    },
    "hospital_b": {
        "name": "Hospital B",
        "institution_type": "Regional Medical Center",
        "region": "Central Clinical Network",
    },
    "hospital_c": {
        "name": "Hospital C",
        "institution_type": "Community Screening Hospital",
        "region": "West Clinical Network",
    },
}


@dataclass
class DatasetSplit:
    feature_names: List[str]
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_raw_x: np.ndarray
    test_y: np.ndarray
    scaler: StandardScaler


@dataclass
class HospitalPartition:
    hospital_id: str
    name: str
    institution_type: str
    region: str
    train_x: np.ndarray
    train_y: np.ndarray
    sample_count: int
    benign_count: int
    malignant_count: int


def prepare_dataset_split(seed: int = 42, test_size: float = 0.2) -> DatasetSplit:
    bunch = load_breast_cancer()
    x_raw = bunch.data.astype(np.float64)
    y = bunch.target.astype(np.int64)
    feature_names = [str(name) for name in bunch.feature_names]

    raw_train_x, raw_test_x, train_y, test_y = train_test_split(
        x_raw,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    train_x = scaler.fit_transform(raw_train_x)
    test_x = scaler.transform(raw_test_x)

    return DatasetSplit(
        feature_names=feature_names,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_raw_x=raw_test_x,
        test_y=test_y,
        scaler=scaler,
    )


def partition_hospital_data(
    train_x: np.ndarray,
    train_y: np.ndarray,
    distribution: str = "balanced",
    seed: int = 42,
) -> Dict[str, HospitalPartition]:
    rng = np.random.default_rng(seed + 303)
    malignant_idx = np.where(train_y == 0)[0]
    benign_idx = np.where(train_y == 1)[0]
    rng.shuffle(malignant_idx)
    rng.shuffle(benign_idx)

    if distribution == "balanced":
        m_splits = np.array_split(malignant_idx, 3)
        b_splits = np.array_split(benign_idx, 3)
        idx_map = {
            "hospital_a": np.concatenate([m_splits[0], b_splits[0]]),
            "hospital_b": np.concatenate([m_splits[1], b_splits[1]]),
            "hospital_c": np.concatenate([m_splits[2], b_splits[2]]),
        }
    else:
        m_a = round(len(malignant_idx) * 0.65)
        m_b = round(len(malignant_idx) * 0.25)
        b_a = round(len(benign_idx) * 0.20)
        b_b = round(len(benign_idx) * 0.40)
        idx_map = {
            "hospital_a": np.concatenate([malignant_idx[:m_a], benign_idx[:b_a]]),
            "hospital_b": np.concatenate(
                [malignant_idx[m_a : m_a + m_b], benign_idx[b_a : b_a + b_b]]
            ),
            "hospital_c": np.concatenate(
                [malignant_idx[m_a + m_b :], benign_idx[b_a + b_b :]]
            ),
        }

    partitions: Dict[str, HospitalPartition] = {}
    for hid, indices in idx_map.items():
        rng.shuffle(indices)
        node_x = train_x[indices].copy()
        node_y = train_y[indices].copy()
        benign_count = int(np.sum(node_y == 1))
        malignant_count = int(len(node_y) - benign_count)
        meta = HOSPITAL_METADATA[hid]
        partitions[hid] = HospitalPartition(
            hospital_id=hid,
            name=meta["name"],
            institution_type=meta["institution_type"],
            region=meta["region"],
            train_x=node_x,
            train_y=node_y,
            sample_count=int(len(node_y)),
            benign_count=benign_count,
            malignant_count=malignant_count,
        )
    return partitions
