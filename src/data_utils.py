from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


CLASSES = ("pull_up", "push_up", "squat")

MEDIAPIPE_LANDMARKS = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky_1",
    "right_pinky_1",
    "left_index_1",
    "right_index_1",
    "left_thumb_2",
    "right_thumb_2",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


@dataclass(frozen=True)
class SplitIds:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class SequenceData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: StandardScaler


@dataclass
class TabularData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    feature_scaler: StandardScaler


def load_labels(data_dir: Path, classes: Iterable[str] = CLASSES) -> pd.DataFrame:
    labels = pd.read_csv(data_dir / "labels.csv")
    labels["class"] = labels["class"].astype(str)
    labels = labels[labels["class"].isin(classes)].copy()
    labels["vid_id"] = labels["vid_id"].astype(int)
    return labels.sort_values("vid_id").reset_index(drop=True)


def make_label_encoder(labels: pd.DataFrame) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(sorted(labels["class"].unique()))
    return encoder


def make_splits(labels: pd.DataFrame, seed: int = 42, test_size: float = 0.2, val_size: float = 0.2) -> SplitIds:
    vid_ids = labels["vid_id"].to_numpy()
    y = labels["class"].to_numpy()
    train_val_ids, test_ids, y_train_val, _ = train_test_split(
        vid_ids, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_ratio = val_size / (1.0 - test_size)
    train_ids, val_ids, _, _ = train_test_split(
        train_val_ids,
        y_train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_train_val,
    )
    return SplitIds(train=np.sort(train_ids), val=np.sort(val_ids), test=np.sort(test_ids))


def labels_for_ids(labels: pd.DataFrame, ids: np.ndarray, encoder: LabelEncoder) -> np.ndarray:
    y = labels.set_index("vid_id").loc[ids, "class"].to_numpy()
    return encoder.transform(y)


def aggregate_frame_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in {"vid_id", "frame_order"}]
    grouped = df.groupby("vid_id", sort=True)[feature_cols]
    aggs = grouped.agg(["mean", "std", "min", "max", "median"])
    aggs.columns = [f"{prefix}_{col}_{stat}" for col, stat in aggs.columns]

    ranges = grouped.max() - grouped.min()
    ranges.columns = [f"{prefix}_{col}_range" for col in ranges.columns]

    first_last = grouped.last() - grouped.first()
    first_last.columns = [f"{prefix}_{col}_last_minus_first" for col in first_last.columns]

    return pd.concat([aggs, ranges, first_last], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_tabular_dataset(data_dir: Path, labels: pd.DataFrame, splits: SplitIds, encoder: LabelEncoder) -> TabularData:
    feature_files = [
        ("angles", "angles.csv"),
        ("dist3d", "calculated_3d_distances.csv"),
        ("xyzdist", "xyz_distances.csv"),
    ]
    frames: list[pd.DataFrame] = []
    valid_ids = set(labels["vid_id"])
    for prefix, filename in feature_files:
        df = pd.read_csv(data_dir / filename)
        df = df[df["vid_id"].isin(valid_ids)].copy()
        frames.append(aggregate_frame_features(df, prefix))

    features = pd.concat(frames, axis=1).sort_index()
    features = features.loc[labels["vid_id"]]
    feature_names = list(features.columns)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(features.loc[splits.train].to_numpy(dtype=np.float32))
    x_val = scaler.transform(features.loc[splits.val].to_numpy(dtype=np.float32))
    x_test = scaler.transform(features.loc[splits.test].to_numpy(dtype=np.float32))

    return TabularData(
        x_train=x_train,
        y_train=labels_for_ids(labels, splits.train, encoder),
        x_val=x_val,
        y_val=labels_for_ids(labels, splits.val, encoder),
        x_test=x_test,
        y_test=labels_for_ids(labels, splits.test, encoder),
        feature_names=feature_names,
        feature_scaler=scaler,
    )


def landmark_columns() -> list[str]:
    return [f"{axis}_{joint}" for joint in MEDIAPIPE_LANDMARKS for axis in ("x", "y", "z")]


def normalize_skeleton_frame(frame: np.ndarray) -> np.ndarray:
    points = frame.reshape(len(MEDIAPIPE_LANDMARKS), 3).astype(np.float32)
    left_hip = points[MEDIAPIPE_LANDMARKS.index("left_hip")]
    right_hip = points[MEDIAPIPE_LANDMARKS.index("right_hip")]
    left_shoulder = points[MEDIAPIPE_LANDMARKS.index("left_shoulder")]
    right_shoulder = points[MEDIAPIPE_LANDMARKS.index("right_shoulder")]
    center = (left_hip + right_hip) / 2.0
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    hip_width = np.linalg.norm(left_hip - right_hip)
    scale = max(float(shoulder_width + hip_width) / 2.0, 1e-6)
    return ((points - center) / scale).reshape(-1)


def resample_sequence(values: np.ndarray, seq_len: int) -> np.ndarray:
    if len(values) == seq_len:
        return values.astype(np.float32)
    if len(values) == 1:
        return np.repeat(values, seq_len, axis=0).astype(np.float32)

    old_x = np.linspace(0.0, 1.0, num=len(values))
    new_x = np.linspace(0.0, 1.0, num=seq_len)
    out = np.empty((seq_len, values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(new_x, old_x, values[:, col])
    return out


def build_sequence_dataset(
    data_dir: Path,
    labels: pd.DataFrame,
    splits: SplitIds,
    encoder: LabelEncoder,
    seq_len: int = 120,
) -> SequenceData:
    wanted_ids = set(labels["vid_id"])
    cols = ["vid_id", "frame_order", *landmark_columns()]
    df = pd.read_csv(data_dir / "landmarks.csv", usecols=cols)
    df = df[df["vid_id"].isin(wanted_ids)].sort_values(["vid_id", "frame_order"])

    sequences: dict[int, np.ndarray] = {}
    for vid_id, group in df.groupby("vid_id", sort=True):
        raw = group[landmark_columns()].to_numpy(dtype=np.float32)
        normalized = np.stack([normalize_skeleton_frame(row) for row in raw], axis=0)
        sequences[int(vid_id)] = resample_sequence(normalized, seq_len)

    def stack(ids: np.ndarray) -> np.ndarray:
        return np.stack([sequences[int(vid_id)] for vid_id in ids], axis=0)

    x_train_raw = stack(splits.train)
    x_val_raw = stack(splits.val)
    x_test_raw = stack(splits.test)

    scaler = StandardScaler()
    flat_train = x_train_raw.reshape(-1, x_train_raw.shape[-1])
    scaler.fit(flat_train)

    def scale(x: np.ndarray) -> np.ndarray:
        flat = x.reshape(-1, x.shape[-1])
        return scaler.transform(flat).reshape(x.shape).astype(np.float32)

    return SequenceData(
        x_train=scale(x_train_raw),
        y_train=labels_for_ids(labels, splits.train, encoder),
        x_val=scale(x_val_raw),
        y_val=labels_for_ids(labels, splits.val, encoder),
        x_test=scale(x_test_raw),
        y_test=labels_for_ids(labels, splits.test, encoder),
        feature_scaler=scaler,
    )


def sequence_to_graph_tensor(x: np.ndarray) -> np.ndarray:
    n, t, _ = x.shape
    v = len(MEDIAPIPE_LANDMARKS)
    return x.reshape(n, t, v, 3).transpose(0, 3, 1, 2).astype(np.float32)

