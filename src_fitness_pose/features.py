from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .constants import JOINTS


STAT_NAMES = ["mean", "std", "min", "max", "range", "delta", "velocity_mean", "velocity_max"]
AXES = ["x", "y", "z"]
NODE_FEATURE_NAMES = [f"{stat}_{axis}" for stat in STAT_NAMES for axis in AXES]


def _points_array(json_path: str) -> np.ndarray:
    with Path(json_path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError(f"No frames in {json_path}")

    values = np.full((len(frames), len(JOINTS), 3), np.nan, dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        points = frame.get("pts", {})
        for joint_index, joint in enumerate(JOINTS):
            point = points.get(joint)
            if point:
                values[frame_index, joint_index] = [
                    float(point.get("x", np.nan)),
                    float(point.get("y", np.nan)),
                    float(point.get("z", np.nan)),
                ]
    return values


def normalize_frames(values: np.ndarray) -> np.ndarray:
    joint_index = {name: index for index, name in enumerate(JOINTS)}
    left_hip = values[:, joint_index["Left Hip"]]
    right_hip = values[:, joint_index["Right Hip"]]
    left_shoulder = values[:, joint_index["Left Shoulder"]]
    right_shoulder = values[:, joint_index["Right Shoulder"]]

    center = np.nanmean(np.stack([left_hip, right_hip], axis=0), axis=0)
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
    hip_width = np.linalg.norm(left_hip - right_hip, axis=1)
    scale = np.nanmean(np.stack([shoulder_width, hip_width], axis=0), axis=0)
    valid_scale = scale[np.isfinite(scale) & (scale > 1e-3)]
    fallback = float(np.median(valid_scale)) if len(valid_scale) else 1.0
    minimum_scale = max(fallback * 0.25, 1e-3)
    scale = np.where(np.isfinite(scale) & (scale >= minimum_scale), scale, fallback)

    normalized = (values - center[:, None, :]) / scale[:, None, None]
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(normalized, -10.0, 10.0).astype(np.float32)


def extract_node_features(json_path: str) -> tuple[np.ndarray, int]:
    values = normalize_frames(_points_array(json_path))
    velocity = np.diff(values, axis=0)
    if len(velocity) == 0:
        velocity = np.zeros_like(values[:1])

    stats = [
        values.mean(axis=0),
        values.std(axis=0),
        values.min(axis=0),
        values.max(axis=0),
        values.max(axis=0) - values.min(axis=0),
        values[-1] - values[0],
        np.abs(velocity).mean(axis=0),
        np.abs(velocity).max(axis=0),
    ]
    # [stats, joints, xyz] -> [joints, stats * xyz]
    node_features = np.stack(stats, axis=0).transpose(1, 0, 2).reshape(len(JOINTS), -1)
    return node_features.astype(np.float32), int(len(values))


def _extract_record(record: tuple[str, str]) -> tuple[str, np.ndarray | None, int, str]:
    sample_id, json_path = record
    try:
        features, frame_count = extract_node_features(json_path)
        return sample_id, features, frame_count, ""
    except Exception as error:
        return sample_id, None, 0, f"{type(error).__name__}: {error}"


def build_or_load_feature_cache(
    metadata: pd.DataFrame,
    cache_path: Path,
    workers: int = 8,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    sample_ids = metadata["sample_id"].astype(str).to_numpy(dtype=str)
    if cache_path.exists() and not force:
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if np.array_equal(cached["requested_sample_ids"], sample_ids):
                failures = pd.DataFrame(
                    {
                        "sample_id": cached["failure_sample_ids"],
                        "error": cached["failure_errors"],
                    }
                )
                return (
                    cached["sample_ids"],
                    cached["node_features"].astype(np.float32),
                    cached["frame_counts"].astype(np.int32),
                    failures,
                )
        except (KeyError, ValueError):
            pass

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    records = list(zip(sample_ids.tolist(), metadata["json_path"].astype(str).tolist()))
    features: list[np.ndarray] = []
    frames: list[int] = []
    extracted_ids: list[str] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        iterator = executor.map(_extract_record, records)
        for sample_id, node_features, frame_count, error in tqdm(iterator, total=len(records), desc="Extract features"):
            if node_features is None:
                failures.append({"sample_id": sample_id, "error": error})
                continue
            extracted_ids.append(sample_id)
            features.append(node_features)
            frames.append(frame_count)

    if not features:
        raise ValueError("No valid JSON samples remained after feature extraction.")

    node_features = np.stack(features).astype(np.float32)
    frame_counts = np.asarray(frames, dtype=np.int32)
    failure_frame = pd.DataFrame(failures, columns=["sample_id", "error"])
    np.savez_compressed(
        cache_path,
        requested_sample_ids=np.asarray(sample_ids, dtype=str),
        sample_ids=np.asarray(extracted_ids, dtype=str),
        node_features=node_features,
        frame_counts=frame_counts,
        node_feature_names=np.asarray(NODE_FEATURE_NAMES),
        joints=np.asarray(JOINTS),
        failure_sample_ids=failure_frame["sample_id"].to_numpy(dtype=str),
        failure_errors=failure_frame["error"].to_numpy(dtype=str),
    )
    return np.asarray(extracted_ids), node_features, frame_counts, failure_frame
