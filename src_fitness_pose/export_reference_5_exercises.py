from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_LABELS = ["버피 테스트", "사이드 런지", "크런치", "푸시업", "플랭크"]
LABEL_TO_EN = {
    "버피 테스트": "burpee_test",
    "사이드 런지": "side_lunge",
    "크런치": "crunch",
    "푸시업": "pushup",
    "플랭크": "plank",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AI-Hub reference JSON/features for five target exercises.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("outputs_fitness_pose") / "bodyweight_17_tuned_all_seed_42" / "data" / "experiment_metadata.csv",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("outputs_fitness_pose") / "_feature_cache" / "features_all.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports") / "aihub_reference_5_exercises",
    )
    return parser.parse_args()


def safe_relative_json_path(row: pd.Series) -> Path:
    label_en = LABEL_TO_EN[row["exercise_label"]]
    return Path("json") / label_en / row["bodyweight_folder"] / row["session_id"] / Path(row["json_path"]).name


def load_feature_subset(feature_cache: Path, sample_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = np.load(feature_cache, allow_pickle=True)
    cache_sample_ids = cache["sample_ids"].astype(str)
    index = {sample_id: position for position, sample_id in enumerate(cache_sample_ids)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in index]
    if missing:
        raise ValueError(f"Feature cache is missing {len(missing)} samples. First missing sample: {missing[0]}")
    positions = np.asarray([index[sample_id] for sample_id in sample_ids], dtype=np.int64)
    return (
        cache["node_features"][positions].astype(np.float32),
        cache["frame_counts"][positions].astype(np.int32),
        cache["joints"].astype(str),
        cache["node_feature_names"].astype(str),
    )


def feature_frame(metadata: pd.DataFrame, node_features: np.ndarray, joints: np.ndarray, feature_names: np.ndarray) -> pd.DataFrame:
    flat = node_features.reshape(len(node_features), -1)
    columns = [f"{joint}_{feature}" for joint in joints for feature in feature_names]
    frame = pd.DataFrame(flat, columns=columns)
    prefix = metadata[["reference_id", "sample_id", "label_ko", "label_en", "session_id", "split", "frame_count"]].reset_index(drop=True)
    return pd.concat([prefix, frame], axis=1)


def write_readme(output_dir: Path, summary: pd.DataFrame) -> None:
    summary_lines = ["| label_ko | label_en | samples | sessions |", "|---|---|---:|---:|"]
    for _, row in summary.iterrows():
        summary_lines.append(f"| {row['label_ko']} | {row['label_en']} | {row['samples']} | {row['sessions']} |")
    readme = f"""# AI-Hub Reference Dataset: 5 Exercises

This export contains AI-Hub `-3d.json` reference samples for five exercise labels:
burpee test, side lunge, crunch, pushup, and plank.

The dataset is intended as a label-ground-truth reference for exercise action
classification. It is not a ground truth dataset for judging whether a posture is
correct or incorrect.

## Files

- `metadata.csv`: exported sample list, original path, copied JSON path, label, session, split, and frame count.
- `label_summary.csv`: sample/session count by exercise label.
- `features_reference.npz`: node feature tensor generated from the same feature pipeline used in the main experiment.
- `features_reference.csv`: flattened feature vectors for spreadsheet inspection or simple similarity experiments.
- `json/`: copied AI-Hub `-3d.json` files grouped by English label name.

## Summary

{chr(10).join(summary_lines)}

## Recommended Use

Use `metadata.csv` as the source of truth for labels. When comparing directly
recorded MediaPipe samples against this reference dataset, convert the recorded
sample into the same feature format as `features_reference.csv` or
`features_reference.npz`, then compare it with the reference feature vectors.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.metadata.exists():
        raise FileNotFoundError(args.metadata)
    if not args.feature_cache.exists():
        raise FileNotFoundError(args.feature_cache)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    selected = metadata.loc[metadata["exercise_label"].isin(DEFAULT_LABELS)].copy()
    if selected.empty:
        raise ValueError("No selected labels were found in metadata.")
    selected["label_ko"] = selected["exercise_label"]
    selected["label_en"] = selected["label_ko"].map(LABEL_TO_EN)
    selected = selected.sort_values(["label_en", "session_id", "serial", "sample_id"]).reset_index(drop=True)
    selected["reference_id"] = [
        f"{row.label_en}_{index + 1:05d}"
        for index, row in enumerate(selected.itertuples(index=False))
    ]
    selected["copied_json_path"] = selected.apply(safe_relative_json_path, axis=1).map(lambda path: path.as_posix())

    for _, row in selected.iterrows():
        source = Path(row["json_path"])
        target = output_dir / row["copied_json_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    node_features, frame_counts, joints, feature_names = load_feature_subset(args.feature_cache, selected["sample_id"].astype(str).tolist())
    selected["frame_count"] = frame_counts

    metadata_columns = [
        "reference_id",
        "sample_id",
        "label_ko",
        "label_en",
        "pose",
        "split",
        "session_id",
        "day_id",
        "camera_id",
        "serial",
        "frame_count",
        "file_size_bytes",
        "json_path",
        "relative_path",
        "copied_json_path",
    ]
    selected[metadata_columns].to_csv(output_dir / "metadata.csv", index=False, encoding="utf-8-sig")

    summary = (
        selected.groupby(["label_ko", "label_en", "pose"], as_index=False)
        .agg(samples=("sample_id", "size"), sessions=("session_id", "nunique"), frames_mean=("frame_count", "mean"))
        .sort_values("label_en")
    )
    summary["frames_mean"] = summary["frames_mean"].map(lambda value: f"{value:.2f}")
    summary.to_csv(output_dir / "label_summary.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        output_dir / "features_reference.npz",
        reference_ids=selected["reference_id"].astype(str).to_numpy(),
        sample_ids=selected["sample_id"].astype(str).to_numpy(),
        labels_ko=selected["label_ko"].astype(str).to_numpy(),
        labels_en=selected["label_en"].astype(str).to_numpy(),
        node_features=node_features,
        frame_counts=frame_counts,
        joints=joints,
        node_feature_names=feature_names,
    )
    feature_frame(selected, node_features, joints, feature_names).to_csv(
        output_dir / "features_reference.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "labels": DEFAULT_LABELS,
        "label_to_en": LABEL_TO_EN,
        "samples": int(len(selected)),
        "metadata_source": str(args.metadata),
        "feature_cache_source": str(args.feature_cache),
        "output_dir": str(output_dir),
        "purpose": "Reference label-ground-truth dataset for five exercise action-classification labels.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir, summary.rename(columns={"exercise_label": "label_ko"}))

    print(f"Completed. Exported {len(selected)} reference samples to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
