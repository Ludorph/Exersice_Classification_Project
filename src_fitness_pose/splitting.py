from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _class_count_matrix(metadata: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(metadata["session_id"], metadata["exercise_label"]).sort_index()


def find_group_split(
    metadata: pd.DataFrame,
    val_groups: int = 3,
    test_groups: int = 3,
    attempts: int = 50000,
    seed: int = 42,
) -> pd.DataFrame:
    """Find a session-disjoint split with all labels represented in every split."""

    matrix = _class_count_matrix(metadata)
    groups = matrix.index.to_numpy()
    if len(groups) <= val_groups + test_groups:
        raise ValueError("Not enough sessions for the requested group split.")

    counts = matrix.to_numpy(dtype=np.float64)
    total = counts.sum(axis=0)
    target_ratios = {
        "train": (len(groups) - val_groups - test_groups) / len(groups),
        "validation": val_groups / len(groups),
        "test": test_groups / len(groups),
    }
    rng = np.random.default_rng(seed)
    best: tuple[float, dict[str, np.ndarray]] | None = None

    for _ in range(attempts):
        order = rng.permutation(len(groups))
        test_idx = order[:test_groups]
        val_idx = order[test_groups : test_groups + val_groups]
        train_idx = order[test_groups + val_groups :]
        indices = {"train": train_idx, "validation": val_idx, "test": test_idx}

        split_counts = {name: counts[idx].sum(axis=0) for name, idx in indices.items()}
        if any(np.any(values == 0) for values in split_counts.values()):
            continue

        score = 0.0
        for name, values in split_counts.items():
            observed = values / total
            score += float(np.mean(np.square(observed - target_ratios[name])))
            score += 0.2 * float(np.std(values / np.maximum(values.sum(), 1.0)))

        if best is None or score < best[0]:
            best = (score, indices)

    if best is None:
        raise ValueError(
            "Could not find a session-disjoint split containing all labels. "
            "Increase attempts or change val/test group counts."
        )

    split_by_session: dict[str, str] = {}
    for split_name, indices in best[1].items():
        for index in indices:
            split_by_session[str(groups[index])] = split_name

    result = metadata.copy()
    result["split"] = result["session_id"].map(split_by_session)
    return result


def validate_split(metadata: pd.DataFrame) -> None:
    expected_labels = set(metadata["exercise_label"])
    session_sets: dict[str, set[str]] = {}
    for split_name, group in metadata.groupby("split"):
        labels = set(group["exercise_label"])
        if labels != expected_labels:
            raise ValueError(f"{split_name} is missing labels: {sorted(expected_labels - labels)}")
        session_sets[split_name] = set(group["session_id"])
    names = list(session_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = session_sets[left] & session_sets[right]
            if overlap:
                raise ValueError(f"Session leakage between {left} and {right}: {sorted(overlap)}")


def save_split_outputs(metadata: pd.DataFrame, output_dir: Path) -> None:
    validate_split(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_dir / "metadata_with_split.csv", index=False, encoding="utf-8-sig")

    distribution = (
        metadata.groupby(["split", "exercise_label"], as_index=False)
        .agg(samples=("sample_id", "count"), sessions=("session_id", "nunique"))
        .sort_values(["split", "exercise_label"])
    )
    distribution.to_csv(output_dir / "split_label_distribution.csv", index=False, encoding="utf-8-sig")

    sessions = (
        metadata.groupby(["split", "session_id"], as_index=False)
        .agg(samples=("sample_id", "count"), labels=("exercise_label", "nunique"))
        .sort_values(["split", "session_id"])
    )
    sessions.to_csv(output_dir / "split_sessions.csv", index=False, encoding="utf-8-sig")

    split_info = {
        split_name: {
            "samples": int(len(group)),
            "sessions": sorted(group["session_id"].unique().tolist()),
            "label_count": int(group["exercise_label"].nunique()),
        }
        for split_name, group in metadata.groupby("split")
    }
    (output_dir / "split_info.json").write_text(
        json.dumps(split_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def balanced_smoke_subset(metadata: pd.DataFrame, per_class_per_split: int, seed: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for (_, _), group in metadata.groupby(["split", "exercise_label"]):
        pieces.append(group.sample(min(per_class_per_split, len(group)), random_state=seed))
    return pd.concat(pieces, ignore_index=True).sort_values(["split", "exercise_label", "sample_id"])

