from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .constants import BODYWEIGHT_LABELS
from .xlsx_reader import read_sheet_rows


FILENAME_PATTERN = re.compile(r"^D(?P<day>\d+)-(?P<camera>\d+)-(?P<serial>\d+)-3d\.json$")


def _serial_rows(workbook: Path, sheet_name: str, source_name: str) -> list[dict]:
    rows: list[dict] = []
    for row in read_sheet_rows(workbook, sheet_name):
        serial_text = row.get("A", "")
        category = row.get("I", "")
        exercise = row.get("K", "")
        if not serial_text.isdigit() or "맨몸" not in category or not exercise:
            continue
        rows.append(
            {
                "serial": int(serial_text),
                "category": category,
                "pose": row.get("J", ""),
                "exercise_label": exercise,
                "status_description": row.get("L", ""),
                "mapping_source": source_name,
            }
        )
    return rows


def _detailed_naming_serial_rows(naming_workbook: Path) -> list[dict]:
    rows: list[dict] = []
    for row in read_sheet_rows(naming_workbook, "맨몸"):
        serial_text = row.get("I", "")
        category = row.get("C", "")
        exercise = row.get("E", "")
        if not serial_text.isdigit() or "맨몸" not in category or not exercise:
            continue
        rows.append(
            {
                "serial": int(serial_text),
                "category": category,
                "pose": row.get("D", ""),
                "exercise_label": exercise,
                "status_description": row.get("H", ""),
                "mapping_source": "fitness_pose_naming_rules:맨몸",
            }
        )
    return rows


def load_bodyweight_serial_map(source_workbook: Path, naming_workbook: Path) -> pd.DataFrame:
    # The revised DB contains the new 473-632 serial range, while some training
    # sessions still use the older 193-352 range documented in the naming rules.
    rows = _serial_rows(source_workbook, "DB", "source_data_list_and_scenario:DB")
    rows += _serial_rows(naming_workbook, "21일 촬영 (old)", "fitness_pose_naming_rules:21일 촬영 (old)")
    rows += _detailed_naming_serial_rows(naming_workbook)

    all_rows = pd.DataFrame(rows)
    conflicts = all_rows.groupby("serial")["exercise_label"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(f"Conflicting labels for serials: {conflicts.index.tolist()}")
    mapping = all_rows.drop_duplicates("serial").sort_values("serial").reset_index(drop=True)
    return mapping


def load_naming_rule_labels(naming_workbook: Path) -> list[str]:
    labels: list[str] = []
    for row in read_sheet_rows(naming_workbook, "맨몸"):
        exercise = row.get("E", "").strip()
        if exercise and exercise != "운동" and exercise not in labels:
            labels.append(exercise)
    return labels


def build_metadata(
    dataset_dir: Path,
    source_workbook: Path,
    naming_workbook: Path,
    labeling_folder: str = "bodyweight_labeling_new_220128",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    serial_map = load_bodyweight_serial_map(source_workbook, naming_workbook)
    naming_labels = load_naming_rule_labels(naming_workbook)

    expected = set(BODYWEIGHT_LABELS)
    if set(naming_labels) != expected:
        missing = sorted(expected - set(naming_labels))
        extra = sorted(set(naming_labels) - expected)
        raise ValueError(f"Naming-rule label mismatch. missing={missing}, extra={extra}")

    label_root = dataset_dir / "1.Training" / "labeling_data" / labeling_folder
    if not label_root.exists():
        raise FileNotFoundError(f"Bodyweight labeling folder not found: {label_root}")

    mapping = serial_map.set_index("serial").to_dict("index")
    records: list[dict] = []
    unmapped: list[str] = []
    for json_path in sorted(label_root.rglob("*-3d.json")):
        match = FILENAME_PATTERN.match(json_path.name)
        if not match:
            continue
        serial = int(match.group("serial"))
        label_info = mapping.get(serial)
        if label_info is None:
            unmapped.append(str(json_path))
            continue
        exercise_label = str(label_info["exercise_label"])
        if exercise_label not in expected:
            continue

        relative_path = json_path.relative_to(dataset_dir)
        bodyweight_folder = next(
            (part for part in relative_path.parts if re.fullmatch(r"bodyweight_\d+", part)),
            "",
        )
        records.append(
            {
                "sample_id": str(relative_path.with_suffix("")).replace("\\", "/"),
                "json_path": str(json_path.resolve()),
                "relative_path": str(relative_path).replace("\\", "/"),
                "bodyweight_folder": bodyweight_folder,
                "session_id": json_path.parent.name,
                "day_id": f"D{match.group('day')}",
                "camera_id": match.group("camera"),
                "serial": serial,
                "exercise_label": exercise_label,
                "pose": label_info["pose"],
                "status_description": label_info["status_description"],
                "file_size_bytes": json_path.stat().st_size,
            }
        )

    if unmapped:
        preview = "\n".join(unmapped[:5])
        raise ValueError(f"{len(unmapped)} JSON files have no serial mapping. First files:\n{preview}")

    metadata = pd.DataFrame(records).sort_values(["session_id", "serial", "sample_id"]).reset_index(drop=True)
    if metadata.empty:
        raise ValueError("No bodyweight -3d.json samples were discovered.")
    if metadata["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values were discovered.")

    discovered_labels = set(metadata["exercise_label"])
    if discovered_labels != expected:
        missing = sorted(expected - discovered_labels)
        raise ValueError(f"Not all 17 bodyweight labels were discovered. missing={missing}")

    relevant_map = serial_map[serial_map["exercise_label"].isin(BODYWEIGHT_LABELS)].copy()
    relevant_map["is_present_in_selected_data"] = relevant_map["serial"].isin(metadata["serial"])
    return metadata, relevant_map


def save_metadata_outputs(metadata: pd.DataFrame, serial_map: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_dir / "metadata.csv", index=False, encoding="utf-8-sig")
    serial_map.to_csv(output_dir / "serial_label_mapping.csv", index=False, encoding="utf-8-sig")

    summary = (
        metadata.groupby(["exercise_label", "pose"], as_index=False)
        .agg(samples=("sample_id", "count"), sessions=("session_id", "nunique"))
        .sort_values("exercise_label")
    )
    summary.to_csv(output_dir / "dataset_summary.csv", index=False, encoding="utf-8-sig")
