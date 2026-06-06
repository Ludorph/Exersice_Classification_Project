from __future__ import annotations

import argparse
from pathlib import Path

from .metadata import build_metadata, save_metadata_outputs
from .splitting import find_group_split, save_split_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate AI-Hub bodyweight metadata.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset") / "fitness_pose")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_fitness_pose") / "metadata_check")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata, serial_map = build_metadata(
        dataset_dir=args.dataset_dir,
        source_workbook=args.dataset_dir / "Document" / "source_data_list_and_scenario" / "source_data_list_and_scenario.xlsx",
        naming_workbook=args.dataset_dir / "Document" / "fitness_pose_naming_rules" / "fitness_pose_naming_rules.xlsx",
    )
    save_metadata_outputs(metadata, serial_map, args.output_dir)
    metadata = find_group_split(metadata, seed=args.seed)
    save_split_outputs(metadata, args.output_dir)
    print(f"Metadata samples: {len(metadata):,}")
    print(f"Labels: {metadata['exercise_label'].nunique()}")
    print(f"Sessions: {metadata['session_id'].nunique()}")
    print(f"Saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

