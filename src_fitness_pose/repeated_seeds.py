from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SEEDS = [42, 7, 21, 100, 2026]
DEFAULT_CONFIG = Path("configs_fitness_pose") / "bodyweight_17.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated-seed experiments and aggregate metrics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--models", nargs="+", default=["xgboost", "svm", "gnn", "transformer"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--force", action="store_true", help="Re-run seed experiments even if metrics already exist.")
    parser.add_argument("--force-features", action="store_true", help="Regenerate the shared feature cache.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_seed(args: argparse.Namespace, base_name: str, output_root: Path, seed: int) -> Path:
    experiment_name = f"{base_name}_seed_{seed}"
    experiment_dir = output_root / experiment_name
    metrics_path = experiment_dir / "results" / "metrics_summary.csv"
    if metrics_path.exists() and not args.force:
        print(f"[skip] seed={seed} metrics already exist: {metrics_path}")
        return metrics_path

    command = [
        sys.executable,
        "-m",
        "src_fitness_pose.train_pipeline",
        "--config",
        str(args.config),
        "--seed",
        str(seed),
        "--experiment-name",
        experiment_name,
        "--device",
        args.device,
        "--models",
        *args.models,
    ]
    if args.force_features:
        command.append("--force-features")
    print(f"[run] seed={seed}: {' '.join(command)}")
    subprocess.run(command, check=True)
    return metrics_path


def aggregate(metrics_paths: list[Path], output_dir: Path) -> None:
    frames: list[pd.DataFrame] = []
    for path in metrics_paths:
        seed = int(path.parents[1].name.rsplit("_seed_", 1)[1])
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        frames.append(frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    by_seed = pd.concat(frames, ignore_index=True).sort_values(["model", "seed"])
    by_seed.to_csv(output_dir / "metrics_by_seed.csv", index=False, encoding="utf-8-sig")

    metric_columns = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    summary = by_seed.groupby("model")[metric_columns].agg(["mean", "std"]).sort_values(("f1_macro", "mean"), ascending=False)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output_dir / "metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    plot_frame = summary[["accuracy_mean", "f1_macro_mean", "f1_weighted_mean"]]
    error_frame = summary[["accuracy_std", "f1_macro_std", "f1_weighted_std"]]
    error_frame.columns = plot_frame.columns
    ax = plot_frame.plot(kind="bar", yerr=error_frame, figsize=(10, 6), ylim=(0, 1), capsize=4, rot=0)
    ax.set_ylabel("Mean score")
    ax.set_title("Repeated-seed Model Performance")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "repeated_seed_model_comparison.png", dpi=200)
    plt.close()

    lines = ["# 반복 seed 실험 요약", ""]
    for _, row in summary.reset_index().iterrows():
        lines.append(
            f"- {row['model']}: Macro-F1 {row['f1_macro_mean']:.4f} ± {row['f1_macro_std']:.4f}, "
            f"Accuracy {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
        )
    (output_dir / "repeated_seed_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(config["paths"]["output_root"])
    base_name = config["experiment_name"]
    metrics_paths = [run_seed(args, base_name, output_root, seed) for seed in args.seeds]
    aggregate(metrics_paths, output_root / f"{base_name}_repeated_seeds")
    print(f"[done] repeated-seed results: {(output_root / f'{base_name}_repeated_seeds').resolve()}")


if __name__ == "__main__":
    main()
