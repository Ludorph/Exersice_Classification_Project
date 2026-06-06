from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .constants import BODYWEIGHT_LABELS
from .features import build_or_load_feature_cache
from .metadata import build_metadata
from .models import GNNClassifier, JointTransformerClassifier, parameter_count
from .splitting import find_group_split, validate_split
from .train_pipeline import model_training_config, predict_torch, select_device, set_seed, train_torch_model


DEFAULT_CONFIG = Path("configs_fitness_pose") / "bodyweight_17.json"


GNN_GRID = [
    {
        "candidate": "baseline_hidden128_layers3_dropout030_lr001",
        "model": {"hidden_dim": 128, "num_layers": 3, "dropout": 0.3},
        "training": {"learning_rate": 0.001, "weight_decay": 0.0001},
    },
    {
        "candidate": "wider_hidden192_layers3_dropout020_lr001",
        "model": {"hidden_dim": 192, "num_layers": 3, "dropout": 0.2},
        "training": {"learning_rate": 0.001, "weight_decay": 0.0001},
    },
    {
        "candidate": "shallower_hidden128_layers2_dropout030_lr0007",
        "model": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.3},
        "training": {"learning_rate": 0.0007, "weight_decay": 0.0001},
    },
]


TRANSFORMER_GRID = [
    {
        "candidate": "baseline_d128_heads4_layers3_ff256_dropout010_lr001",
        "model": {
            "d_model": 128,
            "num_heads": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
        },
        "training": {"learning_rate": 0.001, "weight_decay": 0.0001},
    },
    {
        "candidate": "regularized_d128_heads4_layers3_ff256_dropout020_lr0007",
        "model": {
            "d_model": 128,
            "num_heads": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.2,
        },
        "training": {"learning_rate": 0.0007, "weight_decay": 0.0001},
    },
    {
        "candidate": "ff512_d128_heads4_layers2_dropout010_lr001",
        "model": {
            "d_model": 128,
            "num_heads": 4,
            "num_layers": 2,
            "dim_feedforward": 512,
            "dropout": 0.1,
        },
        "training": {"learning_rate": 0.001, "weight_decay": 0.0001},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small validation-only tuning for GNN and Transformer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_fitness_pose") / "bodyweight_17_full_deep_tuning")
    parser.add_argument("--models", nargs="+", choices=["gnn", "transformer"], default=["gnn", "transformer"])
    parser.add_argument("--seed", type=int, default=None, help="Override config seed. Default uses config seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def load_config(path: Path, seed: int | None) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if seed is not None:
        config["seed"] = int(seed)
    return config


def validation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "validation_accuracy": float(accuracy_score(y_true, y_pred)),
        "validation_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "validation_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "validation_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "validation_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def prepare_data(config: dict, output_dir: Path, force_features: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dataset_dir = Path(config["paths"]["dataset_dir"])
    metadata, _ = build_metadata(
        dataset_dir=dataset_dir,
        source_workbook=Path(config["paths"]["source_workbook"]),
        naming_workbook=Path(config["paths"]["naming_workbook"]),
    )
    metadata = find_group_split(metadata, **config["split"], seed=int(config["seed"]))

    cache_dir = Path(config["features"].get("cache_dir", output_dir / "feature_cache"))
    valid_ids, node_features, frame_counts, failures = build_or_load_feature_cache(
        metadata,
        cache_dir / "features_all.npz",
        workers=int(config["features"]["workers"]),
        force=force_features,
    )
    failures.to_csv(output_dir / "feature_failures.csv", index=False, encoding="utf-8-sig")
    metadata = metadata.set_index("sample_id").loc[valid_ids].reset_index()
    metadata["frame_count"] = frame_counts
    validate_split(metadata)
    metadata.to_csv(output_dir / "tuning_metadata.csv", index=False, encoding="utf-8-sig")

    labels = LabelEncoder().fit(BODYWEIGHT_LABELS).transform(metadata["exercise_label"])
    train_idx = np.flatnonzero(metadata["split"].to_numpy() == "train")

    scaler = StandardScaler()
    scaler.fit(node_features[train_idx].reshape(-1, node_features.shape[-1]))
    scaled_nodes = scaler.transform(node_features.reshape(-1, node_features.shape[-1])).reshape(node_features.shape).astype(np.float32)
    with (output_dir / "tuning_preprocessing.pkl").open("wb") as file:
        pickle.dump({"node_scaler": scaler}, file)
    return metadata, scaled_nodes, labels


def candidate_rows(model_name: str) -> list[dict]:
    if model_name == "gnn":
        return GNN_GRID
    if model_name == "transformer":
        return TRANSFORMER_GRID
    raise ValueError(f"Unsupported model: {model_name}")


def build_model(model_name: str, input_dim: int, num_classes: int, model_config: dict) -> torch.nn.Module:
    if model_name == "gnn":
        return GNNClassifier(input_dim=input_dim, num_classes=num_classes, **model_config)
    if model_name == "transformer":
        return JointTransformerClassifier(input_dim=input_dim, num_classes=num_classes, **model_config)
    raise ValueError(f"Unsupported model: {model_name}")


def run_candidate(
    model_name: str,
    candidate: dict,
    base_config: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    output_dir: Path,
) -> dict:
    config = {
        **base_config,
        "models": {**base_config["models"], model_name: candidate["model"]},
        "training_overrides": {
            **base_config.get("training_overrides", {}),
            model_name: candidate["training"],
        },
    }
    torch_config = model_training_config(config, model_name)
    model = build_model(model_name, x_train.shape[-1], len(BODYWEIGHT_LABELS), candidate["model"])
    model_dir = output_dir / "candidate_checkpoints"
    history_dir = output_dir / "candidate_histories"
    started = time.perf_counter()
    model = train_torch_model(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        torch_config,
        device,
        model_dir / f"{model_name}_{candidate['candidate']}.pt",
        history_dir / f"training_history_{model_name}_{candidate['candidate']}.csv",
    )
    prediction = predict_torch(model, x_val, torch_config["batch_size"], device)
    elapsed = time.perf_counter() - started
    row = {
        "model": model_name,
        "candidate": candidate["candidate"],
        "parameters": parameter_count(model),
        **candidate["model"],
        **torch_config,
        **validation_metrics(y_val, prediction),
        "elapsed_seconds": elapsed,
    }
    print(json.dumps(row, ensure_ascii=False))
    return row


def write_plots(results: pd.DataFrame, output_dir: Path) -> None:
    for model_name, frame in results.groupby("model"):
        plot_frame = frame.sort_values("validation_f1_macro", ascending=False).set_index("candidate")
        ax = plot_frame["validation_f1_macro"].plot(kind="bar", figsize=(11, 5), ylim=(0, 1), rot=35)
        ax.set_ylabel("Validation Macro-F1")
        ax.set_title(f"{model_name} validation tuning")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"tuning_validation_macro_f1_{model_name}.png", dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, args.seed)
    set_seed(int(config["seed"]))
    device = select_device(args.device)
    (args.output_dir / "tuning_config.json").write_text(
        json.dumps({**config, "selected_models": args.models, "device": str(device)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata, nodes, labels = prepare_data(config, args.output_dir, args.force_features)
    split = metadata["split"].to_numpy()
    train_idx = np.flatnonzero(split == "train")
    validation_idx = np.flatnonzero(split == "validation")

    rows: list[dict] = []
    for model_name in args.models:
        for candidate in candidate_rows(model_name):
            set_seed(int(config["seed"]))
            rows.append(
                run_candidate(
                    model_name,
                    candidate,
                    config,
                    nodes[train_idx],
                    labels[train_idx],
                    nodes[validation_idx],
                    labels[validation_idx],
                    device,
                    args.output_dir,
                )
            )

    results = pd.DataFrame(rows)
    results_path = args.output_dir / "tuning_results.csv"
    if results_path.exists():
        previous = pd.read_csv(results_path)
        previous = previous[~previous["model"].isin(args.models)]
        results = pd.concat([previous, results], ignore_index=True, sort=False)
    results = results.sort_values(["model", "validation_f1_macro"], ascending=[True, False])
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    best = results.sort_values("validation_f1_macro", ascending=False).groupby("model", as_index=False).first()
    best.to_csv(args.output_dir / "best_hyperparameters_by_model.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "best_hyperparameters_by_model.json").write_text(
        json.dumps(best.to_dict("records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_plots(results, args.output_dir)
    print("\nBest validation candidates")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
