from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from .constants import BODYWEIGHT_LABELS, JOINTS
from .evaluation import evaluate_model, save_comparison
from .features import NODE_FEATURE_NAMES, build_or_load_feature_cache
from .metadata import build_metadata, save_metadata_outputs
from .models import GNNClassifier, JointTransformerClassifier, parameter_count
from .splitting import balanced_smoke_subset, find_group_split, save_split_outputs, validate_split


DEFAULT_CONFIG = Path("configs_fitness_pose") / "bodyweight_17.json"
ALL_MODELS = ("xgboost", "svm", "gnn", "transformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare four models on AI-Hub bodyweight pose JSON files.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    parser.add_argument("--seed", type=int, default=None, help="Override the seed in the config file.")
    parser.add_argument("--experiment-name", type=str, default=None, help="Override the output experiment name.")
    parser.add_argument("--smoke", action="store_true", help="Run a small end-to-end verification experiment.")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_config(path: Path, smoke: bool, seed: int | None, experiment_name: str | None) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if seed is not None:
        config["seed"] = int(seed)
    if experiment_name is not None:
        config["experiment_name"] = experiment_name
    if smoke:
        for section, overrides in config.get("smoke_overrides", {}).items():
            if isinstance(overrides, dict):
                config.setdefault(section, {}).update(overrides)
            else:
                config[section] = overrides
        config["experiment_name"] = f"{config['experiment_name']}_smoke"
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(argument: str) -> torch.device:
    if argument == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if argument == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def model_training_config(config: dict, model_name: str) -> dict:
    return {
        **config["training"],
        **config.get("training_overrides", {}).get(model_name, {}),
    }


def predict_torch(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = make_loader(x, np.zeros(len(x), dtype=np.int64), batch_size, False)
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            predictions.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def train_torch_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    config: dict,
    device: torch.device,
    checkpoint_path: Path,
    history_path: Path,
) -> nn.Module:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    weights = compute_class_weight("balanced", classes=np.arange(len(np.unique(y_train))), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    model = model.to(device)
    train_loader = make_loader(x_train, y_train, config["batch_size"], True)
    validation_loader = make_loader(x_validation, y_validation, config["batch_size"], False)

    best_loss = float("inf")
    best_state: dict | None = None
    wait = 0
    history: list[dict] = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        validation_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in validation_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                validation_loss += float(criterion(logits, yb).item()) * len(xb)
                correct += int((logits.argmax(dim=1) == yb).sum().item())
        validation_loss /= len(validation_loader.dataset)
        validation_accuracy = correct / len(validation_loader.dataset)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss:.4f} validation_accuracy={validation_accuracy:.4f}"
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            wait = 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            torch.save(best_state, checkpoint_path)
        else:
            wait += 1
            if wait >= config["patience"]:
                break

    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
    if best_state is None:
        raise RuntimeError("No model checkpoint was produced.")
    model.load_state_dict(best_state)
    return model


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.smoke, args.seed, args.experiment_name)
    seed = int(config["seed"])
    set_seed(seed)
    device = select_device(args.device)

    dataset_dir = Path(config["paths"]["dataset_dir"])
    output_dir = Path(config["paths"]["output_root"]) / config["experiment_name"]
    data_output = output_dir / "data"
    result_output = output_dir / "results"
    checkpoint_output = output_dir / "checkpoints"
    for path in (data_output, result_output, checkpoint_output):
        path.mkdir(parents=True, exist_ok=True)

    metadata, serial_map = build_metadata(
        dataset_dir=dataset_dir,
        source_workbook=Path(config["paths"]["source_workbook"]),
        naming_workbook=Path(config["paths"]["naming_workbook"]),
    )
    save_metadata_outputs(metadata, serial_map, data_output)
    metadata = find_group_split(metadata, **config["split"], seed=seed)
    validate_split(metadata)
    save_split_outputs(metadata, data_output)

    if args.smoke:
        metadata = balanced_smoke_subset(
            metadata,
            per_class_per_split=int(config["smoke"]["samples_per_class_per_split"]),
            seed=seed,
        )

    cache_name = "features_smoke.npz" if args.smoke else "features_all.npz"
    cache_dir = config["features"].get("cache_dir")
    cache_path = Path(cache_dir) / cache_name if cache_dir else data_output / cache_name
    valid_sample_ids, node_features, frame_counts, feature_failures = build_or_load_feature_cache(
        metadata,
        cache_path,
        workers=int(config["features"]["workers"]),
        force=args.force_features,
    )
    feature_failures.to_csv(data_output / "feature_failures.csv", index=False, encoding="utf-8-sig")
    metadata = metadata.set_index("sample_id").loc[valid_sample_ids].reset_index()
    metadata["frame_count"] = frame_counts
    validate_split(metadata)
    metadata.to_csv(data_output / ("experiment_metadata_smoke.csv" if args.smoke else "experiment_metadata.csv"), index=False, encoding="utf-8-sig")

    encoder = LabelEncoder()
    encoder.fit(BODYWEIGHT_LABELS)
    labels = encoder.transform(metadata["exercise_label"])
    class_names = encoder.classes_.tolist()
    split_indices = {
        split: np.flatnonzero(metadata["split"].to_numpy() == split)
        for split in ("train", "validation", "test")
    }

    node_scaler = StandardScaler()
    train_node_rows = node_features[split_indices["train"]].reshape(-1, node_features.shape[-1])
    node_scaler.fit(train_node_rows)
    scaled_nodes = node_scaler.transform(node_features.reshape(-1, node_features.shape[-1])).reshape(node_features.shape).astype(np.float32)

    tabular = node_features.reshape(len(node_features), -1)
    tabular_scaler = StandardScaler()
    tabular_scaler.fit(tabular[split_indices["train"]])
    scaled_tabular = tabular_scaler.transform(tabular).astype(np.float32)
    with (data_output / "preprocessing.pkl").open("wb") as file:
        pickle.dump({"node_scaler": node_scaler, "tabular_scaler": tabular_scaler, "label_encoder": encoder}, file)

    actual_config = {
        **config,
        "selected_models": args.models,
        "device": str(device),
        "classes": class_names,
        "node_feature_names": NODE_FEATURE_NAMES,
        "samples_used": int(len(metadata)),
        "excluded_invalid_json_samples": int(len(feature_failures)),
        "split_samples": {name: int(len(indices)) for name, indices in split_indices.items()},
    }
    (output_dir / "experiment_config_actual.json").write_text(
        json.dumps(actual_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results: list[dict] = []
    model_details: list[dict] = []
    for model_name in args.models:
        print(f"\n=== {model_name.upper()} ===")
        started = time.perf_counter()
        if model_name == "xgboost":
            model_config = config["models"]["xgboost"]
            model = XGBClassifier(
                objective="multi:softprob",
                num_class=len(class_names),
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=model_config["n_jobs"],
                **{key: value for key, value in model_config.items() if key != "n_jobs"},
            )
            model.fit(
                scaled_tabular[split_indices["train"]],
                labels[split_indices["train"]],
                eval_set=[(scaled_tabular[split_indices["validation"]], labels[split_indices["validation"]])],
                verbose=False,
            )
            prediction = model.predict(scaled_tabular[split_indices["test"]])
            importance = pd.DataFrame(
                {
                    "feature": [
                        f"{joint}_{feature}"
                        for joint in JOINTS
                        for feature in NODE_FEATURE_NAMES
                    ],
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            importance.to_csv(result_output / "xgboost_feature_importance.csv", index=False, encoding="utf-8-sig")
            with (checkpoint_output / "xgboost.pkl").open("wb") as file:
                pickle.dump(model, file)
            model_details.append({"model": model_name, **model_config})
        elif model_name == "svm":
            model_config = config["models"]["svm"]
            model = SVC(**model_config)
            model.fit(scaled_tabular[split_indices["train"]], labels[split_indices["train"]])
            prediction = model.predict(scaled_tabular[split_indices["test"]])
            with (checkpoint_output / "svm.pkl").open("wb") as file:
                pickle.dump(model, file)
            model_details.append({"model": model_name, **model_config})
        else:
            torch_config = model_training_config(config, model_name)
            if model_name == "gnn":
                model_config = config["models"]["gnn"]
                model = GNNClassifier(
                    input_dim=scaled_nodes.shape[-1],
                    num_classes=len(class_names),
                    **model_config,
                )
            else:
                model_config = config["models"]["transformer"]
                model = JointTransformerClassifier(
                    input_dim=scaled_nodes.shape[-1],
                    num_classes=len(class_names),
                    **model_config,
                )
            model_details.append(
                {"model": model_name, "parameters": parameter_count(model), **model_config, **torch_config}
            )
            model = train_torch_model(
                model,
                scaled_nodes[split_indices["train"]],
                labels[split_indices["train"]],
                scaled_nodes[split_indices["validation"]],
                labels[split_indices["validation"]],
                torch_config,
                device,
                checkpoint_output / f"{model_name}.pt",
                result_output / f"training_history_{model_name}.csv",
            )
            prediction = predict_torch(
                model,
                scaled_nodes[split_indices["test"]],
                torch_config["batch_size"],
                device,
            )

        elapsed = time.perf_counter() - started
        metrics = evaluate_model(
            metadata.iloc[split_indices["test"]].reset_index(drop=True),
            labels[split_indices["test"]],
            prediction,
            class_names,
            model_name,
            result_output,
        )
        metrics["training_and_prediction_seconds"] = elapsed
        results.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    pd.DataFrame(model_details).to_csv(output_dir / "model_hyperparameters.csv", index=False, encoding="utf-8-sig")
    save_comparison(results, result_output)
    print(f"\nCompleted. Results: {result_output.resolve()}")


if __name__ == "__main__":
    main()
