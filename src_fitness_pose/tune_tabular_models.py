from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .constants import BODYWEIGHT_LABELS, JOINTS
from .features import NODE_FEATURE_NAMES, build_or_load_feature_cache
from .metadata import build_metadata
from .splitting import find_group_split, validate_split


DEFAULT_CONFIG = Path("configs_fitness_pose") / "bodyweight_17.json"


SVM_GRID = [
    {"kernel": "rbf", "C": c, "gamma": gamma, "class_weight": "balanced", "cache_size": 4096}
    for c in [3.0, 10.0, 30.0]
    for gamma in ["scale", 0.01, 0.03]
]


XGBOOST_GRID = [
    {
        "name": "baseline_depth6_lr005_est500",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "n_jobs": -1,
    },
    {
        "name": "shallower_depth4_lr005_est500",
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "n_jobs": -1,
    },
    {
        "name": "faster_depth6_lr008_est300",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "n_jobs": -1,
    },
    {
        "name": "deeper_depth8_lr005_est300",
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "n_jobs": -1,
    },
    {
        "name": "regularized_depth5_lr005_est500",
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
        "n_jobs": -1,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small validation-only tuning for SVM and XGBoost.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_fitness_pose") / "bodyweight_17_full_tuning")
    parser.add_argument("--models", nargs="+", choices=["svm", "xgboost"], default=["svm", "xgboost"])
    parser.add_argument("--seed", type=int, default=None, help="Override config seed. Default uses config seed.")
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def load_config(path: Path, seed: int | None) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if seed is not None:
        config["seed"] = seed
    return config


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "validation_accuracy": float(accuracy_score(y_true, y_pred)),
        "validation_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "validation_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "validation_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "validation_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def prepare_data(config: dict, output_dir: Path, force_features: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
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
    tabular = node_features.reshape(len(node_features), -1)
    train_idx = np.flatnonzero(metadata["split"].to_numpy() == "train")
    validation_idx = np.flatnonzero(metadata["split"].to_numpy() == "validation")
    scaler = StandardScaler()
    scaler.fit(tabular[train_idx])
    x = scaler.transform(tabular).astype(np.float32)
    with (output_dir / "tuning_preprocessing.pkl").open("wb") as file:
        pickle.dump({"scaler": scaler}, file)
    return metadata, x, labels, [*map(str, BODYWEIGHT_LABELS)]


def run_svm(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for index, params in enumerate(SVM_GRID, start=1):
        started = time.perf_counter()
        model = SVC(**params)
        model.fit(x_train, y_train)
        prediction = model.predict(x_val)
        elapsed = time.perf_counter() - started
        row = {
            "model": "svm",
            "candidate": f"svm_{index:02d}",
            **params,
            **metrics(y_val, prediction),
            "elapsed_seconds": elapsed,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    return rows


def run_xgboost(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int) -> list[dict]:
    rows: list[dict] = []
    for params in XGBOOST_GRID:
        params = dict(params)
        name = params.pop("name")
        started = time.perf_counter()
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(BODYWEIGHT_LABELS),
            eval_metric="mlogloss",
            random_state=seed,
            **params,
        )
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        prediction = model.predict(x_val)
        elapsed = time.perf_counter() - started
        row = {
            "model": "xgboost",
            "candidate": name,
            **params,
            **metrics(y_val, prediction),
            "elapsed_seconds": elapsed,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, args.seed)
    (args.output_dir / "tuning_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata, x, labels, _ = prepare_data(config, args.output_dir, args.force_features)
    train_idx = np.flatnonzero(metadata["split"].to_numpy() == "train")
    validation_idx = np.flatnonzero(metadata["split"].to_numpy() == "validation")

    rows: list[dict] = []
    if "svm" in args.models:
        rows.extend(run_svm(x[train_idx], labels[train_idx], x[validation_idx], labels[validation_idx]))
    if "xgboost" in args.models:
        rows.extend(run_xgboost(x[train_idx], labels[train_idx], x[validation_idx], labels[validation_idx], int(config["seed"])))

    results = pd.DataFrame(rows)
    results_path = args.output_dir / "tuning_results.csv"
    if results_path.exists():
        previous = pd.read_csv(results_path)
        previous = previous[~previous["model"].isin(args.models)]
        results = pd.concat([previous, results], ignore_index=True, sort=False)
    results = results.sort_values(["model", "validation_f1_macro"], ascending=[True, False])
    results.to_csv(args.output_dir / "tuning_results.csv", index=False, encoding="utf-8-sig")
    best = results.sort_values("validation_f1_macro", ascending=False).groupby("model", as_index=False).first()
    best.to_csv(args.output_dir / "best_hyperparameters_by_model.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "best_hyperparameters_by_model.json").write_text(
        json.dumps(best.to_dict("records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nBest validation candidates")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
