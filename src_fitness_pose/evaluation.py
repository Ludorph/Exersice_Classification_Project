from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def evaluate_model(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = metadata[["sample_id", "session_id", "serial", "exercise_label"]].copy()
    prediction_frame["true_label"] = [class_names[index] for index in y_true]
    prediction_frame["predicted_label"] = [class_names[index] for index in y_pred]
    prediction_frame["is_correct"] = y_true == y_pred
    prediction_frame.to_csv(
        output_dir / f"test_predictions_{model_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        output_dir / f"classification_report_{model_name}.csv",
        encoding="utf-8-sig",
    )

    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        output_dir / f"confusion_matrix_{model_name}.csv",
        encoding="utf-8-sig",
    )
    plt.figure(figsize=(15, 13))
    sns.heatmap(matrix, cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot=False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xticks(rotation=55, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{model_name}.png", dpi=200)
    plt.close()

    pairs = (
        prediction_frame.loc[~prediction_frame["is_correct"]]
        .groupby(["true_label", "predicted_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    pairs.insert(0, "model", model_name)
    pairs.to_csv(output_dir / f"misclassification_pairs_{model_name}.csv", index=False, encoding="utf-8-sig")

    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_comparison(results: list[dict], output_dir: Path) -> None:
    summary = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_frame = summary.set_index("model")[["accuracy", "f1_macro", "f1_weighted"]]
    ax = plot_frame.plot(kind="bar", figsize=(10, 6), ylim=(0, 1), rot=0)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_overall.png", dpi=200)
    plt.close()

    pair_files = [
        path
        for path in sorted(output_dir.glob("misclassification_pairs_*.csv"))
        if path.name != "misclassification_pairs_all_models.csv"
    ]
    if pair_files:
        pairs = pd.concat([pd.read_csv(path) for path in pair_files], ignore_index=True)
        pairs.to_csv(output_dir / "misclassification_pairs_all_models.csv", index=False, encoding="utf-8-sig")

    report_rows: list[pd.DataFrame] = []
    for report_path in sorted(output_dir.glob("classification_report_*.csv")):
        model_name = report_path.stem.replace("classification_report_", "")
        report = pd.read_csv(report_path, index_col=0).reset_index().rename(columns={"index": "exercise_label"})
        report = report[~report["exercise_label"].isin(["accuracy", "macro avg", "weighted avg"])]
        report.insert(0, "model", model_name)
        report_rows.append(report)
    if report_rows:
        per_class = pd.concat(report_rows, ignore_index=True)
        per_class.to_csv(output_dir / "per_class_metrics_long.csv", index=False, encoding="utf-8-sig")
        pivot = per_class.pivot(index="exercise_label", columns="model", values="f1-score")
        ax = pivot.plot(kind="bar", figsize=(18, 8), ylim=(0, 1))
        ax.set_ylabel("F1-score")
        ax.set_title("Per-class F1-score Comparison")
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=55, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison_per_class_f1.png", dpi=200)
        plt.close()

    prediction_files = sorted(output_dir.glob("test_predictions_*.csv"))
    if prediction_files:
        hard_cases: pd.DataFrame | None = None
        for prediction_path in prediction_files:
            model_name = prediction_path.stem.replace("test_predictions_", "")
            frame = pd.read_csv(prediction_path)
            model_errors = frame[["sample_id", "session_id", "serial", "true_label", "is_correct"]].copy()
            model_errors = model_errors.rename(columns={"is_correct": f"{model_name}_correct"})
            hard_cases = model_errors if hard_cases is None else hard_cases.merge(
                model_errors[["sample_id", f"{model_name}_correct"]],
                on="sample_id",
                how="outer",
            )
        if hard_cases is not None:
            correctness_columns = [column for column in hard_cases if column.endswith("_correct")]
            hard_cases["models_incorrect"] = (~hard_cases[correctness_columns].astype(bool)).sum(axis=1)
            hard_cases.sort_values("models_incorrect", ascending=False).to_csv(
                output_dir / "hard_cases_by_sample.csv",
                index=False,
                encoding="utf-8-sig",
            )

    best = summary.iloc[0]
    top_pairs = ""
    combined_pairs_path = output_dir / "misclassification_pairs_all_models.csv"
    if combined_pairs_path.exists():
        combined_pairs = pd.read_csv(combined_pairs_path).sort_values("count", ascending=False).head(10)
        top_pairs = "\n".join(
            f"- {row.model}: {row.true_label} -> {row.predicted_label} ({int(row['count'])})"
            for _, row in combined_pairs.iterrows()
        )
    analysis = (
        "# 실험 결과 자동 요약\n\n"
        f"- 최고 Macro-F1 모델: {best['model']} ({best['f1_macro']:.4f})\n"
        f"- 해당 모델 Accuracy: {best['accuracy']:.4f}\n"
        f"- 해당 모델 Weighted-F1: {best['f1_weighted']:.4f}\n\n"
        "## 주요 오분류 조합\n\n"
        f"{top_pairs or '- 오분류 조합 없음'}\n"
    )
    (output_dir / "analysis_summary.md").write_text(analysis, encoding="utf-8")
