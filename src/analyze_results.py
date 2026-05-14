from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_ORDER = ["xgboost", "lstm", "stgcn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze model comparison and misclassification patterns.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def read_metrics(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run train_pipeline.py first.")
    metrics = pd.read_csv(path)
    metrics["model"] = pd.Categorical(metrics["model"], categories=MODEL_ORDER, ordered=True)
    return metrics.sort_values("model")


def read_class_reports(output_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        path = output_dir / f"classification_report_{model}.csv"
        if not path.exists():
            continue
        report = pd.read_csv(path, index_col=0)
        class_rows = report[~report.index.isin(["accuracy", "macro avg", "weighted avg"])].copy()
        class_rows = class_rows.reset_index().rename(columns={"index": "class"})
        class_rows.insert(0, "model", model)
        frames.append(class_rows)
    if not frames:
        raise FileNotFoundError("No classification_report_*.csv files found.")
    return pd.concat(frames, ignore_index=True)


def read_predictions(output_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        path = output_dir / f"test_predictions_{model}.csv"
        if not path.exists():
            continue
        pred = pd.read_csv(path)
        if pred["is_correct"].dtype == object:
            pred["is_correct"] = pred["is_correct"].astype(str).str.lower().map({"true": True, "false": False})
        pred.insert(0, "model", model)
        frames.append(pred)
    if not frames:
        raise FileNotFoundError("No test_predictions_*.csv files found.")
    return pd.concat(frames, ignore_index=True)


def plot_overall_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    long_df = metrics.melt(id_vars="model", value_vars=metric_cols, var_name="metric", value_name="score")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=long_df, x="metric", y="score", hue="model")
    plt.ylim(0, 1.05)
    plt.xlabel("")
    plt.ylabel("Score")
    plt.title("Overall Model Performance")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_overall.png", dpi=180)
    plt.close()


def plot_per_class_f1(class_reports: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=class_reports, x="class", y="f1-score", hue="model")
    plt.ylim(0, 1.05)
    plt.xlabel("Class")
    plt.ylabel("F1-score")
    plt.title("Per-Class F1-score by Model")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_per_class_f1.png", dpi=180)
    plt.close()


def build_error_pairs(predictions: pd.DataFrame) -> pd.DataFrame:
    errors = predictions[predictions["is_correct"] == False].copy()
    if errors.empty:
        return pd.DataFrame(columns=["model", "true_label", "pred_label", "count"])
    return (
        errors.groupby(["model", "true_label", "pred_label"], observed=False)
        .size()
        .reset_index(name="count")
        .sort_values(["model", "count"], ascending=[True, False])
    )


def plot_error_pair_heatmaps(error_pairs: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    class_names = sorted(predictions["true_label"].unique())
    for model in MODEL_ORDER:
        model_errors = error_pairs[error_pairs["model"] == model]
        matrix = pd.DataFrame(0, index=class_names, columns=class_names)
        for _, row in model_errors.iterrows():
            matrix.loc[row["true_label"], row["pred_label"]] = int(row["count"])

        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Reds")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"Misclassification Pattern - {model}")
        plt.tight_layout()
        plt.savefig(output_dir / f"misclassification_pattern_{model}.png", dpi=180)
        plt.close()


def build_video_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    pivot = predictions.pivot_table(
        index=["vid_id", "true_label"],
        columns="model",
        values="is_correct",
        aggfunc="first",
    ).reset_index()
    for model in MODEL_ORDER:
        if model not in pivot:
            pivot[model] = True
    pivot["num_models_wrong"] = sum((pivot[model] == False).astype(int) for model in MODEL_ORDER)
    return pivot.sort_values(["num_models_wrong", "vid_id"], ascending=[False, True])


def write_markdown_summary(
    metrics: pd.DataFrame,
    class_reports: pd.DataFrame,
    error_pairs: pd.DataFrame,
    video_errors: pd.DataFrame,
    output_dir: Path,
) -> None:
    best_accuracy = metrics.sort_values("accuracy", ascending=False).iloc[0]
    best_f1 = metrics.sort_values("f1_macro", ascending=False).iloc[0]

    lines: list[str] = []
    lines.append("# 모델별 성능 비교 및 오분류 패턴 분석")
    lines.append("")
    lines.append("## 전체 성능 비교")
    lines.append("")
    lines.append(
        f"- Accuracy 기준 최고 모델은 `{best_accuracy['model']}`이며, accuracy는 `{best_accuracy['accuracy']:.4f}`입니다."
    )
    lines.append(f"- Macro F1 기준 최고 모델은 `{best_f1['model']}`이며, macro F1은 `{best_f1['f1_macro']:.4f}`입니다.")
    lines.append("")
    lines.append(metrics.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 동작별 F1-score")
    lines.append("")
    per_class = class_reports.pivot_table(index="class", columns="model", values="f1-score", observed=False)
    lines.append(per_class.to_markdown(floatfmt=".4f"))
    lines.append("")

    lines.append("## 주요 오분류 패턴")
    lines.append("")
    if error_pairs.empty:
        lines.append("- 모든 모델에서 test set 오분류가 발생하지 않았습니다.")
    else:
        for model in MODEL_ORDER:
            subset = error_pairs[error_pairs["model"] == model].head(5)
            if subset.empty:
                lines.append(f"- `{model}`: 오분류 없음")
                continue
            pattern_text = ", ".join(
                f"{row.true_label} -> {row.pred_label} ({int(row['count'])}건)" for _, row in subset.iterrows()
            )
            lines.append(f"- `{model}`: {pattern_text}")
    lines.append("")

    hard_cases = video_errors[video_errors["num_models_wrong"] >= 2]
    lines.append("## 공통 난분류 비디오")
    lines.append("")
    if hard_cases.empty:
        lines.append("- 두 개 이상의 모델이 동시에 틀린 test 비디오는 없습니다.")
    else:
        cols = ["vid_id", "true_label", *MODEL_ORDER, "num_models_wrong"]
        lines.append(hard_cases[cols].to_markdown(index=False))
    lines.append("")

    lines.append("## 해석에 활용할 파일")
    lines.append("")
    lines.append("- `model_comparison_overall.png`: 전체 성능 막대그래프")
    lines.append("- `model_comparison_per_class_f1.png`: 동작별 F1-score 비교")
    lines.append("- `misclassification_pattern_*.png`: 모델별 오분류 방향 heatmap")
    lines.append("- `misclassification_pairs.csv`: 실제 라벨과 예측 라벨 조합별 오분류 횟수")
    lines.append("- `hard_cases_by_video.csv`: 여러 모델이 동시에 틀린 비디오 목록")

    (output_dir / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = read_metrics(output_dir)
    class_reports = read_class_reports(output_dir)
    predictions = read_predictions(output_dir)

    error_pairs = build_error_pairs(predictions)
    video_errors = build_video_error_summary(predictions)

    class_reports.to_csv(output_dir / "per_class_metrics_long.csv", index=False, encoding="utf-8-sig")
    error_pairs.to_csv(output_dir / "misclassification_pairs.csv", index=False, encoding="utf-8-sig")
    video_errors.to_csv(output_dir / "hard_cases_by_video.csv", index=False, encoding="utf-8-sig")

    plot_overall_metrics(metrics, output_dir)
    plot_per_class_f1(class_reports, output_dir)
    plot_error_pair_heatmaps(error_pairs, predictions, output_dir)
    write_markdown_summary(metrics, class_reports, error_pairs, video_errors, output_dir)

    print(f"Analysis files saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
