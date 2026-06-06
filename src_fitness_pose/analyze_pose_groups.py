from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .analyze_misclassifications import DEFAULT_MODELS, DEFAULT_SEEDS, GROUP_NAMES_KO, LABEL_GROUPS, load_predictions


GROUP_ORDER = ["standing_core", "whole_body", "lunge", "hinge", "lying_core", "support_pose"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze model performance by exercise posture/movement group.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs_fitness_pose"))
    parser.add_argument("--experiment-base", type=str, default="bodyweight_17_tuned_all")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_fitness_pose") / "bodyweight_17_tuned_all_pose_group_analysis")
    return parser.parse_args()


def labels_by_group() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for label, group in LABEL_GROUPS.items():
        groups.setdefault(group, []).append(label)
    return groups


def group_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped_labels = labels_by_group()
    rows: list[dict] = []
    for model, model_frame in predictions.groupby("model"):
        for group in GROUP_ORDER:
            frame = model_frame.loc[model_frame["true_group"] == group]
            if frame.empty:
                continue
            labels = grouped_labels[group]
            y_true = frame["true_label"]
            y_pred = frame["predicted_label"]
            rows.append(
                {
                    "model": model,
                    "group": group,
                    "group_ko": GROUP_NAMES_KO[group],
                    "labels": ", ".join(labels),
                    "support": int(len(frame)),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
                    "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
                    "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
                }
            )
    return pd.DataFrame(rows).sort_values(["group", "f1_macro"], ascending=[True, False])


def group_prediction_flow(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, true_group), frame in predictions.groupby(["model", "true_group"]):
        total = len(frame)
        counts = frame.groupby("predicted_group").size()
        for predicted_group, count in counts.items():
            rows.append(
                {
                    "model": model,
                    "true_group": true_group,
                    "true_group_ko": GROUP_NAMES_KO.get(true_group, true_group),
                    "predicted_group": predicted_group,
                    "predicted_group_ko": GROUP_NAMES_KO.get(predicted_group, predicted_group),
                    "count": int(count),
                    "rate_within_true_group": float(count / total),
                    "is_same_group": bool(true_group == predicted_group),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "true_group", "count"], ascending=[True, True, False])


def group_ranking(performance: pd.DataFrame) -> pd.DataFrame:
    ranking = performance.sort_values(["group", "f1_macro", "accuracy"], ascending=[True, False, False]).copy()
    ranking["rank_within_group"] = ranking.groupby("group").cumcount() + 1
    return ranking


def write_plots(performance: pd.DataFrame, flow: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plot_frame = performance.copy()
    plot_frame["group_label"] = plot_frame["group"].map(GROUP_NAMES_KO)
    for metric, filename, title in [
        ("accuracy", "pose_group_accuracy_by_model.png", "Pose Group Accuracy by Model"),
        ("f1_macro", "pose_group_macro_f1_by_model.png", "Pose Group Macro-F1 by Model"),
    ]:
        plt.figure(figsize=(13, 6))
        sns.barplot(data=plot_frame, x="group_label", y=metric, hue="model", order=[GROUP_NAMES_KO[g] for g in GROUP_ORDER])
        plt.ylim(0, 1)
        plt.xlabel("Exercise group")
        plt.ylabel(metric)
        plt.title(title)
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()

    same_group = flow.loc[flow["is_same_group"]].copy()
    same_group["group_label"] = same_group["true_group"].map(GROUP_NAMES_KO)
    pivot = same_group.pivot(index="group_label", columns="model", values="rate_within_true_group")
    pivot = pivot.reindex([GROUP_NAMES_KO[g] for g in GROUP_ORDER])
    plt.figure(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1)
    plt.title("Same-group Prediction Rate")
    plt.xlabel("Model")
    plt.ylabel("True exercise group")
    plt.tight_layout()
    plt.savefig(output_dir / "same_group_prediction_rate_heatmap.png", dpi=200)
    plt.close()


def write_markdown(performance: pd.DataFrame, flow: pd.DataFrame, ranking: pd.DataFrame, output_dir: Path) -> None:
    lines: list[str] = [
        "# 자세 그룹별 분석",
        "",
        "본 분석은 최종 튜닝 설정의 5개 random seed Test 예측 결과를 모두 합산하여 수행하였다. 17개 운동 라벨을 자세와 동작 특성에 따라 6개 운동군으로 묶고, 모델별 성능을 그룹 단위로 비교하였다.",
        "",
        "## 운동군 정의",
        "",
        "| 운동군 | 포함 라벨 |",
        "|---|---|",
    ]
    grouped_labels = labels_by_group()
    for group in GROUP_ORDER:
        lines.append(f"| {GROUP_NAMES_KO[group]} | {', '.join(grouped_labels[group])} |")

    lines.extend(["", "## 그룹별 최고 모델", "", "| 운동군 | 최고 모델 | Macro-F1 | Accuracy |", "|---|---|---:|---:|"])
    for _, row in ranking.loc[ranking["rank_within_group"] == 1].iterrows():
        lines.append(f"| {row['group_ko']} | {row['model']} | {row['f1_macro']:.4f} | {row['accuracy']:.4f} |")

    lines.extend(["", "## 모델별 그룹 성능", ""])
    for model, frame in performance.sort_values(["model", "group"]).groupby("model"):
        lines.extend([f"### {model}", "", "| 운동군 | Accuracy | Macro-F1 | Support |", "|---|---:|---:|---:|"])
        for group in GROUP_ORDER:
            row = frame.loc[frame["group"] == group]
            if row.empty:
                continue
            record = row.iloc[0]
            lines.append(f"| {record['group_ko']} | {record['accuracy']:.4f} | {record['f1_macro']:.4f} | {int(record['support'])} |")
        lines.append("")

    same_group = flow.loc[flow["is_same_group"]].copy()
    lines.extend(["## 그룹 단위 해석", ""])
    for group in GROUP_ORDER:
        group_frame = performance.loc[performance["group"] == group].sort_values("f1_macro", ascending=False)
        best = group_frame.iloc[0]
        worst = group_frame.iloc[-1]
        same_rates = same_group.loc[same_group["true_group"] == group]
        avg_same_rate = same_rates["rate_within_true_group"].mean() if not same_rates.empty else 0.0
        lines.append(
            f"- {GROUP_NAMES_KO[group]}: 가장 높은 Macro-F1은 {best['model']}의 {best['f1_macro']:.4f}이고, "
            f"가장 낮은 Macro-F1은 {worst['model']}의 {worst['f1_macro']:.4f}이다. "
            f"평균적으로 같은 운동군으로 예측된 비율은 {avg_same_rate:.4f}이다."
        )

    lines.extend(
        [
            "",
            "## 논문용 해석 문장",
            "",
            "> 자세 그룹별 분석 결과, 모델 성능은 운동군에 따라 차이를 보였다. 런지 계열, 누운 자세 코어 운동, 상지 지지/엎드린 자세 운동은 세부 라벨 간 관절 움직임이 유사하여 상대적으로 오분류가 많이 발생하였다. 반면 전신 복합 운동이나 고관절 힌지 운동처럼 다른 운동군과 자세 구조가 비교적 뚜렷한 경우에는 높은 분류 성능을 보였다. 이는 관절 좌표 기반 특징이 큰 자세군의 차이는 효과적으로 구분하지만, 같은 자세군 내부의 세밀한 운동 차이를 구분하는 데에는 한계가 있음을 보여준다.",
        ]
    )
    output_dir.joinpath("pose_group_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(args)
    performance = group_performance(predictions)
    flow = group_prediction_flow(predictions)
    ranking = group_ranking(performance)

    performance.to_csv(args.output_dir / "pose_group_performance_by_model.csv", index=False, encoding="utf-8-sig")
    flow.to_csv(args.output_dir / "pose_group_prediction_flow.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(args.output_dir / "pose_group_model_ranking.csv", index=False, encoding="utf-8-sig")
    write_plots(performance, flow, args.output_dir)
    write_markdown(performance, flow, ranking, args.output_dir)
    print(f"Completed. Pose group analysis: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
