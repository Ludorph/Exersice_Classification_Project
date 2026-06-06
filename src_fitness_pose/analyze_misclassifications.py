from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SEEDS = [42, 7, 21, 100, 2026]
DEFAULT_MODELS = ["svm", "xgboost", "transformer", "gnn"]


LABEL_GROUPS = {
    "스탠딩 사이드 크런치": "standing_core",
    "스탠딩 니업": "standing_core",
    "버피 테스트": "whole_body",
    "스텝 포워드 다이나믹 런지": "lunge",
    "스텝 백워드 다이나믹 런지": "lunge",
    "사이드 런지": "lunge",
    "크로스 런지": "lunge",
    "굿모닝": "hinge",
    "라잉 레그 레이즈": "lying_core",
    "크런치": "lying_core",
    "바이시클 크런치": "lying_core",
    "시저크로스": "lying_core",
    "힙쓰러스트": "lying_core",
    "플랭크": "support_pose",
    "푸시업": "support_pose",
    "니푸쉬업": "support_pose",
    "Y - Exercise": "support_pose",
}


GROUP_NAMES_KO = {
    "standing_core": "서서 수행하는 코어 운동",
    "whole_body": "전신 복합 운동",
    "lunge": "런지 계열 운동",
    "hinge": "고관절 힌지 운동",
    "lying_core": "누운 자세 코어 운동",
    "support_pose": "상지 지지/엎드린 자세 운동",
    "unknown": "미분류",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and explain misclassification patterns across seed runs.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs_fitness_pose"))
    parser.add_argument("--experiment-base", type=str, default="bodyweight_17_tuned_all")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_fitness_pose") / "bodyweight_17_tuned_all_error_analysis")
    return parser.parse_args()


def pair_explanation(true_label: str, predicted_label: str) -> str:
    labels = {true_label, predicted_label}
    true_group = LABEL_GROUPS.get(true_label, "unknown")
    predicted_group = LABEL_GROUPS.get(predicted_label, "unknown")

    if labels == {"스텝 포워드 다이나믹 런지", "스텝 백워드 다이나믹 런지"}:
        return "두 동작 모두 런지 계열이며 하체 관절의 굴곡/신전 패턴이 유사하다. 전후 방향 차이는 관절 좌표 통계만으로 약하게 표현될 수 있다."
    if labels == {"푸시업", "니푸쉬업"}:
        return "상체 지지 자세와 팔 굽힘 동작이 거의 동일하며, 무릎 지지 여부가 핵심 차이라 일부 프레임 요약 특징에서는 구분이 어려울 수 있다."
    if labels <= {"크런치", "바이시클 크런치", "시저크로스", "라잉 레그 레이즈", "힙쓰러스트"}:
        return "누운 자세 기반 코어 운동끼리 몸통과 하체 관절 움직임이 겹친다. 특히 다리 움직임과 몸통 굴곡이 동시에 포함될 때 혼동이 발생하기 쉽다."
    if labels == {"사이드 런지", "크로스 런지"}:
        return "두 동작 모두 좌우 방향 하체 이동이 포함된 런지 계열이다. 골반과 무릎의 측면 이동 패턴이 유사하여 오분류가 발생할 수 있다."
    if labels == {"스탠딩 사이드 크런치", "스탠딩 니업"}:
        return "두 동작 모두 서서 수행하며 한쪽 무릎 상승과 몸통 움직임이 포함된다. 상체 측굴 여부가 충분히 분리되지 않으면 혼동될 수 있다."
    if true_group == predicted_group:
        return f"두 라벨 모두 {GROUP_NAMES_KO.get(true_group, true_group)}에 속한다. 시작 자세와 주요 관절 움직임이 유사하여 같은 운동군 내부 오분류로 해석할 수 있다."
    return "두 라벨은 서로 다른 운동군이지만, 특정 구간에서 유사한 관절 배치 또는 정적 자세가 나타나 모델이 혼동한 사례로 볼 수 있다."


def load_predictions(args: argparse.Namespace) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        result_dir = args.output_root / f"{args.experiment_base}_seed_{seed}" / "results"
        for model in args.models:
            path = result_dir / f"test_predictions_{model}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing prediction file: {path}")
            frame = pd.read_csv(path)
            frame.insert(0, "seed", seed)
            frame.insert(1, "model", model)
            frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    predictions["true_group"] = predictions["true_label"].map(LABEL_GROUPS).fillna("unknown")
    predictions["predicted_group"] = predictions["predicted_label"].map(LABEL_GROUPS).fillna("unknown")
    return predictions


def aggregate_pairs(predictions: pd.DataFrame) -> pd.DataFrame:
    errors = predictions.loc[~predictions["is_correct"].astype(bool)].copy()
    true_totals = (
        predictions.groupby(["model", "true_label"], as_index=False)
        .size()
        .rename(columns={"size": "true_total"})
    )
    pairs = (
        errors.groupby(["model", "true_label", "predicted_label", "true_group", "predicted_group"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    pairs = pairs.merge(true_totals, on=["model", "true_label"], how="left")
    pairs["error_rate_within_true_label"] = pairs["count"] / pairs["true_total"]
    pairs["explanation"] = [
        pair_explanation(true_label, predicted_label)
        for true_label, predicted_label in zip(pairs["true_label"], pairs["predicted_label"], strict=True)
    ]
    return pairs.sort_values(["model", "count"], ascending=[True, False])


def aggregate_label_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, true_label), frame in predictions.groupby(["model", "true_label"]):
        errors = frame.loc[~frame["is_correct"].astype(bool)]
        if errors.empty:
            top_wrong_label = ""
            top_wrong_count = 0
        else:
            wrong_counts = errors["predicted_label"].value_counts()
            top_wrong_label = str(wrong_counts.index[0])
            top_wrong_count = int(wrong_counts.iloc[0])
        rows.append(
            {
                "model": model,
                "true_label": true_label,
                "label_group": LABEL_GROUPS.get(true_label, "unknown"),
                "total_test_cases_across_seeds": int(len(frame)),
                "error_count": int(len(errors)),
                "error_rate": float(len(errors) / len(frame)),
                "top_wrong_label": top_wrong_label,
                "top_wrong_count": top_wrong_count,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "error_rate"], ascending=[True, False])


def aggregate_group_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    errors = predictions.loc[~predictions["is_correct"].astype(bool)]
    group_pairs = (
        errors.groupby(["model", "true_group", "predicted_group"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["model", "count"], ascending=[True, False])
    )
    group_pairs["true_group_ko"] = group_pairs["true_group"].map(GROUP_NAMES_KO)
    group_pairs["predicted_group_ko"] = group_pairs["predicted_group"].map(GROUP_NAMES_KO)
    return group_pairs


def aggregate_cross_model_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    cross = (
        pairs.groupby(["true_label", "predicted_label", "true_group", "predicted_group"], as_index=False)
        .agg(
            total_count=("count", "sum"),
            models_involved=("model", lambda values: ", ".join(sorted(set(values)))),
            model_count=("model", lambda values: len(set(values))),
        )
        .sort_values(["total_count", "model_count"], ascending=[False, False])
    )
    cross["explanation"] = [
        pair_explanation(true_label, predicted_label)
        for true_label, predicted_label in zip(cross["true_label"], cross["predicted_label"], strict=True)
    ]
    return cross


def write_markdown(
    output_dir: Path,
    pairs: pd.DataFrame,
    label_errors: pd.DataFrame,
    group_errors: pd.DataFrame,
    cross_pairs: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# 오분류 패턴 심화 분석",
        "",
        "본 분석은 튜닝된 최종 설정의 5개 random seed 실험 결과를 모두 합쳐 수행하였다. 따라서 단일 seed에서 우연히 발생한 오분류보다 반복적으로 나타나는 오분류 조합을 중심으로 해석할 수 있다.",
        "",
        "## 모델 공통 주요 오분류 조합",
        "",
        "| 실제 라벨 | 예측 라벨 | 총 오분류 수 | 관련 모델 | 해석 |",
        "|---|---|---:|---|---|",
    ]
    for _, row in cross_pairs.head(10).iterrows():
        lines.append(
            f"| {row['true_label']} | {row['predicted_label']} | {int(row['total_count'])} | "
            f"{row['models_involved']} | {row['explanation']} |"
        )

    lines.extend(["", "## 모델별 주요 오분류 조합", ""])
    for model, frame in pairs.groupby("model", sort=False):
        lines.extend(
            [
                f"### {model}",
                "",
                "| 실제 라벨 | 예측 라벨 | 오분류 수 | 실제 라벨 내 비율 | 해석 |",
                "|---|---|---:|---:|---|",
            ]
        )
        for _, row in frame.head(8).iterrows():
            lines.append(
                f"| {row['true_label']} | {row['predicted_label']} | {int(row['count'])} | "
                f"{row['error_rate_within_true_label']:.3f} | {row['explanation']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 오분류 경향 요약",
            "",
            "- 가장 반복적으로 나타난 오분류는 런지 계열 내부, 푸시업/니푸쉬업, 누운 자세 코어 운동 내부에서 발생하였다.",
            "- 이는 모델이 전혀 무관한 운동을 무작위로 혼동했다기보다, 시작 자세와 주요 관절 움직임이 유사한 운동군 내부에서 혼동했음을 의미한다.",
            "- SVM과 XGBoost는 전체 성능이 높았지만, 세밀한 자세 차이가 필요한 라벨에서는 여전히 오분류가 발생하였다.",
            "- GNN과 Transformer도 유사 운동군 내부 오분류가 많이 나타났으며, 이는 관절 구조나 attention을 사용하더라도 요약 통계 특징만으로는 방향성·지지 방식 차이를 완전히 분리하기 어렵다는 점을 보여준다.",
            "",
            "## 논문용 해석 문장",
            "",
            "> 오분류 분석 결과, 주요 오분류는 무작위적인 라벨 간 혼동보다는 동작 구조가 유사한 운동군 내부에서 주로 발생하였다. 특히 스텝 포워드/백워드 다이나믹 런지, 푸시업/니푸쉬업, 크런치 계열 운동에서 반복적인 오분류가 확인되었다. 이는 관절 좌표 기반 통계 특징이 전체적인 자세와 움직임 패턴을 효과적으로 반영하는 반면, 운동 방향이나 지지 방식처럼 세밀한 차이를 구분하는 데에는 한계가 있음을 시사한다.",
        ]
    )
    output_dir.joinpath("misclassification_deep_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(args)
    pairs = aggregate_pairs(predictions)
    label_errors = aggregate_label_errors(predictions)
    group_errors = aggregate_group_errors(predictions)
    cross_pairs = aggregate_cross_model_pairs(pairs)

    predictions.to_csv(args.output_dir / "all_seed_test_predictions_long.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(args.output_dir / "misclassification_pairs_by_model.csv", index=False, encoding="utf-8-sig")
    pairs.groupby("model", group_keys=False).head(10).to_csv(
        args.output_dir / "top10_misclassification_pairs_by_model.csv",
        index=False,
        encoding="utf-8-sig",
    )
    label_errors.to_csv(args.output_dir / "label_error_summary_by_model.csv", index=False, encoding="utf-8-sig")
    group_errors.to_csv(args.output_dir / "group_misclassification_summary.csv", index=False, encoding="utf-8-sig")
    cross_pairs.to_csv(args.output_dir / "cross_model_misclassification_pairs.csv", index=False, encoding="utf-8-sig")
    write_markdown(args.output_dir, pairs, label_errors, group_errors, cross_pairs)

    print(f"Completed. Misclassification analysis: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
