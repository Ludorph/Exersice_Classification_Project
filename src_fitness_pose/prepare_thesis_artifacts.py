from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT = Path("outputs_fitness_pose") / "thesis_tables_figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect thesis-ready tables and figures from final experiment outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--final-seed", type=int, default=42)
    return parser.parse_args()


def write_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    path.with_suffix(".md").write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")


def score_pm(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def model_order(frame: pd.DataFrame) -> pd.DataFrame:
    order = ["svm", "xgboost", "transformer", "gnn"]
    return frame.assign(_order=frame["model"].map({model: index for index, model in enumerate(order)})).sort_values("_order").drop(columns="_order")


def hyperparameter_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in model_order(frame).iterrows():
        model = row["model"]
        if model == "svm":
            summary = f"kernel={row['kernel']}, C={row['C']:.1f}, gamma={row['gamma']}, class_weight={row['class_weight']}"
            structure = "RBF kernel SVM"
        elif model == "xgboost":
            summary = (
                f"n_estimators={int(row['n_estimators'])}, max_depth={int(row['max_depth'])}, "
                f"learning_rate={row['learning_rate']:.2f}, subsample={row['subsample']:.2f}, "
                f"colsample_bytree={row['colsample_bytree']:.2f}, reg_lambda={row['reg_lambda']:.1f}"
            )
            structure = "Gradient boosted decision trees"
        elif model == "gnn":
            summary = (
                f"hidden_dim={int(row['hidden_dim'])}, num_layers={int(row['num_layers'])}, dropout={row['dropout']:.1f}, "
                f"learning_rate={row['learning_rate']:.4f}, batch_size={int(row['batch_size'])}, epochs={int(row['epochs'])}, patience={int(row['patience'])}"
            )
            structure = f"Graph neural network, parameters={int(row['parameters'])}"
        elif model == "transformer":
            summary = (
                f"d_model={int(row['d_model'])}, num_heads={int(row['num_heads'])}, num_layers={int(row['num_layers'])}, "
                f"dim_feedforward={int(row['dim_feedforward'])}, dropout={row['dropout']:.1f}, "
                f"learning_rate={row['learning_rate']:.4f}, batch_size={int(row['batch_size'])}, epochs={int(row['epochs'])}, patience={int(row['patience'])}"
            )
            structure = f"Joint-token Transformer, parameters={int(row['parameters'])}"
        else:
            structure = ""
            summary = ""
        rows.append({"model": model, "model_structure": structure, "main_hyperparameters": summary})
    return pd.DataFrame(rows)


def copy_figure(source: Path, figure_dir: Path, new_name: str, rows: list[dict[str, str]], placement: str, note: str) -> None:
    target = figure_dir / new_name
    shutil.copy2(source, target)
    rows.append(
        {
            "figure_file": str(target.relative_to(figure_dir.parent)).replace("\\", "/"),
            "source_file": str(source).replace("\\", "/"),
            "recommended_placement": placement,
            "note": note,
        }
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    for path in (table_dir, figure_dir):
        path.mkdir(parents=True, exist_ok=True)

    final_seed_dir = Path("outputs_fitness_pose") / f"bodyweight_17_tuned_all_seed_{args.final_seed}"
    final_seed_data = final_seed_dir / "data"
    final_seed_results = final_seed_dir / "results"
    repeated_dir = Path("outputs_fitness_pose") / "bodyweight_17_tuned_all_repeated_seeds"
    baseline_repeated_dir = Path("outputs_fitness_pose") / "bodyweight_17_full_repeated_seeds"
    error_dir = Path("outputs_fitness_pose") / "bodyweight_17_tuned_all_error_analysis"
    pose_group_dir = Path("outputs_fitness_pose") / "bodyweight_17_tuned_all_pose_group_analysis"

    experiment_metadata = pd.read_csv(final_seed_data / "experiment_metadata.csv")
    dataset_summary = (
        experiment_metadata.groupby(["exercise_label", "pose"], as_index=False)
        .agg(samples=("sample_id", "size"), sessions=("session_id", "nunique"))
        .sort_values("exercise_label")
    )
    write_table(dataset_summary, table_dir / "table_01_dataset_label_summary")

    split_overall = (
        experiment_metadata.groupby("split", as_index=False)
        .agg(samples=("sample_id", "size"), sessions=("session_id", "nunique"))
    )
    split_order = {"train": 0, "validation": 1, "test": 2}
    split_overall = split_overall.assign(_order=split_overall["split"].map(split_order)).sort_values("_order").drop(columns="_order")
    write_table(split_overall, table_dir / "table_02_split_overall")

    split_distribution = (
        experiment_metadata.groupby(["exercise_label", "split"], as_index=False)
        .agg(samples=("sample_id", "size"))
    )
    split_pivot = split_distribution.pivot(index="exercise_label", columns="split", values="samples").reset_index().fillna(0)
    for column in ("train", "validation", "test"):
        if column in split_pivot:
            split_pivot[column] = split_pivot[column].astype(int)
    ordered_columns = ["exercise_label", "train", "validation", "test"]
    split_pivot = split_pivot[[column for column in ordered_columns if column in split_pivot]]
    split_pivot["total"] = split_pivot[[column for column in ("train", "validation", "test") if column in split_pivot]].sum(axis=1)
    write_table(split_pivot, table_dir / "table_03_split_by_label")

    hyperparameters = pd.read_csv(final_seed_dir / "model_hyperparameters.csv")
    write_table(hyperparameter_summary(hyperparameters), table_dir / "table_04_final_model_hyperparameters")

    final_metrics = model_order(pd.read_csv(repeated_dir / "metrics_mean_std.csv"))
    final_metrics_table = pd.DataFrame(
        {
            "model": final_metrics["model"],
            "accuracy": [score_pm(mean, std) for mean, std in zip(final_metrics["accuracy_mean"], final_metrics["accuracy_std"], strict=True)],
            "macro_f1": [score_pm(mean, std) for mean, std in zip(final_metrics["f1_macro_mean"], final_metrics["f1_macro_std"], strict=True)],
            "weighted_f1": [score_pm(mean, std) for mean, std in zip(final_metrics["f1_weighted_mean"], final_metrics["f1_weighted_std"], strict=True)],
        }
    )
    write_table(final_metrics_table, table_dir / "table_05_final_repeated_seed_performance")

    baseline = pd.read_csv(baseline_repeated_dir / "metrics_mean_std.csv")[["model", "f1_macro_mean"]].rename(columns={"f1_macro_mean": "before_tuning_macro_f1"})
    tuned = pd.read_csv(repeated_dir / "metrics_mean_std.csv")[["model", "f1_macro_mean"]].rename(columns={"f1_macro_mean": "after_tuning_macro_f1"})
    tuning_comparison = model_order(baseline.merge(tuned, on="model"))
    tuning_comparison["change"] = tuning_comparison["after_tuning_macro_f1"] - tuning_comparison["before_tuning_macro_f1"]
    for column in ("before_tuning_macro_f1", "after_tuning_macro_f1", "change"):
        tuning_comparison[column] = tuning_comparison[column].map(lambda value: f"{value:.4f}")
    write_table(tuning_comparison, table_dir / "table_06_tuning_before_after_macro_f1")

    common_errors = pd.read_csv(error_dir / "cross_model_misclassification_pairs.csv").head(12)
    write_table(
        common_errors[["true_label", "predicted_label", "total_count", "models_involved", "explanation"]],
        table_dir / "table_07_common_misclassification_patterns",
    )

    pose_ranking = pd.read_csv(pose_group_dir / "pose_group_model_ranking.csv")
    best_by_group = pose_ranking.loc[pose_ranking["rank_within_group"] == 1, ["group_ko", "model", "accuracy", "f1_macro", "support"]].copy()
    for column in ("accuracy", "f1_macro"):
        best_by_group[column] = best_by_group[column].map(lambda value: f"{value:.4f}")
    write_table(best_by_group, table_dir / "table_08_pose_group_best_models")

    pose_performance = pd.read_csv(pose_group_dir / "pose_group_performance_by_model.csv")
    pose_pivot = pose_performance.pivot(index="group_ko", columns="model", values="f1_macro").reset_index()
    for column in pose_pivot.columns:
        if column != "group_ko":
            pose_pivot[column] = pose_pivot[column].map(lambda value: f"{value:.4f}")
    write_table(pose_pivot, table_dir / "table_09_pose_group_macro_f1_by_model")

    figure_rows: list[dict[str, str]] = []
    copy_figure(
        repeated_dir / "repeated_seed_model_comparison.png",
        figure_dir,
        "figure_01_final_repeated_seed_model_comparison.png",
        figure_rows,
        "3.3 실험 결과 및 분석",
        "최종 튜닝 설정의 5-seed 평균 성능과 표준편차를 보여주는 핵심 성능 비교 그림",
    )
    copy_figure(
        pose_group_dir / "pose_group_macro_f1_by_model.png",
        figure_dir,
        "figure_02_pose_group_macro_f1_by_model.png",
        figure_rows,
        "3.3 실험 결과 및 분석 또는 세부 분석",
        "운동군별로 어떤 모델이 강한지 보여주는 자세 그룹 분석 그림",
    )
    copy_figure(
        pose_group_dir / "same_group_prediction_rate_heatmap.png",
        figure_dir,
        "figure_03_same_group_prediction_rate_heatmap.png",
        figure_rows,
        "오분류/자세 그룹 분석",
        "모델이 실제 운동군을 같은 운동군으로 예측하는 비율을 보여주는 heatmap",
    )
    copy_figure(
        final_seed_results / "confusion_matrix_svm.png",
        figure_dir,
        "figure_04_representative_confusion_matrix_svm_seed42.png",
        figure_rows,
        "오분류 분석",
        "최고 성능 모델인 SVM의 대표 seed 42 혼동행렬. 반복 seed 집계 표와 함께 보조적으로 사용",
    )
    copy_figure(
        final_seed_results / "confusion_matrix_xgboost.png",
        figure_dir,
        "figure_05_representative_confusion_matrix_xgboost_seed42.png",
        figure_rows,
        "오분류 분석 또는 부록",
        "XGBoost의 대표 seed 42 혼동행렬. SVM과 비교할 때 사용",
    )
    copy_figure(
        final_seed_results / "model_comparison_per_class_f1.png",
        figure_dir,
        "figure_06_representative_per_class_f1_seed42.png",
        figure_rows,
        "부록 또는 라벨별 분석",
        "seed 42 기준 라벨별 F1-score. 본문이 길어지면 부록 권장",
    )
    figure_index = pd.DataFrame(figure_rows)
    write_table(figure_index, table_dir / "figure_index")

    readme = f"""# 논문용 표·그림 확정안

이 폴더는 최종 실험 결과를 논문에 넣기 쉽도록 정리한 것이다.

## 본문 우선 첨부 표

1. `tables/table_01_dataset_label_summary.md`: 데이터셋 라벨 구성
2. `tables/table_02_split_overall.md`: Train/Validation/Test 전체 분할
3. `tables/table_04_final_model_hyperparameters.md`: 최종 모델 구조 및 하이퍼파라미터
4. `tables/table_05_final_repeated_seed_performance.md`: 최종 5-seed 성능 비교
5. `tables/table_07_common_misclassification_patterns.md`: 주요 오분류 조합
6. `tables/table_08_pose_group_best_models.md`: 자세 그룹별 최고 모델

## 본문 우선 첨부 그림

1. `figures/figure_01_final_repeated_seed_model_comparison.png`
2. `figures/figure_02_pose_group_macro_f1_by_model.png`
3. `figures/figure_04_representative_confusion_matrix_svm_seed42.png`

## 부록 또는 보조 자료 권장

- `tables/table_03_split_by_label.md`
- `tables/table_06_tuning_before_after_macro_f1.md`
- `tables/table_09_pose_group_macro_f1_by_model.md`
- `figures/figure_03_same_group_prediction_rate_heatmap.png`
- `figures/figure_05_representative_confusion_matrix_xgboost_seed42.png`
- `figures/figure_06_representative_per_class_f1_seed42.png`

## 주의할 점

혼동행렬과 라벨별 F1 그림은 대표 seed {args.final_seed} 결과이다. 최종 성능 수치는 5개 seed 평균을 사용해야 하므로, 본문에서는 `table_05_final_repeated_seed_performance`를 최종 성능 표로 사용한다.
"""
    (output_dir / "README_THESIS_ARTIFACTS.md").write_text(readme, encoding="utf-8")
    print(f"Completed. Thesis artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
