# AI-Hub 맨몸운동 17개 라벨 모델 비교 파이프라인

기존 `src/`와 `outputs/`는 보존한다. 새 실험은 `src_fitness_pose/`,
`configs_fitness_pose/`, `outputs_fitness_pose/`를 사용한다.

## 실험 정의

- 입력: `bodyweight_labeling_new_220128` 폴더의 `-3d.json`
- 샘플 단위: JSON 파일 하나
- 최종 라벨: 맨몸운동 17개 운동명
- 입력 특징: 각 관절 좌표를 신체 중심과 크기로 정규화한 뒤 프레임 통계로 요약
- 모델: XGBoost, SVM, GNN, Transformer
- 분할: 촬영 세션 단위 Train/Validation/Test 분할
- 평가: Accuracy, Macro-F1, Weighted-F1, 분류 보고서, Confusion Matrix, 주요 오분류 조합

공식 Validation 맨몸 데이터는 한 촬영 세션과 9개 라벨만 포함하므로, 17개
주 실험의 Test로 직접 사용하지 않는다. 파이프라인은 학습 데이터의 16개
세션을 조합하여 모든 분할에 17개 라벨이 존재하는 분할을 자동 탐색한다.

## 실행

이 컴퓨터에서 확인된 Python 실행 경로:

```powershell
& 'C:\Users\Ludorph\AppData\Local\Python\bin\python.exe' -m src_fitness_pose.build_metadata
```

전체 파이프라인이 연결되는지 빠르게 확인:

```powershell
& 'C:\Users\Ludorph\AppData\Local\Python\bin\python.exe' -m src_fitness_pose.train_pipeline --smoke
```

전체 실험:

```powershell
& 'C:\Users\Ludorph\AppData\Local\Python\bin\python.exe' -m src_fitness_pose.train_pipeline
```

특정 모델만 실행:

```powershell
& 'C:\Users\Ludorph\AppData\Local\Python\bin\python.exe' -m src_fitness_pose.train_pipeline --models xgboost svm
```

특징 캐시를 다시 생성하려면 `--force-features`를 추가한다.

## 주요 결과 파일

```text
outputs_fitness_pose/bodyweight_17_full/
  experiment_config_actual.json
  model_hyperparameters.csv
  data/
    metadata.csv
    serial_label_mapping.csv
    dataset_summary.csv
    metadata_with_split.csv
    split_sessions.csv
    split_label_distribution.csv
    split_info.json
    features_all.npz
    preprocessing.pkl
  results/
    metrics_summary.csv
    classification_report_*.csv
    confusion_matrix_*.csv
    confusion_matrix_*.png
    test_predictions_*.csv
    misclassification_pairs_*.csv
    model_comparison_overall.png
  checkpoints/
```

`experiment_config_actual.json`과 `model_hyperparameters.csv`는 논문에 기재할
모델 변수, 모델 구조, 하이퍼파라미터 및 실험 조건을 기록한다.
