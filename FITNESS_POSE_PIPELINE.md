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

## 논문 문서 산출물

2026-06-09 기준으로 기존 3개 운동 실험 중심 문서를 AI-Hub 맨몸운동 17개
라벨 실험 기준으로 재작성하였다. 본문에는 연구 배경, 연구 목적, 연구 범위,
연구 방법론, 활용 데이터, 모델 구조와 평가 기준을 새 데이터셋 및 새 모델
구성에 맞게 반영하였다.

- 본문수정본: `공학논문연구_실험결과_AIHub_본문수정본.docx`
- 표그림첨부 검수수정본: `공학논문연구_실험결과_AIHub_표그림첨부본_검수수정본.docx`
- 검수용 추출 텍스트: `outputs_fitness_pose/thesis_docx_review_text_after_formula_fix.txt`

검수수정본에는 논문용 표 9개와 그림 8개를 첨부하였다. 그림 1~2는 방법론
보강을 위한 전체 파이프라인과 모델별 입력 구조 비교 그림이며, 그림 3~8은
최종 실험 결과 그림이다. 표 1~3은 빈 frame을
가진 JSON을 제외한 실제 유효 샘플 16,408개 기준으로 재생성했으며,
Train/Validation/Test 분할은 촬영 세션 단위로 정리하였다. 문서 내부 XML과
그림 네임스페이스를 확인하여 Word에서 열릴 수 있는 docx 구조로 검수하였다.

2026-06-10에는 피드백에 따라 3.1 연구 방법론에 입력 특징 구성 수식,
SVM/XGBoost/GNN/Transformer 모델별 핵심 수식, Cross-Entropy 및 평가 지표
수식을 추가하였다. 수식은 Word 호환성을 위해 본문 텍스트 수식 형태로
작성하였다.

### 2026-06-10 방법론 보강 피드백 및 반영 기록

사용자 피드백/프롬프트:

> 방법론 수식추가 모델설명
> 사용한 모델들에 대한 구체적인 수식 및 모델 설명, 그림 등이 있어야 함.

반영 방향:

- `3.1.1) 입력 데이터 및 특징 구성`을 추가하여 `-3d.json` 파일 하나를
  하나의 운동 수행 샘플로 정의하고, 프레임별 3D 관절 좌표와 관절별 통계
  특징을 수식으로 설명하였다.
- `3.1.2) 모델별 구조 및 수식`을 추가하여 SVM의 RBF kernel, XGBoost의
  boosting 예측식과 목적함수, GNN의 정규화 인접행렬과 그래프 합성곱,
  Transformer의 self-attention 수식을 정리하였다.
- `3.1.3) 학습 및 평가 지표`를 추가하여 Cross-Entropy Loss, Accuracy,
  Precision, Recall, F1-score, Macro-F1, Weighted-F1 수식을 정리하였다.
- 방법론 그림 2개를 생성하여 논문에 첨부하였다.
  - `figure_method_01_pipeline.png`: 관절 좌표 기반 운동 동작 분류 전체 파이프라인
  - `figure_method_02_model_structures.png`: 모델별 입력 구조 비교
- 기존 결과 그림 번호와 충돌하지 않도록 방법론 그림을 그림 1~2로 배치하고,
  기존 실험 결과 그림은 그림 3~8로 재정렬하였다.

최종 검수:

- 최종 문서: `공학논문연구_실험결과_AIHub_표그림첨부본_검수수정본.docx`
- 검수 결과: docx 구조 정상, XML 파싱 정상, 표 9개, 그림 8개, 방법론 수식 포함,
  placeholder 제거, 이전 프로젝트 용어 잔존 없음.
