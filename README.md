# Exercise Classification Project

AI-Hub 피트니스 자세 데이터셋의 3D 관절 좌표를 이용해 운동 동작을 분류하고,
여러 모델의 성능과 오분류 패턴을 비교하는 공학논문용 실험 프로젝트입니다.

프로그래밍에 익숙하지 않은 사람도 흐름을 이해할 수 있도록, 이 문서는
“무엇을 했는지”, “어떤 파일이 필요한지”, “어떤 순서로 실행하는지”를 중심으로
설명합니다.

## 1. 프로젝트가 하는 일

이 프로젝트는 운동 영상 자체를 바로 분석하지 않고, AI-Hub 데이터셋에 포함된
`-3d.json` 파일을 사용합니다. 이 JSON 파일에는 운동 중 여러 프레임에서 추출된
사람의 3D 관절 좌표가 들어 있습니다.

전체 흐름은 다음과 같습니다.

```text
AI-Hub -3d.json
→ 프레임별 3D 관절 좌표 추출
→ 관절 좌표 정규화
→ 관절별 통계 특징 생성
→ SVM / XGBoost / GNN / Transformer 모델 학습
→ 성능 비교와 오분류 분석
→ 논문용 표·그림 생성
```

이 프로젝트에서 비교한 모델은 다음 네 가지입니다.

| 모델 | 간단한 의미 |
|---|---|
| SVM | 관절 특징 벡터를 기준으로 운동 라벨을 구분하는 전통적 머신러닝 모델 |
| XGBoost | 여러 decision tree를 조합해 분류 성능을 높이는 앙상블 모델 |
| GNN | 관절을 그래프의 노드로 보고, 인체 골격 연결 관계를 반영하는 모델 |
| Transformer | 관절을 token처럼 보고, 관절 사이의 관계를 attention으로 학습하는 모델 |

## 2. 사용한 데이터

사용 데이터는 AI-Hub의 **피트니스 자세 이미지 데이터셋**입니다.

원본 데이터셋은 용량과 라이선스 문제 때문에 GitHub에 포함하지 않습니다.
직접 AI-Hub에서 다운로드한 뒤 아래 위치에 넣어야 합니다.

```text
dataset/fitness_pose/
```

코드는 이 폴더 안에서 다음 파일과 폴더를 찾습니다.

```text
dataset/fitness_pose/
  1.Training/
  Document/
    fitness_pose_naming_rules/
    source_data_list_and_scenario/
```

실험 대상은 맨몸운동 17개 라벨입니다. 각 `-3d.json` 파일 하나를 운동 수행
샘플 하나로 보고, 그 안의 여러 프레임 좌표를 요약해 모델 입력으로 사용합니다.

## 3. 폴더 구조

중요한 폴더와 파일은 다음과 같습니다.

```text
configs_fitness_pose/
  bodyweight_17.json
  bodyweight_17_tuned_all.json

src_fitness_pose/
  build_metadata.py
  train_pipeline.py
  repeated_seeds.py
  tune_tabular_models.py
  tune_deep_models.py
  analyze_misclassifications.py
  analyze_pose_groups.py
  prepare_thesis_artifacts.py
  rewrite_thesis_docx.py
  rewrite_thesis_format_docx.py
  attach_results_to_thesis_docx.py
  format_thesis_docx.py
  style_docx_tables.py
  export_reference_5_exercises.py

requirements_fitness_pose.txt
FITNESS_POSE_PIPELINE.md
```

폴더별 의미는 다음과 같습니다.

| 위치 | 의미 |
|---|---|
| `configs_fitness_pose/` | 실험 설정 파일 |
| `src_fitness_pose/` | 실제 실험 코드 |
| `outputs_fitness_pose/` | 모델 실험 결과, 반복 seed 결과, 튜닝 결과, 오분류 분석, 자세 그룹 분석, 논문용 표·그림 저장 위치 |
| `exports/` | 따로 뽑아낸 기준 데이터셋 저장 위치 |
| `dataset/` | AI-Hub 원본 데이터 위치 |
| `FITNESS_POSE_PIPELINE.md` | 날짜별 작업 기록과 자세한 실험 로그 |

`dataset/`, `outputs_fitness_pose/`, `exports/`는 GitHub에 올리지 않는 폴더입니다.
실행하면 각자 컴퓨터에서 생성되거나 직접 준비해야 합니다.

특히 `outputs_fitness_pose/`는 소스코드가 아니라 실험을 실행한 뒤 생기는 결과
폴더입니다. 최종 성능 CSV, 혼동행렬 그림, 오분류 분석 파일, 자세 그룹별 분석
파일, 논문에 첨부한 표와 그림이 이 폴더 아래에 저장됩니다.

## 4. 설치 방법

Python이 설치되어 있어야 합니다. Windows PowerShell 기준 예시는 다음과 같습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_fitness_pose.txt
```

이미 Python 환경이 준비되어 있다면 아래처럼 바로 패키지만 설치해도 됩니다.

```powershell
pip install -r requirements_fitness_pose.txt
```

## 5. 실행 순서

처음에는 전체 실험을 바로 돌리기보다, 작은 확인용 실행부터 하는 것이 좋습니다.

### 5.1. 데이터 구조 확인

```powershell
python -m src_fitness_pose.build_metadata
```

이 명령은 AI-Hub 폴더에서 사용할 수 있는 `-3d.json` 파일과 라벨 정보를
정리합니다.

### 5.2. 빠른 테스트 실행

```powershell
python -m src_fitness_pose.train_pipeline --smoke
```

`--smoke`는 전체 실험 전에 코드가 정상적으로 연결되어 있는지 확인하는
작은 테스트입니다. 시간이 오래 걸리는 전체 학습 전에 먼저 실행하는 것을
권장합니다.

### 5.3. 기본 전체 실험 실행

```powershell
python -m src_fitness_pose.train_pipeline
```

이 명령은 17개 맨몸운동 라벨에 대해 SVM, XGBoost, GNN, Transformer 모델을
학습하고 평가합니다.

### 5.4. 여러 random seed 반복 실험

```powershell
python -m src_fitness_pose.repeated_seeds --config configs_fitness_pose/bodyweight_17_tuned_all.json
```

한 번만 실험하면 우연히 잘 나온 결과인지 알기 어렵습니다. 그래서 여러 seed로
반복 실험을 수행해 평균과 표준편차를 확인합니다.

### 5.5. 오분류 분석

```powershell
python -m src_fitness_pose.analyze_misclassifications
```

모델이 어떤 운동을 어떤 운동으로 헷갈렸는지 분석합니다.

예를 들어 `푸시업`과 `니푸쉬업`, `크런치`와 `바이시클 크런치`처럼 자세가
비슷한 운동끼리 오분류가 자주 발생하는지 확인합니다.

### 5.6. 자세 그룹별 분석

```powershell
python -m src_fitness_pose.analyze_pose_groups
```

운동을 런지 계열, 누운 자세 코어 운동, 상지 지지 운동 등으로 묶어 어떤
운동군에서 모델이 강하거나 약한지 확인합니다.

### 5.7. 논문용 표와 그림 생성

```powershell
python -m src_fitness_pose.prepare_thesis_artifacts
```

논문에 넣기 좋은 형태로 표와 그림을 정리합니다.

생성 위치:

```text
outputs_fitness_pose/thesis_tables_figures/
```

### 5.8. 논문 docx에 표와 그림 첨부

```powershell
python -m src_fitness_pose.attach_results_to_thesis_docx `
  --source "관절 위치 정보 기반 운동 동작 분류 및 성능 및 오분류 패턴 비교 분석_논문포맷적용본.docx" `
  --output "관절 위치 정보 기반 운동 동작 분류 및 성능 및 오분류 패턴 비교 분석_논문포맷적용본.docx"
```

논문 문서에 결과 표와 그림을 자동으로 첨부합니다.

논문 포맷에 맞춘 본문 구조를 다시 생성하고 싶다면 다음 명령을 사용합니다.

```powershell
python -m src_fitness_pose.rewrite_thesis_format_docx `
  --source "관절 위치 정보 기반 운동 동작 분류 및 성능 및 오분류 패턴 비교 분석.docx" `
  --output "관절 위치 정보 기반 운동 동작 분류 및 성능 및 오분류 패턴 비교 분석_논문포맷적용본.docx"
```

표 스타일만 다시 적용하고 싶다면 다음 명령을 사용합니다.

```powershell
python -m src_fitness_pose.style_docx_tables "관절 위치 정보 기반 운동 동작 분류 및 성능 및 오분류 패턴 비교 분석_논문포맷적용본.docx"
```

## 6. 주요 결과 파일

실험 후 중요한 결과는 `outputs_fitness_pose/` 아래에 생성됩니다.

```text
outputs_fitness_pose/
  bodyweight_17_tuned_all_repeated_seeds/
    metrics_by_seed.csv
    metrics_mean_std.csv
    repeated_seed_model_comparison.png

  bodyweight_17_tuned_all_error_analysis/
    cross_model_misclassification_pairs.csv
    error_analysis_report.md

  bodyweight_17_tuned_all_pose_group_analysis/
    pose_group_performance_by_model.csv
    pose_group_model_ranking.csv
    pose_group_analysis.md

  thesis_tables_figures/
    tables/
    figures/
```

가장 자주 확인하는 파일은 다음과 같습니다.

| 파일 | 의미 |
|---|---|
| `metrics_mean_std.csv` | 여러 seed 반복 실험의 평균 성능 |
| `cross_model_misclassification_pairs.csv` | 모델들이 공통으로 헷갈린 운동 조합 |
| `pose_group_performance_by_model.csv` | 자세 그룹별 모델 성능 |
| `thesis_tables_figures/tables/` | 논문에 넣을 표 |
| `thesis_tables_figures/figures/` | 논문에 넣을 그림 |

## 7. 최종 실험 결과 요약

최종 반복 실험에서는 SVM과 XGBoost가 가장 안정적인 성능을 보였습니다.

| 모델 | Accuracy 평균 | Macro-F1 평균 |
|---|---:|---:|
| SVM | 약 0.8985 | 약 0.9044 |
| XGBoost | 약 0.8988 | 약 0.9006 |
| Transformer | 약 0.8777 | 약 0.8770 |
| GNN | 약 0.8462 | 약 0.8462 |

성능이 높게 나온 이유는 관절 좌표 기반 통계 특징이 운동 간 자세와 움직임
차이를 잘 표현했기 때문입니다. 다만 같은 운동 serial이 train, validation, test에
공통으로 존재하므로, 완전히 새로운 운동 변형에 대한 일반화 성능으로 단정하면
안 됩니다.

자세한 검증 내용은 다음 파일에 정리되어 있습니다.

```text
outputs_fitness_pose/burpee_metric_verification.md
FITNESS_POSE_PIPELINE.md
```

## 8. 직접 촬영 데이터와 비교하기

직접 노트북 웹캠과 MediaPipe로 촬영한 운동 데이터를 비교하기 위해, 5개 운동의
AI-Hub 기준 데이터셋을 따로 추출할 수 있습니다.

대상 운동:

- 버피 테스트
- 사이드 런지
- 크런치
- 푸시업
- 플랭크

실행 명령:

```powershell
python -m src_fitness_pose.export_reference_5_exercises
```

생성 위치:

```text
exports/aihub_reference_5_exercises/
```

이 기준 데이터셋은 “운동 종류 라벨 정답”입니다. 즉 직접 촬영한 동작이
AI-Hub의 버피 테스트, 사이드 런지, 크런치, 푸시업, 플랭크 중 어떤 운동과
가까운지 비교하는 데 사용할 수 있습니다.

단, 이것은 “올바른 자세/틀린 자세”를 판정하는 데이터셋은 아닙니다.

## 9. 초보자를 위한 용어 설명

| 용어 | 뜻 |
|---|---|
| JSON | 데이터를 저장하는 파일 형식 |
| `-3d.json` | 프레임별 3D 관절 좌표가 들어 있는 AI-Hub 파일 |
| frame | 영상의 한 장면 |
| sample | 이 프로젝트에서는 JSON 파일 하나, 즉 운동 수행 하나 |
| label | 운동 이름 정답값 |
| train | 모델이 학습하는 데이터 |
| validation | 하이퍼파라미터를 고르는 데 쓰는 데이터 |
| test | 최종 성능을 평가하는 데이터 |
| random seed | 실험을 반복할 때 사용하는 난수 시작값 |
| Accuracy | 전체 중 맞힌 비율 |
| Macro-F1 | 라벨별 F1을 똑같은 비중으로 평균낸 값 |
| Confusion Matrix | 어떤 운동을 어떤 운동으로 헷갈렸는지 보여주는 표 |

## 10. 참고

더 자세한 작업 기록과 논문 작성 과정은 아래 파일에 정리되어 있습니다.

```text
FITNESS_POSE_PIPELINE.md
```

프로젝트의 최종 논문 문서 파일은 루트 폴더의 다음 파일입니다.

```text
관절 위치 정보 기반 운동 동작 분류 및 성능 및 오분류 패턴 비교 분석_논문포맷적용본.docx
```
