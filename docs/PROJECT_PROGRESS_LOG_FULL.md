# 공학논문 운동 동작 분류 프로젝트 통합 진행 로그

생성일: 2026-06-16

이 문서는 아래 두 로그 파일을 하나로 합친 통합본이다.

- `docs/PROJECT_PROGRESS_LOG_2026-06-04_to_2026-06-05.md`: 2026-06-04부터 2026-06-05까지의 상세 실험 진행 기록
- `FITNESS_POSE_PIPELINE.md`: AI-Hub 피트니스 자세 데이터셋 기반 파이프라인 정의, 결과 산출물, 논문 문서 수정 이력

원본 두 파일은 보존하고, 통합본에서는 읽기 편하도록 원본 제목 수준을 한 단계 낮춰 배치하였다.

---

## Part 1. 초기 상세 진행 기록

> 출처: `docs/PROJECT_PROGRESS_LOG_2026-06-04_to_2026-06-05.md`

### 공학논문 운동 동작 분류 프로젝트 진행 기록

기간: 2026-06-04 ~ 2026-06-05  
프로젝트 경로: `C:\Users\Ludorph\Documents\Exersice_Classification_Project`

이 문서는 사용자가 프로젝트 진행 내용을 확인하고, 이후 공학논문 작성 시 실험 방법과 진행 과정을 정리하는 데 사용할 수 있도록 작성한 기록이다.

---

### 1. 프로젝트의 목적

이 프로젝트의 목적은 AI-Hub 피트니스 자세 이미지 데이터셋의 관절 좌표 JSON 파일을 이용해 맨몸운동 동작을 분류하고, 여러 모델의 성능과 오분류 패턴을 비교하는 것이다.

최종 연구 방향은 다음과 같이 정리된다.

> AI-Hub 피트니스 자세 이미지 데이터셋의 관절 위치 정보를 이용하여 맨몸운동 17개 동작을 분류하고, XGBoost, SVM, GNN, Transformer 모델의 분류 성능과 주요 오분류 패턴을 비교한다.

---

### 2. 기존 프로젝트와 새 프로젝트 방향의 차이

기존 프로젝트에서는 `push_up`, `pull_up`, `squat` 세 가지 운동만 사용했다. 데이터도 CSV 형태였고, 기존 모델 구성은 XGBoost, LSTM, ST-GCN 중심이었다.

새 프로젝트에서는 AI-Hub의 `fitness_pose` 데이터셋을 사용한다. 이 데이터셋은 CSV가 아니라 JSON 파일 안에 여러 프레임의 관절 좌표가 들어 있다. 따라서 기존 코드를 그대로 덮어쓰기보다, 기존 코드는 보존하고 새 데이터셋 전용 파이프라인을 별도로 만드는 방식으로 진행했다.

새로 만든 주요 디렉터리는 다음과 같다.

```text
src_fitness_pose/
configs_fitness_pose/
outputs_fitness_pose/
```

기존 `src/`와 기존 `outputs/`는 유지했다.

---

### 3. 데이터셋 분석

사용한 데이터셋은 AI-Hub 피트니스 자세 이미지 데이터셋이다.

데이터셋 경로:

```text
dataset/fitness_pose
```

처음에는 폴더명이 한글로 되어 있었기 때문에 프로젝트 내 사용 편의를 위해 영어 폴더명으로 정리했다. 이후 JSON 라벨명은 폴더 이름만 보고 판단하지 않고, 아래 엑셀 파일을 기준으로 해석하기로 했다.

```text
dataset/fitness_pose/Document/fitness_pose_naming_rules/fitness_pose_naming_rules.xlsx
dataset/fitness_pose/Document/source_data_list_and_scenario/source_data_list_and_scenario.xlsx
```

JSON 파일 하나는 하나의 운동 수행 샘플로 보았다. 각 JSON 내부에는 여러 프레임이 있고, 각 프레임에는 관절별 `x`, `y`, `z` 좌표가 들어 있다.

---

### 4. 실험 범위 결정

데이터셋 전체에는 다양한 운동 종류가 있었지만, 실험 범위를 맨몸운동으로 제한했다.

맨몸운동으로 제한한 이유는 다음과 같다.

- 운동 종류의 범위가 명확하다.
- 17개 라벨이면 너무 작지도, 너무 크지도 않아 모델 비교에 적절하다.
- 푸시업/니푸쉬업, 크런치 계열, 런지 계열처럼 서로 비슷한 동작이 있어 오분류 패턴 분석에 적합하다.
- 기구 운동과 바벨/덤벨 운동까지 포함하면 라벨 수와 운동 환경 차이가 커져 논문 해석이 복잡해진다.

최종 사용 라벨은 총 17개이다.

| 번호 | 운동 라벨 |
|---:|---|
| 1 | 스탠딩 사이드 크런치 |
| 2 | 스탠딩 니업 |
| 3 | 버피 테스트 |
| 4 | 스텝 포워드 다이나믹 런지 |
| 5 | 스텝 백워드 다이나믹 런지 |
| 6 | 사이드 런지 |
| 7 | 크로스 런지 |
| 8 | 굿모닝 |
| 9 | 라잉 레그 레이즈 |
| 10 | 크런치 |
| 11 | 바이시클 크런치 |
| 12 | 시저크로스 |
| 13 | 힙쓰러스트 |
| 14 | 플랭크 |
| 15 | 푸시업 |
| 16 | 니푸쉬업 |
| 17 | Y - Exercise |

---

### 5. `-3d.json`만 사용한 이유

데이터셋에는 일반 `.json`과 `-3d.json` 파일이 함께 존재한다. 실험에서는 `-3d.json`만 사용했다.

이유는 다음과 같다.

- 일반 JSON과 `-3d.json`을 동시에 사용하면 같은 운동 수행이 중복 샘플처럼 들어갈 수 있다.
- 중복 샘플이 들어가면 학습 데이터와 테스트 데이터 사이에 데이터 누수가 발생할 수 있다.
- `-3d.json`은 3차원 관절 좌표를 포함하므로 관절 위치 기반 운동 분류에 적합하다.

따라서 최종 입력 데이터는 다음 폴더의 `-3d.json`으로 제한했다.

```text
dataset/fitness_pose/1.Training/labeling_data/bodyweight_labeling_new_220128
```

---

### 6. 메타데이터 생성

모델 학습 전에 모든 JSON 파일을 하나의 표로 정리했다. 이 표를 메타데이터라고 한다.

메타데이터에는 다음 정보가 들어간다.

| 항목 | 의미 |
|---|---|
| `sample_id` | JSON 샘플 고유 ID |
| `json_path` | 실제 JSON 파일 경로 |
| `session_id` | 촬영 세션 ID |
| `serial` | JSON 파일명에서 추출한 serial 번호 |
| `exercise_label` | 최종 운동 라벨 |
| `pose` | 선 자세, 누운 자세, 엎드린 자세 등 |
| `status_description` | 자세 상태 설명 |

주의할 점은 serial 번호가 곧 운동 라벨은 아니라는 것이다. 같은 번호라도 문서 기준에 따라 해석해야 하므로, 엑셀 파일을 기준으로 serial을 운동명으로 매핑했다.

메타데이터 생성 코드:

```text
src_fitness_pose/metadata.py
src_fitness_pose/build_metadata.py
```

생성 결과:

```text
outputs_fitness_pose/bodyweight_17_full/data/metadata.csv
outputs_fitness_pose/bodyweight_17_full/data/serial_label_mapping.csv
outputs_fitness_pose/bodyweight_17_full/data/dataset_summary.csv
```

---

### 7. Train/Validation/Test 분할

데이터는 단순히 JSON 파일 단위로 무작위 분할하지 않았다. 촬영 세션 단위로 분할했다.

예를 들어 다음 경로에서 `Day05_200925_F`를 하나의 촬영 세션으로 보았다.

```text
bodyweight_01/Day05_200925_F/D05-1-001-3d.json
```

세션 단위로 분할한 이유는 다음과 같다.

- 같은 촬영 세션의 비슷한 샘플이 학습과 테스트에 동시에 들어가는 것을 막기 위함이다.
- 데이터 누수를 줄이고 더 공정한 성능 평가를 하기 위함이다.
- 모든 모델이 같은 분할에서 평가되도록 하기 위함이다.

최종 기준 실험 분할은 다음과 같다.

| 구분 | 샘플 수 |
|---|---:|
| Train | 10,147 |
| Validation | 3,093 |
| Test | 3,168 |

각 분할에는 17개 운동 라벨이 모두 포함되도록 검증했다.

분할 코드:

```text
src_fitness_pose/splitting.py
```

생성 결과:

```text
outputs_fitness_pose/bodyweight_17_full/data/metadata_with_split.csv
outputs_fitness_pose/bodyweight_17_full/data/split_info.json
outputs_fitness_pose/bodyweight_17_full/data/split_sessions.csv
outputs_fitness_pose/bodyweight_17_full/data/split_label_distribution.csv
```

---

### 8. 관절 좌표 전처리

JSON에는 여러 프레임의 관절 좌표가 들어 있다. 이 좌표를 그대로 쓰면 사람의 위치, 크기, 촬영 거리 차이 때문에 모델이 운동 동작이 아니라 위치 차이를 학습할 수 있다.

따라서 다음 전처리를 적용했다.

1. 관절 순서를 고정했다.
2. 골반 중심 또는 허리 기준으로 좌표를 정규화했다.
3. 어깨너비와 골반너비를 이용해 신체 크기 차이를 보정했다.
4. 비정상적으로 작은 신체 크기 기준은 샘플 중앙값으로 대체했다.
5. 정규화 좌표의 지나친 이상치를 제한했다.
6. 각 JSON의 여러 프레임을 통계 특징으로 요약했다.

추출한 특징은 다음과 같다.

| 특징 | 의미 |
|---|---|
| mean | 평균 위치 |
| std | 위치 변화량 |
| min | 최솟값 |
| max | 최댓값 |
| range | 움직임 범위 |
| delta | 마지막 프레임과 첫 프레임 차이 |
| velocity_mean | 평균 이동량 |
| velocity_max | 최대 이동량 |

각 관절마다 `x`, `y`, `z` 축에 대해 위 특징을 계산했다.

전처리 코드:

```text
src_fitness_pose/features.py
```

생성 결과:

```text
outputs_fitness_pose/bodyweight_17_full/data/features_all.npz
outputs_fitness_pose/bodyweight_17_full/data/preprocessing.pkl
```

---

### 9. 빈 JSON 및 데이터 품질 처리

실제 실행 중 일부 JSON 파일에는 `frames`가 비어 있다는 문제가 발견됐다. 이런 파일은 모델 입력으로 사용할 수 없기 때문에 자동 제외했다.

| 항목 | 개수 |
|---|---:|
| 발견된 `-3d.json` | 16,423 |
| 빈 프레임 JSON | 15 |
| 최종 사용 샘플 | 16,408 |

빈 JSON 목록은 다음 파일에 저장했다.

```text
outputs_fitness_pose/bodyweight_17_full/data/feature_failures.csv
```

---

### 10. 사용한 모델

이번 실험에서는 네 개 모델을 비교했다.

| 모델 | 입력 방식 | 역할 |
|---|---|---|
| XGBoost | 관절 통계 특징을 1차원 벡터로 입력 | 강력한 트리 기반 기준 모델 |
| SVM | 표준화된 1차원 특징 벡터 입력 | 고전 머신러닝 비교 모델 |
| GNN | 관절을 노드로 보는 그래프 입력 | 관절 연결 구조 반영 |
| Transformer | 관절을 토큰으로 보는 입력 | 관절 간 관계를 attention으로 학습 |

모델 코드:

```text
src_fitness_pose/models.py
src_fitness_pose/train_pipeline.py
```

모델 하이퍼파라미터 기록:

```text
outputs_fitness_pose/bodyweight_17_full/model_hyperparameters.csv
outputs_fitness_pose/bodyweight_17_full/experiment_config_actual.json
```

---

### 11. 스모크 테스트

전체 실험 전에 작은 데이터로 파이프라인이 정상 동작하는지 확인했다. 이를 스모크 테스트라고 한다.

스모크 테스트 목적은 성능을 확인하는 것이 아니라, 코드가 처음부터 끝까지 정상 실행되는지 확인하는 것이다.

확인한 항목은 다음과 같다.

- 메타데이터 생성
- 라벨 매핑
- 특징 추출
- 세션 단위 분할
- XGBoost/SVM/GNN/Transformer 학습
- 예측 결과 저장
- 성능표 저장
- 혼동행렬 이미지 저장
- 오분류 조합 저장

실행 명령:

```powershell
& 'C:\Users\Ludorph\AppData\Local\Python\bin\python.exe' -m src_fitness_pose.train_pipeline --smoke
```

---

### 12. 기준 실험

스모크 테스트 후 전체 17개 라벨과 전체 유효 샘플을 사용해 기준 실험을 진행했다.

초기 실행에서는 일부 정규화 좌표가 지나치게 커지는 문제가 발견됐다. 이는 어깨너비 또는 골반너비가 거의 0에 가깝게 계산된 일부 프레임 때문이었다.

이 문제를 해결하기 위해 다음 보정을 추가했다.

- 비정상적으로 작은 scale 값은 샘플 중앙값으로 대체
- 정규화 좌표를 일정 범위로 제한
- 특징값이 유한한지 확인

보정 후 기준 실험을 다시 실행했다.

기준 실험 결과는 다음과 같다.

| 모델 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| SVM | 0.8842 | 0.9000 | 0.8818 |
| Transformer | 0.8864 | 0.8931 | 0.8878 |
| XGBoost | 0.8857 | 0.8922 | 0.8817 |
| GNN | 0.8065 | 0.8132 | 0.7969 |

기준 실험 결과 폴더:

```text
outputs_fitness_pose/bodyweight_17_full
```

---

### 13. 시각적 결과물

다음 시각적 결과물을 생성했다.

| 파일 | 의미 |
|---|---|
| `confusion_matrix_xgboost.png` | XGBoost 혼동행렬 |
| `confusion_matrix_svm.png` | SVM 혼동행렬 |
| `confusion_matrix_gnn.png` | GNN 혼동행렬 |
| `confusion_matrix_transformer.png` | Transformer 혼동행렬 |
| `model_comparison_overall.png` | 모델별 전체 성능 비교 |
| `model_comparison_per_class_f1.png` | 운동 라벨별 F1-score 비교 |

위 파일들은 다음 경로에 있다.

```text
outputs_fitness_pose/bodyweight_17_full/results
```

---

### 14. 비시각적 결과물

다음 CSV 및 Markdown 결과물을 생성했다.

| 파일 | 의미 |
|---|---|
| `metrics_summary.csv` | 모델별 Accuracy, Macro-F1, Weighted-F1 |
| `classification_report_*.csv` | 모델별 라벨 단위 precision, recall, F1 |
| `test_predictions_*.csv` | 테스트 샘플별 실제 라벨과 예측 라벨 |
| `misclassification_pairs_all_models.csv` | 전체 모델의 주요 오분류 조합 |
| `hard_cases_by_sample.csv` | 여러 모델이 동시에 틀린 어려운 샘플 |
| `analysis_summary.md` | 자동 실험 요약 |

---

### 15. 주요 오분류 패턴

기준 실험에서 주요 오분류는 다음과 같이 나타났다.

| 오분류 조합 | 해석 |
|---|---|
| 스텝 백워드 다이나믹 런지 → 스텝 포워드 다이나믹 런지 | 두 동작 모두 런지 계열이며 관절 위치 패턴이 유사하다. 방향성 차이가 통계 특징만으로는 약하게 표현될 수 있다. |
| 니푸쉬업 → 푸시업 | 상체 움직임이 유사하고 무릎 지지 여부가 핵심 차이다. |
| 푸시업 → 니푸쉬업 | 위와 같은 이유로 혼동이 발생한다. |
| 크런치 → 바이시클 크런치 | 누운 자세 기반 복부 운동으로 몸통 움직임이 유사하다. |
| 시저크로스 → 바이시클 크런치 | 누운 자세에서 다리 움직임이 포함된 동작이라는 점에서 유사하다. |

이 내용은 논문의 오분류 패턴 분석 절에 사용할 수 있다.

---

### 16. Random Seed 반복 실험

기준 실험은 한 번의 랜덤 조건에서 수행된 결과다. 실험 결과가 우연한 분할이나 초기값에 의존하지 않는지 확인하기 위해 random seed 반복 실험을 수행했다.

사용한 seed는 다음과 같다.

```text
42, 7, 21, 100, 2026
```

각 seed마다 같은 실험을 다시 실행했다. 즉, 같은 데이터셋, 같은 모델, 같은 전처리, 같은 평가 지표를 사용하되 랜덤 분할과 모델 초기값 조건을 다르게 했다.

반복 실험 자동화 코드:

```text
src_fitness_pose/repeated_seeds.py
```

실행 명령:

```powershell
& 'C:\Users\Ludorph\AppData\Local\Python\bin\python.exe' -m src_fitness_pose.repeated_seeds --seeds 42 7 21 100 2026
```

반복 실험 결과는 다음과 같다.

| 모델 | Accuracy 평균 ± 표준편차 | Macro-F1 평균 ± 표준편차 |
|---|---:|---:|
| SVM | 0.8985 ± 0.0151 | 0.9044 ± 0.0142 |
| XGBoost | 0.8974 ± 0.0138 | 0.8989 ± 0.0122 |
| Transformer | 0.8833 ± 0.0078 | 0.8856 ± 0.0107 |
| GNN | 0.8404 ± 0.0222 | 0.8441 ± 0.0210 |

반복 실험 결과 폴더:

```text
outputs_fitness_pose/bodyweight_17_full_repeated_seeds
```

주요 파일:

```text
metrics_by_seed.csv
metrics_mean_std.csv
repeated_seed_summary.md
repeated_seed_model_comparison.png
```

---

### 17. 반복 실험 결과 해석

5개 seed 반복 실험에서도 SVM과 XGBoost가 상위권 성능을 보였다.

Macro-F1 평균 기준 순위는 다음과 같다.

1. SVM
2. XGBoost
3. Transformer
4. GNN

SVM은 평균 Macro-F1이 가장 높았고, XGBoost는 SVM에 근접한 성능을 보였다. Transformer는 비교적 안정적인 성능을 보였지만 평균 성능은 SVM/XGBoost보다 낮았다. GNN은 관절 그래프 구조를 반영했지만 현재 특징 설계와 모델 구조에서는 상대적으로 낮은 성능을 보였다.

논문에서는 다음과 같이 해석할 수 있다.

> 반복 실험 결과 SVM과 XGBoost는 서로 다른 random seed 조건에서도 안정적으로 높은 성능을 보였으며, 특히 SVM은 Macro-F1 평균 기준 가장 우수한 성능을 나타냈다. 이는 본 연구에서 사용한 관절 통계 특징이 고전 머신러닝 모델에도 효과적으로 작용했음을 의미한다.

---

### 18. 논문에 사용할 수 있는 핵심 문장

아래 문장은 논문 실험 방법 절에 사용할 수 있다.

> 본 연구에서는 AI-Hub 피트니스 자세 이미지 데이터셋 중 맨몸운동 17개 라벨을 대상으로 운동 동작 분류 실험을 수행하였다. 각 샘플은 `-3d.json` 파일 하나로 정의하였으며, JSON 내부의 프레임별 관절 좌표를 정규화한 뒤 관절별 통계 특징으로 변환하였다. 데이터 분할은 동일 촬영 세션이 서로 다른 분할에 동시에 포함되지 않도록 세션 단위로 수행하였다. 이후 XGBoost, SVM, GNN, Transformer 모델을 동일한 Train/Validation/Test 분할에서 학습 및 평가하였다.

아래 문장은 반복 실험 설명에 사용할 수 있다.

> 실험 결과의 안정성을 확인하기 위해 5개의 random seed를 사용하여 반복 실험을 수행하였으며, 각 모델의 Accuracy, Macro-F1, Weighted-F1에 대해 평균과 표준편차를 함께 산출하였다.

아래 문장은 결과 분석 절에 사용할 수 있다.

> 반복 실험 결과 SVM은 Macro-F1 평균 0.9044로 가장 높은 성능을 보였으며, XGBoost는 0.8989로 근접한 성능을 나타냈다. Transformer는 0.8856, GNN은 0.8441의 Macro-F1을 기록하였다. 주요 오분류는 런지 계열, 푸시업 계열, 크런치 계열처럼 관절 움직임이 유사한 운동 사이에서 주로 발생하였다.

---

### 19. 현재 완료 상태

현재 완료된 작업은 다음과 같다.

- 새 데이터셋 구조 분석
- 맨몸운동 17개 라벨 확정
- `-3d.json` 기준 입력 데이터 선택
- 라벨 매핑 코드 구현
- 메타데이터 생성
- 세션 단위 데이터 분할
- 관절 좌표 전처리 및 특징 추출
- 빈 JSON 자동 제외
- XGBoost, SVM, GNN, Transformer 구현
- 기준 실험 실행
- 시각적 결과물 생성
- 비시각적 결과물 생성
- random seed 5회 반복 실험
- 평균 및 표준편차 결과 생성

---

### 20. 다음 단계

다음 단계는 소규모 하이퍼파라미터 튜닝이다.

우선 튜닝할 모델은 SVM과 XGBoost를 추천한다. 두 모델이 반복 실험에서 상위권 성능을 보였고, 튜닝 비용도 상대적으로 낮기 때문이다.

추천 순서는 다음과 같다.

1. SVM의 `C`, `gamma` 튜닝
2. XGBoost의 `max_depth`, `learning_rate`, `n_estimators` 튜닝
3. Validation 성능 기준으로 최적 설정 선택
4. 최적 설정으로 5 seed 반복 실험 재수행
5. 최종 논문용 성능 표 확정

주의할 점은 Test set으로 하이퍼파라미터를 고르면 안 된다는 것이다. 튜닝은 Validation set 기준으로 수행하고, 최종 선택된 설정만 Test set에서 평가해야 한다.

---

### 21. SVM/XGBoost 소규모 하이퍼파라미터 튜닝

반복 seed 실험 이후, 성능 상위권 모델인 SVM과 XGBoost를 대상으로 소규모 하이퍼파라미터 튜닝을 수행했다.

튜닝 목적은 다음과 같다.

- 기준 실험에서 사용한 설정이 적절한지 확인한다.
- Validation set 기준으로 더 나은 설정이 있는지 비교한다.
- Test set을 튜닝에 사용하지 않아 최종 평가의 공정성을 유지한다.

튜닝은 seed 42 기준의 Train/Validation 분할에서 수행했다. 즉, Train set으로 모델을 학습하고 Validation set의 Macro-F1을 기준으로 후보 설정을 비교했다. Test set은 하이퍼파라미터 선택 과정에 사용하지 않았다.

튜닝 코드:

```text
src_fitness_pose/tune_tabular_models.py
```

실행 명령:

```powershell
& 'C:\Users\Ludorph\AppData\Local\Python\bin\python.exe' -m src_fitness_pose.tune_tabular_models --models svm
& 'C:\Users\Ludorph\AppData\Local\Python\bin\python.exe' -m src_fitness_pose.tune_tabular_models --models xgboost
```

튜닝 결과 폴더:

```text
outputs_fitness_pose/bodyweight_17_full_tuning
```

주요 결과 파일:

```text
tuning_results.csv
best_hyperparameters_by_model.csv
best_hyperparameters_by_model.json
tuning_validation_macro_f1_svm.png
tuning_validation_macro_f1_xgboost.png
```

#### 21.1 SVM 튜닝

SVM은 RBF kernel을 유지하고, `C`와 `gamma` 값을 비교했다.

비교한 후보는 다음과 같다.

| 항목 | 후보 |
|---|---|
| `C` | 3.0, 10.0, 30.0 |
| `gamma` | `scale`, 0.01, 0.03 |
| `class_weight` | `balanced` |

SVM 튜닝 결과, Validation Macro-F1 기준 최고 설정은 다음과 같다.

| 항목 | 값 |
|---|---|
| `kernel` | `rbf` |
| `C` | 10.0 |
| `gamma` | `scale` |
| `class_weight` | `balanced` |
| Validation Accuracy | 0.8975 |
| Validation Macro-F1 | 0.8994 |
| Validation Weighted-F1 | 0.8977 |

이는 기준 실험에서 사용한 SVM 설정과 동일하다. 따라서 SVM은 현재 설정을 유지하는 것이 적절하다고 판단했다.

#### 21.2 XGBoost 튜닝

XGBoost는 기준 설정을 포함하여 트리 깊이, 학습률, 추정기 수, 샘플링 비율, 정규화 강도를 조금씩 바꾼 5개 후보를 비교했다.

비교한 대표 후보는 다음과 같다.

| 후보명 | 주요 설정 |
|---|---|
| `baseline_depth6_lr005_est500` | 기존 기준 설정 |
| `shallower_depth4_lr005_est500` | 트리 깊이 축소 |
| `faster_depth6_lr008_est300` | 학습률 증가, 추정기 수 감소 |
| `deeper_depth8_lr005_est300` | 트리 깊이 증가 |
| `regularized_depth5_lr005_est500` | 중간 깊이, 샘플링 축소, 정규화 강화 |

XGBoost 튜닝 결과, Validation Macro-F1 기준 최고 설정은 다음과 같다.

| 항목 | 값 |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | 5 |
| `learning_rate` | 0.05 |
| `subsample` | 0.85 |
| `colsample_bytree` | 0.85 |
| `reg_lambda` | 2.0 |
| Validation Accuracy | 0.8978 |
| Validation Macro-F1 | 0.8994 |
| Validation Weighted-F1 | 0.8968 |

기준 XGBoost 설정의 Validation Macro-F1은 0.8979였고, 튜닝 후 최고 설정은 0.8994였다. 따라서 XGBoost는 약간의 성능 향상이 확인되었으며, 이후 최종 실험에서는 튜닝된 XGBoost 설정을 적용하는 것이 적절하다.

#### 21.3 튜닝 결과 해석

SVM은 기존 설정이 이미 가장 좋은 후보로 확인되었다. XGBoost는 기존 설정보다 약간 더 규제된 설정이 Validation Macro-F1 기준으로 더 좋은 결과를 보였다.

이 결과는 다음과 같이 해석할 수 있다.

> SVM의 경우 기준 실험에서 사용한 `C=10`, `gamma=scale` 설정이 Validation Macro-F1 기준 가장 우수하여 추가 변경 없이 유지하였다. XGBoost의 경우 트리 깊이를 6에서 5로 낮추고, `subsample`과 `colsample_bytree`를 0.85로 낮추며, `reg_lambda`를 2.0으로 높인 설정이 가장 높은 Validation Macro-F1을 보였다. 이는 과도하게 복잡한 트리 구조보다 다소 규제된 설정이 본 데이터셋에서 더 안정적인 일반화 성능을 보였음을 의미한다.

#### 21.4 다음 단계

하이퍼파라미터 튜닝 이후에는 다음 순서로 진행하는 것이 적절하다.

1. SVM은 기존 설정 유지
2. XGBoost는 튜닝된 설정 적용
3. 선택된 설정으로 다시 5개 random seed 반복 실험 수행
4. 튜닝 전 결과와 튜닝 후 결과 비교
5. 최종 논문용 성능 표 확정

---

### 22. Transformer/GNN 소규모 하이퍼파라미터 튜닝 및 최종 반복 실험

SVM과 XGBoost만 튜닝할 경우 특정 모델군에 유리한 실험으로 보일 수 있으므로, Transformer와 GNN도 소규모 하이퍼파라미터 튜닝에 포함하였다. 튜닝은 기존과 동일하게 seed 42 기준 Train/Validation 분할에서 수행하였고, Test set은 하이퍼파라미터 선택 과정에 사용하지 않았다.

튜닝 기준은 Validation Macro-F1로 설정하였다. Macro-F1은 각 운동 라벨의 F1-score를 동일한 비중으로 평균낸 값이므로, 라벨별 표본 수 차이가 있는 다중 운동 분류 실험에서 모델의 전반적인 분류 균형을 확인하기에 적절하다.

튜닝 코드:

```text
src_fitness_pose/tune_deep_models.py
```

실행 명령:

```powershell
& 'C:\Users\Ludorph\AppData\Local\Python\bin\python.exe' -m src_fitness_pose.tune_deep_models --models gnn transformer --device auto
```

튜닝 결과 폴더:

```text
outputs_fitness_pose/bodyweight_17_full_deep_tuning
```

주요 결과 파일:

```text
tuning_results.csv
best_hyperparameters_by_model.csv
best_hyperparameters_by_model.json
tuning_validation_macro_f1_gnn.png
tuning_validation_macro_f1_transformer.png
```

#### 22.1 GNN 튜닝

GNN은 관절을 노드로 보고, 인체 골격 연결 관계를 그래프 구조로 반영하는 모델이다. 이번 튜닝에서는 hidden dimension, graph layer 수, dropout, learning rate를 소규모로 변경하였다.

비교한 후보는 다음과 같다.

| 후보 | hidden_dim | num_layers | dropout | learning_rate |
|---|---:|---:|---:|---:|
| baseline_hidden128_layers3_dropout030_lr001 | 128 | 3 | 0.3 | 0.0010 |
| wider_hidden192_layers3_dropout020_lr001 | 192 | 3 | 0.2 | 0.0010 |
| shallower_hidden128_layers2_dropout030_lr0007 | 128 | 2 | 0.3 | 0.0007 |

Validation Macro-F1 기준 최고 설정은 다음과 같다.

| 항목 | 값 |
|---|---:|
| hidden_dim | 128 |
| num_layers | 2 |
| dropout | 0.3 |
| learning_rate | 0.0007 |
| Validation Accuracy | 0.8493 |
| Validation Macro-F1 | 0.8495 |
| Validation Weighted-F1 | 0.8492 |

기존 GNN보다 layer 수를 줄인 설정이 더 좋은 결과를 보였다. 이는 본 실험에서 사용하는 입력 특징이 이미 프레임별 관절 좌표를 통계적으로 요약한 형태이기 때문에, 너무 깊은 그래프 구조가 항상 유리하지 않을 수 있음을 의미한다.

#### 22.2 Transformer 튜닝

Transformer는 관절을 token처럼 보고, 관절 간 관계를 attention으로 학습하는 모델이다. 이번 튜닝에서는 embedding 차원, layer 수, feedforward 차원, dropout, learning rate를 소규모로 변경하였다.

비교한 후보는 다음과 같다.

| 후보 | d_model | num_heads | num_layers | dim_feedforward | dropout | learning_rate |
|---|---:|---:|---:|---:|---:|---:|
| baseline_d128_heads4_layers3_ff256_dropout010_lr001 | 128 | 4 | 3 | 256 | 0.1 | 0.0010 |
| regularized_d128_heads4_layers3_ff256_dropout020_lr0007 | 128 | 4 | 3 | 256 | 0.2 | 0.0007 |
| ff512_d128_heads4_layers2_dropout010_lr001 | 128 | 4 | 2 | 512 | 0.1 | 0.0010 |

Validation Macro-F1 기준 최고 설정은 다음과 같다.

| 항목 | 값 |
|---|---:|
| d_model | 128 |
| num_heads | 4 |
| num_layers | 3 |
| dim_feedforward | 256 |
| dropout | 0.2 |
| learning_rate | 0.0007 |
| Validation Accuracy | 0.9001 |
| Validation Macro-F1 | 0.8954 |
| Validation Weighted-F1 | 0.8989 |

Transformer는 기존 구조를 유지하되 dropout을 높이고 learning rate를 낮춘 설정이 Validation set에서 가장 좋은 결과를 보였다. 이는 약간 더 강한 정규화와 완만한 학습률이 Validation 기준 일반화 성능에 도움이 되었음을 의미한다.

#### 22.3 최종 튜닝 설정 파일 생성

SVM, XGBoost, GNN, Transformer의 튜닝 결과를 하나의 설정 파일로 정리하였다.

최종 설정 파일:

```text
configs_fitness_pose/bodyweight_17_tuned_all.json
```

최종 적용 설정은 다음과 같다.

| 모델 | 최종 적용 설정 |
|---|---|
| SVM | 기존 설정 유지: `C=10`, `gamma=scale`, `kernel=rbf`, `class_weight=balanced` |
| XGBoost | `max_depth=5`, `n_estimators=500`, `learning_rate=0.05`, `subsample=0.85`, `colsample_bytree=0.85`, `reg_lambda=2.0` |
| GNN | `hidden_dim=128`, `num_layers=2`, `dropout=0.3`, `learning_rate=0.0007` |
| Transformer | `d_model=128`, `num_heads=4`, `num_layers=3`, `dim_feedforward=256`, `dropout=0.2`, `learning_rate=0.0007` |

설정 파일 검증을 위해 smoke run을 수행하였고, 네 모델 모두 정상적으로 학습 및 평가 단계까지 실행됨을 확인하였다.

Smoke run 명령:

```powershell
& 'C:\Users\Ludorph\AppData\Local\Python\bin\python.exe' -m src_fitness_pose.train_pipeline --config configs_fitness_pose\bodyweight_17_tuned_all.json --smoke --models xgboost svm gnn transformer --device auto
```

#### 22.4 튜닝된 네 모델의 최종 random seed 반복 실험

최종 튜닝 설정을 사용하여 다시 5개 random seed 반복 실험을 수행하였다.

사용한 seed:

```text
42, 7, 21, 100, 2026
```

실행 명령:

```powershell
& 'C:\Users\Ludorph\AppData\Local\Python\bin\python.exe' -m src_fitness_pose.repeated_seeds --config configs_fitness_pose\bodyweight_17_tuned_all.json --seeds 42 7 21 100 2026 --models xgboost svm gnn transformer --device auto
```

최종 반복 실험 결과 폴더:

```text
outputs_fitness_pose/bodyweight_17_tuned_all_repeated_seeds
```

최종 반복 실험 결과는 다음과 같다.

| 모델 | Accuracy 평균 ± 표준편차 | Macro-F1 평균 ± 표준편차 | Weighted-F1 평균 ± 표준편차 |
|---|---:|---:|---:|
| SVM | 0.8985 ± 0.0151 | 0.9044 ± 0.0142 | 0.8984 ± 0.0167 |
| XGBoost | 0.8988 ± 0.0134 | 0.9006 ± 0.0113 | 0.8971 ± 0.0141 |
| Transformer | 0.8777 ± 0.0112 | 0.8770 ± 0.0119 | 0.8774 ± 0.0122 |
| GNN | 0.8462 ± 0.0092 | 0.8462 ± 0.0113 | 0.8456 ± 0.0107 |

#### 22.5 튜닝 전후 비교

튜닝 전 반복 실험 결과와 튜닝 후 반복 실험 결과를 비교하면 다음과 같다.

| 모델 | 튜닝 전 Macro-F1 | 튜닝 후 Macro-F1 | 변화 |
|---|---:|---:|---:|
| SVM | 0.9044 | 0.9044 | 변화 없음 |
| XGBoost | 0.8989 | 0.9006 | +0.0017 |
| Transformer | 0.8856 | 0.8770 | -0.0086 |
| GNN | 0.8441 | 0.8462 | +0.0021 |

SVM은 튜닝 결과 기존 설정이 최적 후보였으므로 최종 반복 실험에서도 동일한 성능을 보였다. XGBoost는 튜닝 후 Macro-F1이 소폭 상승하였다. GNN 역시 평균 Macro-F1이 약간 상승하고 표준편차가 크게 감소하여, 튜닝 후 결과가 더 안정적인 경향을 보였다.

반면 Transformer는 Validation set 기준으로는 튜닝 설정이 가장 좋았지만, 5개 seed 반복 Test 결과에서는 기존 설정보다 평균 Macro-F1이 낮았다. 따라서 논문에서는 Transformer에 대해 단일 Validation 결과만으로 최종 성능을 판단하지 않고, 반복 seed 결과까지 함께 고려했다는 점을 명시하는 것이 적절하다.

최종 논문 결과 표에서는 튜닝 후 5-seed 반복 실험 결과를 중심으로 제시하되, Transformer의 경우 튜닝 전 설정이 반복 Test 평균에서는 더 높았다는 보조 분석을 함께 언급할 수 있다.

---

### 23. 오분류 패턴 심화 분석

최종 튜닝 설정으로 수행한 5개 random seed 반복 실험 결과를 모두 합쳐 오분류 패턴을 심화 분석하였다. 단일 seed 결과만 보면 특정 분할에서 우연히 발생한 오분류가 과도하게 해석될 수 있으므로, 이번 분석에서는 seed 42, 7, 21, 100, 2026의 Test prediction 결과를 모두 집계하였다.

분석 코드:

```text
src_fitness_pose/analyze_misclassifications.py
```

실행 명령:

```powershell
& 'C:\Users\Ludorph\AppData\Local\Python\bin\python.exe' -m src_fitness_pose.analyze_misclassifications
```

분석 결과 폴더:

```text
outputs_fitness_pose/bodyweight_17_tuned_all_error_analysis
```

생성된 주요 파일은 다음과 같다.

| 파일 | 내용 |
|---|---|
| `all_seed_test_predictions_long.csv` | 5개 seed, 4개 모델의 Test 예측 결과 전체 |
| `misclassification_pairs_by_model.csv` | 모델별 실제 라벨-예측 라벨 오분류 조합 전체 |
| `top10_misclassification_pairs_by_model.csv` | 모델별 상위 10개 주요 오분류 조합 |
| `label_error_summary_by_model.csv` | 모델별 라벨 단위 오분류율 |
| `group_misclassification_summary.csv` | 운동군 단위 오분류 집계 |
| `cross_model_misclassification_pairs.csv` | 4개 모델 전체에서 반복적으로 나타난 공통 오분류 조합 |
| `misclassification_deep_analysis.md` | 논문용 해석을 포함한 오분류 심화 분석 요약 |

#### 23.1 모델 공통 주요 오분류 조합

4개 모델 전체와 5개 seed를 합산했을 때 가장 많이 나타난 오분류 조합은 다음과 같다.

| 실제 라벨 | 예측 라벨 | 총 오분류 수 | 관련 모델 |
|---|---|---:|---|
| 스텝 백워드 다이나믹 런지 | 스텝 포워드 다이나믹 런지 | 1440 | GNN, SVM, Transformer, XGBoost |
| 니푸쉬업 | 푸시업 | 1340 | GNN, SVM, Transformer, XGBoost |
| 푸시업 | 니푸쉬업 | 813 | GNN, SVM, Transformer, XGBoost |
| 시저크로스 | 바이시클 크런치 | 608 | GNN, SVM, Transformer, XGBoost |
| 바이시클 크런치 | 크런치 | 552 | GNN, SVM, Transformer, XGBoost |
| 크런치 | 바이시클 크런치 | 504 | GNN, SVM, Transformer, XGBoost |
| 스탠딩 사이드 크런치 | 스탠딩 니업 | 275 | GNN, SVM, Transformer, XGBoost |
| 시저크로스 | 크런치 | 258 | GNN, SVM, Transformer, XGBoost |
| 바이시클 크런치 | 시저크로스 | 219 | GNN, SVM, Transformer, XGBoost |
| 스탠딩 사이드 크런치 | 사이드 런지 | 157 | GNN, SVM, Transformer, XGBoost |

주요 오분류는 특정 모델 하나에서만 나타난 것이 아니라 네 모델 모두에서 반복적으로 나타났다. 따라서 이 패턴은 단일 모델의 오류라기보다 데이터셋 내 동작 간 구조적 유사성에서 비롯된 결과로 해석할 수 있다.

#### 23.2 모델별 주요 오분류 특징

모델별 상위 오분류를 보면 공통적으로 다음 세 가지 유형이 두드러졌다.

1. 런지 계열 내부 오분류  
   `스텝 백워드 다이나믹 런지 -> 스텝 포워드 다이나믹 런지` 오분류가 가장 크게 나타났다. 두 동작은 모두 런지 계열이며 하체 관절의 굴곡/신전 패턴이 유사하다. 전후 방향 차이는 프레임 요약 통계 특징만으로는 약하게 표현될 수 있다.

2. 상지 지지 운동 내부 오분류  
   `니푸쉬업 -> 푸시업`, `푸시업 -> 니푸쉬업` 오분류가 반복적으로 발생하였다. 두 동작은 팔 굽힘과 상체 지지 구조가 거의 동일하며, 무릎 지지 여부가 핵심 차이다. 관절 좌표 통계 특징에서는 이 차이가 충분히 강하게 분리되지 않을 수 있다.

3. 누운 자세 코어 운동 내부 오분류  
   `시저크로스`, `바이시클 크런치`, `크런치`, `라잉 레그 레이즈` 사이의 오분류가 반복적으로 나타났다. 이들 동작은 누운 자세에서 몸통과 하체 관절을 사용하는 공통점이 있으며, 일부 프레임 구간에서는 관절 배치가 유사하게 나타날 수 있다.

#### 23.3 라벨별 취약 구간

라벨 단위 오분류율을 확인한 결과, 모델별로 취약한 라벨은 다음과 같이 나타났다.

| 모델 | 대표 취약 라벨 | 주요 오분류 방향 |
|---|---|---|
| GNN | 니푸쉬업, 푸시업, 바이시클 크런치, 스텝 백워드 다이나믹 런지 | 푸시업/니푸쉬업, 크런치 계열, 런지 전후 방향 |
| SVM | 스텝 백워드 다이나믹 런지, 바이시클 크런치, 니푸쉬업, 시저크로스 | 런지 전후 방향, 크런치 계열, 푸시업/니푸쉬업 |
| Transformer | 니푸쉬업, 바이시클 크런치, 푸시업, 스텝 백워드 다이나믹 런지 | 푸시업/니푸쉬업, 크런치 계열, 런지 전후 방향 |
| XGBoost | 바이시클 크런치, 니푸쉬업, 스텝 백워드 다이나믹 런지, 푸시업 | 크런치 계열, 푸시업/니푸쉬업, 런지 전후 방향 |

모델별 세부 순위에는 차이가 있으나, 취약한 라벨군 자체는 대체로 일관되었다. 이는 각 모델이 서로 다른 학습 구조를 갖고 있더라도, 관절 좌표 기반 입력에서 구분이 어려운 운동 조합이 공통적으로 존재함을 의미한다.

#### 23.4 운동군 단위 오분류 경향

운동군 단위로 보면 같은 운동군 내부의 오분류가 크게 나타났다.

대표적으로 GNN에서는 상지 지지/엎드린 자세 운동 내부 오분류가 818건, 누운 자세 코어 운동 내부 오분류가 739건, 런지 계열 내부 오분류가 491건으로 집계되었다. SVM, XGBoost, Transformer에서도 유사하게 누운 자세 코어 운동, 런지 계열, 상지 지지 운동 내부 오분류가 주요하게 나타났다.

따라서 본 실험의 오분류는 전혀 다른 운동군 간의 무작위 혼동보다는, 자세와 관절 움직임이 유사한 운동군 내부에서 주로 발생했다고 볼 수 있다.

#### 23.5 논문용 해석

오분류 패턴은 다음과 같이 정리할 수 있다.

> 오분류 분석 결과, 주요 오분류는 무작위적인 라벨 간 혼동보다는 동작 구조가 유사한 운동군 내부에서 주로 발생하였다. 특히 스텝 포워드/백워드 다이나믹 런지, 푸시업/니푸쉬업, 크런치 계열 운동에서 반복적인 오분류가 확인되었다. 이는 관절 좌표 기반 통계 특징이 전체적인 자세와 움직임 패턴을 효과적으로 반영하는 반면, 운동 방향이나 지지 방식처럼 세밀한 차이를 구분하는 데에는 한계가 있음을 시사한다.

이번 단계로 오분류 패턴 심화 분석은 완료되었다. 다음 단계는 운동 라벨을 자세 또는 동작 특성별 그룹으로 나누어 모델 성능을 그룹 단위로 비교하는 자세 그룹별 분석이다.

---

### 24. 자세 그룹별 분석

오분류 패턴 심화 분석 이후, 17개 운동 라벨을 자세와 동작 특성에 따라 6개 운동군으로 묶고 모델별 성능을 그룹 단위로 비교하였다. 이 분석은 개별 라벨 단위의 성능뿐 아니라, 어떤 자세군 또는 동작군에서 모델이 강하거나 약한지를 확인하기 위한 것이다.

분석에는 최종 튜닝 설정으로 수행한 5개 random seed Test 예측 결과를 모두 사용하였다.

분석 코드:

```text
src_fitness_pose/analyze_pose_groups.py
```

실행 명령:

```powershell
& 'C:\Users\Ludorph\AppData\Local\Python\bin\python.exe' -m src_fitness_pose.analyze_pose_groups
```

분석 결과 폴더:

```text
outputs_fitness_pose/bodyweight_17_tuned_all_pose_group_analysis
```

생성된 주요 파일은 다음과 같다.

| 파일 | 내용 |
|---|---|
| `pose_group_analysis.md` | 자세 그룹별 분석 요약 및 논문용 해석 |
| `pose_group_performance_by_model.csv` | 모델별, 운동군별 Accuracy/Macro-F1 |
| `pose_group_model_ranking.csv` | 운동군별 모델 성능 순위 |
| `pose_group_prediction_flow.csv` | 실제 운동군이 어떤 운동군으로 예측되었는지의 흐름 |
| `pose_group_accuracy_by_model.png` | 운동군별 Accuracy 비교 그래프 |
| `pose_group_macro_f1_by_model.png` | 운동군별 Macro-F1 비교 그래프 |
| `same_group_prediction_rate_heatmap.png` | 같은 운동군으로 예측된 비율 heatmap |

#### 24.1 운동군 정의

운동 라벨은 다음과 같이 6개 운동군으로 묶었다.

| 운동군 | 포함 라벨 |
|---|---|
| 서서 수행하는 코어 운동 | 스탠딩 사이드 크런치, 스탠딩 니업 |
| 전신 복합 운동 | 버피 테스트 |
| 런지 계열 운동 | 스텝 포워드 다이나믹 런지, 스텝 백워드 다이나믹 런지, 사이드 런지, 크로스 런지 |
| 고관절 힌지 운동 | 굿모닝 |
| 누운 자세 코어 운동 | 라잉 레그 레이즈, 크런치, 바이시클 크런치, 시저크로스, 힙쓰러스트 |
| 상지 지지/엎드린 자세 운동 | 플랭크, 푸시업, 니푸쉬업, Y - Exercise |

#### 24.2 운동군별 최고 모델

각 운동군에서 Macro-F1 기준 가장 높은 성능을 보인 모델은 다음과 같다.

| 운동군 | 최고 모델 | Macro-F1 | Accuracy |
|---|---|---:|---:|
| 고관절 힌지 운동 | SVM | 0.9955 | 0.9910 |
| 런지 계열 운동 | Transformer | 0.9184 | 0.9134 |
| 누운 자세 코어 운동 | SVM | 0.8722 | 0.8691 |
| 서서 수행하는 코어 운동 | XGBoost | 0.9671 | 0.9674 |
| 상지 지지/엎드린 자세 운동 | SVM | 0.8755 | 0.8393 |
| 전신 복합 운동 | GNN | 1.0000 | 1.0000 |

전신 복합 운동은 `버피 테스트` 단일 라벨로 구성되어 있으며, 고관절 힌지 운동도 `굿모닝` 단일 라벨로 구성되어 있다. 따라서 이 두 운동군의 높은 성능은 운동군 내부 세부 라벨 구분이 쉽다는 의미라기보다, 해당 라벨이 다른 운동군과 비교적 뚜렷하게 구분되었다는 의미로 해석해야 한다.

#### 24.3 모델별 운동군 성능

Macro-F1 기준 모델별 운동군 성능은 다음과 같다.

| 운동군 | GNN | SVM | Transformer | XGBoost |
|---|---:|---:|---:|---:|
| 고관절 힌지 운동 | 0.9531 | 0.9955 | 0.9482 | 0.9645 |
| 누운 자세 코어 운동 | 0.8329 | 0.8722 | 0.8548 | 0.8719 |
| 런지 계열 운동 | 0.8846 | 0.9068 | 0.9184 | 0.9151 |
| 상지 지지/엎드린 자세 운동 | 0.7596 | 0.8755 | 0.8057 | 0.8540 |
| 서서 수행하는 코어 운동 | 0.9175 | 0.9587 | 0.9419 | 0.9671 |
| 전신 복합 운동 | 1.0000 | 1.0000 | 1.0000 | 0.9986 |

전체적으로 SVM과 XGBoost가 여러 운동군에서 안정적인 성능을 보였으며, Transformer는 런지 계열에서 가장 높은 Macro-F1을 보였다. GNN은 전신 복합 운동에서는 높은 성능을 보였으나, 상지 지지/엎드린 자세 운동과 누운 자세 코어 운동에서는 상대적으로 낮은 성능을 보였다.

#### 24.4 자세 그룹별 해석

자세 그룹별 분석 결과는 오분류 패턴 심화 분석과 일관되었다. 런지 계열, 누운 자세 코어 운동, 상지 지지/엎드린 자세 운동은 같은 운동군 내부에 세부 동작이 유사한 라벨이 여러 개 포함되어 있어 상대적으로 분류가 어려웠다.

특히 상지 지지/엎드린 자세 운동에는 `푸시업`, `니푸쉬업`, `플랭크`, `Y - Exercise`가 포함된다. 이 운동군은 손 또는 팔을 바닥에 지지하는 자세가 공통적으로 나타나며, 푸시업과 니푸쉬업처럼 무릎 지지 여부만 다른 동작이 포함되어 있어 모델 간 성능 차이가 크게 나타났다.

누운 자세 코어 운동 역시 `크런치`, `바이시클 크런치`, `시저크로스`, `라잉 레그 레이즈`, `힙쓰러스트`가 포함되어 있다. 이 운동군은 모두 누운 자세에서 몸통 또는 하체 관절 움직임을 포함하므로, 프레임 요약 통계 특징만으로는 일부 세부 동작을 구분하기 어려울 수 있다.

런지 계열에서는 Transformer가 가장 높은 Macro-F1을 보였다. 이는 관절 간 관계를 attention 구조로 학습하는 방식이 하체 관절의 위치 관계와 방향성 차이를 일부 포착하는 데 도움이 되었을 가능성을 시사한다. 다만 전체 평균 성능에서는 SVM과 XGBoost가 더 안정적인 결과를 보였으므로, 특정 운동군에서의 강점과 전체 성능은 구분하여 해석해야 한다.

#### 24.5 논문용 해석

자세 그룹별 분석은 다음과 같이 정리할 수 있다.

> 자세 그룹별 분석 결과, 모델 성능은 운동군에 따라 차이를 보였다. 런지 계열, 누운 자세 코어 운동, 상지 지지/엎드린 자세 운동은 세부 라벨 간 관절 움직임이 유사하여 상대적으로 오분류가 많이 발생하였다. 반면 전신 복합 운동이나 고관절 힌지 운동처럼 다른 운동군과 자세 구조가 비교적 뚜렷한 경우에는 높은 분류 성능을 보였다. 이는 관절 좌표 기반 특징이 큰 자세군의 차이는 효과적으로 구분하지만, 같은 자세군 내부의 세밀한 운동 차이를 구분하는 데에는 한계가 있음을 보여준다.

이번 단계로 자세 그룹별 분석은 완료되었다. 다음 단계는 논문에 첨부할 최종 표와 그림을 확정하는 작업이다.

---

### 25. 논문용 표·그림 확정

기준 실험, 반복 seed 실험, 하이퍼파라미터 튜닝, 최종 모델 재실험, 오분류 패턴 심화 분석, 자세 그룹별 분석 결과를 바탕으로 논문에 첨부할 표와 그림을 선별하였다. 원본 결과 파일은 그대로 유지하고, 논문 작성 시 바로 확인할 수 있도록 별도 폴더에 표와 그림을 모아 정리하였다.

정리 코드:

```text
src_fitness_pose/prepare_thesis_artifacts.py
```

실행 명령:

```powershell
& 'C:\Users\Ludorph\AppData\Local\Python\bin\python.exe' -m src_fitness_pose.prepare_thesis_artifacts
```

논문용 표·그림 정리 폴더:

```text
outputs_fitness_pose/thesis_tables_figures
```

#### 25.1 본문 우선 첨부 표

논문 본문에는 다음 표를 우선 첨부하는 것이 적절하다.

| 표 파일 | 첨부 위치 | 내용 |
|---|---|---|
| `table_01_dataset_label_summary.md` | 데이터셋 설명 | 17개 운동 라벨별 샘플 수와 세션 수 |
| `table_02_split_overall.md` | 실험 설정 | Train/Validation/Test 전체 분할 |
| `table_04_final_model_hyperparameters.md` | 실험 방법 | 최종 모델 구조 및 하이퍼파라미터 |
| `table_05_final_repeated_seed_performance.md` | 실험 결과 | 최종 5-seed 평균 성능 비교 |
| `table_07_common_misclassification_patterns.md` | 오분류 분석 | 4개 모델 공통 주요 오분류 조합 |
| `table_08_pose_group_best_models.md` | 자세 그룹별 분석 | 운동군별 최고 모델 |

위 표들은 다음 폴더에 CSV와 Markdown 형식으로 함께 저장하였다.

```text
outputs_fitness_pose/thesis_tables_figures/tables
```

#### 25.2 본문 우선 첨부 그림

논문 본문에는 다음 그림을 우선 첨부하는 것이 적절하다.

| 그림 파일 | 첨부 위치 | 내용 |
|---|---|---|
| `figure_01_final_repeated_seed_model_comparison.png` | 실험 결과 | 최종 5-seed 평균 성능과 표준편차 비교 |
| `figure_02_pose_group_macro_f1_by_model.png` | 자세 그룹별 분석 | 운동군별 모델 Macro-F1 비교 |
| `figure_04_representative_confusion_matrix_svm_seed42.png` | 오분류 분석 | 최고 성능 모델인 SVM의 대표 혼동행렬 |

위 그림들은 다음 폴더에 복사하였다.

```text
outputs_fitness_pose/thesis_tables_figures/figures
```

#### 25.3 부록 또는 보조 자료 권장

본문에 모두 넣으면 분량이 길어질 수 있으므로 다음 자료는 부록 또는 보조 분석 자료로 두는 것이 적절하다.

| 파일 | 권장 위치 | 이유 |
|---|---|---|
| `table_03_split_by_label.md` | 부록 | 라벨별 Train/Validation/Test 분할 상세 |
| `table_06_tuning_before_after_macro_f1.md` | 부록 또는 실험 방법 보조 | 튜닝 전후 Macro-F1 변화 |
| `table_09_pose_group_macro_f1_by_model.md` | 부록 | 운동군별 모델 성능 상세 |
| `figure_03_same_group_prediction_rate_heatmap.png` | 부록 또는 오분류 분석 보조 | 같은 운동군으로 예측된 비율 |
| `figure_05_representative_confusion_matrix_xgboost_seed42.png` | 부록 | XGBoost 대표 혼동행렬 |
| `figure_06_representative_per_class_f1_seed42.png` | 부록 | 라벨별 F1-score 상세 |

#### 25.4 최종 성능 표

논문에서 최종 성능을 제시할 때는 단일 seed 결과가 아니라 5개 random seed 반복 실험의 평균과 표준편차를 사용한다.

| 모델 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| SVM | 0.8985 +/- 0.0151 | 0.9044 +/- 0.0142 | 0.8984 +/- 0.0167 |
| XGBoost | 0.8988 +/- 0.0134 | 0.9006 +/- 0.0113 | 0.8971 +/- 0.0141 |
| Transformer | 0.8777 +/- 0.0112 | 0.8770 +/- 0.0119 | 0.8774 +/- 0.0122 |
| GNN | 0.8462 +/- 0.0092 | 0.8462 +/- 0.0113 | 0.8456 +/- 0.0107 |

최종 성능 기준으로는 SVM이 가장 높은 Macro-F1을 보였고, XGBoost가 근접한 성능을 보였다. Transformer와 GNN은 관절 간 관계를 반영하는 모델이지만, 본 실험의 통계 요약 특징 기반 입력에서는 SVM/XGBoost보다 낮은 평균 성능을 보였다.

#### 25.5 주의 사항

혼동행렬과 라벨별 F1-score 그림은 대표 seed 42 결과이다. 반면 최종 성능 표는 5개 seed 평균 결과이다. 따라서 논문 본문에서는 최종 성능 수치를 해석할 때 `table_05_final_repeated_seed_performance`를 기준으로 삼고, 혼동행렬은 오분류 양상을 시각적으로 설명하는 보조 그림으로 사용해야 한다.

이번 단계로 논문용 표·그림 확정 작업은 완료되었다. 다음 단계는 정리된 표와 그림을 바탕으로 논문 본문을 작성하거나 기존 논문 문서에 반영하는 작업이다.

---

## Part 2. 파이프라인 정의 및 후속 변경 로그

> 출처: `FITNESS_POSE_PIPELINE.md`

### AI-Hub 맨몸운동 17개 라벨 모델 비교 파이프라인

기존 `src/`와 `outputs/`는 보존한다. 새 실험은 `src_fitness_pose/`,
`configs_fitness_pose/`, `outputs_fitness_pose/`를 사용한다.

### 실험 정의

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

### 실행

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

### 주요 결과 파일

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

### 논문 문서 산출물

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

#### 2026-06-10 방법론 보강 피드백 및 반영 기록

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

#### 2026-06-10 5개 운동 기준 데이터셋 추출

사용자 요청:

> MediaPipe로 버피테스트, 사이드런지, 크런치, 푸쉬업, 플랭크 5가지 운동 영상을
> 촬영하려고 하며, 직접 촬영 데이터와 비교할 기준 정답 데이터셋이 필요함.

반영 결과:

- AI-Hub 피트니스 자세 데이터셋에서 5개 운동 라벨의 유효 `-3d.json` 파일만
  추출하여 기준 데이터셋을 생성하였다.
- 출력 폴더: `exports/aihub_reference_5_exercises`
- 포함 파일:
  - `json/`: 라벨별로 복사한 원본 `-3d.json`
  - `metadata.csv`: reference_id, 원본 경로, 복사 경로, 라벨, 세션, split, frame 수
  - `label_summary.csv`: 라벨별 샘플 수와 세션 수
  - `features_reference.npz`: 기존 실험과 동일한 특징 추출 방식의 관절별 특징 텐서
  - `features_reference.csv`: 직접 비교 또는 확인용 flatten 특징 벡터
  - `manifest.json`, `README.md`

라벨별 기준 샘플 수:

| 라벨 | 영문 폴더명 | 유효 JSON |
|---|---|---:|
| 버피 테스트 | `burpee_test` | 1,604 |
| 사이드 런지 | `side_lunge` | 1,402 |
| 크런치 | `crunch` | 1,207 |
| 푸시업 | `pushup` | 1,011 |
| 플랭크 | `plank` | 256 |

검수 결과:

- `metadata.csv` 행 수: 5,480
- 복사된 JSON 수: 5,480
- `features_reference.npz`의 `node_features` shape: `(5480, 24, 24)`
- 이 기준 데이터셋은 운동 종류 라벨 정답 데이터셋이며, 올바른 자세/틀린 자세
  판정 정답 데이터셋은 아니다.

#### 2026-06-10 자세 그룹명 수정

사용자 요청:

> 결과 중 `전신 복합 운동`은 실제 17개 운동 라벨이 아니므로, 해당 표시명을
> `버피 테스트`로 바꾸고 싶음.

반영 결과:

- 자세 그룹 분석에서 `whole_body`의 표시명을 `전신 복합 운동`에서
  `버피 테스트`로 변경하였다.
- 자세 그룹 분석 결과, 논문용 표 8/표 9, 자세 그룹 관련 그림, 최종 docx를
  재생성하였다.
- 검수 결과 최종 문서와 논문용 표에서 `전신 복합 운동` 문자열은 제거되었고,
  해당 위치는 `버피 테스트`로 표시된다.

#### 2026-06-11 버피 테스트 1.000 지표 검증

사용자 요청:

> `버피 테스트` 지표가 1.000으로 나온 이유를 구체적으로 확인하고, 전체적으로
> 지표가 높게 나온 이유도 검증해달라는 요청.

검증 결과:

- 검증 보고서: `outputs_fitness_pose/burpee_metric_verification.md`
- `버피 테스트` 그룹은 단일 라벨 그룹이므로, 그룹 지표 1.000은 사실상
  실제 버피 테스트 test 샘플을 버피 테스트로 맞힌 조건부 성능이다.
- support 1440은 test 버피 샘플 288개 x 5 seeds로 계산된다.
- SVM, GNN, Transformer는 5개 seed의 버피 테스트 샘플을 모두 맞혔다.
- XGBoost는 같은 버피 테스트 샘플 1개를 4개 seed에서 `바이시클 크런치`로
  오분류하였다.
- train/test `sample_id` 중복, `session_id` 중복, 완전 동일 feature vector 중복은
  확인되지 않았다.
- 다만 각 운동의 serial 범위가 train/validation/test에 모두 존재하므로,
  완전히 새로운 운동 변형 평가라기보다는 세션 분리 평가로 해석해야 한다.

#### 2026-06-11 버피 테스트 1.0000 근거 논문 첨부

사용자 요청:

> 논문에 버피 테스트가 1.0000이 나온 이유에 대한 명확한 근거를 첨부하고,
> 논문 독자가 충분히 신뢰할 수 있도록 해달라는 요청.

반영 결과:

- 논문용 표 `table_10_burpee_metric_validation.csv/md`를 생성하였다.
- 최종 docx에 `표 10. 버피 테스트 1.0000 지표 검증`을 추가하였다.
- 표 10에는 support 계산 근거, 모델별 정답 수, XGBoost 오분류 샘플,
  직접 누수 점검 결과, serial 중복에 따른 해석상 한계를 포함하였다.
- 표 10 뒤에 버피 테스트 1.0000을 해석하는 본문 문단을 추가하였다.
- 최종 문서 검수 결과: 표 10개, 그림 8개, docx 구조 정상.

#### 2026-06-11 논문 표 스타일 정리

사용자 요청:

> 논문 문서의 모든 표에 테두리를 적용하고, 표의 첫 번째 행은 다른 색으로 표시해
> 가시성을 높여달라는 요청.

반영 결과:

- 최종 docx `공학논문연구_실험결과_AIHub_표그림첨부본_검수수정본.docx`의
  모든 표에 검은색 단선 테두리를 적용하였다.
- 모든 표의 첫 번째 행에 연한 파란색 배경색을 적용하였다.
- 이후 표를 다시 생성할 때도 같은 스타일이 적용되도록 `attach_results_to_thesis_docx.py`의
  표 생성 로직을 수정하였다.
- 별도 스타일 적용 스크립트 `src_fitness_pose/style_docx_tables.py`를 추가하였다.
- 검수 결과: 표 10개 모두 테두리 적용, 표 10개 모두 첫 행 음영 적용.

#### 2026-06-15 논문 제출 서식 적용

사용자 요청:

> 논문 작성 제약조건에 따라 글씨체는 Times New Roman, 제목은 18pt,
> 저자명/소속/본문은 10pt, abstract/keyword/참고문헌은 9pt로 맞춰달라는 요청.

반영 결과:

- 대상 파일: `관절 위치 정보 기반 운동 동작 분류 및 성능 및 오분류 패턴 비교 분석.docx`
- 전체 문서 run 글꼴을 Times New Roman으로 적용하였다.
- 첫 번째 제목 문단은 18pt로 적용하였다.
- 저자명과 일반 본문, 표, 그림 캡션은 10pt로 적용하였다.
- 논문 키워드와 참고문헌 영역은 9pt로 적용하였다.
- 서식 적용 스크립트 `src_fitness_pose/format_thesis_docx.py`를 추가하였다.
- 검수 결과: docx 구조 정상, 표 10개와 그림 8개 유지, 주요 문단 크기 적용 확인.

#### 2026-06-15 논문 docx 열림 오류 복구

사용자 요청:

> 서식 적용 후 논문 문서 파일 open 시 오류가 난다는 보고.

반영 결과:

- 최종 문서를 `rewrite_thesis_docx.py`와 `attach_results_to_thesis_docx.py` 기반으로
  다시 생성하여 docx 내부 구조를 복구하였다.
- 제목은 `관절 위치 정보 기반 운동 동작 분류 및 성능 및 오분류 패턴 비교 분석`으로
  고정하였다.
- 별도 후처리 서식 스크립트가 아니라 문서 생성 단계에서 Times New Roman과
  글자 크기 조건이 적용되도록 생성 로직을 수정하였다.
- 검수 결과: zip 구조 정상, XML 파싱 정상, 표 10개, 그림 8개, 표 테두리 및
  첫 행 음영 유지, 제목 18pt와 본문/참고문헌 크기 적용 확인.

#### 2026-06-15 논문 포맷 파일 기반 재구성본 생성

사용자 요청:

> `공학논문연구_논문포맷.docx`는 이번 논문의 포맷이므로, 기존 논문 원본을 직접 수정하지 말고
> 복사본을 만든 뒤 이전 제약조건과 해당 포맷을 참고하여 논문을 재구성해달라는 요청.

반영 결과:

- 원본 문서 `관절 위치 정보 기반 운동 동작 분류 및 성능 및 오분류 패턴 비교 분석.docx`는 유지하였다.
- 복사본 성격의 새 문서 `관절 위치 정보 기반 운동 동작 분류 및 성능 및 오분류 패턴 비교 분석_논문포맷적용본.docx`를 생성하였다.
- 제공된 포맷의 흐름에 맞춰 제목, 저자, 소속, Abstract, Keyword, `1. Introduction`,
  `2. Data`, `3. Methodology`, `4. Numerical results`, `5. Conclusions`, `6. References`
  순서로 논문을 재구성하였다.
- 제목은 18pt, 저자/소속/본문은 10pt, Abstract/Keyword/References는 9pt로 작성하였다.
- 실험 결과 표 10개와 그림 8개를 새 포맷의 `4. Numerical results` 위치에 다시 첨부하였다.
- 결과 절 내부 장 번호를 새 포맷에 맞춰 `4.1`부터 `4.5`까지로 정리하였다.
- Word XML 검사 결과: zip 구조 정상, XML 파싱 정상, 표 10개, 그림 8개, 삽입용 placeholder 미잔류,
  이전 프로젝트 용어인 `LSTM`, `ST-GCN`, `pull_up` 미포함을 확인하였다.

#### 2026-06-15 논문 포맷 적용본 검수 수정 및 README 업데이트

사용자 요청:

> 논문포맷적용본 검수 사항과 README 업데이트 필요 항목을 각각 반영해달라는 요청.

반영 결과:

- `4. Numerical results` 제목은 그대로 유지하고, 결과 절 도입부의 중복 문장을 제거하였다.
- Abstract를 더 짧게 정리하고, 문장 끝 어미가 모두 `하였다`로 반복되지 않도록 수정하였다.
- `2. Data` 절에 맨몸운동 17개 실험 라벨 목록을 추가하였다.
- `Y - Exercise`는 데이터셋 원 라벨명이므로 영문 그대로 사용한다고 명시하였다.
- 방법론 수식을 일반 텍스트 문단이 아니라 Word 수식 XML 객체로 삽입하도록 변경하였다.
- Accuracy를 다중 클래스 분류 기준으로 설명하고, `Accuracy = (1 / N) sum I(y_i = y_hat_i)` 형태로 수정하였다.
- SVM/XGBoost와 GNN/Transformer의 입력 형식 차이가 최종 결과 해석과 연결되도록 문단을 보강하였다.
- README의 최종 논문 파일명을 최신 `관절 위치 정보 기반 운동 동작 분류 및 성능 및 오분류 패턴 비교 분석_논문포맷적용본.docx`로 수정하였다.
- README의 `attach_results_to_thesis_docx`, `style_docx_tables` 실행 예시를 최신 파일명 기준으로 수정하였다.
- README에 `outputs_fitness_pose/`가 실험 결과, 반복 seed 결과, 튜닝 결과, 오분류 분석, 자세 그룹 분석, 논문용 표·그림 저장 폴더임을 보강하였다.
- 검수 결과: docx zip 구조 정상, 표 10개, 그림 8개, Word 수식 객체 20개, 이전 프로젝트 용어 미포함을 확인하였다.
