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

### 2026-06-10 5개 운동 기준 데이터셋 추출

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

### 2026-06-10 자세 그룹명 수정

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

### 2026-06-11 버피 테스트 1.000 지표 검증

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

### 2026-06-11 버피 테스트 1.0000 근거 논문 첨부

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

### 2026-06-11 논문 표 스타일 정리

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

### 2026-06-15 논문 제출 서식 적용

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

### 2026-06-15 논문 docx 열림 오류 복구

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

### 2026-06-15 논문 포맷 파일 기반 재구성본 생성

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
